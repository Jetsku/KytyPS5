#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/CallSplitPlan.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include "gpu_mesh_shaders/gpu_call_bin_spv.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fmt/format.h>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>

// KYTY_SHADER_CALLS_SPLIT (shader/recompiler/CallSplit.h): a dispatch of a kernel with call loops
// becomes
//   R_0 (the dispatch's own workgroups) -> per site k: bin_k -> C_k,t (indirect, one per callee
//   slot) -> R_ak (indirect)
// over one call-state buffer per kernel: the wave header words, the wave blocks, and per site the
// indirect commands and wave lists of the bin pass. The record maps of the sites' tables go
// through an upload ring.
namespace Libs::Graphics {

namespace CallSplit = ShaderRecompiler::CallSplit;

namespace {

struct BinPush {
	uint64_t state;
	uint64_t header;
	uint64_t map;
	uint64_t args;
	uint64_t lists;
	uint32_t stride;
	uint32_t waves;
	uint32_t site;
	uint32_t exec_lo;
	uint32_t exec_hi;
	uint32_t record_dword;
	uint32_t records;
	uint32_t targets;
	uint64_t lanes; // KYTY_SHADER_CALLS_SPLIT_STATS: per-slot lane counts, or 0
};
static_assert(sizeof(BinPush) == 80);

constexpr uint64_t Align(uint64_t value) {
	return (value + 255u) & ~uint64_t {255};
}

struct Layout {
	uint64_t              header = 0;
	uint64_t              state  = 0;
	std::vector<uint64_t> args;
	std::vector<uint64_t> lists;
	std::vector<uint64_t> lanes;       // per site: targets + 2 lane counts (stats)
	uint64_t              control = 0; // first args byte (cleared each dispatch)
	uint64_t              size    = 0;
};

Layout MakeLayout(const CallSplit::Run& run, uint32_t waves) {
	Layout layout;
	layout.header = 0;
	layout.state  = Align(uint64_t {waves} * 4u);
	uint64_t at   = Align(layout.state + uint64_t {waves} * run.wave_dwords * 4u);
	layout.control = at;
	for (const auto& site: run.sites) {
		layout.args.push_back(at);
		at += Align((site.callees.size() + 1u) * 12u);
	}
	for (const auto& site: run.sites) {
		layout.lanes.push_back(at);
		at += Align((site.callees.size() + 2u) * 4u);
	}
	const auto control_end = at;
	for (const auto& site: run.sites) {
		layout.lists.push_back(at);
		at += Align((site.callees.size() + 1u) * uint64_t {waves} * 4u);
	}
	layout.size = at;
	(void)control_end;
	return layout;
}

struct Totals {
	std::atomic<uint64_t> dispatches {0};
	std::atomic<uint64_t> callee_dispatches {0};
	std::atomic<uint64_t> skipped_callees {0};
	std::atomic<uint64_t> skipped {0};
	std::atomic<uint64_t> started {0};
	std::atomic<uint64_t> resumed {0};
	std::atomic<uint64_t> held {0}; // chains stopped before R_a: a needed callee was not ready
	std::atomic<int64_t>  logged_at {0};
	// KYTY_SHADER_CALLS_SPLIT_STATS=1: the bin passes' counts, read back after completion.
	std::atomic<uint64_t> stat_sites {0};       // site passes counted
	std::atomic<uint64_t> stat_waves {0};       // dispatch waves over those passes
	std::atomic<uint64_t> stat_suspended {0};   // waves that suspended at the site
	std::atomic<uint64_t> stat_callee_waves {0};
	std::atomic<uint64_t> stat_present {0};     // callee slots with at least one wave
	std::atomic<uint64_t> stat_slots {0};       // callee slots dispatched
	std::atomic<uint64_t> sparse_skipped {0};   // callee passes left out: no waves last time
	// Lanes at the sites (stats): all, with no callee slot (record out of range or no target),
	// and in slots whose callee pass was left out (sparse) or did not run (not ready / unusable).
	std::atomic<uint64_t> stat_lanes {0};
	std::atomic<uint64_t> stat_lanes_unrouted {0};
	std::atomic<uint64_t> stat_lanes_sparse {0};
	std::atomic<uint64_t> stat_lanes_not_run {0};
};
Totals g_totals;

bool EnvOn(const char* name, bool fallback) {
	const auto* value = std::getenv(name);
	if (value == nullptr || *value == '\0') return fallback;
	return std::strcmp(value, "0") != 0;
}

bool StatsEnabled() {
	static const bool enabled = EnvOn("KYTY_SHADER_CALLS_SPLIT_STATS", false);
	return enabled;
}

// KYTY_SHADER_CALLS_SPLIT_SPARSE (default on): a callee pass is recorded only when the latest
// completed bin pass of the same kernel and site gave its slot waves (read back after completion,
// so a few frames late; slots not seen yet are always recorded) or any earlier one did (sticky).
// Each callee pass costs a program lookup and its bindings on the command processor, and a site
// names about 140 callees of which a scene uses a few. The first frame a material appears in after
// its slot was seen empty, its lanes resume unshaded; from the next counts on it is recorded.
bool SparseEnabled() {
	static const bool enabled = EnvOn("KYTY_SHADER_CALLS_SPLIT_SPARSE", true);
	return enabled;
}

// Why one program of a chain did not run (the RT status line names what held chains wait on).
enum class Wait { None, Planning, Unusable, Translating, Building };

const char* WaitText(Wait wait) {
	switch (wait) {
		case Wait::Planning: return "being planned";
		case Wait::Unusable: return "unusable";
		case Wait::Translating: return "translating";
		case Wait::Building: return "pipeline building";
		case Wait::None: break;
	}
	return "ready";
}

// The guest shader as the status line names it: its first four hex digits (6d9e, 208e).
std::string ShortHash(uint64_t hash) {
	return fmt::format("{:04x}", hash >> 48u);
}

// What the chains of the last 10 s waited on (the RT status line).
struct Interval {
	std::mutex                         mutex;
	uint64_t                           chains   = 0; // split dispatches
	uint64_t                           complete = 0; // chains that ran every step
	uint64_t                           held     = 0;
	uint64_t                           skipped  = 0; // stopped at an R_0 / R_a that was not ready
	uint64_t                           partial  = 0; // ran with a needed callee missing
	std::map<std::string, uint64_t>    waiting;      // "6d9e site 1 callee 0x..: pipeline building" -> count
	bool                               announced_ready = false;
};
Interval g_interval;

void NoteWait(uint64_t hash, uint32_t site, const char* program, uint64_t target, Wait wait) {
	std::string key = target != 0 ? fmt::format("{} site {} callee 0x{:x}: {}", ShortHash(hash), site, target, WaitText(wait))
	                              : fmt::format("{} {}: {}", ShortHash(hash), program, WaitText(wait));
	std::scoped_lock lock(g_interval.mutex);
	g_interval.waiting[key]++;
}

} // namespace

struct RenderExecutor::CallSplitState {
	explicit CallSplitState(RenderContext& context): context(context) {}
	~CallSplitState() {
		auto device = context.GetGraphics().device;
		if (pipeline) device.destroyPipeline(pipeline);
		if (layout) device.destroyPipelineLayout(layout);
	}

	void Initialize() {
		if (pipeline) return;
		auto&                       graphics = context.GetGraphics();
		const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(BinPush)};
		vk::PipelineLayoutCreateInfo info {};
		info.pushConstantRangeCount = 1;
		info.pPushConstantRanges    = &push;
		RequireVulkanSuccess(graphics.device.createPipelineLayout(&info, nullptr, &layout),
		                     "create call-split bin layout");
		const auto                  module = CompileSPV(GPU_CALL_BIN_SPV, graphics.device);
		vk::ComputePipelineCreateInfo create {};
		create.layout       = layout;
		create.stage.stage  = vk::ShaderStageFlagBits::eCompute;
		create.stage.module = module;
		create.stage.pName  = "main";
		const auto result   = graphics.device.createComputePipelines(nullptr, 1, &create, nullptr, &pipeline);
		graphics.device.destroyShaderModule(module);
		RequireVulkanSuccess(result, "create call-split bin pipeline");
		maps = std::make_unique<StreamBuffer>(graphics, context.GetCommandScheduler(), MemoryUsage::Upload,
		                                      uint64_t {16} << 20u, false,
		                                      vk::BufferUsageFlagBits::eShaderDeviceAddress);
		SetVulkanObjectNameF(graphics.device, maps->Handle(), "Kyty.CallSplitMaps");
	}

	Buffer& StateBuffer(uint64_t kernel, uint64_t size) {
		auto& slot = buffers[kernel];
		if (slot == nullptr || slot->Size() < size) {
			// Grown by at least a quarter and to a 64 MiB multiple: dispatch sizes vary a little.
			const auto old_size = slot != nullptr ? slot->Size() : 0u;
			const auto grown    = (std::max(size, old_size + old_size / 4u) + (uint64_t {64} << 20u) - 1u) &
			                   ~((uint64_t {64} << 20u) - 1u);
			if (slot != nullptr) {
				// Released once the recordings that used it completed.
				context.GetCommandScheduler().DeferOperation(
				    [old = std::shared_ptr<Buffer>(std::move(slot))]() mutable { old.reset(); });
			}
			slot = std::make_unique<Buffer>(context.GetGraphics(), context.GetCommandScheduler(),
			                                MemoryUsage::DeviceLocal, 0,
			                                AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, grown);
			SetVulkanObjectNameF(context.GetGraphics().device, slot->Handle(), "Kyty.CallSplitState");
			Log::WriteToConsoleAndLog(fmt::format(
			    "KYTY_SHADER_CALLS_SPLIT: call-state buffer of the {} queue: {} MiB\n", kernel != 0 ? "async compute" : "graphics",
			    grown >> 20u));
		}
		return *slot;
	}

	// A host-readable buffer for one dispatch's bin counts (returned after completion).
	std::shared_ptr<Buffer> TakeReadback(uint64_t size) {
		{
			std::scoped_lock lock(readback_mutex);
			for (size_t i = 0; i < free_readbacks.size(); i++) {
				if (free_readbacks[i]->Size() >= size) {
					auto found = std::move(free_readbacks[i]);
					free_readbacks.erase(free_readbacks.begin() + static_cast<std::ptrdiff_t>(i));
					return found;
				}
			}
		}
		return std::make_shared<Buffer>(context.GetGraphics(), context.GetCommandScheduler(),
		                                MemoryUsage::Download, 0, AllFlags,
		                                std::max<uint64_t>(size, uint64_t {64} << 10u));
	}
	void ReturnReadback(std::shared_ptr<Buffer> readback) {
		std::scoped_lock lock(readback_mutex);
		free_readbacks.push_back(std::move(readback));
	}

	// KYTY_SHADER_CALLS_SPLIT_SPARSE: per kernel and site, whether each callee slot had waves in the
	// latest bin counts read back.
	void SetPresent(uint64_t kernel, size_t site, std::vector<uint8_t> bits) {
		std::scoped_lock lock(present_mutex);
		auto&            sites = present[kernel];
		if (sites.size() <= site) sites.resize(site + 1u);
		// Sticky: a slot that had waves once stays recorded (a material seen in one frame is often
		// seen again a few frames later, when the latest counts may not show it).
		auto& known = sites[site];
		if (known.size() < bits.size()) known.resize(bits.size(), 0u);
		for (size_t t = 0; t < bits.size(); t++) known[t] |= bits[t];
	}
	std::vector<std::vector<uint8_t>> Present(uint64_t kernel) {
		std::scoped_lock lock(present_mutex);
		const auto       found = present.find(kernel);
		return found != present.end() ? found->second : std::vector<std::vector<uint8_t>> {};
	}

	RenderContext&                                         context;
	std::mutex                                             present_mutex;
	std::unordered_map<uint64_t, std::vector<std::vector<uint8_t>>> present;
	std::mutex                                             readback_mutex;
	std::vector<std::shared_ptr<Buffer>>                   free_readbacks;
	vk::PipelineLayout                                     layout   = nullptr;
	vk::Pipeline                                           pipeline = nullptr;
	std::unique_ptr<StreamBuffer>                          maps;
	std::unordered_map<uint64_t, std::unique_ptr<Buffer>> buffers;
};

namespace {

// Every 10 s (with the split totals): whether the ray-tracing shaders are compiled. Example:
//   RT: pipelines ready 6d9e 41/43, 208e 40/40; callee programs planned 118/120 (2 being planned);
//   background translations: 3 queued, 212 done (...); chains in the last 10 s: 192 ran (12 with
//   callees not ready: ...), 0 held, 0 skipped (waiting on: 6d9e site 1 callee 0x11a2b3c400:
//   pipeline building x12)
// and once nothing is pending and every chain of the interval completed: "RT shaders ready".
void LogRtStatus(PipelineCache& pipelines) {
	const auto rt       = pipelines.GetRtPipelineStatus();
	const auto plan     = CallSplit::Planner::Instance().GetStatus();
	const auto expanded = PipelineCache::GetCallTranslationTotals();
	uint64_t   building = 0;
	std::string ready;
	for (const auto& shader: rt) {
		building += shader.building;
		ready += fmt::format("{}{} {}/{}{}", ready.empty() ? "" : ", ", ShortHash(shader.shader_hash), shader.ready,
		                     shader.requested, shader.failed != 0 ? fmt::format(" ({} failed)", shader.failed) : "");
	}
	const auto translating = expanded.queued - std::min(expanded.finished, expanded.queued);
	uint64_t   chains = 0, complete = 0, held = 0, skipped = 0, partial = 0;
	std::string waiting;
	bool        announce_ready = false, announce_again = false;
	{
		std::scoped_lock lock(g_interval.mutex);
		chains   = g_interval.chains;
		complete = g_interval.complete;
		held     = g_interval.held;
		skipped  = g_interval.skipped;
		partial  = g_interval.partial;
		std::vector<std::pair<uint64_t, std::string>> top;
		for (const auto& [what, count]: g_interval.waiting) top.emplace_back(count, what);
		std::ranges::sort(top, std::greater {});
		for (size_t i = 0; i < top.size() && i < 4u; i++) {
			waiting += fmt::format("{}{} x{}", i == 0 ? "" : "; ", top[i].second, top[i].first);
		}
		if (top.size() > 4u) waiting += fmt::format("; {} more", top.size() - 4u);
		const bool pending = building != 0 || translating != 0 || plan.planning != 0 || plan.pending != 0 ||
		                     held != 0 || skipped != 0 || !g_interval.waiting.empty();
		if (!pending && complete != 0 && !g_interval.announced_ready) {
			g_interval.announced_ready = true;
			announce_ready             = true;
		} else if (pending && g_interval.announced_ready) {
			g_interval.announced_ready = false;
			announce_again             = true;
		}
		g_interval.chains = g_interval.complete = g_interval.held = g_interval.skipped = g_interval.partial = 0;
		g_interval.waiting.clear();
	}
	Log::WriteToConsoleAndLog(fmt::format(
	    "RT: pipelines ready {}; callee programs planned {}/{}{}{}; background translations: {} queued, {} done "
	    "({:.1f} M words, {:.1f} s); chains in the last 10 s: {} ran{}, {} held, {} skipped{}\n",
	    ready.empty() ? std::string("none requested") : ready, plan.built, plan.targets,
	    plan.pending != 0 ? fmt::format(" ({} being planned)", plan.pending) : std::string(),
	    plan.unusable != 0 ? fmt::format(" ({} unusable: their lanes resume without the call)", plan.unusable)
	                       : std::string(),
	    translating, expanded.finished, static_cast<double>(expanded.words) / 1.0e6,
	    static_cast<double>(expanded.ms) / 1.0e3, complete,
	    partial != 0 ? fmt::format(" ({} with callees not ready: those lanes resumed unshaded)", partial)
	                 : std::string(),
	    held, skipped,
	    waiting.empty() ? std::string() : " (waiting on: " + waiting + ")"));
	(void)chains;
	if (announce_ready) {
		Log::WriteToConsoleAndLog(
		    "RT shaders ready: every ray-tracing pipeline and callee program the game used so far is compiled; "
		    "reflections are complete from here on (new areas can still bring new materials)\n");
	} else if (announce_again) {
		Log::WriteToConsoleAndLog("RT: new ray-tracing shaders are compiling (new materials); reflections may "
		                          "lag or have gaps until the next \"RT shaders ready\"\n");
	}
}

} // namespace

void RenderExecutor::DispatchCallSplit(CommandBuffer& buffer, const ShaderComputeInputInfo& pristine,
                                       const std::shared_ptr<const void>& run_pointer,
                                       const HW::ComputeShaderInfo& cs_regs,
                                       const HW::ShaderRegisters& sh_regs, uint32_t groups_x,
                                       uint32_t groups_y, uint32_t groups_z) {
	const auto& run   = *static_cast<const CallSplit::Run*>(run_pointer.get());
	const auto  waves = groups_x * groups_y * groups_z;
	if (m_call_split == nullptr) {
		m_call_split = std::make_shared<CallSplitState>(m_context);
	}
	auto& split = *m_call_split;
	split.Initialize();
	const auto layout   = MakeLayout(run, waves);
	// One call-state buffer per queue, shared by every split kernel: the chains of a queue run one
	// after the other (each begins with a barrier against every earlier use of the buffer).
	auto&      state    = split.StateBuffer(pristine.async_compute ? 1u : 0u, layout.size);
	const auto address  = state.BufferDeviceAddress();
	auto&      pipelines = m_context.GetPipelineCache();

	// One program of the chain: its preparation through the pipeline cache as a dispatch of the
	// kernel, then the dispatch (direct: the kernel's workgroups; indirect: a bin pass's command).
	Wait       last_wait = Wait::None; // why the latest dispatch() returned false
	const auto dispatch  = [&](const CallSplit::Code& code, uint64_t list, const vk::Buffer* indirect,
	                          uint64_t indirect_offset) {
		auto info = pristine;
		PipelineCache::SelectCallSplitCode(&code);
		const auto program_handle = pipelines.GetComputeProgram(cs_regs, sh_regs, info, m_compute_prep);
		PipelineCache::SelectCallSplitCode(nullptr);
		if (!program_handle) {
			last_wait = Wait::Translating;
			return false;
		}
		auto* pipeline = pipelines.TryGetComputePipeline(info, program_handle);
		if (pipeline == nullptr) {
			last_wait = Wait::Building;
			ResetBindings();
			return false;
		}
		const auto& program  = *info.stage.program;
		auto&       bindings = m_compute_bindings;
		PrepareBindings(info.stage, bindings);
		if (const auto dword = program.bindings.call_state_dword;
		    dword != ShaderRecompiler::IR::PushData::NoStart) {
			const uint64_t header = address + layout.header;
			const uint64_t base   = address + layout.state;
			const uint32_t params[ShaderRecompiler::IR::BindingLayout::CallStateDwords] = {
			    static_cast<uint32_t>(base),   static_cast<uint32_t>(base >> 32u),
			    static_cast<uint32_t>(list),   static_cast<uint32_t>(list >> 32u),
			    static_cast<uint32_t>(header), static_cast<uint32_t>(header >> 32u),
			    run.wave_dwords};
			std::copy(std::begin(params), std::end(params), bindings.shader_data.begin() + dword);
		}
		bindings.dispatch_groups     = {groups_x, groups_y, groups_z};
		bindings.has_dispatch_groups = indirect == nullptr;
		FindBuffers(bindings);
		if (program.info.uses_dma) {
			m_context.PrepareBda();
		}
		if (program.info.bda_writes) {
			m_context.GetBufferCache().PrepareBdaWrites();
		}
		RebindImages(bindings);
		const std::array<uint32_t, 3> groups {groups_x, groups_y, groups_z};
		BindComputeLds(info, bindings, indirect == nullptr ? groups.data() : nullptr);
		RebindBuffers(bindings);
		PreparedBindings* stage = &bindings;
		buffer.BeginEmission();
		CommitBindings(buffer, vk::PipelineBindPoint::eCompute, *pipeline, std::span {&stage, 1u});
		ShaderWriteHazardBarrier(buffer, vk::PipelineStageFlagBits::eComputeShader);
		buffer.BindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline);
		if (indirect == nullptr) {
			buffer.Sink().dispatch(groups_x, groups_y, groups_z);
		} else {
			buffer.Sink().dispatchIndirect(*indirect, indirect_offset);
		}
		ShaderAccessBarrier(buffer, vk::PipelineStageFlagBits::eComputeShader);
		buffer.NoteGuestDispatch(true);
		if (program.info.bda_writes) {
			m_context.GetBufferCache().SettleBdaWrites(program.shader_hash);
		}
		ResetBindings();
		return true;
	};
	const auto to_compute = [&](vk::PipelineStageFlags2 source, vk::AccessFlags2 access) {
		buffer.RequestMemoryBarrier(source, access,
		                            vk::PipelineStageFlagBits2::eComputeShader |
		                                vk::PipelineStageFlagBits2::eDrawIndirect,
		                            vk::AccessFlagBits2::eShaderStorageRead |
		                                vk::AccessFlagBits2::eShaderStorageWrite |
		                                vk::AccessFlagBits2::eIndirectCommandRead,
		                            BarrierOrigin::IndirectArgs);
	};

	// The headers and the bin commands start at zero; the wave blocks need no clearing (a wave
	// writes its block before any program reads it).
	{
		// Every earlier use of the buffer (the previous dispatch's chain) before the clears.
		buffer.RequestMemoryBarrier(vk::PipelineStageFlagBits2::eComputeShader |
		                                vk::PipelineStageFlagBits2::eDrawIndirect,
		                            vk::AccessFlagBits2::eShaderStorageRead |
		                                vk::AccessFlagBits2::eShaderStorageWrite |
		                                vk::AccessFlagBits2::eIndirectCommandRead,
		                            vk::PipelineStageFlagBits2::eClear |
		                                vk::PipelineStageFlagBits2::eCopy,
		                            vk::AccessFlagBits2::eTransferWrite, BarrierOrigin::IndirectArgs);
		auto sink = buffer.Sink();
		sink.fillBuffer(state.Handle(), layout.header, Align(uint64_t {waves} * 4u), 0);
		sink.fillBuffer(state.Handle(), layout.control,
		                (layout.lists.empty() ? layout.size : layout.lists.front()) - layout.control, 0);
	}
	// The record maps.
	std::vector<uint64_t> maps;
	for (const auto& site: run.sites) {
		const auto bytes            = std::max<uint64_t>(site.record_slots.size() * 4u, 4u);
		const auto [mapped, offset] = split.maps->Map(bytes, 256);
		EXIT_IF(mapped == nullptr);
		if (!site.record_slots.empty()) {
			std::memcpy(mapped, site.record_slots.data(), site.record_slots.size() * 4u);
		}
		split.maps->Commit();
		maps.push_back(split.maps->BufferDeviceAddress() + offset);
	}
	to_compute(vk::PipelineStageFlagBits2::eClear | vk::PipelineStageFlagBits2::eCopy |
	               vk::PipelineStageFlagBits2::eHost,
	           vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eHostWrite);

	const auto ordinal = g_totals.dispatches.fetch_add(1, std::memory_order_relaxed);
	// KYTY_SHADER_CALLS_SPLIT_LOG=N: the steps of the first N split dispatches.
	static const uint64_t log_limit = [] {
		const auto* value = std::getenv("KYTY_SHADER_CALLS_SPLIT_LOG");
		return value != nullptr ? std::strtoull(value, nullptr, 0) : 0ull;
	}();
	std::string steps;
	const auto note = [&](const char* what, bool ran) {
		if (ordinal < log_limit) steps += fmt::format(" {}{}", what, ran ? "" : "(skipped)");
	};
	std::shared_ptr<Buffer> readback;
	std::vector<uint64_t>   readback_at;
	const bool sparse  = SparseEnabled();
	const auto present = sparse ? split.Present(run.kernel) : std::vector<std::vector<uint8_t>> {};
	if (StatsEnabled() || sparse) {
		uint64_t bytes = 0;
		for (const auto& site: run.sites) {
			readback_at.push_back(bytes);
			bytes += Align((site.callees.size() + 1u) * 12u) + Align((site.callees.size() + 2u) * 4u);
		}
		readback = split.TakeReadback(bytes);
	}
	const auto shader_hash = run.hash;
	bool       chain_held  = false;
	bool       chain_partial = false; // resumed although a needed callee did not run
	const bool started     = dispatch(*run.start, 0, nullptr, 0);
	note("R_0", started);
	if (!started) {
		g_totals.skipped.fetch_add(1, std::memory_order_relaxed);
		NoteWait(shader_hash, 0, "R_0", 0, last_wait);
	} else {
		g_totals.started.fetch_add(1, std::memory_order_relaxed);
	}
	bool     go     = started;
	uint32_t binned = 0; // sites whose bin pass ran (their counts are read back)
	// Per binned site and callee slot: 0 ran, 1 left out (sparse), 2 did not run (stats).
	std::vector<std::vector<uint8_t>> slot_outcome;
	for (uint32_t k = 1; go && k <= run.sites.size(); k++) {
		const auto& site    = run.sites[k - 1u];
		const auto  targets = static_cast<uint32_t>(site.callees.size());
		// The callee slots the table's records name (a slot no record names gets no waves).
		std::vector<bool> needed(targets, false);
		for (const auto slot: site.record_slots) {
			if (slot < targets) needed[slot] = true;
		}
		bool incomplete = false;
		to_compute(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite);
		BinPush push {};
		push.state        = address + layout.state;
		push.header       = address + layout.header;
		push.map          = maps[k - 1u];
		push.args         = address + layout.args[k - 1u];
		push.lists        = address + layout.lists[k - 1u];
		push.stride       = run.wave_dwords;
		push.waves        = waves;
		push.site         = k;
		push.exec_lo      = site.exec_lo_slot;
		push.exec_hi      = site.exec_hi_slot;
		push.record_dword = site.record_dword;
		push.records      = static_cast<uint32_t>(site.record_slots.size());
		push.targets      = targets;
		push.lanes        = StatsEnabled() ? address + layout.lanes[k - 1u] : 0u;
		buffer.BeginEmission();
		buffer.BindPipeline(vk::PipelineBindPoint::eCompute, split.pipeline);
		{
			auto sink = buffer.Sink();
			sink.pushConstants(split.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
			sink.dispatch(waves, 1, 1);
		}
		to_compute(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite);
		if (readback != nullptr) {
			buffer.RequestMemoryBarrier(vk::PipelineStageFlagBits2::eComputeShader,
			                            vk::AccessFlagBits2::eShaderStorageWrite,
			                            vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead,
			                            BarrierOrigin::IndirectArgs);
			buffer.BeginEmission();
			const vk::BufferCopy regions[2] = {
			    {layout.args[k - 1u], readback_at[k - 1u], (targets + 1u) * 12u},
			    {layout.lanes[k - 1u], readback_at[k - 1u] + Align((targets + 1u) * 12u), (targets + 2u) * 4u}};
			buffer.Sink().copyBuffer(state.Handle(), readback->Handle(), StatsEnabled() ? 2u : 1u, regions);
		}
		binned = k;
		slot_outcome.emplace_back(targets, uint8_t {0});
		const auto handle = state.Handle();
		for (uint32_t t = 0; t < targets; t++) {
			if (k - 1u < present.size() && t < present[k - 1u].size() && present[k - 1u][t] == 0u) {
				g_totals.sparse_skipped.fetch_add(1, std::memory_order_relaxed);
				slot_outcome.back()[t] = 1u;
				continue;
			}
			if (site.callees[t] == nullptr) {
				g_totals.skipped_callees.fetch_add(1, std::memory_order_relaxed);
				slot_outcome.back()[t] = 2u;
				const bool pending = t < site.pending.size() && site.pending[t];
				incomplete |= needed[t] && pending;
				if (needed[t]) {
					NoteWait(shader_hash, k, nullptr, t < site.targets.size() ? site.targets[t] : t + 1u,
					         pending ? Wait::Planning : Wait::Unusable);
				}
				continue;
			}
			const bool ran = dispatch(*site.callees[t], address + layout.lists[k - 1u] + uint64_t {t} * waves * 4u,
			                          &handle, layout.args[k - 1u] + uint64_t {t} * 12u);
			note("C", ran);
			if (ran) {
				g_totals.callee_dispatches.fetch_add(1, std::memory_order_relaxed);
			} else {
				g_totals.skipped_callees.fetch_add(1, std::memory_order_relaxed);
				slot_outcome.back()[t] = 2u;
				incomplete |= needed[t];
				if (needed[t]) {
					NoteWait(shader_hash, k, nullptr, t < site.targets.size() ? site.targets[t] : t + 1u, last_wait);
				}
			}
		}
		to_compute(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite);
		// KYTY_SHADER_CALLS_SPLIT_PARTIAL (default on) resumes while a callee a record names is still
		// being planned or compiled: its lanes keep their pre-call values (the kernel's "no material"
		// result), every other lane is shaded. KYTY_SHADER_CALLS_SPLIT_PARTIAL=0 stops the chain here
		// instead, which does not keep the outputs as they were: R_0 has already written the waves that
		// did not reach the site, and the suspended ones are never written, so Miles Morales'
		// reflection buffer showed scattered fragments of stale data while callee pipelines built
		// (minutes with a fresh cache).
		static const bool partial = EnvOn("KYTY_SHADER_CALLS_SPLIT_PARTIAL", true);
		chain_partial |= incomplete;
		if (incomplete && !partial) {
			g_totals.held.fetch_add(1, std::memory_order_relaxed);
			note("held", false);
			chain_held = true;
			go         = false;
			break;
		}
		const bool resumed =
		    site.resume != nullptr &&
		    dispatch(*site.resume, address + layout.lists[k - 1u] + uint64_t {targets} * waves * 4u, &handle,
		             layout.args[k - 1u] + uint64_t {targets} * 12u);
		note(k == 1 ? "bin1 R_a1" : "bin R_a", resumed);
		if (!resumed) {
			g_totals.skipped.fetch_add(1, std::memory_order_relaxed);
			if (site.resume != nullptr) {
				NoteWait(shader_hash, k, k == 1 ? "R_a1" : "R_a2+", 0, last_wait);
			}
			go = false;
		} else {
			g_totals.resumed.fetch_add(1, std::memory_order_relaxed);
		}
	}
	if (readback != nullptr) {
		if (started) {
			std::vector<uint32_t> counts;
			for (uint32_t k = 0; k < binned; k++) counts.push_back(static_cast<uint32_t>(run.sites[k].callees.size()));
			auto*          owner  = &split;
			const uint64_t kernel = run.kernel;
			const bool lanes = StatsEnabled();
			m_context.GetCommandScheduler().DeferOperation(
			    [owner, readback, counts = std::move(counts), at = std::move(readback_at), waves, kernel,
			     sparse, lanes, outcome = std::move(slot_outcome)]() mutable {
				    if (!readback->IsCoherent()) readback->Invalidate(0, readback->Size());
				    const auto bytes = readback->Mapped();
				    for (size_t k = 0; k < counts.size(); k++) {
					    const auto* w = reinterpret_cast<const uint32_t*>(bytes.data() + at[k]);
					    uint64_t             present = 0, callee_waves = 0;
					    std::vector<uint8_t> bits(counts[k]);
					    for (uint32_t t = 0; t < counts[k]; t++) {
						    bits[t] = w[t * 3u] != 0u ? 1u : 0u;
						    present += bits[t];
						    callee_waves += w[t * 3u];
					    }
					    if (sparse) owner->SetPresent(kernel, k, std::move(bits));
					    g_totals.stat_sites.fetch_add(1, std::memory_order_relaxed);
					    g_totals.stat_waves.fetch_add(waves, std::memory_order_relaxed);
					    g_totals.stat_suspended.fetch_add(w[counts[k] * 3u], std::memory_order_relaxed);
					    g_totals.stat_callee_waves.fetch_add(callee_waves, std::memory_order_relaxed);
					    g_totals.stat_present.fetch_add(present, std::memory_order_relaxed);
					    g_totals.stat_slots.fetch_add(counts[k], std::memory_order_relaxed);
					    if (lanes) {
						    const auto* l = reinterpret_cast<const uint32_t*>(
						        bytes.data() + at[k] + Align((counts[k] + 1u) * 12u));
						    uint64_t sparse_lanes = 0, not_run = 0;
						    for (uint32_t t = 0; t < counts[k] && t < outcome[k].size(); t++) {
							    if (outcome[k][t] == 1u) sparse_lanes += l[t];
							    if (outcome[k][t] == 2u) not_run += l[t];
						    }
						    g_totals.stat_lanes.fetch_add(l[counts[k] + 1u], std::memory_order_relaxed);
						    g_totals.stat_lanes_unrouted.fetch_add(l[counts[k]], std::memory_order_relaxed);
						    g_totals.stat_lanes_sparse.fetch_add(sparse_lanes, std::memory_order_relaxed);
						    g_totals.stat_lanes_not_run.fetch_add(not_run, std::memory_order_relaxed);
					    }
				    }
				    owner->ReturnReadback(std::move(readback));
			    });
		} else {
			split.ReturnReadback(std::move(readback));
		}
	}
	{
		std::scoped_lock lock(g_interval.mutex);
		g_interval.chains++;
		if (go) {
			g_interval.complete++;
			g_interval.partial += chain_partial ? 1u : 0u;
		} else if (chain_held) {
			g_interval.held++;
		} else {
			g_interval.skipped++;
		}
	}
	// Every 10 s: what the chains ran so far (whether the callee passes execute in game).
	{
		const auto now  = std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::steady_clock::now().time_since_epoch())
		                     .count();
		auto       last = g_totals.logged_at.load(std::memory_order_relaxed);
		if (now - last >= 10 && g_totals.logged_at.compare_exchange_strong(last, now)) {
			// The raw totals: console with KYTY_SHADER_CALLS_SPLIT_STATS, else the log file only (the
			// "RT:" line below is the console's summary).
			const auto write = [](const std::string& text) {
				if (StatsEnabled()) {
					Log::WriteToConsoleAndLog(text);
				} else {
					Log::Write(text);
				}
			};
			write(fmt::format(
			    "KYTY_SHADER_CALLS_SPLIT: totals: {} split dispatches, {} R_0 ran, {} R_a ran, {} callee "
			    "passes ran, {} callee passes skipped, {} left out (sparse), {} steps skipped, {} chains held for "
			    "callees\n",
			    g_totals.dispatches.load(), g_totals.started.load(), g_totals.resumed.load(),
			    g_totals.callee_dispatches.load(), g_totals.skipped_callees.load(),
			    g_totals.sparse_skipped.load(), g_totals.skipped.load(),
			    g_totals.held.load()));
			if (const auto sites = g_totals.stat_sites.load(); sites != 0) {
				write(fmt::format(
				    "KYTY_SHADER_CALLS_SPLIT: stats over {} site passes: {} of {} waves suspended ({:.3f}%), {:.1f} callee "
				    "waves per suspended wave, {:.1f} of {:.1f} callee slots have waves\n",
				    sites, g_totals.stat_suspended.load(), g_totals.stat_waves.load(),
				    100.0 * static_cast<double>(g_totals.stat_suspended.load()) /
				               static_cast<double>(std::max<uint64_t>(g_totals.stat_waves.load(), 1u)),
				    static_cast<double>(g_totals.stat_callee_waves.load()) /
				        static_cast<double>(std::max<uint64_t>(g_totals.stat_suspended.load(), 1u)),
				    static_cast<double>(g_totals.stat_present.load()) / static_cast<double>(sites),
				    static_cast<double>(g_totals.stat_slots.load()) / static_cast<double>(sites)));
				if (const auto lanes = g_totals.stat_lanes.load(); lanes != 0) {
					const auto pct = [&](uint64_t n) { return 100.0 * static_cast<double>(n) / static_cast<double>(lanes); };
					Log::WriteToConsoleAndLog(fmt::format(
					    "KYTY_SHADER_CALLS_SPLIT: lanes at the sites: {}; resumed without their call: {} with no callee "
					    "slot ({:.2f}%), {} left out (sparse, {:.2f}%), {} whose callee did not run ({:.2f}%)\n",
					    lanes, g_totals.stat_lanes_unrouted.load(), pct(g_totals.stat_lanes_unrouted.load()),
					    g_totals.stat_lanes_sparse.load(), pct(g_totals.stat_lanes_sparse.load()),
					    g_totals.stat_lanes_not_run.load(), pct(g_totals.stat_lanes_not_run.load())));
				}
			}
			LogRtStatus(pipelines);
		}
	}
	if (ordinal < log_limit) {
		Log::WriteToConsoleAndLog(fmt::format("KYTY_SHADER_CALLS_SPLIT: dispatch {} of kernel 0x{:016x} ({} waves):{}\n",
		                                      ordinal, run.kernel, waves, steps));
	}
}

} // namespace Libs::Graphics
