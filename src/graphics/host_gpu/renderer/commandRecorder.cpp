#include "graphics/host_gpu/renderer/commandRecorder.h"

#include "common/assert.h"
#include "common/cpuPlacement.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"
#include "graphics/host_gpu/renderer/gpuTiming.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Libs::Graphics {

namespace {

using CommandStream::Op;

uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

uint64_t EnvUnsigned(const char* name, uint64_t default_value, uint64_t max_value) {
	const auto* value = std::getenv(name);
	if (value == nullptr || value[0] == '\0') {
		return default_value;
	}
	char*      end    = nullptr;
	const auto parsed = std::strtoull(value, &end, 10);
	if (end == value) {
		return default_value;
	}
	return std::min<uint64_t>(parsed, max_value);
}

bool EnvFlag(const char* name) {
	const auto* value = std::getenv(name);
	return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool DrainLogEnabled() {
	static const bool enabled = EnvFlag("KYTY_CP_RECORDER_DRAIN_LOG");
	return enabled;
}

uint64_t RingBytes() {
	const auto mib   = EnvUnsigned("KYTY_CP_RECORDER_RING_MB", 16, 1024);
	uint64_t   bytes = std::max<uint64_t>(mib, 1) << 20u;
	// Power of two.
	uint64_t power = 1u << 20u;
	while (power < bytes) {
		power <<= 1u;
	}
	return power;
}

// ---- Verify ownership: which thread may record into which guest command buffer ----

// The recorders of live guest schedulers (tests create several render contexts).
constexpr size_t                                        MaxRecorders = 16;
std::array<std::atomic<CommandRecorder*>, MaxRecorders> g_recorders {};
// Native command buffer each registered recorder is recording (set by its Begin replay).
std::array<std::atomic<VkCommandBuffer>, MaxRecorders> g_recording {};
thread_local bool                                      t_replaying = false;
std::atomic<uint64_t>                                  g_ownership_faults {0};

size_t RegisterRecorder(CommandRecorder* recorder) {
	for (size_t i = 0; i < MaxRecorders; i++) {
		CommandRecorder* expected = nullptr;
		if (g_recorders[i].compare_exchange_strong(expected, recorder)) {
			g_recording[i].store(nullptr, std::memory_order_release);
			return i;
		}
	}
	return MaxRecorders;
}

void UnregisterRecorder(size_t slot) {
	if (slot < MaxRecorders) {
		g_recording[slot].store(nullptr, std::memory_order_release);
		g_recorders[slot].store(nullptr, std::memory_order_release);
	}
}

void ReportOwnershipFault(const char* function, VkCommandBuffer command) {
	g_ownership_faults.fetch_add(1, std::memory_order_relaxed);
	Profiler::CountFrameEvent(Profiler::FrameEvent::CpRecorderVerifyMismatches);
	std::printf("CP recorder verify: %s recorded into the guest command buffer %p outside the "
	            "recorder and outside a CP direct window\n",
	            function, static_cast<void*>(command));
	std::fflush(stdout);
	LOGF("CP recorder verify: %s recorded into the guest command buffer %p outside the recorder "
	     "and outside a CP direct window\n",
	     function, static_cast<void*>(command));
	if (CommandRecorder::VerifyExits()) {
		EXIT("CP recorder verify: ownership fault (%s)\n", function);
	}
}

template <size_t Index, typename Fn>
struct CmdHook;

template <size_t Index, typename R, typename... Args>
struct CmdHook<Index, R(VKAPI_PTR*)(VkCommandBuffer, Args...)> {
	static inline R(VKAPI_PTR* real)(VkCommandBuffer, Args...) = nullptr;
	static inline const char* name                             = nullptr;
	static R VKAPI_PTR        Call(VkCommandBuffer command, Args... args) {
		if (!CommandRecorder::MayRecord(command)) {
			ReportOwnershipFault(name, command);
		}
		return real(command, args...);
	}
};

template <size_t Index, typename Fn>
void InstallHook(Fn& slot, const char* name) {
	if (slot == nullptr) {
		return;
	}
	CmdHook<Index, Fn>::real = slot;
	CmdHook<Index, Fn>::name = name;
	slot                     = &CmdHook<Index, Fn>::Call;
}

// Placement: logical processor index (group * 64 + number) -> physical core ordinal.
uint32_t CoreOf(uint32_t logical) {
#if defined(_WIN32)
	static const std::vector<uint32_t> cores = [] {
		std::vector<uint32_t> map(256, UINT32_MAX);
		DWORD                 length = 0;
		(void)GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
		std::vector<uint8_t> buffer(length);
		auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data());
		if (length == 0 ||
		    !GetLogicalProcessorInformationEx(RelationProcessorCore, info, &length)) {
			return map;
		}
		uint32_t core   = 0;
		DWORD    offset = 0;
		while (offset < length) {
			auto* entry =
			    reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);
			if (entry->Relationship == RelationProcessorCore) {
				for (WORD g = 0; g < entry->Processor.GroupCount; g++) {
					const auto& mask = entry->Processor.GroupMask[g];
					for (uint32_t bit = 0; bit < 64; bit++) {
						if ((mask.Mask >> bit) & 1u) {
							const auto index = static_cast<uint32_t>(mask.Group) * 64u + bit;
							if (index < map.size()) {
								map[index] = core;
							}
						}
					}
				}
				core++;
			}
			offset += entry->Size;
		}
		return map;
	}();
	return logical < cores.size() ? cores[logical] : UINT32_MAX;
#else
	return logical;
#endif
}

uint32_t CurrentLogicalProcessor() {
#if defined(_WIN32)
	PROCESSOR_NUMBER number {};
	GetCurrentProcessorNumberEx(&number);
	return static_cast<uint32_t>(number.Group) * 64u + number.Number;
#else
	return UINT32_MAX;
#endif
}

void CurrentSites(const void** site, const void** scope) {
	GpuOpProfiler::Detail::CurrentSites(site, scope);
}

} // namespace

// ------------------------------------------------------------------------------------------------
// Configuration

CommandRecorder::Mode CommandRecorder::ConfiguredMode() {
	static const Mode mode = [] {
		const auto* value = std::getenv("KYTY_CP_RECORDER");
		Mode        m     = Mode::Off;
		if (value != nullptr &&
		    (std::strcmp(value, "1") == 0 || std::strcmp(value, "thread") == 0 ||
		     std::strcmp(value, "on") == 0)) {
			m = Mode::Thread;
		} else if (value != nullptr && std::strcmp(value, "inline") == 0) {
			m = Mode::Inline;
		}
		if (m != Mode::Off && !BarrierBatchEnabled()) {
			std::printf("Kyty CP recorder: off (it needs the barrier batcher; KYTY_BARRIER_BATCH=0 "
			            "is set)\n");
			m = Mode::Off;
		}
		if (m != Mode::Off) {
			std::printf("Kyty CP recorder: %s (ring %" PRIu64 " MiB, verify %s)\n",
			            m == Mode::Thread ? "recorder thread" : "inline replay", RingBytes() >> 20u,
			            VerifyEnabled() ? (VerifyExits() ? "exit" : "log") : "off");
			std::fflush(stdout);
		}
		return m;
	}();
	return mode;
}

bool CommandRecorder::VerifyEnabled() {
	static const bool enabled = EnvFlag("KYTY_CP_RECORDER_VERIFY");
	return enabled;
}

bool CommandRecorder::VerifyExits() {
	static const bool exits = [] {
		const auto* value = std::getenv("KYTY_CP_RECORDER_VERIFY");
		return value != nullptr && std::strcmp(value, "exit") == 0;
	}();
	return exits;
}

// ------------------------------------------------------------------------------------------------
// Native executor: replays packets into the guest command buffer (recorder thread, or the CP in
// inline mode). Command-buffer begin/end and the queue hand-off happen here too.

struct CommandRecorder::NativeExecutor {
	explicit NativeExecutor(CommandRecorder& owner): owner(owner) {}

	CommandRecorder&  owner;
	vk::CommandBuffer command = nullptr;
	// GpuOpProfiler attribution adopted from the producer (EnterSite/LeaveSite).
	GpuOpProfiler::Site* previous[2] = {};
	bool                 owners[2]   = {};
	uint32_t             depth       = 0;

	void Begin(const CommandStream::BeginPacket& p) {
		EXIT_IF(command != nullptr || p.command == nullptr);
		command = p.command;
		if (owner.m_registry_slot < MaxRecorders) {
			g_recording[owner.m_registry_slot].store(static_cast<VkCommandBuffer>(command),
			                                         std::memory_order_release);
		}
		vk::CommandBufferBeginInfo begin_info {};
		begin_info.flags  = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
		const auto result = command.begin(&begin_info);
		EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
		if (owner.m_timing != nullptr) {
			// The ring's single producer is this thread (gpuTiming.h): collect completed pairs,
			// then stamp this buffer, with the CP's recording start time.
			owner.m_timing->Collect(owner.m_master.KnownGpuTick());
			owner.m_timing->BeginCommand(command, p.record_ns);
		}
		if (owner.m_gpu_ops) {
			GpuOpProfiler::OnBeginCommand(owner.m_graphics, command, p.tick,
			                              owner.m_master.KnownGpuTick());
		}
	}

	void Submit(const CommandStream::SubmitPacket& p) {
		EXIT_IF(command == nullptr);
		uint64_t* dispatch_ns = nullptr;
		if (owner.m_timing != nullptr) {
			owner.m_timing->EndCommand(command);
			dispatch_ns = owner.m_timing->Submitted(p.tick, p.submit_ns);
		}
		{
			const auto result = command.end();
			EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
		}
		auto& graphics = owner.m_graphics;
		if (graphics.submission_queue.Enabled()) {
			QueuedSubmission queued {.submit              = p.submit,
			                         .progress            = owner.m_master.GetSubmissionProgress(),
			                         .master_semaphore    = owner.m_master.Handle(),
			                         .command             = command,
			                         .tick                = p.tick,
			                         .preserve_completion = p.preserve_completion != 0,
			                         .dispatch_ns         = dispatch_ns,
			                         .debug_op            = p.debug_op,
			                         .debug_submit        = p.debug_submit,
			                         .debug_arg0          = p.debug_arg0,
			                         .debug_arg1          = p.debug_arg1,
			                         .debug_arg2          = p.debug_arg2,
			                         .debug_arg3          = p.debug_arg3,
			                         .debug_arg4          = p.debug_arg4};
			graphics.submission_queue.Enqueue(std::move(queued));
		} else {
			const auto&                     submit = p.submit;
			vk::TimelineSemaphoreSubmitInfo timeline_info {};
			timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
			timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
			timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
			timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();
			vk::SubmitInfo submit_info {};
			submit_info.pNext                = &timeline_info;
			submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
			submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
			submit_info.pWaitDstStageMask    = submit.wait_stages.data();
			submit_info.commandBufferCount   = 1;
			submit_info.pCommandBuffers      = &command;
			submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
			submit_info.pSignalSemaphores    = submit.signal_semaphores.data();
			vk::Result result;
			{
				Common::LockGuard         lock(graphics.queue_mutex);
				Profiler::ScopedFrameWait frame_wait(Profiler::FrameWait::DriverSubmit);
				result = graphics.queue.submit(1, &submit_info, nullptr);
			}
			if (result == vk::Result::eErrorDeviceLost) DumpDeviceLossDiagnostics(graphics, p.tick);
			if (result != vk::Result::eSuccess) {
				std::printf("vkQueueSubmit (CP recorder) failed: %s (%d), tick=%" PRIu64
				            " debug_op=%u debug_submit=%" PRIu64 "\n",
				            vk::to_string(result).c_str(), static_cast<int>(result), p.tick,
				            p.debug_op, p.debug_submit);
				std::fflush(stdout);
			}
			EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
			if (dispatch_ns != nullptr) {
				*dispatch_ns = GpuTiming::NowNs();
			}
			// The recorder owns a SubmissionProgress in synchronous mode (MasterSemaphore): the
			// tick counts as known complete only after this host call returned.
			const auto& progress = owner.m_master.GetSubmissionProgress();
			EXIT_IF(progress == nullptr);
			progress->dispatched_tick.store(p.tick, std::memory_order_release);
			progress->dispatched_tick.notify_all();
		}
		if (owner.m_registry_slot < MaxRecorders) {
			g_recording[owner.m_registry_slot].store(nullptr, std::memory_order_release);
		}
		command = nullptr;
		owner.m_recorded_tick.store(p.tick, std::memory_order_seq_cst);
		if (owner.m_recorded_waiters.load(std::memory_order_seq_cst) != 0) {
			owner.m_recorded_tick.notify_all();
		}
		owner.PublishConsumerCounters();
	}

	void DrainMarker(uint64_t /*serial*/) {}

	void EnterSite(const void* site, const void* scope) {
		auto* s = static_cast<GpuOpProfiler::Site*>(const_cast<void*>(site));
		auto* c = static_cast<GpuOpProfiler::Site*>(const_cast<void*>(scope));
		depth   = 0;
		if (!GpuOpProfiler::Active()) {
			return;
		}
		if (c != nullptr) {
			GpuOpProfiler::Detail::EnterSite(*c, previous[0], owners[0]);
			depth = 1;
		}
		if (s != nullptr && s != c) {
			GpuOpProfiler::Detail::EnterSite(*s, previous[1], owners[1]);
			depth = 2;
		}
	}
	void LeaveSite() {
		if (depth == 2) {
			GpuOpProfiler::Detail::LeaveSite(previous[1], owners[1]);
		}
		if (depth >= 1) {
			GpuOpProfiler::Detail::LeaveSite(previous[0], owners[0]);
		}
		depth = 0;
	}

	// ---- vk::CommandBuffer forwarding ----
	void beginRendering(const vk::RenderingInfo& info) { command.beginRendering(info); }
	void endRendering() { command.endRendering(); }
	void pipelineBarrier2(const vk::DependencyInfo& info) { command.pipelineBarrier2(info); }
	void pipelineBarrier(vk::PipelineStageFlags src, vk::PipelineStageFlags dst,
	                     vk::DependencyFlags flags, uint32_t memory_count,
	                     const vk::MemoryBarrier* memory, uint32_t buffer_count,
	                     const vk::BufferMemoryBarrier* buffers, uint32_t image_count,
	                     const vk::ImageMemoryBarrier* images) {
		command.pipelineBarrier(src, dst, flags, memory_count, memory, buffer_count, buffers,
		                        image_count, images);
	}
	void copyBuffer(vk::Buffer source, vk::Buffer destination, uint32_t count,
	                const vk::BufferCopy* regions) {
		command.copyBuffer(source, destination, count, regions);
	}
	void copyBufferToImage(vk::Buffer source, vk::Image destination, vk::ImageLayout layout,
	                       uint32_t count, const vk::BufferImageCopy* regions) {
		command.copyBufferToImage(source, destination, layout, count, regions);
	}
	void copyImageToBuffer(vk::Image source, vk::ImageLayout layout, vk::Buffer destination,
	                       uint32_t count, const vk::BufferImageCopy* regions) {
		command.copyImageToBuffer(source, layout, destination, count, regions);
	}
	void copyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout destination_layout, uint32_t count,
	               const vk::ImageCopy* regions) {
		command.copyImage(source, source_layout, destination, destination_layout, count, regions);
	}
	void fillBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size, uint32_t value) {
		command.fillBuffer(buffer, offset, size, value);
	}
	void bindPipeline(vk::PipelineBindPoint point, vk::Pipeline pipeline) {
		command.bindPipeline(point, pipeline);
	}
	void bindDescriptorSets(vk::PipelineBindPoint point, vk::PipelineLayout layout,
	                        uint32_t first_set, uint32_t set_count, const vk::DescriptorSet* sets,
	                        uint32_t dynamic_count, const uint32_t* dynamic_offsets) {
		command.bindDescriptorSets(point, layout, first_set, set_count, sets, dynamic_count,
		                           dynamic_offsets);
	}
	// KYTY_RECORDER_DESCRIPTOR_SETS: the CP allocated the set and binds it in a later packet.
	void updateDescriptorSets(vk::DescriptorSet /*set*/, uint32_t count,
	                          const vk::WriteDescriptorSet* writes) {
		owner.m_graphics.device.updateDescriptorSets(count, writes, 0, nullptr);
	}
	void pushDescriptorSetKHR(vk::PipelineBindPoint point, vk::PipelineLayout layout, uint32_t set,
	                          uint32_t count, const vk::WriteDescriptorSet* writes) {
		command.pushDescriptorSetKHR(point, layout, set, count, writes);
	}
	void pushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, uint32_t offset,
	                   uint32_t size, const void* data) {
		command.pushConstants(layout, stages, offset, size, data);
	}
	void bindVertexBuffers2(uint32_t first, uint32_t count, const vk::Buffer* buffers,
	                        const vk::DeviceSize* offsets, const vk::DeviceSize* sizes,
	                        const vk::DeviceSize* strides) {
		command.bindVertexBuffers2(first, count, buffers, offsets, sizes, strides);
	}
	void bindIndexBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type) {
		command.bindIndexBuffer(buffer, offset, type);
	}
	void setViewportWithCount(uint32_t count, const vk::Viewport* viewports) {
		command.setViewportWithCount(count, viewports);
	}
	void setScissorWithCount(uint32_t count, const vk::Rect2D* scissors) {
		command.setScissorWithCount(count, scissors);
	}
	void setLineWidth(float width) { command.setLineWidth(width); }
	void setBlendConstants(const float constants[4]) { command.setBlendConstants(constants); }
	void setDepthTestEnable(vk::Bool32 enable) { command.setDepthTestEnable(enable); }
	void setDepthWriteEnable(vk::Bool32 enable) { command.setDepthWriteEnable(enable); }
	void setDepthCompareOp(vk::CompareOp op) { command.setDepthCompareOp(op); }
	void setDepthBiasEnable(vk::Bool32 enable) { command.setDepthBiasEnable(enable); }
	void setDepthBias(float constant, float clamp, float slope) {
		command.setDepthBias(constant, clamp, slope);
	}
	void setStencilTestEnable(vk::Bool32 enable) { command.setStencilTestEnable(enable); }
	void setStencilOp(vk::StencilFaceFlags faces, vk::StencilOp fail, vk::StencilOp pass,
	                  vk::StencilOp depth_fail, vk::CompareOp compare) {
		command.setStencilOp(faces, fail, pass, depth_fail, compare);
	}
	void setStencilCompareMask(vk::StencilFaceFlags faces, uint32_t mask) {
		command.setStencilCompareMask(faces, mask);
	}
	void setStencilWriteMask(vk::StencilFaceFlags faces, uint32_t mask) {
		command.setStencilWriteMask(faces, mask);
	}
	void setStencilReference(vk::StencilFaceFlags faces, uint32_t reference) {
		command.setStencilReference(faces, reference);
	}
	void setCullMode(vk::CullModeFlags mode) { command.setCullMode(mode); }
	void setFrontFace(vk::FrontFace face) { command.setFrontFace(face); }
	void setDepthBoundsTestEnable(vk::Bool32 enable) { command.setDepthBoundsTestEnable(enable); }
	void setDepthBounds(float min, float max) { command.setDepthBounds(min, max); }
	void setColorWriteEnableEXT(uint32_t count, const vk::Bool32* enables) {
		command.setColorWriteEnableEXT(count, enables);
	}
	void setAttachmentFeedbackLoopEnableEXT(vk::ImageAspectFlags aspects) {
		command.setAttachmentFeedbackLoopEnableEXT(aspects);
	}
	void draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex,
	          uint32_t first_instance) {
		command.draw(vertex_count, instance_count, first_vertex, first_instance);
	}
	void drawIndexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index,
	                 int32_t vertex_offset, uint32_t first_instance) {
		command.drawIndexed(index_count, instance_count, first_index, vertex_offset,
		                    first_instance);
	}
	void drawMeshTasksEXT(uint32_t x, uint32_t y, uint32_t z) { command.drawMeshTasksEXT(x, y, z); }
	void drawMeshTasksIndirectEXT(vk::Buffer buffer, vk::DeviceSize offset, uint32_t draw_count,
	                              uint32_t stride) {
		command.drawMeshTasksIndirectEXT(buffer, offset, draw_count, stride);
	}
	void drawMeshTasksIndirectCountEXT(vk::Buffer buffer, vk::DeviceSize offset,
	                                   vk::Buffer count_buffer, vk::DeviceSize count_offset,
	                                   uint32_t max_count, uint32_t stride) {
		command.drawMeshTasksIndirectCountEXT(buffer, offset, count_buffer, count_offset, max_count,
		                                      stride);
	}
	void drawIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t draw_count,
	                  uint32_t stride) {
		command.drawIndirect(buffer, offset, draw_count, stride);
	}
	void drawIndexedIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t draw_count,
	                         uint32_t stride) {
		command.drawIndexedIndirect(buffer, offset, draw_count, stride);
	}
	void drawIndirectCount(vk::Buffer buffer, vk::DeviceSize offset, vk::Buffer count_buffer,
	                       vk::DeviceSize count_offset, uint32_t max_count, uint32_t stride) {
		command.drawIndirectCount(buffer, offset, count_buffer, count_offset, max_count, stride);
	}
	void drawIndexedIndirectCount(vk::Buffer buffer, vk::DeviceSize offset, vk::Buffer count_buffer,
	                              vk::DeviceSize count_offset, uint32_t max_count,
	                              uint32_t stride) {
		command.drawIndexedIndirectCount(buffer, offset, count_buffer, count_offset, max_count,
		                                 stride);
	}
	void dispatch(uint32_t x, uint32_t y, uint32_t z) { command.dispatch(x, y, z); }
	void dispatchIndirect(vk::Buffer buffer, vk::DeviceSize offset) {
		command.dispatchIndirect(buffer, offset);
	}
	void resetQueryPool(vk::QueryPool pool, uint32_t first, uint32_t count) {
		command.resetQueryPool(pool, first, count);
	}
	void beginQuery(vk::QueryPool pool, uint32_t query, vk::QueryControlFlags flags) {
		command.beginQuery(pool, query, flags);
	}
	void endQuery(vk::QueryPool pool, uint32_t query) { command.endQuery(pool, query); }
	void copyQueryPoolResults(vk::QueryPool pool, uint32_t first, uint32_t count,
	                          vk::Buffer destination, vk::DeviceSize offset, vk::DeviceSize stride,
	                          vk::QueryResultFlags flags) {
		command.copyQueryPoolResults(pool, first, count, destination, offset, stride, flags);
	}
	void writeTimestamp2(vk::PipelineStageFlags2 stage, vk::QueryPool pool, uint32_t query) {
		command.writeTimestamp2(stage, pool, query);
	}
	void beginConditionalRenderingEXT(const vk::ConditionalRenderingBeginInfoEXT& info) {
		command.beginConditionalRenderingEXT(info);
	}
	void endConditionalRenderingEXT() { command.endConditionalRenderingEXT(); }
};

// ------------------------------------------------------------------------------------------------

namespace {
CommandStream::Encoder::Options MakeOptions(CommandRecorder::Mode mode, void* self,
                                            void (*after_commit)(void*)) {
	CommandStream::Encoder::Options options;
	options.verify         = CommandRecorder::VerifyEnabled();
	options.all_sites      = GpuOpProfiler::CaptureEnabled();
	options.barrier_sites  = GpuOpProfiler::Enabled();
	options.current_site   = &CurrentSites;
	options.policy.spin_ns = EnvUnsigned("KYTY_CP_RECORDER_SPIN_US", 30, 1000000) * 1000u;
	if (mode == CommandRecorder::Mode::Inline) {
		options.after_commit         = after_commit;
		options.after_commit_context = self;
	}
	return options;
}
} // namespace

CommandRecorder::CommandRecorder(GraphicContext& graphics, MasterSemaphore& master,
                                 GpuTimestampRing* timing, bool gpu_ops, Mode mode)
    : m_graphics(graphics), m_master(master), m_timing(timing), m_gpu_ops(gpu_ops), m_mode(mode),
      m_ring(RingBytes()), m_encoder(m_ring, MakeOptions(mode, this, &AfterCommitThunk)) {
	EXIT_IF(mode == Mode::Off);
	m_idle_policy.spin_ns  = EnvUnsigned("KYTY_CP_RECORDER_SPIN_US", 30, 1000000) * 1000u;
	m_drain_policy.spin_ns = EnvUnsigned("KYTY_CP_RECORDER_DRAIN_SPIN_US", 2000, 10000000) * 1000u;
	m_replay.verify        = VerifyEnabled();
	m_exec                 = std::make_unique<NativeExecutor>(*this);
	m_registry_slot        = VerifyEnabled() ? RegisterRecorder(this) : MaxRecorders;
	if (mode == Mode::Thread) {
		m_thread = std::jthread([this](std::stop_token stop) { Run(stop); });
	}
}

CommandRecorder::~CommandRecorder() {
	Stop();
	UnregisterRecorder(m_registry_slot);
}

void CommandRecorder::AfterCommitThunk(void* self) {
	static_cast<CommandRecorder*>(self)->AfterCommit();
}

void CommandRecorder::AfterCommit() {
	// Inline mode: replay what was just committed, on the producer thread.
	const bool previous = t_replaying;
	t_replaying         = true;
	ConsumeAvailable();
	t_replaying = previous;
}

void CommandRecorder::Begin(vk::CommandBuffer command, uint64_t tick, uint64_t record_ns) {
	m_encoder.Begin(command, tick, record_ns);
}

void CommandRecorder::Submit(const CommandStream::SubmitPacket& submit) {
	m_cp_processor.store(CurrentLogicalProcessor(), std::memory_order_relaxed);
	m_encoder.Submit(submit);
	PublishCounters();
}

void CommandRecorder::Drain(const void* site_key, bool is_site) {
	const auto start = NowNs();
	// Idle: the recorder released everything encoded (published or not), so it executed every
	// packet and records nothing until the next one. The window opens without a marker and
	// without waking a parked recorder (the consumer's seq_cst release of the position orders its
	// native calls before ours). Inline mode is always idle here.
	const bool idle = m_ring.Consumed() == m_ring.WritePosition();
	if (idle) {
		m_idle_drains++;
	} else {
		const auto end = m_encoder.DrainMarker(++m_drain_serial);
		if (m_mode == Mode::Thread) {
			m_ring.WaitConsumed(end, m_drain_policy, m_drain_stats);
		}
	}
	const auto now = NowNs();
	m_drains++;
	m_drain_ns += now - start;
	if (DrainLogEnabled()) {
		auto& entry = m_drain_log[DrainKey {site_key, is_site}];
		entry.count++;
		entry.ns += now - start;
		if (idle) {
			entry.idle++;
		} else {
			entry.waited_ns += now - start;
		}
		// Also every 10 s: a game session usually ends without a clean shutdown.
		if (now - m_drain_log_printed_ns >= 10'000'000'000ull) {
			m_drain_log_printed_ns = now;
			PrintDrainLog();
		}
	}
}

void CommandRecorder::WaitRecorded(uint64_t tick, bool from_producer) {
	if (m_recorded_tick.load(std::memory_order_acquire) >= tick) {
		return;
	}
	EXIT_IF(m_mode != Mode::Thread);
	if (from_producer) {
		m_ring.Kick(m_encoder.Stats());
	}
	const auto start = NowNs();
	while (m_recorded_tick.load(std::memory_order_acquire) < tick) {
		if (NowNs() - start >= m_drain_policy.spin_ns) {
			m_recorded_waiters.fetch_add(1, std::memory_order_seq_cst);
			for (;;) {
				const auto value = m_recorded_tick.load(std::memory_order_seq_cst);
				if (value >= tick) {
					break;
				}
				m_recorded_tick.wait(value, std::memory_order_acquire);
			}
			m_recorded_waiters.fetch_sub(1, std::memory_order_seq_cst);
			return;
		}
		for (uint32_t i = 0; i < 64; i++) {
#if defined(_M_X64) || defined(__x86_64__)
			_mm_pause();
#endif
		}
	}
}

void CommandRecorder::Stop() {
	if (m_stopped) {
		return;
	}
	m_stopped = true;
	m_test_stall.store(false, std::memory_order_release);
	if (m_mode == Mode::Thread) {
		m_ring.WaitConsumed(m_ring.WritePosition(), m_drain_policy, m_drain_stats);
		m_stop.store(true, std::memory_order_seq_cst);
		m_ring.WakeConsumer();
		m_thread.request_stop();
		if (m_thread.joinable()) {
			m_thread.join();
		}
	}
	PublishCounters();
	if (m_encoder.Packets() != 0) {
		std::printf("Kyty CP recorder: %" PRIu64 " packets, %" PRIu64 " MiB, %" PRIu64
		            " drains (%" PRIu64 " idle, %.3f ms), ring-full waits %" PRIu64
		            ", recorder parks %" PRIu64 ", verify checks %" PRIu64 ", mismatches %" PRIu64
		            ", ownership faults %" PRIu64 "\n",
		            m_encoder.Packets(), m_encoder.Bytes() >> 20u, m_drains, m_idle_drains,
		            static_cast<double>(m_drain_ns) / 1e6, m_encoder.Stats().spins,
		            m_consumer_stats.blocks, m_replay.checks, m_replay.mismatches,
		            g_ownership_faults.load(std::memory_order_relaxed));
		std::fflush(stdout);
	}
	PrintDrainLog();
}

void CommandRecorder::Run(std::stop_token stop) {
	KYTY_PROFILER_THREAD("CP recorder");
	// On the critical path of every drain and submission, like the CP itself.
	Common::RaiseCurrentThreadPriority();
	// KYTY_CPU_RESERVE=cp+recorder: its own physical core; cp: off the CP's core, with the other
	// threads (common/cpuPlacement.h).
	Common::PlaceCurrentThread(Common::ThreadRole::Recorder);
#if defined(_WIN32)
	if (const auto* ideal = std::getenv("KYTY_CP_RECORDER_IDEAL_CPU"); ideal != nullptr) {
		const auto       cpu = static_cast<uint32_t>(std::strtoul(ideal, nullptr, 10));
		PROCESSOR_NUMBER number {};
		number.Group  = static_cast<WORD>(cpu / 64u);
		number.Number = static_cast<BYTE>(cpu % 64u);
		(void)SetThreadIdealProcessorEx(GetCurrentThread(), &number, nullptr);
	}
#endif
	t_replaying = true;
	while (!stop.stop_requested() && !m_stop.load(std::memory_order_acquire)) {
		if (!m_ring.WaitPublished(m_stop, m_idle_policy, m_consumer_stats)) {
			continue;
		}
		while (m_test_stall.load(std::memory_order_acquire) &&
		       !m_stop.load(std::memory_order_acquire)) {
			std::this_thread::sleep_for(std::chrono::microseconds(100));
		}
		const auto start = NowNs();
		ConsumeAvailable();
		m_busy_ns.fetch_add(NowNs() - start, std::memory_order_release);
	}
	ConsumeAvailable();
}

uint64_t CommandRecorder::OwnershipFaults() noexcept {
	return g_ownership_faults.load(std::memory_order_acquire);
}

void CommandRecorder::ConsumeAvailable() {
	std::optional<Profiler::ScopedFrameWait> busy;
	if (m_mode == Mode::Thread && Profiler::AggregateEnabled()) {
		busy.emplace(Profiler::FrameWait::CpRecorderExecute);
	}
	constexpr uint64_t ReleaseEvery = 64u << 10u;
	uint64_t           since        = 0;
	while (const auto* header = m_ring.Peek()) {
		if (header->op != Op::Wrap) {
			CommandStream::Replay(*header, *m_exec, m_replay, &OnMismatch, this);
		}
		m_ring.Advance(header->size);
		since += header->size;
		if (since >= ReleaseEvery) {
			m_ring.Release(m_consumer_stats);
			since = 0;
		}
	}
	m_ring.Release(m_consumer_stats);
	if (m_mode == Mode::Thread && (++m_placement_batches & 255u) == 0) {
		SamplePlacement();
	}
}

void CommandRecorder::OnMismatch(void* self, uint32_t kind, Op op, uint64_t sequence,
                                 uint64_t expected, uint64_t actual) {
	auto*       recorder = static_cast<CommandRecorder*>(self);
	const char* what     = kind == 0   ? "argument hash"
	                       : kind == 1 ? "sequence"
	                                   : "command-buffer digest";
	Profiler::CountFrameEvent(Profiler::FrameEvent::CpRecorderVerifyMismatches);
	std::printf("CP recorder verify: %s mismatch at %s packet %" PRIu64 ": producer 0x%016" PRIx64
	            ", recorder 0x%016" PRIx64 " (mode %s)\n",
	            what, CommandStream::OpName(op), sequence, expected, actual,
	            recorder->m_mode == Mode::Thread ? "thread" : "inline");
	std::fflush(stdout);
	LOGF("CP recorder verify: %s mismatch at %s packet %" PRIu64 ": producer 0x%016" PRIx64
	     ", recorder 0x%016" PRIx64 "\n",
	     what, CommandStream::OpName(op), sequence, expected, actual);
	if (VerifyExits()) {
		EXIT("CP recorder verify: %s mismatch at %s packet %" PRIu64 "\n", what,
		     CommandStream::OpName(op), sequence);
	}
}

void CommandRecorder::PublishCounters() {
	// Producer side (CP thread): packets, bytes, ring-full waits, wakes, drains.
	const auto  packets = m_encoder.Packets();
	const auto  bytes   = m_encoder.Bytes();
	const auto& stats   = m_encoder.Stats();
	if (packets != m_published_packets) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::CpRecorderPackets,
		                          packets - m_published_packets);
		Profiler::CountFrameEvent(Profiler::FrameEvent::CpRecorderBytes, bytes - m_published_bytes);
		m_published_packets = packets;
		m_published_bytes   = bytes;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::CpRecorderSubmits);
	if (stats.spins != m_published_spins) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::CpRecorderRingFullWaits,
		                          stats.spins - m_published_spins);
		Profiler::AddFrameWait(Profiler::FrameWait::CpRecorderRingFull,
		                       stats.spins - m_published_spins,
		                       stats.wait_ns - m_published_wait_ns);
		m_published_spins   = stats.spins;
		m_published_wait_ns = stats.wait_ns;
	}
	const auto wakes = stats.wakes + m_drain_stats.wakes;
	if (wakes != m_published_wakes) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::CpRecorderWakes, wakes - m_published_wakes);
		m_published_wakes = wakes;
	}
	if (m_drains != m_drains_published) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::CpRecorderDrains,
		                          m_drains - m_drains_published);
		if (m_idle_drains != m_idle_drains_published) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::CpRecorderIdleDrains,
			                          m_idle_drains - m_idle_drains_published);
			m_idle_drains_published = m_idle_drains;
		}
		Profiler::AddFrameWait(Profiler::FrameWait::CpRecorderDrain, m_drains - m_drains_published,
		                       m_drain_ns - m_drain_ns_published);
		m_drains_published   = m_drains;
		m_drain_ns_published = m_drain_ns;
	}
}

void CommandRecorder::PublishConsumerCounters() {
	// Consumer side (recorder thread, or the CP in inline mode): parks and verify checks.
	if (m_consumer_stats.blocks != m_published_parks) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::CpRecorderParks,
		                          m_consumer_stats.blocks - m_published_parks);
		m_published_parks = m_consumer_stats.blocks;
	}
	if (m_replay.checks != m_published_checks) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::CpRecorderVerifyChecks,
		                          m_replay.checks - m_published_checks);
		m_published_checks = m_replay.checks;
	}
}

void CommandRecorder::SamplePlacement() {
	// The placement histogram (hang trace placement.csv) and CpuPlacementRecorderOffCore.
	Common::SamplePlacement(Common::ThreadRole::Recorder);
	const auto cp = m_cp_processor.load(std::memory_order_relaxed);
	if (cp == UINT32_MAX) {
		return;
	}
	const auto mine = CurrentLogicalProcessor();
	Profiler::CountFrameEvent(Profiler::FrameEvent::CpRecorderPlacementSamples);
	const auto core = CoreOf(mine);
	if (core != UINT32_MAX && core == CoreOf(cp)) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::CpRecorderSameCoreSamples);
	}
}

void CommandRecorder::PrintDrainLog() {
	if (!DrainLogEnabled() || m_drain_log.empty()) {
		return;
	}
	std::vector<std::pair<DrainKey, DrainStats>> sorted(m_drain_log.begin(), m_drain_log.end());
	std::sort(sorted.begin(), sorted.end(),
	          [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
	if (sorted.size() > 40) {
		sorted.resize(40);
	}
	uintptr_t base = 0;
#if defined(_WIN32)
	base = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
#endif
	char line[256];
	const auto emit = [&] {
		std::printf("%s", line);
		LOGF("%s", line);
	};
	std::snprintf(line, sizeof(line),
	              "Kyty CP recorder drains by site (%" PRIu64 " total, %" PRIu64
	              " idle, %.3f ms; count, idle, us each, us per waited drain):\n",
	              m_drains, m_idle_drains, static_cast<double>(m_drain_ns) / 1e6);
	emit();
	const auto per = [](uint64_t ns, uint64_t count) {
		return count != 0 ? static_cast<double>(ns) / 1e3 / static_cast<double>(count) : 0.0;
	};
	for (const auto& [key, stats]: sorted) {
		const double each_us   = per(stats.ns, stats.count);
		const double waited_us = per(stats.waited_ns, stats.count - stats.idle);
		if (key.is_site) {
			const auto* site = static_cast<const GpuOpProfiler::Site*>(key.key);
			std::snprintf(line, sizeof(line), "  %10" PRIu64 " %10" PRIu64 " %8.2f %8.2f  site %s\n",
			              stats.count, stats.idle, each_us, waited_us,
			              site != nullptr ? site->name : "?");
		} else {
			const auto address = reinterpret_cast<uintptr_t>(key.key);
			std::snprintf(line, sizeof(line),
			              "  %10" PRIu64 " %10" PRIu64 " %8.2f %8.2f  caller exe+0x%" PRIxPTR "\n",
			              stats.count, stats.idle, each_us, waited_us,
			              address >= base ? address - base : address);
		}
		emit();
	}
	std::fflush(stdout);
}

// ------------------------------------------------------------------------------------------------
// Verify ownership hooks

bool CommandRecorder::MayRecord(VkCommandBuffer command) noexcept {
	if (command == nullptr) {
		return true;
	}
	for (size_t i = 0; i < MaxRecorders; i++) {
		if (g_recording[i].load(std::memory_order_acquire) != command) {
			continue;
		}
		const auto* recorder = g_recorders[i].load(std::memory_order_acquire);
		if (recorder == nullptr) {
			return true;
		}
		// The replayer records while no direct window is open; everyone else only inside one
		// (the recorder is then idle with every earlier packet executed).
		const bool window = recorder->m_window_open.load(std::memory_order_acquire);
		return t_replaying ? !window : window;
	}
	return true;
}

void CommandRecorder::InstallVerifyHooks() {
	if (ConfiguredMode() == Mode::Off || !VerifyEnabled()) {
		return;
	}
	static std::once_flag once;
	std::call_once(once, [] {
		auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
#define KYTY_RECORDER_HOOK(member) InstallHook<__COUNTER__>(d.member, #member)
		KYTY_RECORDER_HOOK(vkBeginCommandBuffer);
		KYTY_RECORDER_HOOK(vkEndCommandBuffer);
		KYTY_RECORDER_HOOK(vkCmdBeginRendering);
		KYTY_RECORDER_HOOK(vkCmdEndRendering);
		KYTY_RECORDER_HOOK(vkCmdPipelineBarrier);
		KYTY_RECORDER_HOOK(vkCmdPipelineBarrier2);
		KYTY_RECORDER_HOOK(vkCmdCopyBuffer);
		KYTY_RECORDER_HOOK(vkCmdCopyBufferToImage);
		KYTY_RECORDER_HOOK(vkCmdCopyImageToBuffer);
		KYTY_RECORDER_HOOK(vkCmdCopyImage);
		KYTY_RECORDER_HOOK(vkCmdBlitImage);
		KYTY_RECORDER_HOOK(vkCmdResolveImage);
		KYTY_RECORDER_HOOK(vkCmdClearColorImage);
		KYTY_RECORDER_HOOK(vkCmdClearDepthStencilImage);
		KYTY_RECORDER_HOOK(vkCmdClearAttachments);
		KYTY_RECORDER_HOOK(vkCmdFillBuffer);
		KYTY_RECORDER_HOOK(vkCmdUpdateBuffer);
		KYTY_RECORDER_HOOK(vkCmdBindPipeline);
		KYTY_RECORDER_HOOK(vkCmdBindDescriptorSets);
		KYTY_RECORDER_HOOK(vkCmdPushDescriptorSetKHR);
		KYTY_RECORDER_HOOK(vkCmdPushConstants);
		KYTY_RECORDER_HOOK(vkCmdBindVertexBuffers);
		KYTY_RECORDER_HOOK(vkCmdBindVertexBuffers2);
		KYTY_RECORDER_HOOK(vkCmdBindIndexBuffer);
		KYTY_RECORDER_HOOK(vkCmdSetViewport);
		KYTY_RECORDER_HOOK(vkCmdSetScissor);
		KYTY_RECORDER_HOOK(vkCmdSetViewportWithCount);
		KYTY_RECORDER_HOOK(vkCmdSetScissorWithCount);
		KYTY_RECORDER_HOOK(vkCmdSetLineWidth);
		KYTY_RECORDER_HOOK(vkCmdSetBlendConstants);
		KYTY_RECORDER_HOOK(vkCmdSetDepthTestEnable);
		KYTY_RECORDER_HOOK(vkCmdSetDepthWriteEnable);
		KYTY_RECORDER_HOOK(vkCmdSetDepthCompareOp);
		KYTY_RECORDER_HOOK(vkCmdSetDepthBiasEnable);
		KYTY_RECORDER_HOOK(vkCmdSetDepthBias);
		KYTY_RECORDER_HOOK(vkCmdSetStencilTestEnable);
		KYTY_RECORDER_HOOK(vkCmdSetStencilOp);
		KYTY_RECORDER_HOOK(vkCmdSetStencilCompareMask);
		KYTY_RECORDER_HOOK(vkCmdSetStencilWriteMask);
		KYTY_RECORDER_HOOK(vkCmdSetStencilReference);
		KYTY_RECORDER_HOOK(vkCmdSetCullMode);
		KYTY_RECORDER_HOOK(vkCmdSetFrontFace);
		KYTY_RECORDER_HOOK(vkCmdSetDepthBoundsTestEnable);
		KYTY_RECORDER_HOOK(vkCmdSetDepthBounds);
		KYTY_RECORDER_HOOK(vkCmdSetColorWriteEnableEXT);
		KYTY_RECORDER_HOOK(vkCmdSetAttachmentFeedbackLoopEnableEXT);
		KYTY_RECORDER_HOOK(vkCmdDraw);
		KYTY_RECORDER_HOOK(vkCmdDrawIndexed);
		KYTY_RECORDER_HOOK(vkCmdDrawIndirect);
		KYTY_RECORDER_HOOK(vkCmdDrawIndexedIndirect);
		KYTY_RECORDER_HOOK(vkCmdDrawIndirectCount);
		KYTY_RECORDER_HOOK(vkCmdDrawIndexedIndirectCount);
		KYTY_RECORDER_HOOK(vkCmdDrawMeshTasksEXT);
		KYTY_RECORDER_HOOK(vkCmdDrawMeshTasksIndirectEXT);
		KYTY_RECORDER_HOOK(vkCmdDrawMeshTasksIndirectCountEXT);
		KYTY_RECORDER_HOOK(vkCmdDispatch);
		KYTY_RECORDER_HOOK(vkCmdDispatchIndirect);
		KYTY_RECORDER_HOOK(vkCmdResetQueryPool);
		KYTY_RECORDER_HOOK(vkCmdBeginQuery);
		KYTY_RECORDER_HOOK(vkCmdEndQuery);
		KYTY_RECORDER_HOOK(vkCmdCopyQueryPoolResults);
		KYTY_RECORDER_HOOK(vkCmdWriteTimestamp);
		KYTY_RECORDER_HOOK(vkCmdWriteTimestamp2);
		KYTY_RECORDER_HOOK(vkCmdBeginConditionalRenderingEXT);
		KYTY_RECORDER_HOOK(vkCmdEndConditionalRenderingEXT);
#undef KYTY_RECORDER_HOOK
		std::printf("Kyty CP recorder verify: ownership hooks installed\n");
		std::fflush(stdout);
	});
}

} // namespace Libs::Graphics
