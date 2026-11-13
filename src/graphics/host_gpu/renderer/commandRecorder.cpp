#include "graphics/host_gpu/renderer/commandRecorder.h"

#include "common/assert.h"
#include "common/cpuPlacement.h"
#include "common/hangWatchdog.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/drawPrep/cpGaps.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"
#include "graphics/host_gpu/renderer/gpuTiming.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/mtRecord.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/watchdogSubmit.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <chrono>
#include <condition_variable>
#include <deque>
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
			// The work the batch reads, before queue_mutex (SubmitDependency).
			submit.WaitHostDependencies();
			vk::Result result;
			{
				Common::LockGuard         lock(graphics.queue_mutex);
				Profiler::ScopedFrameWait frame_wait(Profiler::FrameWait::DriverSubmit);
				HangWatchdog::Scope       submit(
				    "vkQueueSubmit-recorder",
				    reinterpret_cast<uint64_t>(static_cast<VkQueue>(graphics.queue)), p.tick, 0, 0,
				    p.debug_submit);
				HangWatchdog::DebugDelay("submit", p.tick);
				NoteWatchdogSubmit(graphics.queue, submit_info, p.tick);
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
	// KYTY_MT_RECORD: added to the pass flags (secondary command buffer contents).
	vk::RenderingFlags extra_rendering_flags {};
	void beginRendering(const vk::RenderingInfo& info) {
		if (extra_rendering_flags) {
			auto copy  = info;
			copy.flags |= extra_rendering_flags;
			command.beginRendering(copy);
			return;
		}
		command.beginRendering(info);
	}
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
	void clearColorImage(vk::Image image, vk::ImageLayout layout, const vk::ClearColorValue* color,
	                     uint32_t count, const vk::ImageSubresourceRange* ranges) {
		command.clearColorImage(image, layout, color, count, ranges);
	}
	void clearDepthStencilImage(vk::Image image, vk::ImageLayout layout,
	                            const vk::ClearDepthStencilValue* value, uint32_t count,
	                            const vk::ImageSubresourceRange* ranges) {
		command.clearDepthStencilImage(image, layout, value, count, ranges);
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
	void dispatchBase(uint32_t bx, uint32_t by, uint32_t bz, uint32_t x, uint32_t y, uint32_t z) {
		command.dispatchBase(bx, by, bz, x, y, z);
	}
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
	void writeBufferMarkerAMD(vk::PipelineStageFlagBits stage, vk::Buffer buffer, vk::DeviceSize offset,
	                          uint32_t marker) {
		command.writeBufferMarkerAMD(stage, buffer, offset, marker);
	}
	// KYTY_STREAM_COPY_DEFER (BufferCache::ObtainBufferNow): the stream copy the CP reserved.
	void hostCopy(void* destination, uint64_t source, uint64_t size) {
		if (!Libs::LibKernel::Memory::TryReadBackingDirect(source, destination, size)) {
			std::memset(destination, 0, static_cast<size_t>(size));
		}
	}
};

// ------------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------------
// KYTY_MT_RECORD (mtRecord.h): render passes recorded into secondary command buffers in parallel.

namespace {

using CommandStream::Header;

constexpr uint32_t MtMaxSets   = 8;
constexpr uint32_t MtMaxVb     = 32;
constexpr uint32_t MtMaxPc     = 16;
constexpr uint32_t MtMaxColors = 8;
constexpr uint32_t MtNoKey     = UINT32_MAX;
constexpr uint32_t MtDynCount  = static_cast<uint32_t>(Op::SetAttachmentFeedbackLoopEnable) -
                                static_cast<uint32_t>(Op::SetViewportWithCount) + 1u;
enum : uint32_t {
	MtKeyGfxPipeline = 0,
	MtKeyCsPipeline  = 1,
	MtKeyGfxSets     = 2,
	MtKeyCsSets      = MtKeyGfxSets + MtMaxSets,
	MtKeyGfxPush     = MtKeyCsSets + MtMaxSets,
	MtKeyCsPush      = MtKeyGfxPush + MtMaxSets,
	MtKeyVb          = MtKeyCsPush + MtMaxSets,
	MtKeyIndex       = MtKeyVb + MtMaxVb,
	MtKeyDyn         = MtKeyIndex + 1,
	MtKeyStencil     = MtKeyDyn + MtDynCount, // 4 face-keyed stencil commands x 3 face masks
	MtKeyPc          = MtKeyStencil + 12,
	MtKeyCount       = MtKeyPc + MtMaxPc,
};

bool MtComputeKey(uint32_t key) {
	return key == MtKeyCsPipeline || (key >= MtKeyCsSets && key < MtKeyCsSets + MtMaxSets) ||
	       (key >= MtKeyCsPush && key < MtKeyCsPush + MtMaxSets);
}

enum class MtReason : uint32_t { Small, Incomplete, Op, Drain, Query, View, Flags, Large, State, Count };
constexpr const char* MtReasonNames[] = {"small", "incomplete", "op",    "drain", "query",
                                         "view",  "flags",      "large", "state"};
static_assert(std::size(MtReasonNames) == static_cast<size_t>(MtReason::Count));

bool MtIsDraw(Op op) {
	switch (op) {
		case Op::Draw:
		case Op::DrawIndexed:
		case Op::DrawMeshTasks:
		case Op::DrawMeshTasksIndirect:
		case Op::DrawMeshTasksIndirectCount:
		case Op::DrawIndirect:
		case Op::DrawIndexedIndirect:
		case Op::DrawIndirectCount:
		case Op::DrawIndexedIndirectCount: return true;
		default: return false;
	}
}

bool MtIsDynamic(Op op) {
	return static_cast<uint32_t>(op) >= static_cast<uint32_t>(Op::SetViewportWithCount) &&
	       static_cast<uint32_t>(op) <= static_cast<uint32_t>(Op::SetAttachmentFeedbackLoopEnable);
}

template <typename T>
const T& MtGet(const Header* header) {
	const CommandStream::SiteBlock*   site   = nullptr;
	const CommandStream::VerifyBlock* verify = nullptr;
	const auto offset = CommandStream::Detail::PayloadOffset(*header, &site, &verify);
	return *reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(header) + offset);
}

// The latest packet per state key since the primary's Begin. Replaying the kept packets in their
// original order reproduces the state the full sequence leaves behind.
struct MtTracker {
	struct Entry {
		uint64_t             seq     = 0;
		const Header*        ptr     = nullptr;
		bool                 in_ring = false; // points into the ring (segment being recorded)
		std::vector<uint8_t> owned;
	};
	struct PcKey {
		uint32_t stages = 0;
		uint32_t offset = 0;
		uint32_t size   = 0;
	};
	std::array<Entry, MtKeyCount> entries;
	std::array<PcKey, MtMaxPc>    pc_keys {};
	uint32_t                      pc_count       = 0;
	uint64_t                      seq            = 0;
	uint64_t                      tick           = 0;
	uint32_t                      active_queries = 0;
	bool                          unknown        = false;
	std::vector<std::pair<uint64_t, const Header*>> scratch;

	void Reset() {
		for (auto& e: entries) {
			e.seq     = 0;
			e.ptr     = nullptr;
			e.in_ring = false;
		}
		pc_count       = 0;
		active_queries = 0;
		unknown        = false;
	}

	uint32_t BindKey(vk::PipelineBindPoint point, uint32_t index, uint32_t gfx, uint32_t cs,
	                 uint32_t limit) {
		if (index >= limit) {
			unknown = true;
			return MtNoKey;
		}
		if (point == vk::PipelineBindPoint::eGraphics) {
			return gfx + index;
		}
		if (point == vk::PipelineBindPoint::eCompute) {
			return cs + index;
		}
		unknown = true;
		return MtNoKey;
	}

	uint32_t StencilKey(uint32_t command, vk::StencilFaceFlags faces) {
		const auto f = static_cast<uint32_t>(static_cast<VkStencilFaceFlags>(faces));
		if (f < 1u || f > 3u) {
			unknown = true;
			return MtNoKey;
		}
		return MtKeyStencil + command * 3u + (f - 1u);
	}

	uint32_t Key(const Header* h) {
		switch (h->op) {
			case Op::BindPipeline: {
				const auto& p = MtGet<CommandStream::BindPipelinePacket>(h);
				return BindKey(p.point, 0, MtKeyGfxPipeline, MtKeyCsPipeline, 1);
			}
			case Op::BindDescriptorSets: {
				const auto& p = MtGet<CommandStream::BindDescriptorSetsPacket>(h);
				return BindKey(p.point, p.first_set, MtKeyGfxSets, MtKeyCsSets, MtMaxSets);
			}
			case Op::PushDescriptorSet: {
				const auto& p = MtGet<CommandStream::PushDescriptorSetPacket>(h);
				return BindKey(p.point, p.set, MtKeyGfxPush, MtKeyCsPush, MtMaxSets);
			}
			case Op::BindVertexBuffers2: {
				const auto& p = MtGet<CommandStream::BindVertexBuffers2Packet>(h);
				if (p.first >= MtMaxVb) {
					unknown = true;
					return MtNoKey;
				}
				return MtKeyVb + p.first;
			}
			case Op::BindIndexBuffer: return MtKeyIndex;
			case Op::SetStencilOp:
				return StencilKey(0, MtGet<CommandStream::StencilOpPacket>(h).faces);
			case Op::SetStencilCompareMask:
				return StencilKey(1, MtGet<CommandStream::StencilValuePacket>(h).faces);
			case Op::SetStencilWriteMask:
				return StencilKey(2, MtGet<CommandStream::StencilValuePacket>(h).faces);
			case Op::SetStencilReference:
				return StencilKey(3, MtGet<CommandStream::StencilValuePacket>(h).faces);
			case Op::PushConstants: {
				const auto& p      = MtGet<CommandStream::PushConstantsPacket>(h);
				const auto  stages = static_cast<uint32_t>(static_cast<VkShaderStageFlags>(p.stages));
				for (uint32_t i = 0; i < pc_count; i++) {
					if (pc_keys[i].stages == stages && pc_keys[i].offset == p.offset &&
					    pc_keys[i].size == p.size) {
						return MtKeyPc + i;
					}
				}
				if (pc_count == MtMaxPc) {
					unknown = true;
					return MtNoKey;
				}
				pc_keys[pc_count] = {stages, p.offset, p.size};
				return MtKeyPc + pc_count++;
			}
			default:
				if (MtIsDynamic(h->op)) {
					return MtKeyDyn + (static_cast<uint32_t>(h->op) -
					                   static_cast<uint32_t>(Op::SetViewportWithCount));
				}
				return MtNoKey;
		}
	}

	void Track(const Header* h, bool in_ring) {
		switch (h->op) {
			case Op::Begin:
				Reset();
				tick = MtGet<CommandStream::BeginPacket>(h).tick;
				return;
			case Op::BeginQuery: active_queries++; return;
			case Op::EndQuery:
				if (active_queries != 0) {
					active_queries--;
				}
				return;
			default: break;
		}
		const auto key = Key(h);
		if (key == MtNoKey) {
			return;
		}
		auto& e = entries[key];
		e.seq   = ++seq;
		if (in_ring) {
			e.ptr     = h;
			e.in_ring = true;
		} else {
			const auto* bytes = reinterpret_cast<const uint8_t*>(h);
			e.owned.assign(bytes, bytes + h->size);
			e.ptr     = reinterpret_cast<const Header*>(e.owned.data());
			e.in_ring = false;
		}
	}

	// Copies the entries that point into the ring (before the ring region is released).
	void Materialize() {
		for (auto& e: entries) {
			if (e.in_ring && e.ptr != nullptr) {
				const auto* bytes = reinterpret_cast<const uint8_t*>(e.ptr);
				e.owned.assign(bytes, bytes + e.ptr->size);
				e.ptr     = reinterpret_cast<const Header*>(e.owned.data());
				e.in_ring = false;
			}
		}
	}

	void Snapshot(std::vector<const Header*>& out, bool graphics_only) {
		scratch.clear();
		for (uint32_t key = 0; key < MtKeyCount; key++) {
			const auto& e = entries[key];
			if (e.ptr != nullptr && !(graphics_only && MtComputeKey(key))) {
				scratch.emplace_back(e.seq, e.ptr);
			}
		}
		std::sort(scratch.begin(), scratch.end(),
		          [](const auto& a, const auto& b) { return a.first < b.first; });
		out.clear();
		for (const auto& [s, p]: scratch) {
			out.push_back(p);
		}
	}
};

struct MtChunk {
	std::vector<const Header*> prefix;
	std::vector<const Header*> packets;
	uint32_t                   draws = 0;
	uint64_t                   tick  = 0;
	vk::CommandBuffer          command = nullptr;
	bool                       pushed  = false; // handed to the workers
	std::atomic<bool>          done {false};
};

uint64_t MtEnvUs(const char* name, uint64_t default_value) {
	const auto* value = std::getenv(name);
	return value != nullptr && value[0] != '\0' ? std::strtoull(value, nullptr, 10) : default_value;
}

void MtPause() {
#if defined(_M_X64) || defined(__x86_64__)
	_mm_pause();
#endif
}

} // namespace

struct CommandRecorder::MtState {
	struct Slot {
		vk::CommandPool                                     pool = nullptr;
		std::vector<vk::CommandBuffer>                      free;
		std::deque<std::pair<uint64_t, vk::CommandBuffer>> pending;
		std::unique_ptr<NativeExecutor>                     exec;
		CommandStream::ReplayState                          replay;
	};

	explicit MtState(CommandRecorder& owner): owner(owner) {
		chunk_draws = static_cast<uint32_t>(std::clamp<uint64_t>(MtEnvUs("KYTY_MT_RECORD_CHUNK", 48), 4, 4096));
		min_draws   = static_cast<uint32_t>(std::clamp<uint64_t>(MtEnvUs("KYTY_MT_RECORD_MIN_DRAWS", 16), 1, 1u << 20u));
		wait_ns     = MtEnvUs("KYTY_MT_RECORD_WAIT_US", 4000) * 1000u;
		spin_ns     = MtEnvUs("KYTY_MT_RECORD_SPIN_US", 300) * 1000u;
		const auto workers = MtRecord::Threads();
		slots.resize(workers + 1u); // the last slot is the recorder thread's
		for (auto& slot: slots) {
			vk::CommandPoolCreateInfo info {};
			info.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer |
			             vk::CommandPoolCreateFlagBits::eTransient;
			info.queueFamilyIndex = owner.m_graphics.queue_family;
			EXIT_NOT_IMPLEMENTED(owner.m_graphics.device.createCommandPool(&info, nullptr, &slot.pool) !=
			                     vk::Result::eSuccess);
			slot.exec = std::make_unique<NativeExecutor>(owner);
		}
		for (uint32_t i = 0; i < workers; i++) {
			threads.emplace_back([this, i] { WorkerRun(i); });
		}
		last_print_ns = MtRecord::NowNs();
		std::printf("Kyty MtRecord: enabled, %u worker threads, chunk %u draws, min %u draws, wait %" PRIu64
		            " us\n",
		            workers, chunk_draws, min_draws, wait_ns / 1000u);
		std::fflush(stdout);
	}

	~MtState() { StopWorkers(); }

	void StopWorkers() {
		{
			std::lock_guard lock(queue_mutex);
			stop.store(true, std::memory_order_release);
		}
		queue_cv.notify_all();
		for (auto& t: threads) {
			if (t.joinable()) {
				t.join();
			}
		}
		threads.clear();
		// The command pools are left alive: the GPU may still execute their buffers.
	}

	void Push(MtChunk* chunk) {
		chunk->pushed = true;
		{
			std::lock_guard lock(queue_mutex);
			queue.push_back(chunk);
			queued.fetch_add(1, std::memory_order_release);
		}
		queue_cv.notify_one();
	}

	MtChunk* TryPop() {
		if (queued.load(std::memory_order_acquire) == 0) {
			return nullptr;
		}
		std::lock_guard lock(queue_mutex);
		if (queue.empty()) {
			return nullptr;
		}
		auto* chunk = queue.front();
		queue.pop_front();
		queued.fetch_sub(1, std::memory_order_release);
		return chunk;
	}

	void WorkerRun(uint32_t index) {
		Common::PlaceCurrentThread(Common::ThreadRole::Host);
		Common::RaiseCurrentThreadPriority();
		for (;;) {
			if (auto* chunk = TryPop()) {
				Record(index, *chunk);
				continue;
			}
			const auto start = MtRecord::NowNs();
			bool       found = false;
			while (MtRecord::NowNs() - start < spin_ns) {
				if (queued.load(std::memory_order_acquire) != 0 ||
				    stop.load(std::memory_order_acquire)) {
					found = true;
					break;
				}
				for (int i = 0; i < 32; i++) {
					MtPause();
				}
			}
			if (found) {
				std::lock_guard lock(queue_mutex);
				if (stop && queue.empty()) {
					return;
				}
				continue;
			}
			std::unique_lock lock(queue_mutex);
			queue_cv.wait(lock, [this] { return stop || !queue.empty(); });
			if (stop && queue.empty()) {
				return;
			}
		}
	}

	// Records one chunk into a secondary command buffer of slot `index` (one thread per slot).
	void Record(uint32_t index, MtChunk& chunk) {
		auto&      slot  = slots[index];
		const auto known = owner.m_master.KnownGpuTick();
		while (!slot.pending.empty() && slot.pending.front().first <= known) {
			slot.free.push_back(slot.pending.front().second);
			slot.pending.pop_front();
		}
		if (slot.free.empty()) {
			std::array<vk::CommandBuffer, 16> buffers {};
			vk::CommandBufferAllocateInfo     allocate {};
			allocate.commandPool        = slot.pool;
			allocate.level              = vk::CommandBufferLevel::eSecondary;
			allocate.commandBufferCount = static_cast<uint32_t>(buffers.size());
			EXIT_NOT_IMPLEMENTED(owner.m_graphics.device.allocateCommandBuffers(&allocate, buffers.data()) !=
			                     vk::Result::eSuccess);
			slot.free.insert(slot.free.end(), buffers.begin(), buffers.end());
		}
		const auto command = slot.free.back();
		slot.free.pop_back();

		vk::CommandBufferInheritanceRenderingInfo rendering {};
		rendering.flags                   = pass_flags;
		rendering.viewMask                = view_mask;
		rendering.colorAttachmentCount    = color_count;
		rendering.pColorAttachmentFormats = color_formats.data();
		rendering.depthAttachmentFormat   = depth_format;
		rendering.stencilAttachmentFormat = stencil_format;
		rendering.rasterizationSamples    = samples;
		vk::CommandBufferInheritanceInfo inheritance {};
		inheritance.pNext = &rendering;
		vk::CommandBufferBeginInfo begin {};
		begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit |
		              vk::CommandBufferUsageFlagBits::eRenderPassContinue;
		begin.pInheritanceInfo = &inheritance;
		EXIT_NOT_IMPLEMENTED(command.begin(&begin) != vk::Result::eSuccess);
		slot.exec->command = command;
		for (const auto* h: chunk.prefix) {
			CommandStream::Replay(*h, *slot.exec, slot.replay, nullptr, nullptr);
		}
		for (const auto* h: chunk.packets) {
			CommandStream::Replay(*h, *slot.exec, slot.replay, nullptr, nullptr);
		}
		slot.exec->command = nullptr;
		EXIT_NOT_IMPLEMENTED(command.end() != vk::Result::eSuccess);
		slot.pending.emplace_back(chunk.tick, command);
		chunk.command = command;
		chunk.done.store(true, std::memory_order_release);
	}

	MtChunk* NewChunk() {
		if (chunks_used == chunk_pool.size()) {
			chunk_pool.push_back(std::make_unique<MtChunk>());
		}
		auto* chunk = chunk_pool[chunks_used++].get();
		chunk->prefix.clear();
		chunk->packets.clear();
		chunk->draws   = 0;
		chunk->tick    = tracker.tick;
		chunk->command = nullptr;
		chunk->pushed  = false;
		chunk->done.store(false, std::memory_order_relaxed);
		return chunk;
	}

	// Recorder thread: waits for every dispatched chunk, recording queued ones itself meanwhile.
	void WaitChunks() {
		for (size_t i = 0; i < chunks_used; i++) {
			auto& chunk = *chunk_pool[i];
			if (chunk.pushed) {
				while (!chunk.done.load(std::memory_order_acquire)) {
					if (auto* other = TryPop()) {
						Record(static_cast<uint32_t>(slots.size() - 1u), *other);
					} else {
						MtPause();
					}
				}
			}
		}
	}

	CommandRecorder&                       owner;
	std::vector<Slot>                      slots;
	std::vector<std::thread>               threads;
	std::mutex                             queue_mutex;
	std::condition_variable                queue_cv;
	std::deque<MtChunk*>                   queue;
	std::atomic<uint32_t>                  queued {0};
	std::atomic<bool>                      stop {false};
	std::vector<std::unique_ptr<MtChunk>> chunk_pool;
	size_t                                 chunks_used = 0;
	MtTracker                              tracker;
	std::vector<const Header*>             emit;

	// Inheritance of the current segment (immutable while its chunks are recorded).
	vk::RenderingFlags                     pass_flags {};
	uint32_t                               view_mask   = 0;
	uint32_t                               color_count = 0;
	std::array<vk::Format, MtMaxColors>    color_formats {};
	vk::Format                             depth_format   = vk::Format::eUndefined;
	vk::Format                             stencil_format = vk::Format::eUndefined;
	vk::SampleCountFlagBits                samples        = vk::SampleCountFlagBits::e1;

	// UpdateDescriptorSets / HostCopy packets executed during the look-ahead of an abandoned
	// segment: skipped by the serial replay (ring positions, ascending).
	std::vector<uint64_t> executed;
	size_t                executed_next = 0;

	uint32_t chunk_draws = 48;
	uint32_t min_draws   = 16;
	uint64_t wait_ns     = 0;
	uint64_t spin_ns     = 0;

	// Stats (recorder thread).
	uint64_t                                               segments      = 0;
	uint64_t                                               chunks        = 0;
	uint64_t                                               draws_off     = 0;
	uint64_t                                               draws_serial  = 0;
	uint64_t                                               abandoned     = 0; // after chunks ran
	uint64_t                                               lookahead_wait_ns = 0;
	std::array<uint64_t, static_cast<size_t>(MtReason::Count)> fallbacks {};
	std::array<uint64_t, static_cast<size_t>(Op::Count)>    fallback_ops {};
	uint64_t                                               last_print_ns = 0;
	uint64_t                                               last_busy_ns  = 0;
	uint64_t                                               last_cp_ns    = 0;
	uint64_t                                               last_cp_draws = 0;
};

bool CommandRecorder::MtTrySegment(const Header* begin) {
	auto&      mt   = *m_mt;
	const auto fail = [&mt](MtReason reason) {
		mt.fallbacks[static_cast<size_t>(reason)]++;
		return false;
	};
	if (mt.tracker.unknown) {
		return fail(MtReason::State);
	}
	if (mt.tracker.active_queries != 0) {
		return fail(MtReason::Query);
	}
	if ((begin->flags & CommandStream::FlagVerify) != 0) {
		return fail(MtReason::Op);
	}
	// Inheritance: flags, formats and the sample count of the attachments.
	{
		const CommandStream::SiteBlock*   site   = nullptr;
		const CommandStream::VerifyBlock* verify = nullptr;
		CommandStream::Reader reader(begin, CommandStream::Detail::PayloadOffset(*begin, &site, &verify));
		const auto& p      = reader.Get<CommandStream::BeginRenderingPacket>();
		const auto* colors = reader.GetArray<vk::RenderingAttachmentInfo>(p.colors);
		const vk::RenderingAttachmentInfo* depth =
		    p.has_depth != 0 ? &reader.Get<vk::RenderingAttachmentInfo>() : nullptr;
		const vk::RenderingAttachmentInfo* stencil =
		    p.has_stencil != 0 ? &reader.Get<vk::RenderingAttachmentInfo>() : nullptr;
		if (p.flags & (vk::RenderingFlagBits::eSuspending | vk::RenderingFlagBits::eResuming |
		               vk::RenderingFlagBits::eContentsSecondaryCommandBuffers)) {
			return fail(MtReason::Flags);
		}
		if (p.colors > MtMaxColors) {
			return fail(MtReason::View);
		}
		bool have_samples = false;
		bool ok           = true;
		const auto lookup = [&](vk::ImageView view, vk::Format* format) {
			*format = vk::Format::eUndefined;
			if (!view) {
				return;
			}
			MtRecord::ViewInfo info;
			if (!MtRecord::LookupView(view, &info)) {
				ok = false;
				return;
			}
			if (have_samples && info.samples != mt.samples) {
				ok = false;
			}
			have_samples = true;
			mt.samples   = info.samples;
			*format      = info.format;
		};
		for (uint32_t i = 0; i < p.colors; i++) {
			lookup(colors[i].imageView, &mt.color_formats[i]);
		}
		lookup(depth != nullptr ? depth->imageView : vk::ImageView {}, &mt.depth_format);
		lookup(stencil != nullptr ? stencil->imageView : vk::ImageView {}, &mt.stencil_format);
		if (!ok || !have_samples) {
			return fail(MtReason::View);
		}
		mt.pass_flags  = p.flags;
		mt.view_mask   = p.view_mask;
		mt.color_count = p.colors;
	}

	// Look-ahead to the EndRendering: chunks are dispatched to the workers as they fill up, while
	// the CP still encodes the rest of the pass.
	const uint64_t start_pos  = m_ring.ReadPosition();
	uint64_t       pos        = start_pos + begin->size;
	const Header*  end        = nullptr;
	uint64_t       end_pos    = 0;
	uint32_t       draws      = 0;
	uint64_t       idle_since = 0;
	uint64_t       wait_start = 0;
	bool           dispatched = false;
	const size_t   executed_mark = mt.executed.size();
	MtReason       reason        = MtReason::Count;
	Op             bad_op        = Op::Wrap;
	mt.chunks_used               = 0;
	MtChunk* chunk               = mt.NewChunk();
	mt.tracker.Snapshot(chunk->prefix, true);
	while (end == nullptr) {
		const auto* h = m_ring.PeekAt(pos);
		if (h == nullptr) {
			if (m_stop.load(std::memory_order_acquire)) {
				reason = MtReason::Incomplete;
				break;
			}
			const auto now = MtRecord::NowNs();
			if (idle_since == 0) {
				idle_since = now;
				wait_start = now;
			} else if (now - idle_since >= mt.wait_ns) {
				mt.lookahead_wait_ns += now - wait_start;
				reason = MtReason::Incomplete;
				break;
			}
			if (auto* other = mt.TryPop()) {
				mt.Record(static_cast<uint32_t>(mt.slots.size() - 1u), *other);
			} else {
				for (int i = 0; i < 16; i++) {
					MtPause();
				}
			}
			continue;
		}
		if (idle_since != 0) {
			mt.lookahead_wait_ns += MtRecord::NowNs() - wait_start;
		}
		idle_since = 0;
		if (pos - start_pos > m_ring.Capacity() / 4u) {
			reason = MtReason::Large;
			break;
		}
		const auto op = h->op;
		if (op == Op::Wrap) {
			pos += h->size;
			continue;
		}
		if ((h->flags & CommandStream::FlagVerify) != 0) {
			reason = MtReason::Op;
			bad_op = op;
			break;
		}
		if (op == Op::EndRendering) {
			end     = h;
			end_pos = pos + h->size;
			break;
		}
		if (op == Op::DrainMarker) {
			reason = MtReason::Drain;
			break;
		}
		if (op == Op::UpdateDescriptorSets || op == Op::HostCopy) {
			// Device-side work in stream order, before any later command that binds the set is
			// recorded (by a worker). Executed now; skipped by a serial replay if abandoned.
			CommandStream::Replay(*h, *m_exec, m_replay, nullptr, nullptr);
			mt.executed.push_back(pos);
			pos += h->size;
			continue;
		}
		if (MtIsDraw(op)) {
			chunk->packets.push_back(h);
			chunk->draws++;
			draws++;
			if (chunk->draws >= mt.chunk_draws) {
				mt.Push(chunk);
				dispatched = true;
				chunk      = mt.NewChunk();
				mt.tracker.Snapshot(chunk->prefix, true);
			}
		} else if (op == Op::BindPipeline || op == Op::BindDescriptorSets ||
		           op == Op::PushDescriptorSet || op == Op::PushConstants ||
		           op == Op::BindVertexBuffers2 || op == Op::BindIndexBuffer || MtIsDynamic(op)) {
			if ((op == Op::BindPipeline &&
			     MtGet<CommandStream::BindPipelinePacket>(h).point != vk::PipelineBindPoint::eGraphics) ||
			    (op == Op::BindDescriptorSets &&
			     MtGet<CommandStream::BindDescriptorSetsPacket>(h).point != vk::PipelineBindPoint::eGraphics) ||
			    (op == Op::PushDescriptorSet &&
			     MtGet<CommandStream::PushDescriptorSetPacket>(h).point != vk::PipelineBindPoint::eGraphics)) {
				reason = MtReason::Op;
				bad_op = op;
				break;
			}
			chunk->packets.push_back(h);
			mt.tracker.Track(h, true);
			if (mt.tracker.unknown) {
				reason = MtReason::State;
				break;
			}
		} else {
			reason = MtReason::Op;
			bad_op = op;
			break;
		}
		pos += h->size;
	}
	if (end != nullptr && !dispatched && draws < mt.min_draws) {
		reason = MtReason::Small;
	}

	if (reason != MtReason::Count) {
		// Abandoned: finish what the workers started (their buffers are recycled with the tick),
		// then replay the pass serially. The tracker's in-ring entries are all packets of this
		// pass: the serial replay tracks them again (owned copies) in the same order.
		if (dispatched) {
			mt.WaitChunks();
			mt.abandoned++;
		}
		mt.chunks_used = 0;
		mt.fallbacks[static_cast<size_t>(reason)]++;
		if (reason == MtReason::Op) {
			mt.fallback_ops[static_cast<size_t>(bad_op)]++;
		}
		if (mt.tracker.unknown) {
			// Track() set it on a packet of this pass; the serial replay sets it again.
			mt.tracker.unknown = false;
		}
		return false;
	}

	if (!chunk->packets.empty()) {
		mt.Push(chunk);
	}
	// Packets executed ahead belong to this segment: no serial replay will see them.
	mt.executed.resize(executed_mark);

	// Primary: BeginRendering (secondary contents), the chunks in order, EndRendering.
	m_exec->extra_rendering_flags = vk::RenderingFlagBits::eContentsSecondaryCommandBuffers;
	CommandStream::Replay(*begin, *m_exec, m_replay, nullptr, nullptr);
	m_exec->extra_rendering_flags = {};
	mt.WaitChunks();
	static thread_local std::vector<vk::CommandBuffer> buffers;
	buffers.clear();
	for (size_t i = 0; i < mt.chunks_used; i++) {
		const auto& c = *mt.chunk_pool[i];
		if (c.command) {
			buffers.push_back(c.command);
			mt.draws_off += c.draws;
		}
	}
	if (!buffers.empty()) {
		m_exec->command.executeCommands(static_cast<uint32_t>(buffers.size()), buffers.data());
	}
	CommandStream::Replay(*end, *m_exec, m_replay, nullptr, nullptr);
	// The primary's state is undefined after vkCmdExecuteCommands: re-emit every tracked state
	// (graphics and compute) so the serial commands that follow see what they expect.
	mt.tracker.Snapshot(mt.emit, false);
	for (const auto* h: mt.emit) {
		CommandStream::Replay(*h, *m_exec, m_replay, nullptr, nullptr);
	}
	mt.tracker.Materialize();
	mt.segments++;
	mt.chunks += buffers.size();
	mt.chunks_used = 0;
	m_ring.Advance(static_cast<uint32_t>(end_pos - start_pos));
	m_ring.Release(m_consumer_stats);
	return true;
}

void CommandRecorder::MtMaybePrintStats() {
	auto&      mt  = *m_mt;
	const auto now = MtRecord::NowNs();
	if (now - mt.last_print_ns < 5'000'000'000ull) {
		return;
	}
	const double seconds = static_cast<double>(now - mt.last_print_ns) / 1e9;
	mt.last_print_ns     = now;
	const auto busy      = m_busy_ns.load(std::memory_order_acquire);
	const auto cp_ns     = MtRecord::g_cp_draw_ns.load(std::memory_order_relaxed);
	const auto cp_draws  = MtRecord::g_cp_draws.load(std::memory_order_relaxed);
	const auto d_busy    = busy - mt.last_busy_ns;
	const auto d_cp_ns   = cp_ns - mt.last_cp_ns;
	const auto d_cp      = cp_draws - mt.last_cp_draws;
	mt.last_busy_ns      = busy;
	mt.last_cp_ns        = cp_ns;
	mt.last_cp_draws     = cp_draws;
	const auto draws     = mt.draws_off + mt.draws_serial;
	// Top three ops that made a pass fall back.
	std::array<size_t, 3> top {SIZE_MAX, SIZE_MAX, SIZE_MAX};
	for (size_t i = 0; i < mt.fallback_ops.size(); i++) {
		if (mt.fallback_ops[i] == 0) {
			continue;
		}
		for (size_t t = 0; t < top.size(); t++) {
			if (top[t] == SIZE_MAX || mt.fallback_ops[i] > mt.fallback_ops[top[t]]) {
				for (size_t m = top.size() - 1; m > t; m--) {
					top[m] = top[m - 1];
				}
				top[t] = i;
				break;
			}
		}
	}
	std::string ops;
	for (const auto i: top) {
		if (i != SIZE_MAX) {
			char item[64];
			std::snprintf(item, sizeof(item), "%s%s=%" PRIu64, ops.empty() ? "" : ",",
			              CommandStream::OpName(static_cast<Op>(i)), mt.fallback_ops[i]);
			ops += item;
		}
	}
	std::string reasons;
	for (size_t i = 0; i < mt.fallbacks.size(); i++) {
		char item[48];
		std::snprintf(item, sizeof(item), " %s=%" PRIu64, MtReasonNames[i], mt.fallbacks[i]);
		reasons += item;
	}
	std::printf("MtRecord: %.1fs segments=%" PRIu64 " chunks=%" PRIu64 " draws_off=%" PRIu64
	            " draws_serial=%" PRIu64 " (%.1f%% off) abandoned=%" PRIu64 " fallbacks:%s [ops %s] lookahead_wait=%.1fms"
	            " | recorder busy %.2f us/draw | CP %.2f us/draw (%" PRIu64 " draws)\n",
	            seconds, mt.segments, mt.chunks, mt.draws_off, mt.draws_serial,
	            draws != 0 ? 100.0 * static_cast<double>(mt.draws_off) / static_cast<double>(draws) : 0.0,
	            mt.abandoned, reasons.c_str(), ops.empty() ? "-" : ops.c_str(),
	            static_cast<double>(mt.lookahead_wait_ns) / 1e6,
	            draws != 0 ? static_cast<double>(d_busy) / 1e3 / static_cast<double>(draws) : 0.0,
	            d_cp != 0 ? static_cast<double>(d_cp_ns) / 1e3 / static_cast<double>(d_cp) : 0.0, d_cp);
	std::fflush(stdout);
	mt.segments = mt.chunks = mt.draws_off = mt.draws_serial = mt.abandoned = mt.lookahead_wait_ns = 0;
	mt.fallbacks.fill(0);
	mt.fallback_ops.fill(0);
}

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
	if (MtRecord::Enabled()) {
		const auto* guard = std::getenv("KYTY_CMD_GUARD");
		const bool  guard_on = guard != nullptr && guard[0] != '\0' && std::strcmp(guard, "0") != 0;
		const char* why = mode != Mode::Thread                ? "recorder not in thread mode"
		                  : VerifyEnabled()                   ? "KYTY_CP_RECORDER_VERIFY"
		                  : gpu_ops || GpuOpProfiler::Enabled() || GpuOpProfiler::CaptureEnabled()
		                      ? "GPU op profiler"
		                  : guard_on ? "KYTY_CMD_GUARD"
		                             : nullptr;
		if (why == nullptr) {
			m_mt = std::make_unique<MtState>(*this);
		} else {
			std::printf("Kyty MtRecord: disabled (%s)\n", why);
			std::fflush(stdout);
		}
	}
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
	const CpGaps::Scope gap(CpGaps::Cat::RecorderWait);
	HangWatchdog::Scope wait("recorder-drain", reinterpret_cast<uint64_t>(this),
	                         HangWatchdog::Enabled() ? m_ring.WritePosition() : 0,
	                         HangWatchdog::Enabled() ? m_ring.Consumed() : 0);
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
	const CpGaps::Scope gap(CpGaps::Cat::RecorderWait);
	HangWatchdog::Scope wait("recorder-recorded-tick", reinterpret_cast<uint64_t>(this), tick,
	                         HangWatchdog::Enabled() ? m_recorded_tick.load() : 0);
	if (from_producer) {
		m_ring.Kick(m_encoder.Stats());
	}
	const auto start = NowNs();
	while (m_recorded_tick.load(std::memory_order_acquire) < tick) {
		if (NowNs() - start >= m_drain_policy.spin_ns) {
			m_recorded_waiters.fetch_add(1, std::memory_order_seq_cst);
			for (;;) {
				const auto value = m_recorded_tick.load(std::memory_order_seq_cst);
				wait.Observed(value);
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
	if (m_mt) {
		m_mt->StopWorkers();
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
		if (m_mt) {
			if (header->op == Op::BeginRendering && MtTrySegment(header)) {
				since = 0;
				continue;
			}
			if (header->op != Op::Wrap) {
				auto& mt = *m_mt;
				if (mt.executed_next < mt.executed.size() &&
				    mt.executed[mt.executed_next] == m_ring.ReadPosition()) {
					// Executed during an abandoned look-ahead (MtTrySegment).
					if (++mt.executed_next == mt.executed.size()) {
						mt.executed.clear();
						mt.executed_next = 0;
					}
				} else {
					CommandStream::Replay(*header, *m_exec, m_replay, &OnMismatch, this);
					mt.tracker.Track(header, false);
					if (MtIsDraw(header->op)) {
						mt.draws_serial++;
					}
				}
			}
		} else if (header->op != Op::Wrap) {
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
	if (m_mt) {
		MtMaybePrintStats();
	}
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
#undef KYTY_RECORDER_HOOK
		std::printf("Kyty CP recorder verify: ownership hooks installed\n");
		std::fflush(stdout);
	});
}

// ---- KYTY_CMD_GUARD: command-buffer state machine on the dispatcher ----

namespace {

enum class GuardState : uint32_t { Initial = 0, Recording, Executable, Pending, Freed };

struct GuardEntry {
	std::atomic<VkCommandBuffer> key {nullptr};
	std::atomic<uint32_t>        pool {0};
	std::atomic<uint32_t>        state {0};
	std::atomic<VkSemaphore>     semaphore {VK_NULL_HANDLE};
	std::atomic<uint64_t>        value {0};
	std::atomic<uint32_t>        last_tid {0};
	std::atomic<uint32_t>        generation {0}; // +1 at every begin
};

struct GuardPool {
	std::atomic<VkCommandPool> handle {VK_NULL_HANDLE};
	std::atomic<uint32_t>      owner {0};
};

constexpr size_t GuardSlots      = 16384;
constexpr size_t GuardPools      = 64;
constexpr size_t GuardTimelines  = 64;
constexpr uint64_t GuardMaxPrint = 8;

std::array<GuardEntry, GuardSlots>                   g_guard_entries;
std::array<GuardPool, GuardPools>                    g_guard_pools;
std::array<std::atomic<VkSemaphore>, GuardTimelines> g_guard_timelines {};
std::atomic<uint64_t>                                g_guard_violations {0};
std::mutex                                           g_guard_mutex; // inserts only
VkDevice                                             g_guard_device = VK_NULL_HANDLE;
int                                                  g_guard_mode   = 0; // 1 report, 2 exit, 3 wait
PFN_vkGetSemaphoreCounterValue                       g_guard_counter = nullptr;
PFN_vkWaitSemaphores                                 g_guard_wait    = nullptr;

uint32_t GuardTid() noexcept {
#if defined(_WIN32)
	static thread_local const uint32_t tid = static_cast<uint32_t>(GetCurrentThreadId());
#else
	static thread_local const uint32_t tid =
	    static_cast<uint32_t>(std::hash<std::thread::id> {}(std::this_thread::get_id()) | 1u);
#endif
	return tid;
}

std::string GuardThreadName(uint32_t tid) {
#if defined(_WIN32)
	std::string name = std::to_string(tid);
	if (HANDLE thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, tid); thread != nullptr) {
		PWSTR description = nullptr;
		if (SUCCEEDED(GetThreadDescription(thread, &description)) && description != nullptr) {
			char narrow[128] = {};
			WideCharToMultiByte(CP_UTF8, 0, description, -1, narrow, sizeof(narrow) - 1, nullptr, nullptr);
			LocalFree(description);
			if (narrow[0] != '\0') {
				name += std::string(" '") + narrow + "'";
			}
		}
		CloseHandle(thread);
	}
	return name;
#else
	return std::to_string(tid);
#endif
}

size_t GuardHash(const void* key) noexcept {
	return static_cast<size_t>((reinterpret_cast<uintptr_t>(key) >> 4u) * 0x9e3779b97f4a7c15ull >> 50u) %
	       GuardSlots;
}

GuardEntry* GuardFind(VkCommandBuffer command) noexcept {
	if (command == nullptr) {
		return nullptr;
	}
	for (size_t i = 0, slot = GuardHash(command); i < GuardSlots; i++, slot = (slot + 1) % GuardSlots) {
		const auto key = g_guard_entries[slot].key.load(std::memory_order_acquire);
		if (key == command) {
			return &g_guard_entries[slot];
		}
		if (key == nullptr) {
			return nullptr;
		}
	}
	return nullptr;
}

uint32_t GuardPoolIndex(VkCommandPool pool) {
	std::lock_guard lock(g_guard_mutex);
	for (uint32_t i = 0; i < GuardPools; i++) {
		const auto handle = g_guard_pools[i].handle.load(std::memory_order_acquire);
		if (handle == pool) {
			return i;
		}
		if (handle == VK_NULL_HANDLE) {
			g_guard_pools[i].handle.store(pool, std::memory_order_release);
			return i;
		}
	}
	return GuardPools - 1;
}

void GuardInsert(VkCommandBuffer command, uint32_t pool) {
	std::lock_guard lock(g_guard_mutex);
	for (size_t i = 0, slot = GuardHash(command); i < GuardSlots; i++, slot = (slot + 1) % GuardSlots) {
		auto&      entry = g_guard_entries[slot];
		const auto key   = entry.key.load(std::memory_order_acquire);
		if (key == command || key == nullptr) {
			entry.pool.store(pool, std::memory_order_relaxed);
			entry.state.store(static_cast<uint32_t>(GuardState::Initial), std::memory_order_relaxed);
			entry.semaphore.store(VK_NULL_HANDLE, std::memory_order_relaxed);
			entry.value.store(0, std::memory_order_relaxed);
			entry.key.store(command, std::memory_order_release);
			return;
		}
	}
}

bool GuardIsTimeline(VkSemaphore semaphore) noexcept {
	for (const auto& slot: g_guard_timelines) {
		const auto value = slot.load(std::memory_order_acquire);
		if (value == semaphore) {
			return true;
		}
		if (value == VK_NULL_HANDLE) {
			return false;
		}
	}
	return false;
}

const char* GuardStateName(uint32_t state) {
	switch (static_cast<GuardState>(state)) {
		case GuardState::Initial: return "initial";
		case GuardState::Recording: return "recording";
		case GuardState::Executable: return "executable";
		case GuardState::Pending: return "pending";
		case GuardState::Freed: return "freed";
	}
	return "?";
}

void GuardViolation(const char* what, const char* function, VkCommandBuffer command,
                    const GuardEntry* entry, uint32_t other_tid, uint64_t reached) {
	const auto count = g_guard_violations.fetch_add(1, std::memory_order_relaxed) + 1;
	// Per kind (the what string is a literal): the first few of each.
	static std::array<std::atomic<const char*>, 16> kinds {};
	static std::array<std::atomic<uint64_t>, 16>    printed {};
	size_t kind = kinds.size() - 1;
	for (size_t i = 0; i < kinds.size(); i++) {
		const char* expected = nullptr;
		if (kinds[i].load() == what || kinds[i].compare_exchange_strong(expected, what) || expected == what) {
			kind = i;
			break;
		}
	}
	const auto of_kind = printed[kind].fetch_add(1, std::memory_order_relaxed) + 1;
	if (of_kind % 1000 == 0) {
		std::printf("Kyty command guard: %" PRIu64 " x \"%s\" so far (%" PRIu64 " violations in all)\n",
		            of_kind, what, count);
		std::fflush(stdout);
	}
	if (of_kind <= GuardMaxPrint) {
		const auto self  = GuardThreadName(GuardTid());
		const auto other = other_tid != 0 ? GuardThreadName(other_tid) : std::string("-");
		char       line[512];
		std::snprintf(line, sizeof(line),
		              "Kyty command guard #%" PRIu64 ": %s in %s: buffer %p pool #%u state %s, "
		              "thread %s, other thread %s, signal value %" PRIu64 " reached %" PRIu64 "\n",
		              count, what, function, static_cast<void*>(command),
		              entry != nullptr ? entry->pool.load() : 0u,
		              entry != nullptr ? GuardStateName(entry->state.load()) : "?", self.c_str(),
		              other.c_str(), entry != nullptr ? entry->value.load() : 0ull, reached);
		std::printf("%s", line);
		std::fflush(stdout);
		LOGF("%s", line);
	}
	if (g_guard_mode == 2) {
		EXIT("Kyty command guard: %s in %s\n", what, function);
	}
}

// The pool is externally synchronized: no two threads inside one of its buffers' commands.
uint32_t GuardEnter(GuardEntry* entry, const char* function, VkCommandBuffer command) noexcept {
	const auto tid  = GuardTid();
	const auto prev = g_guard_pools[entry->pool.load(std::memory_order_relaxed)].owner.exchange(
	    tid, std::memory_order_acq_rel);
	if (prev != 0 && prev != tid) {
		GuardViolation("two threads in one command pool", function, command, entry, prev, 0);
	}
	entry->last_tid.store(tid, std::memory_order_relaxed);
	return tid;
}

void GuardLeave(GuardEntry* entry, uint32_t tid) noexcept {
	auto expected = tid;
	g_guard_pools[entry->pool.load(std::memory_order_relaxed)].owner.compare_exchange_strong(
	    expected, 0, std::memory_order_acq_rel);
}

// Begin / reset / free: the buffer's last submission must have completed.
void GuardCheckIdle(GuardEntry* entry, const char* function, VkCommandBuffer command) {
	if (entry->state.load(std::memory_order_acquire) != static_cast<uint32_t>(GuardState::Pending)) {
		return;
	}
	const auto semaphore = entry->semaphore.load(std::memory_order_acquire);
	const auto value     = entry->value.load(std::memory_order_acquire);
	if (semaphore == VK_NULL_HANDLE || g_guard_counter == nullptr) {
		return;
	}
	uint64_t reached = 0;
	if (g_guard_counter(g_guard_device, semaphore, &reached) != VK_SUCCESS) {
		return;
	}
	if (reached >= value) {
		return;
	}
	GuardViolation("reuse of a buffer whose submission has not completed", function, command, entry,
	               0, reached);
	if (g_guard_mode == 3 && g_guard_wait != nullptr) {
		VkSemaphoreWaitInfo wait {};
		wait.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
		wait.semaphoreCount = 1;
		wait.pSemaphores    = &semaphore;
		wait.pValues        = &value;
		(void)g_guard_wait(g_guard_device, &wait, UINT64_MAX);
	}
}

template <size_t Index, typename Fn>
struct GuardCmdHook;

template <size_t Index, typename... Args>
struct GuardCmdHook<Index, void(VKAPI_PTR*)(VkCommandBuffer, Args...)> {
	static inline void(VKAPI_PTR* real)(VkCommandBuffer, Args...) = nullptr;
	static inline const char* name                                = nullptr;
	static void VKAPI_PTR     Call(VkCommandBuffer command, Args... args) {
		auto* entry = GuardFind(command);
		if (entry == nullptr) {
			real(command, args...);
			return;
		}
		const auto tid = GuardEnter(entry, name, command);
		if (entry->state.load(std::memory_order_acquire) != static_cast<uint32_t>(GuardState::Recording)) {
			GuardViolation("command into a buffer that is not recording", name, command, entry, 0, 0);
		}
		real(command, args...);
		GuardLeave(entry, tid);
	}
};

template <size_t Index, typename Fn>
void InstallGuardHook(Fn& slot, const char* name) {
	if (slot == nullptr) {
		return;
	}
	GuardCmdHook<Index, Fn>::real = slot;
	GuardCmdHook<Index, Fn>::name = name;
	slot                          = &GuardCmdHook<Index, Fn>::Call;
}

PFN_vkBeginCommandBuffer     g_guard_real_begin    = nullptr;
PFN_vkEndCommandBuffer       g_guard_real_end      = nullptr;
PFN_vkResetCommandBuffer     g_guard_real_reset    = nullptr;
PFN_vkAllocateCommandBuffers g_guard_real_allocate = nullptr;
PFN_vkFreeCommandBuffers     g_guard_real_free     = nullptr;
PFN_vkQueueSubmit            g_guard_real_submit   = nullptr;
PFN_vkQueueSubmit2           g_guard_real_submit2  = nullptr;
PFN_vkCreateSemaphore        g_guard_real_semaphore = nullptr;
PFN_vkQueuePresentKHR        g_guard_real_present   = nullptr;
PFN_vkQueueBindSparse        g_guard_real_sparse    = nullptr;
PFN_vkQueueWaitIdle          g_guard_real_idle      = nullptr;
PFN_vkDestroyCommandPool     g_guard_real_destroy_pool = nullptr;
PFN_vkResetCommandPool       g_guard_real_reset_pool   = nullptr;

// Queues are externally synchronized too: no two threads in vkQueue* of one queue at once.
struct GuardQueue {
	std::atomic<VkQueue>  handle {VK_NULL_HANDLE};
	std::atomic<uint32_t> owner {0};
};
std::array<GuardQueue, 8> g_guard_queues;

GuardQueue& GuardQueueOf(VkQueue queue) {
	for (auto& slot: g_guard_queues) {
		const auto handle = slot.handle.load(std::memory_order_acquire);
		if (handle == queue) {
			return slot;
		}
		if (handle == VK_NULL_HANDLE) {
			std::lock_guard lock(g_guard_mutex);
			auto            expected = VkQueue {VK_NULL_HANDLE};
			if (slot.handle.compare_exchange_strong(expected, queue) || expected == queue) {
				return slot;
			}
		}
	}
	return g_guard_queues.back();
}

struct GuardQueueScope {
	GuardQueue& queue;
	uint32_t    tid;
	GuardQueueScope(VkQueue handle, const char* function): queue(GuardQueueOf(handle)), tid(GuardTid()) {
		const auto prev = queue.owner.exchange(tid, std::memory_order_acq_rel);
		if (prev != 0 && prev != tid) {
			GuardViolation("two threads in one queue", function, nullptr, nullptr, prev, 0);
		}
	}
	~GuardQueueScope() {
		auto expected = tid;
		queue.owner.compare_exchange_strong(expected, 0, std::memory_order_acq_rel);
	}
	GuardQueueScope(const GuardQueueScope&)            = delete;
	GuardQueueScope& operator=(const GuardQueueScope&) = delete;
};

VkResult VKAPI_PTR GuardPresent(VkQueue queue, const VkPresentInfoKHR* info) {
	GuardQueueScope scope(queue, "vkQueuePresentKHR");
	return g_guard_real_present(queue, info);
}

VkResult VKAPI_PTR GuardBindSparse(VkQueue queue, uint32_t count, const VkBindSparseInfo* info, VkFence fence) {
	GuardQueueScope scope(queue, "vkQueueBindSparse");
	return g_guard_real_sparse(queue, count, info, fence);
}

VkResult VKAPI_PTR GuardWaitIdle(VkQueue queue) {
	GuardQueueScope scope(queue, "vkQueueWaitIdle");
	return g_guard_real_idle(queue);
}

void GuardCheckPoolIdle(VkCommandPool pool, const char* function) {
	uint32_t index = GuardPools;
	for (uint32_t i = 0; i < GuardPools; i++) {
		if (g_guard_pools[i].handle.load(std::memory_order_acquire) == pool) {
			index = i;
			break;
		}
	}
	if (index == GuardPools) {
		return;
	}
	for (auto& entry: g_guard_entries) {
		const auto key = entry.key.load(std::memory_order_acquire);
		if (key != nullptr && entry.pool.load(std::memory_order_relaxed) == index) {
			GuardCheckIdle(&entry, function, key);
		}
	}
}

void VKAPI_PTR GuardDestroyPool(VkDevice device, VkCommandPool pool, const VkAllocationCallbacks* allocator) {
	GuardCheckPoolIdle(pool, "vkDestroyCommandPool");
	g_guard_real_destroy_pool(device, pool, allocator);
}

VkResult VKAPI_PTR GuardResetPool(VkDevice device, VkCommandPool pool, VkCommandPoolResetFlags flags) {
	GuardCheckPoolIdle(pool, "vkResetCommandPool");
	return g_guard_real_reset_pool(device, pool, flags);
}

VkResult VKAPI_PTR GuardBegin(VkCommandBuffer command, const VkCommandBufferBeginInfo* info) {
	auto* entry = GuardFind(command);
	if (entry == nullptr) {
		return g_guard_real_begin(command, info);
	}
	const auto tid = GuardEnter(entry, "vkBeginCommandBuffer", command);
	GuardCheckIdle(entry, "vkBeginCommandBuffer", command);
	if (entry->state.load() == static_cast<uint32_t>(GuardState::Recording)) {
		GuardViolation("begin of a buffer that is recording", "vkBeginCommandBuffer", command, entry, 0, 0);
	}
	entry->generation.fetch_add(1, std::memory_order_acq_rel);
	entry->state.store(static_cast<uint32_t>(GuardState::Recording), std::memory_order_release);
	const auto result = g_guard_real_begin(command, info);
	GuardLeave(entry, tid);
	return result;
}

VkResult VKAPI_PTR GuardEnd(VkCommandBuffer command) {
	auto* entry = GuardFind(command);
	if (entry == nullptr) {
		return g_guard_real_end(command);
	}
	const auto tid = GuardEnter(entry, "vkEndCommandBuffer", command);
	if (entry->state.load() != static_cast<uint32_t>(GuardState::Recording)) {
		GuardViolation("end of a buffer that is not recording", "vkEndCommandBuffer", command, entry, 0, 0);
	}
	const auto result = g_guard_real_end(command);
	entry->state.store(static_cast<uint32_t>(GuardState::Executable), std::memory_order_release);
	GuardLeave(entry, tid);
	return result;
}

VkResult VKAPI_PTR GuardReset(VkCommandBuffer command, VkCommandBufferResetFlags flags) {
	auto* entry = GuardFind(command);
	if (entry == nullptr) {
		return g_guard_real_reset(command, flags);
	}
	const auto tid = GuardEnter(entry, "vkResetCommandBuffer", command);
	GuardCheckIdle(entry, "vkResetCommandBuffer", command);
	entry->state.store(static_cast<uint32_t>(GuardState::Initial), std::memory_order_release);
	const auto result = g_guard_real_reset(command, flags);
	GuardLeave(entry, tid);
	return result;
}

VkResult VKAPI_PTR GuardAllocate(VkDevice device, const VkCommandBufferAllocateInfo* info,
                                 VkCommandBuffer* buffers) {
	const auto pool   = GuardPoolIndex(info->commandPool);
	const auto tid    = GuardTid();
	const auto prev   = g_guard_pools[pool].owner.exchange(tid, std::memory_order_acq_rel);
	if (prev != 0 && prev != tid) {
		GuardViolation("two threads in one command pool", "vkAllocateCommandBuffers", nullptr, nullptr, prev, 0);
	}
	const auto result = g_guard_real_allocate(device, info, buffers);
	if (result == VK_SUCCESS) {
		for (uint32_t i = 0; i < info->commandBufferCount; i++) {
			GuardInsert(buffers[i], pool);
		}
	}
	auto expected = tid;
	g_guard_pools[pool].owner.compare_exchange_strong(expected, 0, std::memory_order_acq_rel);
	return result;
}

void VKAPI_PTR GuardFree(VkDevice device, VkCommandPool pool, uint32_t count, const VkCommandBuffer* buffers) {
	for (uint32_t i = 0; i < count; i++) {
		if (auto* entry = GuardFind(buffers[i]); entry != nullptr) {
			GuardCheckIdle(entry, "vkFreeCommandBuffers", buffers[i]);
			entry->state.store(static_cast<uint32_t>(GuardState::Freed), std::memory_order_release);
		}
	}
	g_guard_real_free(device, pool, count, buffers);
}

void GuardSubmitted(VkCommandBuffer command, VkSemaphore semaphore, uint64_t value, const char* function) {
	auto* entry = GuardFind(command);
	if (entry == nullptr) {
		return;
	}
	if (entry->state.load() != static_cast<uint32_t>(GuardState::Executable)) {
		GuardViolation("submit of a buffer that is not executable", function, command, entry,
		               entry->last_tid.load(), 0);
	}
	entry->value.store(value, std::memory_order_relaxed);
	entry->semaphore.store(semaphore, std::memory_order_relaxed);
	entry->state.store(static_cast<uint32_t>(GuardState::Pending), std::memory_order_release);
}

VkResult VKAPI_PTR GuardSubmit(VkQueue queue, uint32_t count, const VkSubmitInfo* submits, VkFence fence) {
	for (uint32_t s = 0; s < count; s++) {
		const auto& submit    = submits[s];
		VkSemaphore semaphore = VK_NULL_HANDLE;
		uint64_t    value     = 0;
		for (const auto* next = static_cast<const VkBaseInStructure*>(submit.pNext); next != nullptr;
		     next             = next->pNext) {
			if (next->sType != VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO) {
				continue;
			}
			const auto* timeline = reinterpret_cast<const VkTimelineSemaphoreSubmitInfo*>(next);
			for (uint32_t i = 0; i < submit.signalSemaphoreCount && i < timeline->signalSemaphoreValueCount; i++) {
				if (GuardIsTimeline(submit.pSignalSemaphores[i])) {
					semaphore = submit.pSignalSemaphores[i];
					value     = timeline->pSignalSemaphoreValues[i];
				}
			}
		}
		for (uint32_t i = 0; i < submit.commandBufferCount; i++) {
			GuardSubmitted(submit.pCommandBuffers[i], semaphore, value, "vkQueueSubmit");
		}
	}
	GuardQueueScope scope(queue, "vkQueueSubmit");
	return g_guard_real_submit(queue, count, submits, fence);
}

VkResult VKAPI_PTR GuardSubmit2(VkQueue queue, uint32_t count, const VkSubmitInfo2* submits, VkFence fence) {
	for (uint32_t s = 0; s < count; s++) {
		const auto& submit    = submits[s];
		VkSemaphore semaphore = VK_NULL_HANDLE;
		uint64_t    value     = 0;
		for (uint32_t i = 0; i < submit.signalSemaphoreInfoCount; i++) {
			if (GuardIsTimeline(submit.pSignalSemaphoreInfos[i].semaphore)) {
				semaphore = submit.pSignalSemaphoreInfos[i].semaphore;
				value     = submit.pSignalSemaphoreInfos[i].value;
			}
		}
		for (uint32_t i = 0; i < submit.commandBufferInfoCount; i++) {
			GuardSubmitted(submit.pCommandBufferInfos[i].commandBuffer, semaphore, value, "vkQueueSubmit2");
		}
	}
	GuardQueueScope scope(queue, "vkQueueSubmit2");
	return g_guard_real_submit2(queue, count, submits, fence);
}

VkResult VKAPI_PTR GuardCreateSemaphore(VkDevice device, const VkSemaphoreCreateInfo* info,
                                        const VkAllocationCallbacks* allocator, VkSemaphore* semaphore) {
	const auto result = g_guard_real_semaphore(device, info, allocator, semaphore);
	bool       timeline = false;
	for (const auto* next = static_cast<const VkBaseInStructure*>(info->pNext); next != nullptr; next = next->pNext) {
		if (next->sType == VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO &&
		    reinterpret_cast<const VkSemaphoreTypeCreateInfo*>(next)->semaphoreType == VK_SEMAPHORE_TYPE_TIMELINE) {
			timeline = true;
		}
	}
	if (result == VK_SUCCESS && timeline) {
		std::lock_guard lock(g_guard_mutex);
		for (auto& slot: g_guard_timelines) {
			if (slot.load() == VK_NULL_HANDLE) {
				slot.store(*semaphore, std::memory_order_release);
				break;
			}
		}
	}
	return result;
}

// ---- Object lifetime: a handle a command buffer still uses must not be destroyed ----
// Every buffer, image, image view, buffer view and pipeline a hooked command names is noted with
// the command buffer and its recording generation; its destruction is a violation while that
// recording is not yet submitted or its submission has not completed. (Descriptor-set contents
// are not tracked.)

struct GuardRef {
	GuardEntry* entry      = nullptr;
	uint32_t    generation = 0;
	const char* function   = nullptr;
};
std::mutex                                 g_guard_ref_mutex;
std::unordered_map<uint64_t, GuardRef>     g_guard_refs;
std::atomic<uint64_t>                      g_guard_ref_checks {0};

template <typename Handle>
uint64_t GuardKey(Handle handle) noexcept {
	return reinterpret_cast<uint64_t>(handle);
}

void GuardNoteRef(VkCommandBuffer command, uint64_t key, const char* function) {
	if (key == 0) {
		return;
	}
	auto* entry = GuardFind(command);
	if (entry == nullptr) {
		return;
	}
	const GuardRef ref {entry, entry->generation.load(std::memory_order_acquire), function};
	std::lock_guard lock(g_guard_ref_mutex);
	g_guard_refs[key] = ref;
}

void GuardNoteDestroy(uint64_t key, const char* function) {
	if (key == 0) {
		return;
	}
	GuardRef ref;
	{
		std::lock_guard lock(g_guard_ref_mutex);
		const auto found = g_guard_refs.find(key);
		if (found == g_guard_refs.end()) {
			return;
		}
		ref = found->second;
		g_guard_refs.erase(found);
	}
	if (const auto checks = g_guard_ref_checks.fetch_add(1, std::memory_order_relaxed) + 1;
	    (checks & (checks - 1)) == 0 && checks >= 64) {
		std::size_t live = 0;
		{
			std::lock_guard lock(g_guard_ref_mutex);
			live = g_guard_refs.size();
		}
		std::printf("Kyty command guard: %" PRIu64 " destroyed handles checked against their last use (%zu tracked)\n",
		            checks, live);
		std::fflush(stdout);
	}
	auto* entry = ref.entry;
	if (entry->generation.load(std::memory_order_acquire) != ref.generation) {
		return; // the buffer was begun again, so that use was submitted and completed (begin check)
	}
	const auto state = static_cast<GuardState>(entry->state.load(std::memory_order_acquire));
	char       what[160];
	if (state == GuardState::Recording || state == GuardState::Executable) {
		std::snprintf(what, sizeof(what), "destroy of a handle a not-yet-submitted buffer uses (%s, %s)", function,
		              ref.function);
	} else if (state == GuardState::Pending) {
		const auto semaphore = entry->semaphore.load(std::memory_order_acquire);
		uint64_t   reached   = 0;
		if (semaphore == VK_NULL_HANDLE || g_guard_counter == nullptr ||
		    g_guard_counter(g_guard_device, semaphore, &reached) != VK_SUCCESS ||
		    reached >= entry->value.load(std::memory_order_acquire)) {
			return;
		}
		std::snprintf(what, sizeof(what), "destroy of a handle a pending buffer uses (%s, %s)", function,
		              ref.function);
	} else {
		return;
	}
	// The kind key must be a stable literal: one per destroy function.
	const char* kind = function;
	const auto  count = g_guard_violations.load(std::memory_order_relaxed);
	if (count < 64) {
		std::printf("Kyty command guard: %s: handle 0x%016" PRIx64 " buffer %p (thread %u)\n", what, key,
		            static_cast<void*>(entry->key.load()), GuardTid());
		std::fflush(stdout);
		LOGF("Kyty command guard: %s: handle 0x%016" PRIx64 "\n", what, key);
	}
	GuardViolation("destroy of a handle still used by a command buffer", kind,
	               entry->key.load(), entry, 0, 0);
}

#define KYTY_GUARD_REAL(name) PFN_##name g_guard_next_##name = nullptr

KYTY_GUARD_REAL(vkCmdBindPipeline);
KYTY_GUARD_REAL(vkCmdBindVertexBuffers);
KYTY_GUARD_REAL(vkCmdBindVertexBuffers2);
KYTY_GUARD_REAL(vkCmdBindIndexBuffer);
KYTY_GUARD_REAL(vkCmdCopyBuffer);
KYTY_GUARD_REAL(vkCmdCopyBufferToImage);
KYTY_GUARD_REAL(vkCmdCopyImageToBuffer);
KYTY_GUARD_REAL(vkCmdCopyImage);
KYTY_GUARD_REAL(vkCmdBlitImage);
KYTY_GUARD_REAL(vkCmdResolveImage);
KYTY_GUARD_REAL(vkCmdClearColorImage);
KYTY_GUARD_REAL(vkCmdClearDepthStencilImage);
KYTY_GUARD_REAL(vkCmdFillBuffer);
KYTY_GUARD_REAL(vkCmdUpdateBuffer);
KYTY_GUARD_REAL(vkCmdDrawIndirect);
KYTY_GUARD_REAL(vkCmdDrawIndexedIndirect);
KYTY_GUARD_REAL(vkCmdDrawIndirectCount);
KYTY_GUARD_REAL(vkCmdDrawIndexedIndirectCount);
KYTY_GUARD_REAL(vkCmdDrawMeshTasksIndirectEXT);
KYTY_GUARD_REAL(vkCmdDrawMeshTasksIndirectCountEXT);
KYTY_GUARD_REAL(vkCmdDispatchIndirect);
KYTY_GUARD_REAL(vkCmdBeginRendering);
KYTY_GUARD_REAL(vkCmdPushDescriptorSetKHR);
KYTY_GUARD_REAL(vkCmdCopyQueryPoolResults);
KYTY_GUARD_REAL(vkDestroyBuffer);
KYTY_GUARD_REAL(vkDestroyImage);
KYTY_GUARD_REAL(vkDestroyImageView);
KYTY_GUARD_REAL(vkDestroyBufferView);
KYTY_GUARD_REAL(vkDestroyPipeline);
KYTY_GUARD_REAL(vkUpdateDescriptorSets);
KYTY_GUARD_REAL(vkCmdBindDescriptorSets);
KYTY_GUARD_REAL(vkAllocateDescriptorSets);

// Descriptor sets: the handles each (binding, element) of a set was last written with; a bind
// notes them all as used by the command buffer.
std::mutex                                                                   g_guard_set_mutex;
std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, uint64_t>>>     g_guard_sets;

void VKAPI_PTR GuardRefUpdateDescriptorSets(VkDevice d, uint32_t n, const VkWriteDescriptorSet* w, uint32_t nc,
                                            const VkCopyDescriptorSet* c) {
	{
		std::lock_guard lock(g_guard_set_mutex);
		for (uint32_t i = 0; i < n; i++) {
			auto& slots = g_guard_sets[GuardKey(w[i].dstSet)];
			for (uint32_t k = 0; k < w[i].descriptorCount; k++) {
				uint64_t handle = 0;
				if (w[i].pBufferInfo != nullptr) {
					handle = GuardKey(w[i].pBufferInfo[k].buffer);
				} else if (w[i].pImageInfo != nullptr) {
					handle = GuardKey(w[i].pImageInfo[k].imageView);
				} else if (w[i].pTexelBufferView != nullptr) {
					handle = GuardKey(w[i].pTexelBufferView[k]);
				}
				const uint64_t slot = (uint64_t {w[i].dstBinding} << 32u) | (w[i].dstArrayElement + k);
				bool           found = false;
				for (auto& entry: slots) {
					if (entry.first == slot) {
						entry.second = handle;
						found        = true;
						break;
					}
				}
				if (!found) {
					slots.emplace_back(slot, handle);
				}
			}
		}
	}
	g_guard_next_vkUpdateDescriptorSets(d, n, w, nc, c);
}

// A set (re)allocated from a reset pool starts with no written descriptors.
VkResult VKAPI_PTR GuardRefAllocateDescriptorSets(VkDevice d, const VkDescriptorSetAllocateInfo* info,
                                                  VkDescriptorSet* sets) {
	const auto result = g_guard_next_vkAllocateDescriptorSets(d, info, sets);
	if (result == VK_SUCCESS) {
		std::lock_guard lock(g_guard_set_mutex);
		for (uint32_t i = 0; i < info->descriptorSetCount; i++) {
			g_guard_sets.erase(GuardKey(sets[i]));
		}
	}
	return result;
}

void VKAPI_PTR GuardRefBindDescriptorSets(VkCommandBuffer c, VkPipelineBindPoint p, VkPipelineLayout l,
                                          uint32_t first, uint32_t n, const VkDescriptorSet* sets,
                                          uint32_t nd, const uint32_t* dynamic) {
	std::vector<uint64_t> handles;
	{
		std::lock_guard lock(g_guard_set_mutex);
		for (uint32_t i = 0; i < n; i++) {
			const auto found = g_guard_sets.find(GuardKey(sets[i]));
			if (found != g_guard_sets.end()) {
				for (const auto& entry: found->second) {
					handles.push_back(entry.second);
				}
			}
		}
	}
	for (const auto handle: handles) {
		GuardNoteRef(c, handle, "vkCmdBindDescriptorSets");
	}
	g_guard_next_vkCmdBindDescriptorSets(c, p, l, first, n, sets, nd, dynamic);
}
#undef KYTY_GUARD_REAL

void VKAPI_PTR GuardRefBindPipeline(VkCommandBuffer c, VkPipelineBindPoint p, VkPipeline pipeline) {
	GuardNoteRef(c, GuardKey(pipeline), "vkCmdBindPipeline");
	g_guard_next_vkCmdBindPipeline(c, p, pipeline);
}
void VKAPI_PTR GuardRefBindVertexBuffers(VkCommandBuffer c, uint32_t first, uint32_t n, const VkBuffer* b,
                                         const VkDeviceSize* o) {
	for (uint32_t i = 0; i < n; i++) GuardNoteRef(c, GuardKey(b[i]), "vkCmdBindVertexBuffers");
	g_guard_next_vkCmdBindVertexBuffers(c, first, n, b, o);
}
void VKAPI_PTR GuardRefBindVertexBuffers2(VkCommandBuffer c, uint32_t first, uint32_t n, const VkBuffer* b,
                                          const VkDeviceSize* o, const VkDeviceSize* s, const VkDeviceSize* st) {
	for (uint32_t i = 0; i < n; i++) GuardNoteRef(c, GuardKey(b[i]), "vkCmdBindVertexBuffers2");
	g_guard_next_vkCmdBindVertexBuffers2(c, first, n, b, o, s, st);
}
void VKAPI_PTR GuardRefBindIndexBuffer(VkCommandBuffer c, VkBuffer b, VkDeviceSize o, VkIndexType t) {
	GuardNoteRef(c, GuardKey(b), "vkCmdBindIndexBuffer");
	g_guard_next_vkCmdBindIndexBuffer(c, b, o, t);
}
void VKAPI_PTR GuardRefCopyBuffer(VkCommandBuffer c, VkBuffer s, VkBuffer d, uint32_t n, const VkBufferCopy* r) {
	GuardNoteRef(c, GuardKey(s), "vkCmdCopyBuffer");
	GuardNoteRef(c, GuardKey(d), "vkCmdCopyBuffer");
	g_guard_next_vkCmdCopyBuffer(c, s, d, n, r);
}
void VKAPI_PTR GuardRefCopyBufferToImage(VkCommandBuffer c, VkBuffer s, VkImage d, VkImageLayout l, uint32_t n,
                                         const VkBufferImageCopy* r) {
	GuardNoteRef(c, GuardKey(s), "vkCmdCopyBufferToImage");
	GuardNoteRef(c, GuardKey(d), "vkCmdCopyBufferToImage");
	g_guard_next_vkCmdCopyBufferToImage(c, s, d, l, n, r);
}
void VKAPI_PTR GuardRefCopyImageToBuffer(VkCommandBuffer c, VkImage s, VkImageLayout l, VkBuffer d, uint32_t n,
                                         const VkBufferImageCopy* r) {
	GuardNoteRef(c, GuardKey(s), "vkCmdCopyImageToBuffer");
	GuardNoteRef(c, GuardKey(d), "vkCmdCopyImageToBuffer");
	g_guard_next_vkCmdCopyImageToBuffer(c, s, l, d, n, r);
}
void VKAPI_PTR GuardRefCopyImage(VkCommandBuffer c, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl,
                                 uint32_t n, const VkImageCopy* r) {
	GuardNoteRef(c, GuardKey(s), "vkCmdCopyImage");
	GuardNoteRef(c, GuardKey(d), "vkCmdCopyImage");
	g_guard_next_vkCmdCopyImage(c, s, sl, d, dl, n, r);
}
void VKAPI_PTR GuardRefBlitImage(VkCommandBuffer c, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl,
                                 uint32_t n, const VkImageBlit* r, VkFilter f) {
	GuardNoteRef(c, GuardKey(s), "vkCmdBlitImage");
	GuardNoteRef(c, GuardKey(d), "vkCmdBlitImage");
	g_guard_next_vkCmdBlitImage(c, s, sl, d, dl, n, r, f);
}
void VKAPI_PTR GuardRefResolveImage(VkCommandBuffer c, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl,
                                    uint32_t n, const VkImageResolve* r) {
	GuardNoteRef(c, GuardKey(s), "vkCmdResolveImage");
	GuardNoteRef(c, GuardKey(d), "vkCmdResolveImage");
	g_guard_next_vkCmdResolveImage(c, s, sl, d, dl, n, r);
}
void VKAPI_PTR GuardRefClearColorImage(VkCommandBuffer c, VkImage i, VkImageLayout l, const VkClearColorValue* v,
                                       uint32_t n, const VkImageSubresourceRange* r) {
	GuardNoteRef(c, GuardKey(i), "vkCmdClearColorImage");
	g_guard_next_vkCmdClearColorImage(c, i, l, v, n, r);
}
void VKAPI_PTR GuardRefClearDepthStencilImage(VkCommandBuffer c, VkImage i, VkImageLayout l,
                                              const VkClearDepthStencilValue* v, uint32_t n,
                                              const VkImageSubresourceRange* r) {
	GuardNoteRef(c, GuardKey(i), "vkCmdClearDepthStencilImage");
	g_guard_next_vkCmdClearDepthStencilImage(c, i, l, v, n, r);
}
void VKAPI_PTR GuardRefFillBuffer(VkCommandBuffer c, VkBuffer b, VkDeviceSize o, VkDeviceSize s, uint32_t d) {
	GuardNoteRef(c, GuardKey(b), "vkCmdFillBuffer");
	g_guard_next_vkCmdFillBuffer(c, b, o, s, d);
}
void VKAPI_PTR GuardRefUpdateBuffer(VkCommandBuffer c, VkBuffer b, VkDeviceSize o, VkDeviceSize s, const void* d) {
	GuardNoteRef(c, GuardKey(b), "vkCmdUpdateBuffer");
	g_guard_next_vkCmdUpdateBuffer(c, b, o, s, d);
}
void VKAPI_PTR GuardRefDrawIndirect(VkCommandBuffer c, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s) {
	GuardNoteRef(c, GuardKey(b), "vkCmdDrawIndirect");
	g_guard_next_vkCmdDrawIndirect(c, b, o, n, s);
}
void VKAPI_PTR GuardRefDrawIndexedIndirect(VkCommandBuffer c, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s) {
	GuardNoteRef(c, GuardKey(b), "vkCmdDrawIndexedIndirect");
	g_guard_next_vkCmdDrawIndexedIndirect(c, b, o, n, s);
}
void VKAPI_PTR GuardRefDrawIndirectCount(VkCommandBuffer c, VkBuffer b, VkDeviceSize o, VkBuffer cb,
                                         VkDeviceSize co, uint32_t m, uint32_t s) {
	GuardNoteRef(c, GuardKey(b), "vkCmdDrawIndirectCount");
	GuardNoteRef(c, GuardKey(cb), "vkCmdDrawIndirectCount");
	g_guard_next_vkCmdDrawIndirectCount(c, b, o, cb, co, m, s);
}
void VKAPI_PTR GuardRefDrawIndexedIndirectCount(VkCommandBuffer c, VkBuffer b, VkDeviceSize o, VkBuffer cb,
                                                VkDeviceSize co, uint32_t m, uint32_t s) {
	GuardNoteRef(c, GuardKey(b), "vkCmdDrawIndexedIndirectCount");
	GuardNoteRef(c, GuardKey(cb), "vkCmdDrawIndexedIndirectCount");
	g_guard_next_vkCmdDrawIndexedIndirectCount(c, b, o, cb, co, m, s);
}
void VKAPI_PTR GuardRefDrawMeshIndirect(VkCommandBuffer c, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s) {
	GuardNoteRef(c, GuardKey(b), "vkCmdDrawMeshTasksIndirectEXT");
	g_guard_next_vkCmdDrawMeshTasksIndirectEXT(c, b, o, n, s);
}
void VKAPI_PTR GuardRefDrawMeshIndirectCount(VkCommandBuffer c, VkBuffer b, VkDeviceSize o, VkBuffer cb,
                                             VkDeviceSize co, uint32_t m, uint32_t s) {
	GuardNoteRef(c, GuardKey(b), "vkCmdDrawMeshTasksIndirectCountEXT");
	GuardNoteRef(c, GuardKey(cb), "vkCmdDrawMeshTasksIndirectCountEXT");
	g_guard_next_vkCmdDrawMeshTasksIndirectCountEXT(c, b, o, cb, co, m, s);
}
void VKAPI_PTR GuardRefDispatchIndirect(VkCommandBuffer c, VkBuffer b, VkDeviceSize o) {
	GuardNoteRef(c, GuardKey(b), "vkCmdDispatchIndirect");
	g_guard_next_vkCmdDispatchIndirect(c, b, o);
}
void VKAPI_PTR GuardRefBeginRendering(VkCommandBuffer c, const VkRenderingInfo* info) {
	for (uint32_t i = 0; i < info->colorAttachmentCount; i++) {
		GuardNoteRef(c, GuardKey(info->pColorAttachments[i].imageView), "vkCmdBeginRendering");
		GuardNoteRef(c, GuardKey(info->pColorAttachments[i].resolveImageView), "vkCmdBeginRendering");
	}
	if (info->pDepthAttachment != nullptr) {
		GuardNoteRef(c, GuardKey(info->pDepthAttachment->imageView), "vkCmdBeginRendering");
	}
	if (info->pStencilAttachment != nullptr) {
		GuardNoteRef(c, GuardKey(info->pStencilAttachment->imageView), "vkCmdBeginRendering");
	}
	g_guard_next_vkCmdBeginRendering(c, info);
}
void VKAPI_PTR GuardRefPushDescriptorSet(VkCommandBuffer c, VkPipelineBindPoint p, VkPipelineLayout l, uint32_t set,
                                         uint32_t n, const VkWriteDescriptorSet* w) {
	for (uint32_t i = 0; i < n; i++) {
		for (uint32_t k = 0; k < w[i].descriptorCount; k++) {
			if (w[i].pBufferInfo != nullptr) {
				GuardNoteRef(c, GuardKey(w[i].pBufferInfo[k].buffer), "vkCmdPushDescriptorSetKHR");
			}
			if (w[i].pImageInfo != nullptr) {
				GuardNoteRef(c, GuardKey(w[i].pImageInfo[k].imageView), "vkCmdPushDescriptorSetKHR");
			}
			if (w[i].pTexelBufferView != nullptr) {
				GuardNoteRef(c, GuardKey(w[i].pTexelBufferView[k]), "vkCmdPushDescriptorSetKHR");
			}
		}
	}
	g_guard_next_vkCmdPushDescriptorSetKHR(c, p, l, set, n, w);
}
void VKAPI_PTR GuardRefCopyQueryPoolResults(VkCommandBuffer c, VkQueryPool q, uint32_t f, uint32_t n, VkBuffer b,
                                            VkDeviceSize o, VkDeviceSize s, VkQueryResultFlags fl) {
	GuardNoteRef(c, GuardKey(b), "vkCmdCopyQueryPoolResults");
	g_guard_next_vkCmdCopyQueryPoolResults(c, q, f, n, b, o, s, fl);
}
void VKAPI_PTR GuardRefDestroyBuffer(VkDevice d, VkBuffer b, const VkAllocationCallbacks* a) {
	GuardNoteDestroy(GuardKey(b), "vkDestroyBuffer");
	g_guard_next_vkDestroyBuffer(d, b, a);
}
void VKAPI_PTR GuardRefDestroyImage(VkDevice d, VkImage i, const VkAllocationCallbacks* a) {
	GuardNoteDestroy(GuardKey(i), "vkDestroyImage");
	g_guard_next_vkDestroyImage(d, i, a);
}
void VKAPI_PTR GuardRefDestroyImageView(VkDevice d, VkImageView v, const VkAllocationCallbacks* a) {
	GuardNoteDestroy(GuardKey(v), "vkDestroyImageView");
	g_guard_next_vkDestroyImageView(d, v, a);
}
void VKAPI_PTR GuardRefDestroyBufferView(VkDevice d, VkBufferView v, const VkAllocationCallbacks* a) {
	GuardNoteDestroy(GuardKey(v), "vkDestroyBufferView");
	g_guard_next_vkDestroyBufferView(d, v, a);
}
void VKAPI_PTR GuardRefDestroyPipeline(VkDevice d, VkPipeline p, const VkAllocationCallbacks* a) {
	GuardNoteDestroy(GuardKey(p), "vkDestroyPipeline");
	g_guard_next_vkDestroyPipeline(d, p, a);
}

void InstallGuardRefHooks() {
	auto&      d    = VULKAN_HPP_DEFAULT_DISPATCHER;
	const auto swap = [](auto& slot, auto& next, auto hook) {
		if (slot != nullptr) {
			next = slot;
			slot = hook;
		}
	};
#define KYTY_GUARD_REF(name, hook) swap(d.name, g_guard_next_##name, &hook)
	KYTY_GUARD_REF(vkCmdBindPipeline, GuardRefBindPipeline);
	KYTY_GUARD_REF(vkCmdBindVertexBuffers, GuardRefBindVertexBuffers);
	KYTY_GUARD_REF(vkCmdBindVertexBuffers2, GuardRefBindVertexBuffers2);
	KYTY_GUARD_REF(vkCmdBindIndexBuffer, GuardRefBindIndexBuffer);
	KYTY_GUARD_REF(vkCmdCopyBuffer, GuardRefCopyBuffer);
	KYTY_GUARD_REF(vkCmdCopyBufferToImage, GuardRefCopyBufferToImage);
	KYTY_GUARD_REF(vkCmdCopyImageToBuffer, GuardRefCopyImageToBuffer);
	KYTY_GUARD_REF(vkCmdCopyImage, GuardRefCopyImage);
	KYTY_GUARD_REF(vkCmdBlitImage, GuardRefBlitImage);
	KYTY_GUARD_REF(vkCmdResolveImage, GuardRefResolveImage);
	KYTY_GUARD_REF(vkCmdClearColorImage, GuardRefClearColorImage);
	KYTY_GUARD_REF(vkCmdClearDepthStencilImage, GuardRefClearDepthStencilImage);
	KYTY_GUARD_REF(vkCmdFillBuffer, GuardRefFillBuffer);
	KYTY_GUARD_REF(vkCmdUpdateBuffer, GuardRefUpdateBuffer);
	KYTY_GUARD_REF(vkCmdDrawIndirect, GuardRefDrawIndirect);
	KYTY_GUARD_REF(vkCmdDrawIndexedIndirect, GuardRefDrawIndexedIndirect);
	KYTY_GUARD_REF(vkCmdDrawIndirectCount, GuardRefDrawIndirectCount);
	KYTY_GUARD_REF(vkCmdDrawIndexedIndirectCount, GuardRefDrawIndexedIndirectCount);
	KYTY_GUARD_REF(vkCmdDrawMeshTasksIndirectEXT, GuardRefDrawMeshIndirect);
	KYTY_GUARD_REF(vkCmdDrawMeshTasksIndirectCountEXT, GuardRefDrawMeshIndirectCount);
	KYTY_GUARD_REF(vkCmdDispatchIndirect, GuardRefDispatchIndirect);
	KYTY_GUARD_REF(vkCmdBeginRendering, GuardRefBeginRendering);
	KYTY_GUARD_REF(vkCmdPushDescriptorSetKHR, GuardRefPushDescriptorSet);
	KYTY_GUARD_REF(vkCmdCopyQueryPoolResults, GuardRefCopyQueryPoolResults);
	KYTY_GUARD_REF(vkDestroyBuffer, GuardRefDestroyBuffer);
	KYTY_GUARD_REF(vkDestroyImage, GuardRefDestroyImage);
	KYTY_GUARD_REF(vkDestroyImageView, GuardRefDestroyImageView);
	KYTY_GUARD_REF(vkDestroyBufferView, GuardRefDestroyBufferView);
	KYTY_GUARD_REF(vkDestroyPipeline, GuardRefDestroyPipeline);
	KYTY_GUARD_REF(vkUpdateDescriptorSets, GuardRefUpdateDescriptorSets);
	KYTY_GUARD_REF(vkCmdBindDescriptorSets, GuardRefBindDescriptorSets);
	KYTY_GUARD_REF(vkAllocateDescriptorSets, GuardRefAllocateDescriptorSets);
#undef KYTY_GUARD_REF
}

} // namespace

uint64_t CommandRecorder::GuardViolations() noexcept {
	return g_guard_violations.load(std::memory_order_relaxed);
}

void CommandRecorder::InstallGuardHooks(vk::Device device) {
	const auto* value = std::getenv("KYTY_CMD_GUARD");
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
		return;
	}
	static std::once_flag once;
	std::call_once(once, [device, value] {
		g_guard_mode   = std::strcmp(value, "exit") == 0 ? 2 : std::strcmp(value, "wait") == 0 ? 3 : 1;
		g_guard_device = static_cast<VkDevice>(device);
		auto& d        = VULKAN_HPP_DEFAULT_DISPATCHER;
		g_guard_counter = d.vkGetSemaphoreCounterValue != nullptr ? d.vkGetSemaphoreCounterValue
		                                                          : d.vkGetSemaphoreCounterValueKHR;
		g_guard_wait    = d.vkWaitSemaphores != nullptr ? d.vkWaitSemaphores : d.vkWaitSemaphoresKHR;
		const auto swap = [](auto& slot, auto& real, auto hook) {
			if (slot != nullptr) {
				real = slot;
				slot = hook;
			}
		};
		swap(d.vkBeginCommandBuffer, g_guard_real_begin, &GuardBegin);
		swap(d.vkEndCommandBuffer, g_guard_real_end, &GuardEnd);
		swap(d.vkResetCommandBuffer, g_guard_real_reset, &GuardReset);
		swap(d.vkAllocateCommandBuffers, g_guard_real_allocate, &GuardAllocate);
		swap(d.vkFreeCommandBuffers, g_guard_real_free, &GuardFree);
		swap(d.vkQueueSubmit, g_guard_real_submit, &GuardSubmit);
		if (d.vkQueueSubmit2 != nullptr) {
			swap(d.vkQueueSubmit2, g_guard_real_submit2, &GuardSubmit2);
		} else if (d.vkQueueSubmit2KHR != nullptr) {
			swap(d.vkQueueSubmit2KHR, g_guard_real_submit2, &GuardSubmit2);
		}
		swap(d.vkCreateSemaphore, g_guard_real_semaphore, &GuardCreateSemaphore);
		swap(d.vkQueuePresentKHR, g_guard_real_present, &GuardPresent);
		swap(d.vkQueueBindSparse, g_guard_real_sparse, &GuardBindSparse);
		swap(d.vkQueueWaitIdle, g_guard_real_idle, &GuardWaitIdle);
		swap(d.vkDestroyCommandPool, g_guard_real_destroy_pool, &GuardDestroyPool);
		swap(d.vkResetCommandPool, g_guard_real_reset_pool, &GuardResetPool);
#define KYTY_GUARD_HOOK(member) InstallGuardHook<__COUNTER__>(d.member, #member)
		KYTY_GUARD_HOOK(vkCmdBeginRendering);
		KYTY_GUARD_HOOK(vkCmdEndRendering);
		KYTY_GUARD_HOOK(vkCmdPipelineBarrier);
		KYTY_GUARD_HOOK(vkCmdPipelineBarrier2);
		KYTY_GUARD_HOOK(vkCmdCopyBuffer);
		KYTY_GUARD_HOOK(vkCmdCopyBufferToImage);
		KYTY_GUARD_HOOK(vkCmdCopyImageToBuffer);
		KYTY_GUARD_HOOK(vkCmdCopyImage);
		KYTY_GUARD_HOOK(vkCmdBlitImage);
		KYTY_GUARD_HOOK(vkCmdResolveImage);
		KYTY_GUARD_HOOK(vkCmdClearColorImage);
		KYTY_GUARD_HOOK(vkCmdClearDepthStencilImage);
		KYTY_GUARD_HOOK(vkCmdClearAttachments);
		KYTY_GUARD_HOOK(vkCmdFillBuffer);
		KYTY_GUARD_HOOK(vkCmdUpdateBuffer);
		KYTY_GUARD_HOOK(vkCmdBindPipeline);
		KYTY_GUARD_HOOK(vkCmdBindDescriptorSets);
		KYTY_GUARD_HOOK(vkCmdPushDescriptorSetKHR);
		KYTY_GUARD_HOOK(vkCmdPushConstants);
		KYTY_GUARD_HOOK(vkCmdBindVertexBuffers);
		KYTY_GUARD_HOOK(vkCmdBindVertexBuffers2);
		KYTY_GUARD_HOOK(vkCmdBindIndexBuffer);
		KYTY_GUARD_HOOK(vkCmdSetViewport);
		KYTY_GUARD_HOOK(vkCmdSetScissor);
		KYTY_GUARD_HOOK(vkCmdSetViewportWithCount);
		KYTY_GUARD_HOOK(vkCmdSetScissorWithCount);
		KYTY_GUARD_HOOK(vkCmdSetBlendConstants);
		KYTY_GUARD_HOOK(vkCmdSetDepthTestEnable);
		KYTY_GUARD_HOOK(vkCmdSetDepthWriteEnable);
		KYTY_GUARD_HOOK(vkCmdSetDepthCompareOp);
		KYTY_GUARD_HOOK(vkCmdSetDepthBiasEnable);
		KYTY_GUARD_HOOK(vkCmdSetDepthBias);
		KYTY_GUARD_HOOK(vkCmdSetStencilTestEnable);
		KYTY_GUARD_HOOK(vkCmdSetStencilOp);
		KYTY_GUARD_HOOK(vkCmdSetStencilReference);
		KYTY_GUARD_HOOK(vkCmdSetCullMode);
		KYTY_GUARD_HOOK(vkCmdSetFrontFace);
		KYTY_GUARD_HOOK(vkCmdSetColorWriteEnableEXT);
		KYTY_GUARD_HOOK(vkCmdDraw);
		KYTY_GUARD_HOOK(vkCmdDrawIndexed);
		KYTY_GUARD_HOOK(vkCmdDrawIndirect);
		KYTY_GUARD_HOOK(vkCmdDrawIndexedIndirect);
		KYTY_GUARD_HOOK(vkCmdDrawIndirectCount);
		KYTY_GUARD_HOOK(vkCmdDrawIndexedIndirectCount);
		KYTY_GUARD_HOOK(vkCmdDrawMeshTasksEXT);
		KYTY_GUARD_HOOK(vkCmdDrawMeshTasksIndirectEXT);
		KYTY_GUARD_HOOK(vkCmdDrawMeshTasksIndirectCountEXT);
		KYTY_GUARD_HOOK(vkCmdDispatch);
		KYTY_GUARD_HOOK(vkCmdDispatchIndirect);
		KYTY_GUARD_HOOK(vkCmdResetQueryPool);
		KYTY_GUARD_HOOK(vkCmdBeginQuery);
		KYTY_GUARD_HOOK(vkCmdEndQuery);
		KYTY_GUARD_HOOK(vkCmdCopyQueryPoolResults);
		KYTY_GUARD_HOOK(vkCmdWriteTimestamp);
		KYTY_GUARD_HOOK(vkCmdWriteTimestamp2);
		KYTY_GUARD_HOOK(vkCmdWriteBufferMarkerAMD);
#undef KYTY_GUARD_HOOK
		InstallGuardRefHooks();
		std::printf("Kyty command guard (KYTY_CMD_GUARD=%s): command-buffer state hooks installed\n", value);
		std::fflush(stdout);
	});
}

} // namespace Libs::Graphics
