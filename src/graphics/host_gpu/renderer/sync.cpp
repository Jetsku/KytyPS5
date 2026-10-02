#include "graphics/host_gpu/renderer/sync.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/presentation/videoOut.h"
#include "kernel/eventQueue.h"
#include "kernel/eventQueueFilters.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <cstring>
#include <limits>

namespace Libs::Graphics::Sync {

uint64_t ReadReferenceClock() {
	const auto host_frequency = LibKernel::KernelGetTscFrequency();
	const auto host_ticks     = LibKernel::KernelReadTsc();
	uint64_t   value          = 0;
	if (!ScaleReferenceClock(host_ticks, host_frequency, value)) {
		EXIT("cannot scale host clock, ticks=0x%016" PRIx64 " frequency=%" PRIu64 "\n", host_ticks,
		     host_frequency);
	}
	return value;
}

enum class EndOfPipeWriteSize : uint32_t { Dword = 4, Qword = 8 };
enum class EndOfPipeWriteAction { Write, WriteBack, Interrupt, InterruptWriteBack };

static CommandBufferDebugOp DebugOperation(EndOfPipeWriteAction action) {
	switch (action) {
		case EndOfPipeWriteAction::Write: return CommandBufferDebugOp::EopWrite;
		case EndOfPipeWriteAction::WriteBack:
		case EndOfPipeWriteAction::InterruptWriteBack: return CommandBufferDebugOp::EopWriteBack;
		case EndOfPipeWriteAction::Interrupt: return CommandBufferDebugOp::EopInterrupt;
	}
	EXIT("unsupported end-of-pipe write action\n");
	return CommandBufferDebugOp::Unknown;
}

static bool TriggersInterrupt(EndOfPipeWriteAction action) {
	return action == EndOfPipeWriteAction::Interrupt ||
	       action == EndOfPipeWriteAction::InterruptWriteBack;
}

static void RecordEndOfPipeWrite(uint64_t submit_id, CommandBuffer& buffer, uint64_t destination,
                                 uint64_t value, EndOfPipeWriteSize size,
                                 EndOfPipeWriteAction action, int interrupt_event_id = 0,
                                 uint32_t context_id = 0) {
	EXIT_IF(destination == 0);
	// Validation only: the write happens on the CPU once the recording completes, so it is
	// not a barrier flush point.
	EXIT_IF(buffer.IsInvalid());

	const auto width      = static_cast<uint32_t>(size);
	const auto value_low  = static_cast<uint32_t>(value);
	const auto value_high = static_cast<uint32_t>(value >> 32u);
	const auto operation  = static_cast<uint32_t>(DebugOperation(action));
	if (TriggersInterrupt(action)) {
		buffer.SetDebugInfo(operation, submit_id, width, context_id, value_low, value_high,
		                    destination);
		TriggerEopEventAtEndOfPipe(buffer, interrupt_event_id, context_id);
	} else {
		buffer.SetDebugInfo(operation, submit_id, width, value_low, value_high, 0, destination);
	}
}

void WriteAtEndOfPipe32(uint64_t submit_id, CommandBuffer& buffer, uint32_t* dst_gpu_addr,
                        uint32_t value) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Dword, EndOfPipeWriteAction::Write);
}

void WriteAtEndOfPipeGds32(uint64_t submit_id, CommandBuffer& buffer, uint32_t* dst_gpu_addr,
                           uint32_t dw_offset, uint32_t dw_num) {
	EXIT_IF(dst_gpu_addr == nullptr);
	// Validation only: the write happens on the CPU once the recording completes, so it is
	// not a barrier flush point.
	EXIT_IF(buffer.IsInvalid());
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::EopWrite), submit_id,
	                    dw_offset, dw_num, 0, 0, reinterpret_cast<uint64_t>(dst_gpu_addr));
}

void WriteAtEndOfPipe64(uint64_t submit_id, CommandBuffer& buffer, uint64_t* dst_gpu_addr,
                        uint64_t value) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Qword, EndOfPipeWriteAction::Write);
}

void WriteAtEndOfPipeWithWriteBack64(uint64_t submit_id, CommandBuffer& buffer,
                                     uint64_t* dst_gpu_addr, uint64_t value) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Qword, EndOfPipeWriteAction::WriteBack);
}

void WriteAtEndOfPipeWithWriteBack32(uint64_t submit_id, CommandBuffer& buffer,
                                     uint32_t* dst_gpu_addr, uint32_t value) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Dword, EndOfPipeWriteAction::WriteBack);
}

void WriteAtEndOfPipeWithInterruptWriteBack64(uint64_t submit_id, CommandBuffer& buffer,
                                              uint64_t* dst_gpu_addr, uint64_t value, int event_id,
                                              uint32_t context_id) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Qword, EndOfPipeWriteAction::InterruptWriteBack,
	                     event_id, context_id);
}

void WriteAtEndOfPipeWithInterruptWriteBack32(uint64_t submit_id, CommandBuffer& buffer,
                                              uint32_t* dst_gpu_addr, uint32_t value, int event_id,
                                              uint32_t context_id) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Dword, EndOfPipeWriteAction::InterruptWriteBack,
	                     event_id, context_id);
}

void WriteAtEndOfPipeWithInterrupt64(uint64_t submit_id, CommandBuffer& buffer,
                                     uint64_t* dst_gpu_addr, uint64_t value, int event_id,
                                     uint32_t context_id) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Qword, EndOfPipeWriteAction::Interrupt, event_id,
	                     context_id);
}

void WriteAtEndOfPipeWithInterrupt32(uint64_t submit_id, CommandBuffer& buffer,
                                     uint32_t* dst_gpu_addr, uint32_t value, int event_id,
                                     uint32_t context_id) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Dword, EndOfPipeWriteAction::Interrupt, event_id,
	                     context_id);
}

uint64_t PrepareVideoOutFlip(CommandBuffer& buffer, int handle, int index, int flip_mode,
                             int64_t flip_arg) {
	for (;;) {
		uint64_t   request_id = 0;
		auto&      video_out  = buffer.GetContext().GetVideoOut();
		const auto result =
		    video_out.SubmitFlipFromGpu(buffer, handle, index, flip_mode, flip_arg, request_id);
		if (result == OK) {
			EXIT_IF(request_id == 0);
			return request_id;
		}
		if (result != VideoOut::VIDEO_OUT_ERROR_FLIP_QUEUE_FULL) {
			EXIT("GPU flip submission failed, result=%d handle=%d index=%d mode=%d arg=%" PRId64
			     "\n",
			     result, handle, index, flip_mode, flip_arg);
		}
		video_out.WaitForSubmitSlot(handle);
	}
}

void WriteAtEndOfPipeWithInterruptWriteBackFlip32(uint64_t submit_id, CommandBuffer& buffer,
                                                  uint32_t* dst_gpu_addr, uint32_t value,
                                                  int handle, int index, int flip_mode,
                                                  int64_t flip_arg, uint64_t request_id,
                                                  int event_id) {
	EXIT_IF(dst_gpu_addr == nullptr);
	// Validation only: the write happens on the CPU once the recording completes, so it is
	// not a barrier flush point.
	EXIT_IF(buffer.IsInvalid());
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::EopWriteBackFlip), submit_id,
	                    static_cast<uint32_t>(handle), static_cast<uint32_t>(index),
	                    static_cast<uint32_t>(flip_mode), value, static_cast<uint64_t>(flip_arg));

	auto& renderer  = buffer.GetContext();
	auto& scheduler = renderer.GetCommandScheduler();
	EXIT_IF(!scheduler.Active() || &buffer != &scheduler.Current());
	scheduler.DeferPriorityOperation([&renderer, event_id, request_id] {
		renderer.GetVideoOut().CompleteFlip(request_id);
		renderer.TriggerInterrupt(event_id, 0);
	});
}

void WriteAtEndOfPipeWithFlip32(uint64_t submit_id, CommandBuffer& buffer, uint32_t* dst_gpu_addr,
                                uint32_t value, int handle, int index, int flip_mode,
                                int64_t flip_arg, uint64_t request_id) {
	EXIT_IF(dst_gpu_addr == nullptr);
	// Validation only: the write happens on the CPU once the recording completes, so it is
	// not a barrier flush point.
	EXIT_IF(buffer.IsInvalid());
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::EopFlip), submit_id,
	                    static_cast<uint32_t>(handle), static_cast<uint32_t>(index),
	                    static_cast<uint32_t>(flip_mode), value, static_cast<uint64_t>(flip_arg));

	auto& renderer  = buffer.GetContext();
	auto& scheduler = renderer.GetCommandScheduler();
	EXIT_IF(!scheduler.Active() || &buffer != &scheduler.Current());
	scheduler.DeferPriorityOperation(
	    [&renderer, request_id] { renderer.GetVideoOut().CompleteFlip(request_id); });
}

void WriteAtEndOfPipeOnlyFlip(uint64_t submit_id, CommandBuffer& buffer, int handle, int index,
                              int flip_mode, int64_t flip_arg, uint64_t request_id) {
	// Validation only: the write happens on the CPU once the recording completes, so it is
	// not a barrier flush point.
	EXIT_IF(buffer.IsInvalid());
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::EopOnlyFlip), submit_id,
	                    static_cast<uint32_t>(handle), static_cast<uint32_t>(index),
	                    static_cast<uint32_t>(flip_mode), 0, static_cast<uint64_t>(flip_arg));

	auto& renderer  = buffer.GetContext();
	auto& scheduler = renderer.GetCommandScheduler();
	EXIT_IF(!scheduler.Active() || &buffer != &scheduler.Current());
	scheduler.DeferPriorityOperation(
	    [&renderer, request_id] { renderer.GetVideoOut().CompleteFlip(request_id); });
}

void TriggerEopEventAtEndOfPipe(CommandBuffer& buffer, int event_id, uint32_t context_id) {
	// Validation only: the write happens on the CPU once the recording completes, so it is
	// not a barrier flush point.
	EXIT_IF(buffer.IsInvalid());
	auto& renderer  = buffer.GetContext();
	auto& scheduler = renderer.GetCommandScheduler();
	EXIT_IF(!scheduler.Active() || &buffer != &scheduler.Current());
	scheduler.DeferPriorityOperation(
	    [&renderer, event_id, context_id] { renderer.TriggerInterrupt(event_id, context_id); },
	    CommandScheduler::PriorityOperationKind::EopInterrupt);
}

static void InterruptEventResetFunc(LibKernel::EventQueue::KernelEqueueEvent* event) {
	EXIT_IF(event == nullptr);
	event->triggered    = false;
	event->event.fflags = 0;
	event->event.data   = 0;
}

// Interrupts the guest has not consumed yet merge into its pending event (fflags counts them, data
// is the newest context id) instead of queueing one copy each, which grew without bound: the
// guest's waits drain with a zero timeout and ignore the contents (eboot 0x4a5eb0 / 0x4a65f0).
// KYTY_EQUEUE_COALESCE=0 restores the queue.
static void InterruptEventTriggerFunc(LibKernel::EventQueue::KernelEqueueEvent* event,
                                      void*                                     trigger_data) {
	EXIT_IF(event == nullptr);

	LibKernel::EventQueue::KernelEqueueApplyTrigger(
	    event, LibKernel::EventQueue::GraphicsInterruptNextState(
	               event->event, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(trigger_data))));
}

int AddEqEvent(RenderContext& renderer, LibKernel::EventQueue::KernelEqueue eq, int id,
               void* udata) {
	LibKernel::EventQueue::KernelEqueueEvent event;
	event.triggered           = false;
	event.event.ident         = static_cast<uintptr_t>(id);
	event.event.filter        = LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS;
	event.event.udata         = udata;
	event.event.fflags        = 0;
	event.event.data          = id;
	event.filter.reset_func   = InterruptEventResetFunc;
	event.filter.trigger_func = InterruptEventTriggerFunc;

	int result = LibKernel::EventQueue::KernelAddEvent(eq, event);

	if (result == 0) {
		renderer.AddInterruptEq(eq, id);
	}

	return result;
}

int DeleteEqEvent(RenderContext& renderer, LibKernel::EventQueue::KernelEqueue eq, int id) {
	int result = LibKernel::EventQueue::KernelDeleteEvent(
	    eq, static_cast<uintptr_t>(id), LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS);
	if (result == OK || result == LibKernel::KERNEL_ERROR_ENOENT) {
		renderer.DeleteInterruptEq(eq, id);
	}

	return result;
}

void ReadGds(const Buffer& gds, uint32_t* dst, uint32_t dw_offset, uint32_t dw_size) {
	const auto offset = uint64_t {dw_offset} * sizeof(uint32_t);
	const auto size   = uint64_t {dw_size} * sizeof(uint32_t);
	EXIT_IF(dst == nullptr || offset > gds.Size() || size > gds.Size() - offset ||
	        gds.Mapped().empty());
	std::memcpy(dst, gds.Mapped().data() + offset, static_cast<size_t>(size));
}

} // namespace Libs::Graphics::Sync
