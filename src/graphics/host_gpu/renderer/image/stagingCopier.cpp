#include "graphics/host_gpu/renderer/image/stagingCopier.h"

#include "common/assert.h"
#include "common/hangWatchdog.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "kernel/memory.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>

namespace Libs::Graphics {

StagingCopier::StagingCopier(GraphicContext& graphics): m_graphics(graphics) {
	if (SubmitWaitBeforeSignal()) {
		vk::SemaphoreTypeCreateInfo type_info {};
		type_info.semaphoreType = vk::SemaphoreType::eTimeline;
		type_info.initialValue  = 0;
		vk::SemaphoreCreateInfo create_info {};
		create_info.pNext = &type_info;
		const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
		EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
	}
	m_worker = std::jthread([this](std::stop_token stop) { Worker(stop); });
}

StagingCopier::~StagingCopier() {
	{
		std::scoped_lock lock(m_mutex);
		m_stopping = true;
	}
	m_available.notify_all();
	if (m_worker.joinable()) {
		m_worker.join();
	}
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

void StagingCopier::Enqueue(std::vector<Range> ranges, Buffer* flush_buffer, uint64_t flush_offset,
                            uint64_t flush_size) {
	uint64_t bytes = 0;
	for (const auto& range: ranges) {
		bytes += range.size;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::TextureAsyncCopies);
	Profiler::CountFrameEvent(Profiler::FrameEvent::TextureAsyncCopyBytes, bytes);
	{
		std::scoped_lock lock(m_mutex);
		EXIT_IF(m_stopping);
		m_jobs.push_back({std::move(ranges), flush_buffer, flush_offset, flush_size, ++m_enqueued});
	}
	m_available.notify_one();
}

uint64_t StagingCopier::PendingValue() {
	// Acquire: when the worker has finished, its host writes happen-before the caller's
	// submission, which makes them available to the device (host write ordering guarantee).
	return m_completed.load(std::memory_order_acquire) >= m_enqueued ? 0 : m_enqueued;
}

void StagingCopier::WaitHost(uint64_t value) {
	auto completed = m_completed.load(std::memory_order_acquire);
	if (completed >= value) {
		return;
	}
	HangWatchdog::Scope scope("texture-staging-copy", reinterpret_cast<uint64_t>(this), value,
	                          completed);
	while (completed < value) {
		m_completed.wait(completed, std::memory_order_acquire);
		completed = m_completed.load(std::memory_order_acquire);
	}
}

void StagingCopier::Run(Job& job) {
	Profiler::ScopedFrameWait timing(Profiler::FrameWait::TextureStagingCopy);
	// Short slices: TryReadBacking holds the guest address-space lock while it copies.
	constexpr uint64_t Slice = 256ull * 1024;
	for (const auto& range: job.ranges) {
		for (uint64_t done = 0; done < range.size;) {
			const auto bytes = std::min(Slice, range.size - done);
			auto*      dst   = range.destination + done;
			const auto src   = range.guest_address + done;
			if (!LibKernel::Memory::TryReadBacking(src, dst, bytes) &&
			    !LibKernel::Memory::TryReadSparseBacking(src, dst, bytes)) {
				// The range was mapped when the refresh was recorded; unmapping drains the GPU
				// (and so this job) first. Never leave a submission waiting on a failed copy.
				std::memset(dst, 0, bytes);
				if (++m_read_failures <= 16) {
					LOGF("StagingCopier: failed to read guest image backing 0x%016" PRIx64
					     " size=0x%" PRIx64 "\n",
					     src, bytes);
				}
			}
			done += bytes;
		}
	}
	if (job.flush_buffer != nullptr && job.flush_size != 0) {
		job.flush_buffer->Flush(job.flush_offset, job.flush_size);
	}
}

void StagingCopier::Worker(std::stop_token stop) {
	KYTY_PROFILER_THREAD("Texture staging copies");
	(void)stop;
	for (;;) {
		Job job;
		{
			std::unique_lock lock(m_mutex);
			m_available.wait(lock, [this] { return m_stopping || !m_jobs.empty(); });
			if (m_jobs.empty()) {
				return; // stopping and drained
			}
			job = std::move(m_jobs.front());
			m_jobs.pop_front();
		}
		Run(job);
		m_completed.store(job.value, std::memory_order_release);
		m_completed.notify_all();
		if (m_semaphore != nullptr) {
			// KYTY_SUBMIT_WAIT_BEFORE_SIGNAL=1 only: batches already queued may wait for it.
			vk::SemaphoreSignalInfo signal {};
			signal.semaphore = m_semaphore;
			signal.value     = job.value;
			const auto result = m_graphics.device.signalSemaphore(&signal);
			EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
		}
	}
}

} // namespace Libs::Graphics
