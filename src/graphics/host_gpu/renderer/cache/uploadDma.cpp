#include "graphics/host_gpu/renderer/cache/uploadDma.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/hangWatchdog.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/gpuBreadcrumbs.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB         = 1024ull * 1024;
constexpr uint64_t Alignment   = 256;
constexpr size_t   BatchBuffers = 16;

uint64_t EnvU64(const char* name, uint64_t default_value) {
	const auto* value = std::getenv(name);
	if (value == nullptr || value[0] == '\0') {
		return default_value;
	}
	return std::strtoull(value, nullptr, 10);
}

} // namespace

bool UploadDmaRequested() {
	static const bool requested = EnvU64("KYTY_UPLOAD_DMA", 1) != 0;
	return requested;
}

bool UploadDmaVerify() {
	static const bool verify = EnvU64("KYTY_UPLOAD_DMA_VERIFY", 0) != 0;
	return verify;
}

bool UploadDmaHostCopyEnabled() {
	static const bool enabled = EnvU64("KYTY_UPLOAD_DMA_HOST_COPY", 1) != 0;
	return enabled;
}

std::unique_ptr<UploadDma> UploadDma::Create(GraphicContext& graphics, CommandScheduler& scheduler) {
	if (!UploadDmaRequested() || graphics.transfer_queue == nullptr) {
		return nullptr;
	}
	const auto ring_mib = std::max<uint64_t>(EnvU64("KYTY_UPLOAD_DMA_MB", 64), 1);
	const auto min_kib  = EnvU64("KYTY_UPLOAD_DMA_MIN_KB", 64);
	LOGF("Upload DMA: on, device-local ring %" PRIu64 " MiB, uploads from %" PRIu64
	     " KiB, transfer family %u\n",
	     ring_mib, min_kib, graphics.transfer_queue_family);
	return std::make_unique<UploadDma>(graphics, scheduler, ring_mib * MiB, min_kib * 1024);
}

UploadDma::UploadDma(GraphicContext& graphics, CommandScheduler& scheduler, uint64_t ring_size,
                     uint64_t min_bytes)
    : m_graphics(graphics), m_scheduler(scheduler), m_ring_size(ring_size),
      m_min_bytes(std::max<uint64_t>(min_bytes, 1)) {
	EXIT_IF(m_graphics.transfer_queue == nullptr);
	m_ring = std::make_unique<Buffer>(
	    graphics, scheduler, MemoryUsage::DeviceLocal, 0,
	    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, ring_size,
	    true);
	SetVulkanObjectNameF(m_graphics.device, m_ring->Handle(), "Kyty.UploadDmaRing");

	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;
	vk::SemaphoreCreateInfo semaphore_info {};
	semaphore_info.pNext = &type_info;
	RequireVulkanSuccess(m_graphics.device.createSemaphore(&semaphore_info, nullptr, &m_semaphore),
	                     "create upload DMA semaphore");

	vk::CommandPoolCreateInfo pool_info {};
	pool_info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	pool_info.queueFamilyIndex = m_graphics.transfer_queue_family;
	RequireVulkanSuccess(m_graphics.device.createCommandPool(&pool_info, nullptr, &m_pool),
	                     "create upload DMA command pool");
	std::vector<vk::CommandBuffer> buffers(BatchBuffers);
	vk::CommandBufferAllocateInfo  allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = static_cast<uint32_t>(buffers.size());
	RequireVulkanSuccess(m_graphics.device.allocateCommandBuffers(&allocate, buffers.data()),
	                     "allocate upload DMA command buffers");
	m_batches.reserve(buffers.size());
	for (auto buffer: buffers) {
		m_batches.push_back({buffer, 0});
	}
	m_worker = std::jthread([this](std::stop_token stop) { Worker(stop); });
}

UploadDma::~UploadDma() {
	{
		std::scoped_lock lock(m_mutex);
		m_stopping = true;
	}
	m_available.notify_all();
	if (m_worker.joinable()) {
		m_worker.join();
	}
	// The worker submitted every queued copy before it stopped; wait for them on the device.
	if (m_enqueued != 0) {
		WaitHost(m_enqueued);
	}
	if (m_pool != nullptr) {
		m_graphics.device.destroyCommandPool(m_pool, nullptr);
	}
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

vk::Buffer UploadDma::RingHandle() const noexcept {
	return m_ring->Handle();
}

std::optional<uint64_t> UploadDma::Allocate(uint64_t size) {
	size = Common::AlignUp(size, Alignment);
	if (size > m_ring_size) {
		return std::nullopt;
	}
	// Ranges whose reading tick completed are free again. That tick waited for the transfer that
	// wrote them, so both sides are done with them.
	while (!m_spans.empty() && m_scheduler.IsFree(m_spans.front().tick)) {
		m_reuse_tick = std::max(m_reuse_tick, m_spans.front().tick);
		m_spans.pop_front();
	}
	uint64_t offset = 0;
	if (m_spans.empty()) {
		offset = m_head + size <= m_ring_size ? m_head : 0;
	} else {
		const auto tail = m_spans.front().begin;
		if (m_head > tail) {
			// In use: [tail, head). Free: [head, ring) and [0, tail).
			if (m_head + size <= m_ring_size) {
				offset = m_head;
			} else if (size <= tail) {
				offset = 0;
			} else {
				return std::nullopt;
			}
		} else if (m_head < tail) {
			// In use: [tail, ring) and [0, head). Free: [head, tail).
			if (m_head + size <= tail) {
				offset = m_head;
			} else {
				return std::nullopt;
			}
		} else {
			// head == tail with ranges in use: the ring is full.
			return std::nullopt;
		}
	}
	m_head          = offset + size;
	const auto tick = m_scheduler.CurrentTick();
	if (!m_spans.empty() && m_spans.back().tick == tick && m_spans.back().end == offset) {
		m_spans.back().end = m_head;
	} else {
		m_spans.push_back({offset, m_head, tick});
	}
	return offset;
}

std::optional<uint64_t> UploadDma::Stage(vk::Buffer source, uint64_t source_offset, uint64_t size,
                                         std::vector<UploadHostCopy>* host_copies) {
	if (source == nullptr || size < m_min_bytes) {
		return std::nullopt;
	}
	const auto offset = Allocate(size);
	if (!offset.has_value()) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::UploadDmaRingFull);
		return std::nullopt;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::UploadDmaCopies);
	Profiler::CountFrameEvent(Profiler::FrameEvent::UploadDmaBytes, size);
	std::vector<UploadHostCopy> copies;
	if (host_copies != nullptr && !host_copies->empty()) {
		uint64_t bytes = 0;
		for (const auto& copy: *host_copies) {
			bytes += copy.size;
		}
		m_host_bytes_queued += bytes;
		Profiler::CountFrameEvent(Profiler::FrameEvent::UploadDmaHostCopies, host_copies->size());
		Profiler::CountFrameEvent(Profiler::FrameEvent::UploadDmaHostCopyBytes, bytes);
		copies.swap(*host_copies);
	}
	{
		std::scoped_lock lock(m_mutex);
		EXIT_IF(m_stopping);
		m_jobs.push_back(
		    {source, source_offset, *offset, size, ++m_enqueued, m_reuse_tick, std::move(copies)});
	}
	m_available.notify_one();
	m_stage_tick  = m_scheduler.CurrentTick();
	m_stage_value = m_enqueued;
	return offset;
}

uint64_t UploadDma::PendingValue() {
	// Called when the guest scheduler submits its current recording.
	if (m_stage_tick != m_scheduler.CurrentTick() || m_known_value >= m_stage_value) {
		return 0;
	}
	uint64_t value = 0;
	if (m_graphics.device.getSemaphoreCounterValue(m_semaphore, &value) == vk::Result::eSuccess) {
		m_known_value = value;
	}
	if (m_known_value >= m_stage_value) {
		return 0;
	}
	// The submission containing the graphics copies waits for the transfer on the device.
	Profiler::CountFrameEvent(Profiler::FrameEvent::UploadDmaSubmitWaits);
	return m_stage_value;
}

void UploadDma::WaitSubmittable(uint64_t value) {
	auto submitted = m_submitted.load(std::memory_order_acquire);
	if (submitted >= value) {
		return;
	}
	HangWatchdog::Scope wait_scope("upload-dma-submit", reinterpret_cast<uint64_t>(this), value,
	                               submitted);
	while (submitted < value) {
		m_submitted.wait(submitted, std::memory_order_acquire);
		submitted = m_submitted.load(std::memory_order_acquire);
	}
}

void UploadDma::WaitHost(uint64_t value) {
	HangWatchdog::Scope wait_scope(
	    "upload-dma-gpu", reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(m_semaphore)), value);
	vk::SemaphoreWaitInfo wait {};
	wait.semaphoreCount = 1;
	wait.pSemaphores    = &m_semaphore;
	wait.pValues        = &value;
	RequireVulkanSuccess(m_graphics.device.waitSemaphores(&wait, UINT64_MAX),
	                     "wait upload DMA semaphore");
}

void UploadDma::SubmitBatch(std::vector<Job>& jobs) {
	auto& batch = m_batches[m_next_batch];
	m_next_batch = (m_next_batch + 1) % m_batches.size();
	if (batch.value != 0) {
		// The command buffer's previous batch must have finished before it is reset.
		WaitHost(batch.value);
	}
	const auto command = batch.command;
	RequireVulkanSuccess(command.reset({}), "reset upload DMA command buffer");
	vk::CommandBufferBeginInfo begin {};
	begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	RequireVulkanSuccess(command.begin(&begin), "begin upload DMA command buffer");
	if (GpuBreadcrumbs::Active()) {
		GpuBreadcrumbs::Write(command, GpuBreadcrumbs::Queue::Transfer,
		                      GpuBreadcrumbs::Note(GpuBreadcrumbs::Queue::Transfer,
		                                           {.op   = GpuBreadcrumbs::OpUploadDma,
		                                            .arg0 = static_cast<uint32_t>(jobs.size()),
		                                            .arg4 = jobs.back().value}));
	}
	std::vector<vk::BufferCopy> regions;
	uint64_t                    reuse_tick = 0;
	for (size_t first = 0; first < jobs.size();) {
		// One copy command per run of jobs with the same source buffer.
		size_t last = first;
		regions.clear();
		while (last < jobs.size() && jobs[last].source == jobs[first].source) {
			regions.push_back({jobs[last].source_offset, jobs[last].ring_offset, jobs[last].size});
			reuse_tick = std::max(reuse_tick, jobs[last].reuse_tick);
			last++;
		}
		command.copyBuffer(jobs[first].source, m_ring->Handle(), static_cast<uint32_t>(regions.size()),
		                   regions.data());
		first = last;
	}
	if (GpuBreadcrumbs::Active()) {
		GpuBreadcrumbs::Write(command, GpuBreadcrumbs::Queue::Transfer,
		                      GpuBreadcrumbs::Note(GpuBreadcrumbs::Queue::Transfer,
		                                           {.op = GpuBreadcrumbs::OpUploadDmaEnd, .arg4 = jobs.back().value}));
	}
	RequireVulkanSuccess(command.end(), "end upload DMA command buffer");

	const auto value = jobs.back().value;
	vk::CommandBufferSubmitInfo command_info {};
	command_info.commandBuffer = command;
	vk::SemaphoreSubmitInfo signal {};
	signal.semaphore = m_semaphore;
	signal.value     = value;
	signal.stageMask = vk::PipelineStageFlagBits2::eAllTransfer;
	// Ring bytes reused from completed ticks: the graphics reads of them happened before those
	// ticks' signals, which have been observed; the wait makes that order explicit on the device.
	vk::SemaphoreSubmitInfo reuse {};
	reuse.semaphore = m_scheduler.GetMasterSemaphore().Handle();
	reuse.value     = reuse_tick;
	reuse.stageMask = vk::PipelineStageFlagBits2::eAllTransfer;
	vk::SubmitInfo2 submit {};
	submit.waitSemaphoreInfoCount   = reuse_tick != 0 ? 1u : 0u;
	submit.pWaitSemaphoreInfos      = &reuse;
	submit.commandBufferInfoCount   = 1;
	submit.pCommandBufferInfos      = &command_info;
	submit.signalSemaphoreInfoCount = 1;
	submit.pSignalSemaphoreInfos    = &signal;
	// Only this worker submits to the transfer queue.
	HangWatchdog::Scope native(
	    "vkQueueSubmit2-transfer",
	    reinterpret_cast<uint64_t>(static_cast<VkQueue>(m_graphics.transfer_queue)), value,
	    reuse_tick);
	if (HangWatchdog::Enabled()) {
		const HangWatchdog::SemaphoreValue waited {
		    reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(reuse.semaphore)), reuse.value,
		    static_cast<uint64_t>(static_cast<VkPipelineStageFlags2>(reuse.stageMask))};
		const HangWatchdog::SemaphoreValue signalled {
		    reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(signal.semaphore)), signal.value,
		    static_cast<uint64_t>(static_cast<VkPipelineStageFlags2>(signal.stageMask))};
		HangWatchdog::NoteNativeSubmit(
		    reinterpret_cast<uint64_t>(static_cast<VkQueue>(m_graphics.transfer_queue)), 0,
		    reinterpret_cast<uint64_t>(static_cast<VkCommandBuffer>(command)),
		    std::span(&waited, submit.waitSemaphoreInfoCount), std::span(&signalled, 1));
	}
	RequireVulkanSuccess(m_graphics.transfer_queue.submit2(1, &submit, nullptr),
	                     "submit upload DMA copies");
	// Graphics batches waiting for this value may reach their queue now (SubmitDependency).
	m_submitted.store(value, std::memory_order_release);
	m_submitted.notify_all();
	batch.value = value;
	Profiler::CountFrameEvent(Profiler::FrameEvent::UploadDmaSubmits);
}

void UploadDma::HoldWorkerForTest(bool hold) {
	{
		std::scoped_lock lock(m_mutex);
		m_hold = hold;
	}
	m_available.notify_all();
}

void UploadDma::Worker(std::stop_token stop) {
	KYTY_PROFILER_THREAD("Upload DMA");
	(void)stop;
	std::vector<Job> jobs;
	for (;;) {
		{
			std::unique_lock lock(m_mutex);
			m_available.wait(lock,
			                 [this] { return m_stopping || (!m_hold && !m_jobs.empty()); });
			if (m_jobs.empty()) {
				return; // stopping and drained
			}
			jobs.swap(m_jobs);
		}
		// KYTY_UPLOAD_DMA_HOST_COPY: the staged bytes the command processor left to this worker,
		// written before the submission below makes them visible to the transfer.
		uint64_t copied = 0;
		for (const auto& job: jobs) {
			for (const auto& copy: job.host_copies) {
				std::memcpy(copy.destination, copy.source, static_cast<size_t>(copy.size));
				copied += copy.size;
			}
		}
		if (copied != 0) {
			m_host_bytes_done.fetch_add(copied, std::memory_order_release);
		}
		SubmitBatch(jobs);
		jobs.clear();
	}
}

} // namespace Libs::Graphics
