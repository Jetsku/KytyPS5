#pragma once
#include "common/hangWatchdog.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>

namespace Libs::Graphics {
// Copy only host submission operands; the watchdog never follows Vulkan or guest pointers.
inline void NoteWatchdogSubmit(vk::Queue queue, const vk::SubmitInfo& info, uint64_t tick = 0) {
	if (!HangWatchdog::Enabled()) return;
	const VkTimelineSemaphoreSubmitInfo* timeline = nullptr;
	auto*                                chain = static_cast<const VkBaseInStructure*>(info.pNext);
	while (chain) {
		if (chain->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO) {
			timeline = reinterpret_cast<const VkTimelineSemaphoreSubmitInfo*>(chain);
			break;
		}
		chain = chain->pNext;
	}
	std::array<HangWatchdog::SemaphoreValue, 8> waits {}, signals {};
	const auto nw = std::min<uint32_t>(info.waitSemaphoreCount, waits.size());
	const auto ns = std::min<uint32_t>(info.signalSemaphoreCount, signals.size());
	for (uint32_t i = 0; i < nw; ++i)
		waits[i] = {reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(info.pWaitSemaphores[i])),
		            timeline && i < timeline->waitSemaphoreValueCount
		                ? timeline->pWaitSemaphoreValues[i]
		                : 0,
		            static_cast<VkPipelineStageFlags>(info.pWaitDstStageMask[i])};
	for (uint32_t i = 0; i < ns; ++i)
		signals[i] = {
		    reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(info.pSignalSemaphores[i])),
		    timeline && i < timeline->signalSemaphoreValueCount
		        ? timeline->pSignalSemaphoreValues[i]
		        : 0,
		    0};
	HangWatchdog::NoteNativeSubmit(
	    reinterpret_cast<uint64_t>(static_cast<VkQueue>(queue)), tick,
	    info.commandBufferCount
	        ? reinterpret_cast<uint64_t>(static_cast<VkCommandBuffer>(info.pCommandBuffers[0]))
	        : 0,
	    std::span(waits).first(nw), std::span(signals).first(ns), info.waitSemaphoreCount,
	    info.signalSemaphoreCount);
}
} // namespace Libs::Graphics
