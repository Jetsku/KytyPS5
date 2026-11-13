#include "graphics/host_gpu/renderer/mtRecord.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace Libs::Graphics::MtRecord {

std::atomic<uint64_t> g_cp_draw_ns {0};
std::atomic<uint64_t> g_cp_draws {0};

uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

bool Enabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_MT_RECORD");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

uint32_t Threads() {
	static const uint32_t threads = [] {
		const auto* value = std::getenv("KYTY_MT_RECORD_THREADS");
		uint32_t    n     = value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 2u;
		return n < 1u ? 1u : (n > 8u ? 8u : n);
	}();
	return threads;
}

namespace {

struct ViewEntry {
	vk::Format format = vk::Format::eUndefined;
	vk::Image  image  = nullptr;
};

std::shared_mutex                                                g_mutex;
std::unordered_map<VkImage, vk::SampleCountFlagBits>            g_images;
std::unordered_map<VkImageView, ViewEntry>                       g_views;
PFN_vkCreateImageView                                            g_real_create_view  = nullptr;
PFN_vkDestroyImageView                                           g_real_destroy_view = nullptr;

VKAPI_ATTR VkResult VKAPI_CALL HookCreateImageView(VkDevice device, const VkImageViewCreateInfo* info,
                                                   const VkAllocationCallbacks* allocator,
                                                   VkImageView*                 view) {
	const auto result = g_real_create_view(device, info, allocator, view);
	if (result == VK_SUCCESS && info != nullptr && view != nullptr) {
		std::unique_lock lock(g_mutex);
		g_views[*view] = ViewEntry {static_cast<vk::Format>(info->format), vk::Image(info->image)};
	}
	return result;
}

VKAPI_ATTR void VKAPI_CALL HookDestroyImageView(VkDevice device, VkImageView view,
                                                const VkAllocationCallbacks* allocator) {
	if (view != VK_NULL_HANDLE) {
		std::unique_lock lock(g_mutex);
		g_views.erase(view);
	}
	g_real_destroy_view(device, view, allocator);
}

} // namespace

void InstallHooks() {
	if (!Enabled()) {
		return;
	}
	static std::once_flag once;
	std::call_once(once, [] {
		auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
		if (d.vkCreateImageView != nullptr && d.vkDestroyImageView != nullptr) {
			g_real_create_view  = d.vkCreateImageView;
			g_real_destroy_view = d.vkDestroyImageView;
			d.vkCreateImageView  = &HookCreateImageView;
			d.vkDestroyImageView = &HookDestroyImageView;
		}
	});
}

void NoteImage(vk::Image image, vk::SampleCountFlagBits samples) {
	if (!Enabled() || !image) {
		return;
	}
	std::unique_lock lock(g_mutex);
	g_images[static_cast<VkImage>(image)] = samples;
}

bool LookupView(vk::ImageView view, ViewInfo* info) {
	std::shared_lock lock(g_mutex);
	const auto       v = g_views.find(static_cast<VkImageView>(view));
	if (v == g_views.end()) {
		return false;
	}
	const auto i = g_images.find(static_cast<VkImage>(v->second.image));
	if (i == g_images.end()) {
		return false;
	}
	info->format  = v->second.format;
	info->samples = i->second;
	return true;
}

} // namespace Libs::Graphics::MtRecord
