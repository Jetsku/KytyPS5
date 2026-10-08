#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_WINDOWINTERNAL_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_WINDOWINTERNAL_H_

#include <SDL3/SDL.h>

#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace Libs::Graphics {

class Presenter;
class RenderContext;

struct SurfaceCapabilities {
	vk::SurfaceCapabilitiesKHR        capabilities {};
	std::vector<vk::SurfaceFormatKHR> formats;
	std::vector<vk::PresentModeKHR>   present_modes;
};

// KYTY_PRESENT_BOX_DOWNSCALE=1 (default off; BryanKAdams/KytyPS5 6b00e83): a linear blit that
// shrinks by less than 2x (a 4K frame in a maximized 2560x1369 window) reads texel pairs at uneven
// phases, so a per-pixel dither survives at up to half strength as a static diagonal beat.
// PresentFilter averages each target pixel over a two-texel box instead, which cancels one-texel
// patterns at any phase. An exact 2:1 blit already is that box, so it, 1:1 and upscales keep the
// blit; so do downscales beyond 2:1 (the presented frame has no mip chain here).
[[nodiscard]] bool PresentNeedsFilter(vk::Extent2D source, vk::Extent2D target) noexcept;

class PresentFilter final {
public:
	PresentFilter()  = default;
	~PresentFilter() = default;
	KYTY_CLASS_NO_COPY(PresentFilter);

	// Draws level `level` (extent `source`, at most twice `target` on each axis) of `source_view`
	// over all of `target_view`. The source view is in eShaderReadOnlyOptimal; the target is in
	// eColorAttachmentOptimal.
	void Record(vk::Device device, vk::CommandBuffer command, vk::ImageView source_view,
	            uint32_t level, vk::Extent2D source, vk::ImageView target_view,
	            vk::Format target_format, vk::Extent2D target);
	void Release(vk::Device device);

private:
	vk::Format              m_format      = vk::Format::eUndefined;
	vk::Sampler             m_sampler     = nullptr;
	vk::DescriptorSetLayout m_descriptors = nullptr;
	vk::PipelineLayout      m_layout      = nullptr;
	vk::Pipeline            m_pipeline    = nullptr;
};

struct WindowLoopState {
	SDL_Event        event {};
	bool             need_exit = false;
	std::atomic_bool paused    = false;
	std::atomic<uint64_t> presented_frames {0};
};

struct WindowContext {
	WindowContext();
	~WindowContext();
	KYTY_CLASS_NO_COPY(WindowContext);

	[[nodiscard]] static vk::PhysicalDeviceVulkan11Features RequiredVulkan11Features() noexcept;
	[[nodiscard]] static vk::PhysicalDeviceVulkan12Features RequiredVulkan12Features() noexcept;
	[[nodiscard]] static vk::PhysicalDeviceVulkan13Features RequiredVulkan13Features() noexcept;
	[[nodiscard]] static uint32_t InitialWindowFlags(bool fullscreen) noexcept;
	void                                                    CreateVulkan();
	void                                                    RecreateSurface();
	void                                                    RefreshSurfaceCapabilities();
	void                                                    UpdateIcon();
	void UpdateTitle(uint64_t frame_num, double current_fps);
	/// Resizes the drawable surface to the given pixel dimensions.
	/// Sets `minimized = false` on a positive size; sets `minimized = true` and returns early on a
	/// nonpositive one.
	void Resize(int width, int height);
	/// Dispatches a single SDL window event and keeps `minimized` up to date.
	void ProcessWindowEvent(const SDL_WindowEvent& event);
	void ProcessDisplayEvent(const SDL_DisplayEvent& event);
	void ProcessEvent(double time_seconds);
	void Run();

	GraphicContext                 graphic_ctx;
	SDL_Window*                    window        = nullptr;
	vk::SurfaceKHR                 surface       = nullptr;
	SurfaceCapabilities            surface_capabilities;
	std::unique_ptr<RenderContext> render_context;
	std::unique_ptr<Presenter>     presenter;
	WindowLoopState                loop;
	std::atomic_bool               minimized     = false;

	Common::Mutex mutex;

private:
	/// Queries the current drawable size via SDL and calls Resize() to clear `minimized`.
	/// Does nothing when the window is still minimised or the reported size is non-positive.
	void RefreshSizeFromWindow();
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_WINDOWINTERNAL_H_
