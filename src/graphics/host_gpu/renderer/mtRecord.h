#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MTRECORD_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MTRECORD_H_

// KYTY_MT_RECORD=1 (default off; prototype): multi-threaded replay of the CP recorder's command
// stream (KYTY_CP_RECORDER=1, commandRecorder.h).
//
// The recorder thread looks ahead at every dynamic render pass (BeginRendering .. EndRendering).
// When the whole pass is published and contains only draw-state commands and draws, it is split
// into chunks of draws; each chunk is recorded into a secondary command buffer by one of
// KYTY_MT_RECORD_THREADS (2) worker threads, or by the recorder itself, in parallel. The recorder
// then records BeginRendering (contents: secondary command buffers), one vkCmdExecuteCommands with
// the chunks in stream order, and EndRendering into the guest primary command buffer. Each chunk
// starts by re-emitting the graphics state (pipeline, descriptor sets, push descriptors, push
// constants, vertex/index buffers, every dynamic state) that is current at its first packet: the
// recorder tracks the latest packet per state key since the primary's Begin.
//
// Anything else falls back to the unchanged serial replay, counted by reason: barriers, queries,
// timestamps, markers, compute or copies inside the pass, a drain marker before its end, a pass not
// fully published within KYTY_MT_RECORD_WAIT_US (300), too few draws (KYTY_MT_RECORD_MIN_DRAWS,
// 16), suspending/resuming passes, an occlusion query active across the pass, attachments whose
// format or sample count is unknown, or a state key the tracker cannot represent.
//
// Disabled (whole feature) unless the recorder runs in thread mode, and with
// KYTY_CP_RECORDER_VERIFY, KYTY_GPU_OP_PROFILE/CAPTURE, KYTY_CMD_GUARD.
//
// KYTY_MT_RECORD_CHUNK (48): draws per secondary command buffer. KYTY_MT_RECORD_SPIN_US (300):
// worker spin before parking. Stats line every 5 s ("MtRecord:").

#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <cstdint>

namespace Libs::Graphics::MtRecord {

// KYTY_MT_RECORD=1 (evaluated once).
[[nodiscard]] bool Enabled();
// KYTY_MT_RECORD_THREADS (2, 1..8).
[[nodiscard]] uint32_t Threads();

// Image and image-view registry for the secondary command buffers' inheritance info (formats and
// sample count of the pass attachments). Only populated when Enabled().
void InstallHooks();
void NoteImage(vk::Image image, vk::SampleCountFlagBits samples);
struct ViewInfo {
	vk::Format              format  = vk::Format::eUndefined;
	vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;
};
// False when the view (or its image) is unknown.
[[nodiscard]] bool LookupView(vk::ImageView view, ViewInfo* info);

// CP-side timing of the draw entry points (DrawIndex/DrawAuto/DrawIndirectNative), for the stats.
extern std::atomic<uint64_t> g_cp_draw_ns;
extern std::atomic<uint64_t> g_cp_draws;
uint64_t NowNs();
class CpDrawScope {
public:
	CpDrawScope(): m_start(Enabled() ? NowNs() : 0) {}
	~CpDrawScope() {
		if (m_start != 0) {
			g_cp_draw_ns.fetch_add(NowNs() - m_start, std::memory_order_relaxed);
			g_cp_draws.fetch_add(1, std::memory_order_relaxed);
		}
	}
	CpDrawScope(const CpDrawScope&)            = delete;
	CpDrawScope& operator=(const CpDrawScope&) = delete;

private:
	uint64_t m_start;
};

} // namespace Libs::Graphics::MtRecord

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MTRECORD_H_
