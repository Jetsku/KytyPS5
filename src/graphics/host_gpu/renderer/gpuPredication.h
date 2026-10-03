#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUPREDICATION_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUPREDICATION_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>

namespace Libs::Graphics {

class Buffer;
class CommandBuffer;
class RenderContext;

// Guest SET_PREDICATION with a boolean memory source (op 3) that asks to wait for the value.
//
// The command processor used to submit everything recorded and wait for the GPU to go idle
// before reading the 8 bytes (CommandProcessor::ExecPredication), about 13 times a frame in the
// slow Crash Bandicoot 4 scene, so the GPU sat idle while the CPU waited and the other way round.
//
// KYTY_PREDICATION_MODE:
//   drain            the wait above, then a CPU read; predicated packets are skipped on the CPU.
//   precise          read the value as WAIT_REG_MEM does (ReadGuestForCp's ownership checks): a
//                    clean value needs no wait, a GPU-written one waits for its own readback.
//   gpu (default)    precise when the value is clean on the CPU; otherwise a one-invocation
//                    compute dispatch (gpu_predicate.comp) reads the 8 bytes where the command
//                    processor meets the packet and writes a 32-bit predicate slot, and the
//                    predicated direct draws are recorded inside VK_EXT_conditional_rendering
//                    scopes on that slot instead of being skipped. Without the extension it is
//                    precise.
//
// What conditional rendering cannot gate takes the decision on the CPU (Resolve: wait for the
// recording that wrote the slot, read its host copy), so it matches what the GPU did:
//  - predicated packets other than direct draws (register writes, labels, dispatches, indirect
//    draws): the command processor resolves before running them;
//  - draws that run a target operation instead of drawing (metadata colour modes, resolves,
//    depth/stencil copies) or clear depth/stencil through rendering load operations: the
//    renderer resolves before them (NeedsCpuDecision).
// The renderer's write tracking of a draw the GPU then discards stays conservative: memory it
// marks GPU-written keeps its old bytes, which a later readback returns unchanged.
namespace GpuPredication {

enum class Mode : uint8_t { Drain, Precise, Gpu };
[[nodiscard]] Mode GetMode();
// The device enables VK_EXT_conditional_rendering only for mode gpu.
[[nodiscard]] bool ExtensionRequested();

struct Totals {
	std::atomic<uint64_t> recorded {0};     // predicates snapshotted on the GPU
	std::atomic<uint64_t> gated_draws {0};  // draws recorded inside a conditional scope
	std::atomic<uint64_t> resolves {0};     // CPU decisions taken (each waits for the GPU once)
	std::atomic<uint64_t> slot_waits {0};   // slot reuse that had to wait for an older recording
};
[[nodiscard]] Totals& GetTotals();

// Owned by the RenderExecutor. GPU thread.
class Predicates {
public:
	explicit Predicates(RenderContext& context);
	~Predicates();
	Predicates(const Predicates&)            = delete;
	Predicates& operator=(const Predicates&) = delete;

	[[nodiscard]] bool Supported() const;

	// Render mutex held, recording command buffer. Ends the active rendering instance and
	// records the snapshot of the 8 bytes at (source, source_offset) for `condition`, ordered
	// after every earlier write and before the conditional-rendering and host reads. Returns the
	// predicate id (nonzero).
	[[nodiscard]] uint32_t Record(CommandBuffer& buffer, const Buffer& source,
	                              uint64_t source_offset, uint32_t condition);
	// The conditional-rendering scope of a draw predicated on `id`, recorded now.
	[[nodiscard]] vk::ConditionalRenderingBeginInfoEXT Use(uint32_t id);
	// Whether the predicated packets of `id` run (the GPU's decision). The first call per id
	// waits for the recording that wrote the slot (submitting the current one if needed). Not
	// with the render mutex held.
	[[nodiscard]] bool Resolve(uint32_t id);

private:
	static constexpr uint32_t Slots = 4096;

	struct Entry {
		uint32_t id         = 0;
		int32_t  resolved   = -1; // -1: unknown, else 0/1 (draw)
		uint64_t write_tick = 0;  // recording that wrote the slot
		uint64_t last_tick  = 0;  // last recording that reads it (conditional rendering)
	};

	void   Initialize();
	Entry& Lookup(uint32_t id);

	RenderContext&                  m_context;
	vk::DescriptorSetLayout         m_descriptors = nullptr;
	vk::PipelineLayout              m_layout      = nullptr;
	vk::Pipeline                    m_pipeline    = nullptr;
	std::unique_ptr<Buffer>         m_predicates; // device-local, conditional-rendering source
	std::unique_ptr<Buffer>         m_readback;   // host copy for Resolve
	std::array<Entry, Slots>        m_entries {};
	uint32_t                        m_next_id = 0;
};

} // namespace GpuPredication
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUPREDICATION_H_
