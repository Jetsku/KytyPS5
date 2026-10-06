#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHINDIRECT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHINDIRECT_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <span>

namespace Libs::Graphics {

class CommandBuffer;
class RenderContext;
class StreamBuffer;

// Native indirect mesh draws (KYTY_NATIVE_INDIRECT_MESH).
//
// An indirect draw whose vertex stage runs as a mesh shader (NGG) fell back to the CPU path:
// the command processor read the argument record, which a shader of the recording just submitted
// had written, so it waited for the GPU to reach that shader (GpuWaitSideCopy, about 2 ms per
// flip in the desert, every flip). The CPU path derives the mesh dispatches (workgroups per
// instance, instances per dispatch) and the six draw dwords the mesh shader reads (index or
// vertex count, vertex offset, first instance, index size, index address) from those counts.
//
// Here a helper compute dispatch derives exactly the same values on the GPU from the same bytes,
// right before the draw, and the draw becomes Records indirect mesh dispatches whose shaders read
// their draw dwords from memory: mesh push dword 3 (the index size, 0/1/2/4 on the CPU path) is
// IR::PushData::MeshIndirectSentinel and dwords 0-1 hold the device address of the dispatch's
// parameter block (MeshDrawParameter in the SPIR-V backend, CodegenOptions::mesh_indirect_params).
// Convert() is the C++ twin of the shader (gpu_mesh_indirect.comp) and states the CPU path's
// decisions; a dispatch the CPU path would not record has no workgroups.
//
// Differences from the CPU path, all counted in the slot's status word:
//  - more than Records dispatches (instances per dispatch x Records < instance count): the CPU
//    path records them all, the GPU conversion drops the rest (StatusOverflow);
//  - a draw the host cannot dispatch at all (more workgroups than the host limits): the CPU path
//    stops the emulator, or with ShaderMeshInputInfo::split_groups dispatches the groups in parts,
//    the conversion draws nothing (StatusLimits);
//  - an indexed record whose start + clamped count passes INDEX_BUFFER_SIZE: both read the same
//    index bytes, but only [INDEX_BASE, +INDEX_BUFFER_SIZE) was registered and synchronized for
//    the mesh shader's address reads (the CPU path registers the exact range) (StatusIndexRange).
//
// Provably empty draws. The per-flip indirect mesh draw Astro Bot issues in every scene (argument
// record 0x56ddaffa0, INDEX_BASE 0x56ddaffc0) binds INDEX_BUFFER_SIZE = 1 index: the CPU path
// clamps its count to 1, which is below one input primitive, and so draws nothing whatever the
// record holds. AlwaysEmpty() recognises such a draw from the registers; it is then consumed
// without reading the record (no drain), as the CPU path would draw it: nothing (target
// operations and custom restart values still take the CPU path, which may act on them). The
// command processor keeps the record as the pending NUM_INSTANCES source, as for every native
// indirect draw.
//
// KYTY_NATIVE_INDIRECT_MESH:
//   empty          provably empty indirect mesh draws only; every other one reads its record
//                  (the mesh shaders are generated without the indirect branch);
//   unset | 1 | on also the GPU conversion for the other ones. The default since int16: at the
//                  Sky Garden start view the CPU path's record reads (SideReadbackState::Wait,
//                  5.5% of command-processor samples) cost about 2.1 ms per flip
//                  (30.5 -> 28.5 ms/flip, 32.7 -> 35.1 fps, occlusion off, no game patch);
//   verify         on, and every conversion is compared after its recording completes with
//                  Convert() of the argument bytes the GPU read (MeshIndirectVerifyMismatches);
//   exit           verify, stopping on the first difference;
//   0              off: every indirect mesh draw reads its record on the CPU.
namespace MeshIndirect {

enum class Mode : uint8_t { Off, Empty, On, Verify, VerifyExit };
[[nodiscard]] Mode GetMode();
// The GPU conversion is enabled (On, Verify, VerifyExit).
[[nodiscard]] bool ConversionEnabled();

// Slot layout (dwords): Records commands {x, y, z}, Records parameter blocks of 8 dwords
// {count, vertex offset, first instance, index size, index address lo, hi, 0, 0}, the status
// word, the 5 argument dwords as read, and the generation the dispatch was recorded with.
inline constexpr uint32_t Records        = 4;
inline constexpr uint32_t CommandDwords  = 3;
inline constexpr uint32_t ParamDwords    = 8;
inline constexpr uint32_t ParamsWord     = Records * CommandDwords;
inline constexpr uint32_t StatusWord     = ParamsWord + Records * ParamDwords;
inline constexpr uint32_t ArgsCopyWord   = StatusWord + 1;
inline constexpr uint32_t GenerationWord = ArgsCopyWord + 5;
inline constexpr uint32_t SlotDwords     = 64;
inline constexpr uint32_t SlotBytes      = SlotDwords * 4;
static_assert(GenerationWord < SlotDwords);

inline constexpr uint32_t StatusOverflow   = 1;
inline constexpr uint32_t StatusLimits     = 2;
inline constexpr uint32_t StatusIndexRange = 4;

struct Inputs {
	bool     indexed              = false;
	uint64_t index_base           = 0;
	uint32_t index_size           = 0; // bytes per index (indexed records)
	uint32_t index_buffer_size    = 0; // INDEX_BUFFER_SIZE in indices; 0: no clamp
	uint32_t primitive_size       = 3; // ShaderMeshInputInfo::InputPrimitiveSize
	uint32_t primitive_step       = 3; // ShaderMeshInputInfo::InputPrimitiveStep
	uint32_t primitives_per_group = 1;
	uint32_t max_groups_x         = 0; // VkPhysicalDeviceMeshShaderPropertiesEXT
	uint32_t max_groups_y         = 0;
	uint32_t max_groups_total     = 0;
};

struct Conversion {
	std::array<std::array<uint32_t, CommandDwords>, Records> commands {};
	std::array<std::array<uint32_t, ParamDwords>, Records>   params {};
	uint32_t                                                 status = 0;

	bool operator==(const Conversion&) const = default;
};

// The CPU path's decisions for one argument record: 5 dwords (DrawIndexedIndirectArgs), or the
// first 4 (DrawIndirectArgs) when not indexed.
[[nodiscard]] Conversion Convert(const Inputs& inputs, std::span<const uint32_t, 5> args);

// No argument record can yield a primitive: INDEX_BUFFER_SIZE clamps every index count below one
// input primitive, so the CPU path draws nothing whatever the record holds.
[[nodiscard]] bool AlwaysEmpty(const Inputs& inputs);

// Process-wide outcomes (tests read them; the MeshIndirect* frame events need a profiler).
struct Totals {
	std::atomic<uint64_t> draws {0};
	std::atomic<uint64_t> always_empty {0};
	std::atomic<uint64_t> checks {0};
	std::atomic<uint64_t> mismatches {0};
	std::atomic<uint64_t> status_overflow {0};
	std::atomic<uint64_t> status_limits {0};
	std::atomic<uint64_t> status_index_range {0};
	std::atomic<uint64_t> stale {0};
};
[[nodiscard]] Totals& GetTotals();

// Owned by the RenderExecutor; GPU thread except the completion checks.
class Converter {
public:
	explicit Converter(RenderContext& context);
	~Converter();
	Converter(const Converter&)            = delete;
	Converter& operator=(const Converter&) = delete;

	struct Slot {
		vk::Buffer        buffer     = nullptr;
		uint64_t          offset     = 0; // bytes, in `buffer`
		vk::DeviceAddress address    = 0; // of the slot
		uint32_t          generation = 0;

		[[nodiscard]] uint64_t CommandOffset(uint32_t record) const {
			return offset + uint64_t {record} * CommandDwords * 4u;
		}
		[[nodiscard]] vk::DeviceAddress ParamsAddress(uint32_t record) const {
			return address + (uint64_t {ParamsWord} + uint64_t {record} * ParamDwords) * 4u;
		}
	};

	// Preparation phase: may submit the recording (ring wrap), like any stream allocation.
	[[nodiscard]] Slot Reserve();
	// Ends the active rendering instance and records the conversion of the record at
	// (args, args_offset) into `slot`, ordered after every earlier write (the batched barrier the
	// dispatch flushes) and before the draw's indirect reads, mesh-shader reads and a host read
	// (requested, recorded at the draw's BeginRendering). Queues the completion check.
	void Record(CommandBuffer& buffer, const Slot& slot, const Inputs& inputs, vk::Buffer args,
	            uint64_t args_offset);

	// Tests: the slot's words once its recording has completed.
	[[nodiscard]] std::array<uint32_t, SlotDwords> ReadSlot(const Slot& slot) const;

private:
	void Initialize();
	void Check(const Slot& slot, const Inputs& inputs);

	RenderContext&                m_context;
	vk::DescriptorSetLayout       m_descriptors = nullptr;
	vk::PipelineLayout            m_layout      = nullptr;
	vk::Pipeline                  m_pipeline    = nullptr;
	std::unique_ptr<StreamBuffer> m_ring;
	uint32_t                      m_generation = 0;
};

} // namespace MeshIndirect
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHINDIRECT_H_
