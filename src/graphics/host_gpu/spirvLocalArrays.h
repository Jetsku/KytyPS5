#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_SPIRVLOCALARRAYS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_SPIRVLOCALARRAYS_H_

#include <cstdint>
#include <span>
#include <string>
#include <vector>

// Per-invocation (Function storage) arrays in a SPIR-V module, shrunk to the indices the module can
// reach (KYTY_FUNCTION_ARRAY_SHRINK).
//
// Drivers back Function-storage arrays that stay indexed at run time with local (scratch) memory and
// size it for every thread the GPU can keep resident: the NVIDIA driver reserves the largest
// per-thread footprint of any pipeline times ~126K threads on an RTX 3090 when the pipeline is
// created, and keeps it for the life of the device. The recompiler emulates LDS in vertex and pixel
// shaders (and GCN scratch) with such arrays, sized for the whole 32 KiB LDS (8192 dwords): eight
// Astro Bot pixel shaders that only touch LDS[lane] and LDS[lane + 64] made the driver hold 3.9 GiB
// of device memory outside every allocation the emulator makes.
//
// Shrink() proves an upper bound for every index of a Function-storage array variable with an
// interval analysis over the module's SSA values (constants, SubgroupLocalInvocationId and
// SubgroupSize, integer arithmetic, shifts, masks, selects, phis) and gives the variable an array
// type of that length. It changes nothing when any use of the variable is not an access chain with
// a provably bounded first index, so every load and store keeps its element: the module behaves the
// same. Only the module given to the driver changes (the program cache keeps the emitted words).
namespace Libs::Graphics::SpirvLocalArrays {

struct Shrunk {
	uint32_t    variable   = 0; // result id of the OpVariable
	std::string name;           // OpName, if any
	uint32_t    old_length = 0; // elements
	uint32_t    new_length = 0;
	uint32_t    element_bytes = 0;
};

struct Result {
	bool                changed = false;
	std::vector<Shrunk> arrays;              // the arrays that were shrunk
	uint32_t            unbounded_arrays = 0; // Function-storage arrays left as they were
	uint64_t            bytes_before     = 0; // per invocation, all Function-storage arrays
	uint64_t            bytes_after      = 0;
};

// The upper bound assumed for SubgroupLocalInvocationId (+1) and SubgroupSize: 128, Vulkan's
// largest subgroup size, unless set lower (the device's maxSubgroupSize).
void SetMaxSubgroupSize(uint32_t size) noexcept;

// What a shrunk array starts with. None: undefined, as emitted (Function variables have no
// initializer). Zero: OpConstantNull (the driver zero-fills it at function entry). Poison: every
// element 0x7fc00000 (a float NaN), a diagnostic that makes reads of never-written elements visible.
enum class Init : uint8_t { None, Zero, Poison };

// `out` receives the rewritten module when Result::changed, else it is left empty.
Result Shrink(std::span<const uint32_t> module, std::vector<uint32_t>& out, Init init = Init::None);

} // namespace Libs::Graphics::SpirvLocalArrays

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_SPIRVLOCALARRAYS_H_ */
