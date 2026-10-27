#pragma once

#include <cstdint>
#include <span>
#include <string>

namespace Libs::Graphics {
struct ShaderTessellationInputInfo;
}

namespace Libs::Graphics::ShaderRecompiler {

struct CompileOptions;
namespace IR {
struct Program;
}

// False when the LS/HS pair cannot be modelled (KYTY_SKIP_INVALID on; with it off the analysis
// ends the emulator as before): the caller skips the set's patch draws. `rejection` gets the reason.
bool AnalyzeTessellationPrograms(std::span<const uint32_t> local, std::span<const uint32_t> control,
                                 ShaderTessellationInputInfo& info, uint64_t local_hash = 0,
                                 uint64_t control_hash = 0,
                                 std::span<const uint32_t> evaluation = {},
                                 std::string* rejection = nullptr);
namespace Decoder {
struct Program;
}
// `decoded` (optional) is the guest program: its HS ring stores with constant addresses past the
// per-vertex area are patch constants even when the IR address does not fold.
void LowerTessellationMemory(IR::Program& program, const CompileOptions& options,
                             const Decoder::Program* decoded = nullptr);

} // namespace Libs::Graphics::ShaderRecompiler
