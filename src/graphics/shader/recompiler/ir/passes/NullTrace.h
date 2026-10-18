#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_NULLTRACE_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_NULLTRACE_H_

#include <cstdint>

// KYTY_NULL_TRACE=1 (diagnostics, default off): counts, per guest shader hash, what was nulled or
// skipped for its draws and dispatches (buffer V#s specialized as null, image T#s that are not
// descriptors, programs whose draws or dispatches are skipped, failed materializations) and writes
// the top shaders of each kind every 10 s to stderr ("Kyty null trace"). For finding the shaders of
// something missing on screen: stand at it and compare with a reference spot.
namespace Libs::Graphics::ShaderRecompiler::IR {

enum class NullTraceKind : uint32_t {
	NullBuffer = 0,  // buffer V# specialized as null (unmapped base)
	NullImage,       // image T# that is not a descriptor, bound as null
	SkippedProgram,  // a draw or dispatch skipped because its program is skipped
	SkippedResources, // a draw or dispatch skipped because its resources did not materialize
	Count
};

[[nodiscard]] bool NullTraceEnabled();
// The shader whose resources the calling thread materializes (for NullBuffer, which has no hash).
void NullTraceSetShader(uint64_t shader_hash);
void NullTraceNote(NullTraceKind kind, uint64_t shader_hash = 0);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_NULLTRACE_H_ */
