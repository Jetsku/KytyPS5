#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PASSCENSUS_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PASSCENSUS_H_

#include <cstdint>
#include <string>

// KYTY_TRACE_PASS_CENSUS=1 (diagnostic, default off; idea from Erlou93/KytyPS5-Yotei passCensus.cpp): the game's
// debug-marker passes ("outer>inner", two levels of the command processor's marker stack) with, per pass and frame,
// the draw/dispatch packets the CP parsed, the draws/dispatches that reached the renderer (by shader hash), the ones
// it dropped or skipped (with the shader hash and reason) and the resources bound as null. Every 5 s the whole table
// goes to KYTY_TRACE_PASS_CENSUS_FILE (default _pass-census.txt in the working directory).
// Attribution is exact only when the CP parses on the GPU thread (KYTY_CP_SEQ=0 KYTY_CP_SEQ_PREFETCH=0): renderer
// events take the pass of the last marker/packet event on the GPU thread; events on other threads are not counted.
namespace Libs::Graphics::PassCensus {

[[nodiscard]] bool Enabled();
// A marker push (`push_name`) or pop (null) of command processor `cp`.
void Marker(const void* cp, const char* push_name);
// A draw (or dispatch) packet of `cp`.
void Packet(const void* cp, bool dispatch);
// A draw ('d': a = VS hash, b = PS hash) or dispatch ('c': a = CS hash) prepared for recording.
void Exec(char kind, uint64_t a, uint64_t b);
// A draw ('d') or dispatch ('c') the renderer dropped or skipped; `hash` is 0 when unknown (the line then keeps
// hash 0 and names the thread's last materialized shader only in the reason, as a hint).
void Drop(char kind, uint64_t hash, const char* reason);
// A resource of the shader being prepared bound as null, or a specialization failure (`reason` copied).
void Null(const char* reason);
// The shader whose resources are materialized on this thread now (named by Null and Drop).
void SetShader(uint64_t hash);
// One presented frame.
void Frame();
// The "outer>inner" marker pass of the GPU thread's current packet ("-" outside markers; KYTY_PASS_DUMP).
[[nodiscard]] std::string CurrentKey();

} // namespace Libs::Graphics::PassCensus

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PASSCENSUS_H_
