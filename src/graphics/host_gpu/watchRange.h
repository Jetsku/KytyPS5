#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_WATCHRANGE_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_WATCHRANGE_H_

#include <cstdint>

// KYTY_WATCH_RANGE=<addr>:<bytes>[,<addr>:<bytes>...] (diagnostic, hex or decimal): every change
// of the tracker's CPU/GPU ownership of a page in a watched range, and every guest mapping that
// overlaps one, is logged (console and log, KYTY_WATCH_RANGE_LINES lines, default 400) with the
// GPU operation that ran last. Finds who writes guest memory (CPU fault, a GPU binding, a BDA
// write) and when it is mapped. Off unless set: one relaxed load per call.
namespace Libs::Graphics::WatchRange {

[[nodiscard]] bool Enabled() noexcept;
// Logs `what` when [address, address + size) overlaps a watched range.
void Note(const char* what, uint64_t address, uint64_t size);
// The GPU operation being recorded (GPU thread): kind ("CS", "draw") and the shader hash.
void SetCurrentOp(const char* kind, uint64_t shader_hash) noexcept;
// KYTY_WATCH_SHADERS=<hash>[,<hash>...] (hex): logs each dispatch of those compute shaders with its
// queue, groups and the GPU-op ordinal (KYTY_WATCH_RANGE_LINES lines), to see their order.
void NoteDispatch(uint64_t shader_hash, bool async_compute, uint32_t groups_x, uint32_t groups_y,
                  uint32_t groups_z, bool indirect);

} // namespace Libs::Graphics::WatchRange

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_WATCHRANGE_H_ */
