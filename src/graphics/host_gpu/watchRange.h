#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_WATCHRANGE_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_WATCHRANGE_H_

#include <cstdint>
#include <string>

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
// The last SetCurrentOp (tracked with KYTY_WATCH_RANGE or KYTY_CP_FAULT_WRITERS=1).
[[nodiscard]] uint64_t CurrentOpHash() noexcept;
[[nodiscard]] const char* CurrentOpKind() noexcept;
[[nodiscard]] bool CpFaultWritersEnabled() noexcept;
// KYTY_WATCH_SHADERS=<hash>[,<hash>...] (hex): logs each dispatch of those compute shaders with its
// queue, groups and the GPU-op ordinal (KYTY_WATCH_RANGE_LINES lines), to see their order.
void NoteDispatch(uint64_t shader_hash, bool async_compute, uint32_t groups_x, uint32_t groups_y,
                  uint32_t groups_z, bool indirect);

// KYTY_WATCH_WRITERS=<addr>:<bytes> (diagnostic, hex or decimal): every write registration of a page
// in the range (GPU bindings, BDA settles, image bytes, CPU write faults) is kept in a per-page history
// (last 8: op ordinal, frame, kind, hash, groups, the first 16 user-data words of the op); consumers
// (KYTY_DRAW_TRACE_CB) print the history of the pages a draw reads.
[[nodiscard]] bool WritersEnabled() noexcept;
// Records the current op as a writer of [address, address + size) when it overlaps the range.
void NoteWriter(const char* what, uint64_t address, uint64_t size);
// The details of the op being recorded (GPU thread), kept with its writes.
void SetCurrentOpDetail(uint32_t frame, const uint32_t* user_data, uint32_t user_data_count, uint32_t groups_x,
                        uint32_t groups_y, uint32_t groups_z) noexcept;
// The writer history of the page holding `address` ("" when untracked).
[[nodiscard]] std::string DescribeWriters(uint64_t address);
// The first two user-data words of the newest writer of the page holding `address` (false: none).
bool NewestWriterUd(uint64_t address, uint32_t* ud0, uint32_t* ud1);

} // namespace Libs::Graphics::WatchRange

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_WATCHRANGE_H_ */
