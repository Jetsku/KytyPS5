#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>

// Independent of tracing/profiling. No file I/O, stack walking or driver queries before a stall.
// Writers publish bounded, per-thread records; the watchdog never takes a guest or renderer lock.
namespace HangWatchdog {

extern std::atomic<bool>  g_enabled;
[[nodiscard]] inline bool Enabled() noexcept {
	return g_enabled.load(std::memory_order_relaxed);
}
void     Initialize(std::string_view trace_directory = {});
// KYTY_HANG_WATCHDOG=auto (the release preset): the watchdog stays off at Initialize and turns on
// here, once the GPU is selected, only for NVIDIA RTX 50 (Blackwell) GPUs, where the emulator froze
// at the title screen -> galaxy map step. The caller logs the result.
enum class AutoResult { NotAuto, On, Off };
[[nodiscard]] bool       IsNvidiaBlackwell(uint32_t vendor_id, uint32_t device_id, std::string_view name);
[[nodiscard]] AutoResult ResolveAutoForDevice(uint32_t vendor_id, uint32_t device_id, std::string_view name);
void     Shutdown();
void     SetThreadName(std::string_view name);
void     SetGuestThread(uint64_t guest_thread, std::string_view name);
void     SetCpContext(uint32_t queue, uint64_t submission);
uint32_t CurrentCpQueue();
uint64_t CurrentCpSubmission();
void     RegisterGuestCode(uint64_t base, uint64_t size, std::string_view name);
void     NoteFatal(std::string_view text, std::string_view file, uint32_t line);
void     NoteFlip();
void     NoteSubmission();
void     NotePacket(uint32_t queue, uint64_t submission, uint64_t address, uint32_t opcode,
                    uint64_t a = 0, uint64_t b = 0, uint64_t c = 0, uint64_t d = 0,
                    const char* kind = "pm4");
void     NoteQueue(uint32_t queue, uint64_t submission, const char* state, uint64_t fence = 0);
void     NoteQueueWait(uint32_t queue, const char* kind, uint64_t address, uint64_t expected,
                       uint64_t observed, uint64_t mask, uint64_t compare, uint64_t size);
void     ClearQueueWait(uint32_t queue);
// Timeline slots are never recycled, so a snapshot cannot dereference a destroyed renderer.
uint32_t RegisterTimeline(uint64_t semaphore);
void     UpdateTimeline(uint32_t slot, uint64_t current, uint64_t gpu, uint64_t dispatched);

struct SemaphoreValue {
	uint64_t semaphore = 0, value = 0, stages = 0;
};
void NoteNativeSubmit(uint64_t queue, uint64_t tick, uint64_t command,
                      std::span<const SemaphoreValue> waits,
                      std::span<const SemaphoreValue> signals, uint32_t full_wait_count = 0,
                      uint32_t full_signal_count = 0);

// Names must be static strings. address identifies the waited resource (guest address, host
// object or Vulkan handle); expected/observed/mask/aux retain the operation-specific operands.
// Scopes nest: a readback -> recorder drain -> atomic wait remains visible at all three levels.
class Scope {
public:
	explicit Scope(const char* kind, uint64_t address = 0, uint64_t expected = 0,
	               uint64_t observed = 0, uint64_t mask = 0, uint64_t aux = 0);
	~Scope();
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;
	void   Observed(uint64_t value);

private:
	void* m_record = nullptr;
};

// Copies event registrations while their owner already holds its own lock. This is called only
// on event add/change, not by the watchdog. Names are copied, never kept as dangling pointers.
void NoteEvent(uint64_t queue, std::string_view name, uint64_t ident, int16_t filter,
               bool triggered, uint64_t data, uint64_t udata, bool deleted = false);

// Injects a bounded delay at a named site (default off). A selective shader hash or minimum
// process age can target the loading transition. The scope identifies an intentional stall.
void DebugDelay(const char* site, uint64_t key = 0);

// Pure decision model and bounded snapshot entry points used by the regression tests.
struct StallDetector {
	uint64_t flips = 0, submissions = 0, idle_since_ms = 0;
	bool     armed = false, fired = false;
	bool Poll(uint64_t now_ms, uint64_t new_flips, uint64_t new_submissions, uint64_t timeout_ms);
};
std::string SnapshotForTest();
bool        WriteSnapshotForTest(const std::string& directory);

} // namespace HangWatchdog
