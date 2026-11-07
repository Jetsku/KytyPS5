#pragma once

#include <atomic>
#include <cstdint>
#include <string>

// Console progress while the emulator waits for shader and pipeline compiles (on by default,
// KYTY_COMPILE_WATCH=0 turns it off). Builds and the threads blocked on them publish small
// lock-free records; a monitor thread checks them once a second and prints, rate-limited:
//   - "Compile wait: draw blocked for 2.0 s on ..." once per wait that lasts over 2 s (and a line
//     when it ends),
//   - "Compile wait: N builds in flight ..." every KYTY_COMPILE_WATCH_PERIOD_S (default 10) seconds
//     while a build has run for 5 s or more or a wait is blocked,
//   - "Frame stall: no flip for 10 s ..." with what the command processor waits on, when the guest
//     has flipped before and then stops flipping (again every 30 s while it lasts).
namespace CompileWatch {

enum class Kind : uint8_t {
	GraphicsPipeline,
	ComputePipeline,
	RtPipeline,     // a software ray-tracing kernel (background)
	SplitPipeline,  // a call-split program (KYTY_SHADER_CALLS_SPLIT, background)
	HugeCompute,    // KYTY_COMPUTE_BACKGROUND_WORDS
	Shader,         // a shader translation (guest code to SPIR-V and module)
	Count
};
[[nodiscard]] const char* KindName(Kind kind);
[[nodiscard]] bool        Enabled();

// A build running on this thread, from construction to destruction.
class Build {
public:
	Build(Kind kind, uint64_t hash, uint64_t hash2 = 0);
	~Build();
	Build(const Build&)            = delete;
	Build& operator=(const Build&) = delete;

private:
	int      m_slot     = -1;
	uint64_t m_begin_ns = 0; // KYTY_STALL_TRACE (PaceLog::NoteStall)
	uint64_t m_hash     = 0;
	Kind     m_kind     = Kind::Count;
};

// This thread (who: "draw", "dispatch", ...) cannot go on until a build of `hash` is done.
class Wait {
public:
	Wait(const char* who, Kind kind, uint64_t hash, uint64_t hash2 = 0);
	~Wait();
	Wait(const Wait&)            = delete;
	Wait& operator=(const Wait&) = delete;

private:
	int         m_slot     = -1;
	uint64_t    m_begin_ns = 0; // KYTY_STALL_TRACE (PaceLog::NoteStall)
	uint64_t    m_hash     = 0;
	Kind        m_kind     = Kind::Count;
	const char* m_who      = nullptr;
};

// Builds queued and not yet started, shown by name in the periodic line. Registered for its
// lifetime; Add/Sub are relaxed atomics.
class QueueCounter {
public:
	explicit QueueCounter(const char* name);
	~QueueCounter();
	QueueCounter(const QueueCounter&)            = delete;
	QueueCounter& operator=(const QueueCounter&) = delete;
	void          Add() { m_queued.fetch_add(1, std::memory_order_relaxed); }
	void          Sub() { m_queued.fetch_sub(1, std::memory_order_relaxed); }
	[[nodiscard]] int64_t     Queued() const { return m_queued.load(std::memory_order_relaxed); }
	[[nodiscard]] const char* Name() const { return m_name; }

private:
	const char*          m_name;
	std::atomic<int64_t> m_queued {0};
};

void NotePrecompile(uint64_t done, uint64_t total);
void NoteFlip();
// The renderer submitted work to Vulkan (the stall line says how long ago).
void NoteSubmission();
// A command-processor queue waits (WAIT_REG_MEM, WAIT_FLIP_DONE, ...); kind is a static string.
void NoteQueueWait(uint32_t queue, const char* kind, uint64_t address);
void ClearQueueWait(uint32_t queue);

// The monitor's decisions and lines, for the tests: one check at now_ms (0 = the clock), returns
// what it printed.
std::string CheckForTest(uint64_t now_ms);
void        ResetForTest();
uint64_t    NowMsForTest();

} // namespace CompileWatch
