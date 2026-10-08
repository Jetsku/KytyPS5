#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_CPGAPS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_CPGAPS_H_

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <x86intrin.h>
#endif

#include <chrono>

// KYTY_CP_GAP_STATS=1 (diagnostic, default off): where the command processor thread's time goes
// between draw commits (KYTY_CP_COMMIT_STATS' "gap"), as exclusive time per cause.
//
// Every instrumented region on the GPU thread is a scope with a category; a scope's time minus the
// time of the scopes nested in it is its own (exclusive) time, so the categories add up to the
// thread's wall time; what no scope covers is the scheduler loop ("loop"). Regions:
//  - commit: DrawPrep::Engine::Commit / ExecuteIndirect's draw (what CommitStats calls a commit);
//  - headself / headwait: ReadyHead preparing the head itself or waiting for (and stealing next to)
//    the worker that holds it, by the window state the wait began with (barrier: the head is the
//    first draw after a draw-prep stop; start: the first after a run start; shallow: at most two
//    slots published; deep: more);
//  - ops by kind (CommandProcessor::ExecuteOp): draw ops outside their commit (serial fallbacks,
//    slot bookkeeping, indirect argument reads), dispatches split into program (GetComputeProgram:
//    SRT materialization), pipeline, bind (PrepareBindings + FindBuffers), bda (PrepareBda and BDA
//    write preparation), rebind (RebindImages/Buffers), emit (descriptors, barriers, the dispatch,
//    the BDA settle) and the rest; end-of-pipe/release-mem, event writes, WRITE_DATA, DMA_DATA,
//    waits, flips and other ops;
//  - the resolver's own loop (KYTY_CP_SEQ=1), its starvation (the sequencer has not parsed the
//    next op yet), service commands, other queues' slices, command-buffer flushes, the garbage
//    collector;
//  - nested anywhere: GPU waits (MasterSemaphore::Wait, by ScopedGpuWaitReason: drains = readbacks,
//    occlusion publication, other), CP recorder drains and ring-space waits, submit-dependency
//    waits, texture uploads, protection calls and write faults on this thread (FaultCost);
//  - idle: no submission or command at all; blocked: every queue suspended on a wait.
// The nested categories also report the part of them inside commits ("in commit").
// One console line every 10 s (KYTY_CP_GAP_STATS_INTERVAL_MS) at a guest frame boundary,
// "CpGaps <seconds>s: ..." in ms per frame.
// Costs two TSC reads per op and region (~0.2-0.4 ms per frame at 6,000 draws).
namespace Libs::Graphics::CpGaps {

enum class Cat : uint8_t {
	Commit,
	HeadSelf,
	HeadWait,        // a wait that began right after a stop, with a shallow or a deep window:
	HeadWaitStart,   // (DrawPrepCommitWaits{Barrier,Start,Shallow,Deep}); HeadWait = barrier
	HeadWaitShallow,
	HeadWaitDeep,
	DrawOp,
	DispatchProgram,
	ProgPrepare,     // nested program-lookup parts (any caller on this thread): PrepareProgram,
	ProgKey,         // ProgramCache key + source memo, materialization (SRT walk), permutation
	ProgMaterialize, // lookup
	ProgPermutation,
	DispatchPipeline,
	DispatchBind,
	DispatchBda,
	DispatchRebind,
	DispatchEmit,
	DispatchOp,
	EndOfPipe,
	EventWrite,
	WriteData,
	DmaData,
	WaitOp,
	FlipOp,
	OtherOp,
	Resolver,
	Starved,
	Service,
	Slice,
	Flush,
	Gc,
	GpuWaitDrain,
	GpuWaitOcclusion,
	GpuWaitOther,
	RecorderWait,
	SubmitWait,
	TextureUpload,
	Protect,
	Fault,
	Idle,
	Blocked,
	Count
};

constexpr size_t CatCount = static_cast<size_t>(Cat::Count);

[[nodiscard]] const char* CatName(Cat cat) noexcept;

// The accounting itself (one thread; the clock is passed in, so tests drive it).
class Ledger {
public:
	static constexpr uint32_t MaxDepth = 24;

	void Enter(Cat cat, uint64_t now) noexcept {
		m_calls[Index(cat)]++;
		if (m_depth >= MaxDepth) {
			m_depth++;
			m_overflows++;
			return;
		}
		auto& frame   = m_stack[m_depth++];
		frame.cat     = cat;
		frame.start   = now;
		frame.segment = now;
		frame.child   = 0;
		if (cat == Cat::Commit) {
			m_commit_depth++;
		}
	}
	// The innermost scope's category from `now` on (its time so far stays with the old one).
	void Switch(Cat cat, uint64_t now) noexcept {
		if (m_depth == 0 || m_depth > MaxDepth) {
			return;
		}
		auto& frame = m_stack[m_depth - 1];
		if (frame.cat == cat) {
			return;
		}
		Account(frame, now, InsideCommit(frame));
		if (frame.cat == Cat::Commit) {
			m_commit_depth--;
		}
		if (cat == Cat::Commit) {
			m_commit_depth++;
		}
		frame.cat     = cat;
		frame.segment = now;
		frame.child   = 0;
	}
	void Exit(uint64_t now) noexcept {
		if (m_depth == 0) {
			return;
		}
		if (m_depth > MaxDepth) {
			m_depth--;
			return;
		}
		auto& frame = m_stack[m_depth - 1];
		Account(frame, now, InsideCommit(frame));
		if (frame.cat == Cat::Commit) {
			m_commit_depth--;
		}
		const auto duration = now - frame.start;
		m_depth--;
		if (m_depth != 0 && m_depth <= MaxDepth) {
			m_stack[m_depth - 1].child += duration;
		}
	}
	// The innermost scope's whole time so far and from now on belongs to `cat` (a category known
	// only at the scope's end). Not for commits.
	void Relabel(Cat cat) noexcept {
		if (m_depth == 0 || m_depth > MaxDepth || cat == Cat::Commit) {
			return;
		}
		auto& frame = m_stack[m_depth - 1];
		if (frame.cat != Cat::Commit) {
			frame.cat = cat;
		}
	}
	// Time measured elsewhere (`ticks`) that belongs to `cat` and lies inside the current scope.
	void AddNested(Cat cat, uint64_t ticks) noexcept {
		m_calls[Index(cat)]++;
		m_ticks[Index(cat)] += ticks;
		if (m_commit_depth != 0) {
			m_in_commit[Index(cat)] += ticks;
		}
		if (m_depth != 0 && m_depth <= MaxDepth) {
			m_stack[m_depth - 1].child += ticks;
		}
	}

	[[nodiscard]] uint64_t Ticks(Cat cat) const noexcept { return m_ticks[Index(cat)]; }
	[[nodiscard]] uint64_t InCommit(Cat cat) const noexcept { return m_in_commit[Index(cat)]; }
	[[nodiscard]] uint64_t Calls(Cat cat) const noexcept { return m_calls[Index(cat)]; }
	[[nodiscard]] uint32_t Depth() const noexcept { return m_depth; }
	[[nodiscard]] uint64_t Overflows() const noexcept { return m_overflows; }
	// Totals of the closed (and closed parts of the open) scopes; open scopes keep running.
	void ResetTotals() noexcept {
		m_ticks     = {};
		m_in_commit = {};
		m_calls     = {};
		m_overflows = 0;
	}
	// Brings every open scope's time up to `now` into the totals (before a report) so that a long
	// open scope (an idle wait across the report) is not reported in a later interval at once.
	void Flush(uint64_t now) noexcept {
		const auto depth = m_depth < MaxDepth ? m_depth : MaxDepth;
		// Outer scopes first: an open inner scope's time so far is nested time of its parent.
		uint32_t commits = 0;
		for (uint32_t i = 0; i < depth; i++) {
			auto& frame = m_stack[i];
			if (i + 1 < depth) {
				frame.child += now - m_stack[i + 1].start;
			}
			Account(frame, now, commits != 0);
			commits += frame.cat == Cat::Commit ? 1u : 0u;
		}
		// Everything up to `now` is accounted: every open scope continues from here.
		for (uint32_t i = 0; i < depth; i++) {
			m_stack[i].start   = now;
			m_stack[i].segment = now;
			m_stack[i].child   = 0;
		}
	}
	// The scope chain's categories, innermost last (tests).
	[[nodiscard]] Cat At(uint32_t level) const noexcept { return m_stack[level].cat; }

private:
	struct Frame {
		Cat      cat     = Cat::Count;
		uint64_t start   = 0; // the scope's start (its parent's child time at exit)
		uint64_t segment = 0; // the current category's start
		uint64_t child   = 0; // nested time inside the current segment
	};
	static constexpr size_t Index(Cat cat) noexcept { return static_cast<size_t>(cat); }
	// The innermost scope `frame` lies inside a commit (an enclosing one, not itself).
	[[nodiscard]] bool InsideCommit(const Frame& frame) const noexcept {
		return m_commit_depth > (frame.cat == Cat::Commit ? 1u : 0u);
	}
	void Account(const Frame& frame, uint64_t now, bool inside_commit) noexcept {
		const auto elapsed = now - frame.segment;
		const auto own     = elapsed > frame.child ? elapsed - frame.child : 0;
		m_ticks[Index(frame.cat)] += own;
		if (inside_commit) {
			m_in_commit[Index(frame.cat)] += own;
		}
	}

	std::array<Frame, MaxDepth>      m_stack {};
	uint32_t                         m_depth        = 0;
	uint32_t                         m_commit_depth = 0;
	uint64_t                         m_overflows    = 0;
	std::array<uint64_t, CatCount>   m_ticks {};
	std::array<uint64_t, CatCount>   m_in_commit {};
	std::array<uint64_t, CatCount>   m_calls {};
};

[[nodiscard]] inline bool Enabled() noexcept {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_CP_GAP_STATS");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

[[nodiscard]] inline uint64_t Tsc() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
	return __rdtsc();
#else
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
#endif
}

// Counted events (per frame in the line): guest readbacks the command processor served by a
// drain (a write; bytes of the current recording; an address-writing shader in it; another
// reason; a read of this thread), and KYTY_READBACK_FLUSH_SIDE's flushes and the side copies
// they made possible.
enum class Event : uint8_t {
	DrainWrite,
	DrainCurrentWriter,
	DrainUnbounded,
	DrainOther,
	DrainGpuThread,
	SideFlushes,
	SideFlushCopies,
	Count
};
constexpr size_t EventCount = static_cast<size_t>(Event::Count);

// The GPU thread's ledger (null on every other thread, and while the diagnostic is off).
struct ThreadState {
	Ledger                           ledger;
	double                           ticks_per_ns = 1.0;
	std::array<uint64_t, EventCount> events {};
};
inline constinit thread_local ThreadState* t_state = nullptr;

class Scope {
public:
	explicit Scope(Cat cat) noexcept {
		if (Enabled()) [[unlikely]] {
			m_state = t_state;
			if (m_state != nullptr) {
				m_state->ledger.Enter(cat, Tsc());
			}
		}
	}
	~Scope() {
		if (m_state != nullptr) [[unlikely]] {
			m_state->ledger.Exit(Tsc());
		}
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	ThreadState* m_state = nullptr;
};

// The innermost scope continues as `cat` (dispatch phases).
inline void Phase(Cat cat) noexcept {
	if (Enabled()) [[unlikely]] {
		if (auto* state = t_state; state != nullptr) {
			state->ledger.Switch(cat, Tsc());
		}
	}
}

inline void Note(Event event) noexcept {
	if (Enabled()) [[unlikely]] {
		if (auto* state = t_state; state != nullptr) {
			state->events[static_cast<size_t>(event)]++;
		}
	}
}

// The innermost scope is `cat` (all of its time).
inline void Relabel(Cat cat) noexcept {
	if (Enabled()) [[unlikely]] {
		if (auto* state = t_state; state != nullptr) {
			state->ledger.Relabel(cat);
		}
	}
}

// Externally timed work of this thread (steady_clock nanoseconds) inside the current scope.
inline void NestedNs(Cat cat, uint64_t ns) noexcept {
	if (Enabled()) [[unlikely]] {
		if (auto* state = t_state; state != nullptr) {
			state->ledger.AddNested(cat, static_cast<uint64_t>(static_cast<double>(ns) *
			                                                   state->ticks_per_ns));
		}
	}
}

// GPU thread: starts the accounting on the calling thread (when enabled).
void AttachThread();
// GPU thread: a guest frame boundary (the processor reset after sceAgcSuspendPoint); prints the
// line every 10 s.
void NoteFrame();

} // namespace Libs::Graphics::CpGaps

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_CPGAPS_H_
