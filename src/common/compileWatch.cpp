#include "common/compileWatch.h"

#include "common/kytyEvents.h"
#include "common/logging/log.h"
#include "common/paceLog.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace CompileWatch {
namespace {

constexpr uint64_t BlockedMs    = 2000;  // a wait is reported once after this long
constexpr uint64_t LongBuildMs  = 5000;  // builds this old keep the periodic line going
constexpr uint64_t StallMs      = 10000; // no flip for this long is a frame stall
constexpr uint64_t StallRepeat  = 30000; // the stall line again while it lasts
constexpr size_t   MaxBuilds    = 256;
constexpr size_t   MaxWaits     = 64;
constexpr size_t   MaxQueues    = 64;
constexpr size_t   MaxCounters  = 8;
constexpr size_t   ListedBuilds = 4;

const auto g_start = std::chrono::steady_clock::now();
uint64_t   ClockMs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - g_start).count());
}

uint64_t EnvU64(const char* name, uint64_t fallback) {
	const char* value = std::getenv(name);
	if (value == nullptr || value[0] == '\0') return fallback;
	return std::strtoull(value, nullptr, 10);
}

struct Slot {
	// 0 free, 1 being filled, 2 published. version changes on every claim, so the monitor can tell
	// a slot that was released and claimed again while it read it.
	std::atomic<uint32_t>    state {0};
	std::atomic<uint32_t>    version {0};
	std::atomic<uint8_t>     kind {0};
	std::atomic<const char*> who {nullptr};
	std::atomic<uint64_t>    hash {0}, hash2 {0}, start_ms {0};
	std::atomic<bool>        reported {false};
};

struct QueueWait {
	std::atomic<const char*> kind {nullptr};
	std::atomic<uint64_t>    address {0}, since_ms {0};
};

struct State {
	std::array<Slot, MaxBuilds>      builds;
	std::array<Slot, MaxWaits>       waits;
	std::array<QueueWait, MaxQueues> queues;
	std::atomic<uint64_t>            precompile_done {0}, precompile_total {0};
	std::atomic<uint64_t>            flips {0}, last_flip_ms {0};
	std::atomic<uint64_t>            submissions {0}, last_submission_ms {0};
	std::mutex                       counters_mutex;
	std::array<QueueCounter*, MaxCounters> counters {};
	// Monitor-only state.
	uint64_t last_periodic_ms = UINT64_MAX; // never
	uint64_t stall_flips      = UINT64_MAX; // flip count of the stall last reported
	uint64_t last_stall_ms    = 0;
	uint64_t period_ms        = 10000;
};
State& S() {
	static State state;
	return state;
}

bool ComputeEnabled() {
	const char* value = std::getenv("KYTY_COMPILE_WATCH");
	return value == nullptr || std::strcmp(value, "0") != 0;
}

void Print(const std::string& line) {
	Log::WriteToConsoleAndLog(line + "\n");
}

std::string Check(uint64_t now);
extern std::atomic<bool> g_no_monitor;

void MonitorThread() {
#ifdef _WIN32
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		try {
			(void)Check(ClockMs());
		} catch (...) {
			// Logging must never change emulation.
		}
	}
}

void EnsureMonitor() {
	static std::once_flag once;
	std::call_once(once, [] {
		S().period_ms = std::clamp<uint64_t>(EnvU64("KYTY_COMPILE_WATCH_PERIOD_S", 10), 2, 3600) * 1000;
		if (g_no_monitor.load()) return;
		try {
			std::thread(MonitorThread).detach();
		} catch (...) {
		}
	});
}

template <size_t N>
int Claim(std::array<Slot, N>& slots, Kind kind, const char* who, uint64_t hash, uint64_t hash2) {
	static thread_local size_t hint = 0;
	for (size_t n = 0; n < N; n++) {
		const size_t i        = (hint + n) % N;
		uint32_t     expected = 0;
		if (!slots[i].state.compare_exchange_strong(expected, 1, std::memory_order_acquire)) continue;
		auto& slot = slots[i];
		slot.version.fetch_add(1, std::memory_order_relaxed);
		slot.kind.store(static_cast<uint8_t>(kind), std::memory_order_relaxed);
		slot.who.store(who, std::memory_order_relaxed);
		slot.hash.store(hash, std::memory_order_relaxed);
		slot.hash2.store(hash2, std::memory_order_relaxed);
		slot.start_ms.store(ClockMs(), std::memory_order_relaxed);
		slot.reported.store(false, std::memory_order_relaxed);
		slot.state.store(2, std::memory_order_release);
		hint = i;
		return static_cast<int>(i);
	}
	return -1; // full: not shown
}

struct Snap {
	Kind        kind;
	const char* who;
	uint64_t    hash, hash2, start_ms;
	size_t      index;
};

template <size_t N>
std::vector<Snap> Collect(std::array<Slot, N>& slots) {
	std::vector<Snap> out;
	for (size_t i = 0; i < N; i++) {
		auto& slot = slots[i];
		if (slot.state.load(std::memory_order_acquire) != 2) continue;
		const auto version = slot.version.load(std::memory_order_relaxed);
		Snap       snap {static_cast<Kind>(slot.kind.load(std::memory_order_relaxed)),
                   slot.who.load(std::memory_order_relaxed), slot.hash.load(std::memory_order_relaxed),
                   slot.hash2.load(std::memory_order_relaxed), slot.start_ms.load(std::memory_order_relaxed), i};
		if (slot.state.load(std::memory_order_acquire) != 2 || slot.version.load(std::memory_order_relaxed) != version) {
			continue;
		}
		out.push_back(snap);
	}
	return out;
}

std::string Hashes(Kind kind, uint64_t hash, uint64_t hash2) {
	if (kind == Kind::GraphicsPipeline) {
		return hash2 != 0 ? fmt::format("VS 0x{:016x} PS 0x{:016x}", hash, hash2) : fmt::format("VS 0x{:016x}", hash);
	}
	return fmt::format("0x{:016x}", hash);
}

std::string Seconds(uint64_t ms) {
	return fmt::format("{:.1f} s", static_cast<double>(ms) / 1000.0);
}

// "rt-pipeline 0x... 312.0 s, ..." for the oldest builds.
std::string ListBuilds(std::vector<Snap> builds, uint64_t now, size_t limit) {
	std::sort(builds.begin(), builds.end(), [](const Snap& a, const Snap& b) { return a.start_ms < b.start_ms; });
	std::string text;
	for (size_t i = 0; i < builds.size() && i < limit; i++) {
		const auto& b = builds[i];
		text += fmt::format("{}{} {} {}", i == 0 ? "" : ", ", KindName(b.kind), Hashes(b.kind, b.hash, b.hash2),
		                    Seconds(now - std::min(now, b.start_ms)));
	}
	if (builds.size() > limit) text += fmt::format(", +{} more", builds.size() - limit);
	return text;
}

std::string QueuedText() {
	auto&            s = S();
	std::string      text;
	std::scoped_lock lock(s.counters_mutex);
	for (auto* counter: s.counters) {
		if (counter == nullptr) continue;
		const auto queued = counter->Queued();
		if (queued <= 0) continue;
		text += fmt::format("{}{} {} queued", text.empty() ? "" : ", ", queued, counter->Name());
	}
	return text;
}

std::string PrecompileText() {
	const auto total = S().precompile_total.load(std::memory_order_relaxed);
	const auto done  = S().precompile_done.load(std::memory_order_relaxed);
	if (total == 0 || done >= total) return {};
	return fmt::format("; precompile {}/{}", done, total);
}

std::string CpWaitsText(uint64_t now) {
	std::string text;
	for (size_t q = 0; q < MaxQueues; q++) {
		auto&       wait = S().queues[q];
		const char* kind = wait.kind.load(std::memory_order_acquire);
		if (kind == nullptr) continue;
		const auto since = wait.since_ms.load(std::memory_order_relaxed);
		text += fmt::format("{}queue {} {} 0x{:x} for {}", text.empty() ? "" : ", ", q, kind,
		                    wait.address.load(std::memory_order_relaxed), Seconds(now - std::min(now, since)));
	}
	return text;
}

// The build a wait is for (same kind family and hash), or null.
const Snap* BuildFor(const std::vector<Snap>& builds, const Snap& wait) {
	for (const auto& b: builds) {
		if (b.hash == wait.hash && (wait.hash2 == 0 || b.hash2 == wait.hash2)) return &b;
	}
	return nullptr;
}

std::string WaitText(const Snap& w, const std::vector<Snap>& builds, uint64_t now) {
	const auto* b = BuildFor(builds, w);
	return fmt::format("{} blocked for {} on {} {} ({})", w.who != nullptr ? w.who : "thread",
	                   Seconds(now - std::min(now, w.start_ms)), KindName(w.kind), Hashes(w.kind, w.hash, w.hash2),
	                   b != nullptr ? "building for " + Seconds(now - std::min(now, b->start_ms))
	                                : std::string("not started yet"));
}

std::atomic<bool> g_no_monitor {false}; // ResetForTest: the tests drive Check themselves
std::mutex        g_check_mutex;

std::string Check(uint64_t now) {
	std::scoped_lock check_lock(g_check_mutex);
	auto&       s = S();
	std::string printed;
	const auto  emit = [&](const std::string& line) {
        Print(line);
        printed += line + "\n";
	};
	const auto builds = Collect(s.builds);
	const auto waits  = Collect(s.waits);

	// 1. One line per wait that passes BlockedMs.
	bool blocked = false;
	for (const auto& w: waits) {
		if (now - std::min(now, w.start_ms) < BlockedMs) continue;
		blocked = true;
		auto& slot = s.waits[w.index];
		if (slot.reported.exchange(true, std::memory_order_relaxed)) continue;
		emit("Compile wait: " + WaitText(w, builds, now));
	}

	// 2. The periodic summary while a build is long or a wait is blocked.
	bool long_build = false;
	for (const auto& b: builds) long_build |= now - std::min(now, b.start_ms) >= LongBuildMs;
	if ((long_build || blocked) && (s.last_periodic_ms == UINT64_MAX || now - std::min(now, s.last_periodic_ms) >= s.period_ms)) {
		s.last_periodic_ms = now;
		std::string line   = fmt::format("Compile wait: {} build(s) in flight", builds.size());
		if (!builds.empty()) line += " (oldest: " + ListBuilds(builds, now, ListedBuilds) + ")";
		if (const auto queued = QueuedText(); !queued.empty()) line += "; " + queued;
		line += PrecompileText();
		std::string blocked_text;
		for (const auto& w: waits) {
			if (now - std::min(now, w.start_ms) < BlockedMs) continue;
			blocked_text += (blocked_text.empty() ? "" : "; ") + WaitText(w, builds, now);
		}
		if (!blocked_text.empty()) line += "; " + blocked_text;
		emit(line);
	}

	// 3. Frame stall: the guest flipped before and has not flipped for StallMs.
	const auto flips = s.flips.load(std::memory_order_relaxed);
	const auto last  = s.last_flip_ms.load(std::memory_order_relaxed);
	if (flips != 0 && now - std::min(now, last) >= StallMs &&
	    (s.stall_flips != flips || now - s.last_stall_ms >= StallRepeat)) {
		s.stall_flips   = flips;
		s.last_stall_ms = now;
		std::string line = fmt::format("Frame stall: no flip for {} (after flip {}).", Seconds(now - std::min(now, last)), flips);
		if (s.submissions.load(std::memory_order_relaxed) != 0) {
			const auto submitted = s.last_submission_ms.load(std::memory_order_relaxed);
			line += fmt::format(" Last Vulkan submission {} ago.", Seconds(now - std::min(now, submitted)));
		}
		const auto  cp   = CpWaitsText(now);
		line += cp.empty() ? " Command processor: no queue wait." : " Command processor waits: " + cp + ".";
		std::string blocked_text;
		for (const auto& w: waits) blocked_text += (blocked_text.empty() ? "" : "; ") + WaitText(w, builds, now);
		line += blocked_text.empty() ? " No draw or dispatch waits for a compile." : " Compile waits: " + blocked_text + ".";
		if (!builds.empty()) line += fmt::format(" {} build(s) in flight: {}.", builds.size(), ListBuilds(builds, now, ListedBuilds));
		if (const auto queued = QueuedText(); !queued.empty()) line += " " + queued + ".";
		emit(line);
	}
	return printed;
}

} // namespace

const char* KindName(Kind kind) {
	switch (kind) {
		case Kind::GraphicsPipeline: return "graphics pipeline";
		case Kind::ComputePipeline: return "compute pipeline";
		case Kind::RtPipeline: return "RT pipeline";
		case Kind::SplitPipeline: return "call-split pipeline";
		case Kind::HugeCompute: return "huge compute pipeline";
		case Kind::Shader: return "shader";
		case Kind::Count: break;
	}
	return "build";
}

bool Enabled() {
	static const bool enabled = ComputeEnabled();
	return enabled;
}

Build::Build(Kind kind, uint64_t hash, uint64_t hash2) {
	KytyEvents::NoteBuildStart();
	if (PaceLog::StallTraceEnabled()) {
		m_begin_ns = PaceLog::NowNs();
		m_hash     = hash;
		m_kind     = kind;
	}
	if (!Enabled()) return;
	EnsureMonitor();
	m_slot = Claim(S().builds, kind, nullptr, hash, hash2);
}

Build::~Build() {
	KytyEvents::NoteBuildEnd();
	if (m_begin_ns != 0) {
		PaceLog::NoteStall(m_begin_ns, PaceLog::NowNs(), PaceLog::StallBuild, m_hash, static_cast<int>(m_kind),
		                   KindName(m_kind));
	}
	if (m_slot >= 0) S().builds[static_cast<size_t>(m_slot)].state.store(0, std::memory_order_release);
}

Wait::Wait(const char* who, Kind kind, uint64_t hash, uint64_t hash2) {
	if (PaceLog::StallTraceEnabled()) {
		m_begin_ns = PaceLog::NowNs();
		m_hash     = hash;
		m_kind     = kind;
		m_who      = who;
	}
	if (!Enabled()) return;
	EnsureMonitor();
	m_slot = Claim(S().waits, kind, who, hash, hash2);
}

Wait::~Wait() {
	if (m_begin_ns != 0) {
		PaceLog::NoteStall(m_begin_ns, PaceLog::NowNs(), PaceLog::StallCompileWait, m_hash, static_cast<int>(m_kind),
		                   m_who);
	}
	if (m_slot < 0) return;
	auto& slot = S().waits[static_cast<size_t>(m_slot)];
	if (slot.reported.load(std::memory_order_relaxed)) {
		const auto now   = ClockMs();
		const auto start = slot.start_ms.load(std::memory_order_relaxed);
		const auto kind  = static_cast<Kind>(slot.kind.load(std::memory_order_relaxed));
		const char* who  = slot.who.load(std::memory_order_relaxed);
		Print(fmt::format("Compile wait: {} unblocked after {} ({} {})", who != nullptr ? who : "thread",
		                  Seconds(now - std::min(now, start)), KindName(kind),
		                  Hashes(kind, slot.hash.load(std::memory_order_relaxed), slot.hash2.load(std::memory_order_relaxed))));
	}
	slot.state.store(0, std::memory_order_release);
}

QueueCounter::QueueCounter(const char* name): m_name(name) {
	if (!Enabled()) return;
	auto&            s = S();
	std::scoped_lock lock(s.counters_mutex);
	for (auto*& counter: s.counters) {
		if (counter == nullptr) {
			counter = this;
			return;
		}
	}
}

QueueCounter::~QueueCounter() {
	if (!Enabled()) return;
	auto&            s = S();
	std::scoped_lock lock(s.counters_mutex);
	for (auto*& counter: s.counters) {
		if (counter == this) counter = nullptr;
	}
}

void NotePrecompile(uint64_t done, uint64_t total) {
	if (!Enabled()) return;
	S().precompile_total.store(total, std::memory_order_relaxed);
	S().precompile_done.store(done, std::memory_order_relaxed);
}

void NoteFlip() {
	if (!Enabled()) return;
	auto& s = S();
	s.last_flip_ms.store(ClockMs(), std::memory_order_relaxed);
	if (s.flips.fetch_add(1, std::memory_order_relaxed) == 0) EnsureMonitor();
}

void NoteSubmission() {
	if (!Enabled()) return;
	auto& s = S();
	s.last_submission_ms.store(ClockMs(), std::memory_order_relaxed);
	s.submissions.fetch_add(1, std::memory_order_relaxed);
}

void NoteQueueWait(uint32_t queue, const char* kind, uint64_t address) {
	if (!Enabled() || queue >= MaxQueues) return;
	auto& wait = S().queues[queue];
	if (wait.kind.load(std::memory_order_relaxed) == kind && wait.address.load(std::memory_order_relaxed) == address) {
		return;
	}
	wait.address.store(address, std::memory_order_relaxed);
	wait.since_ms.store(ClockMs(), std::memory_order_relaxed);
	wait.kind.store(kind, std::memory_order_release);
}

void ClearQueueWait(uint32_t queue) {
	if (!Enabled() || queue >= MaxQueues) return;
	auto& wait = S().queues[queue];
	if (wait.kind.load(std::memory_order_relaxed) != nullptr) wait.kind.store(nullptr, std::memory_order_release);
}

std::string CheckForTest(uint64_t now_ms) {
	return Check(now_ms != 0 ? now_ms : ClockMs());
}

void ResetForTest() {
	g_no_monitor.store(true);
	std::scoped_lock check_lock(g_check_mutex);
	auto& s = S();
	for (auto& slot: s.builds) slot.state.store(0);
	for (auto& slot: s.waits) slot.state.store(0);
	for (auto& q: s.queues) q.kind.store(nullptr);
	s.precompile_done  = 0;
	s.precompile_total = 0;
	s.flips            = 0;
	s.last_flip_ms     = 0;
	s.submissions      = 0;
	s.last_submission_ms = 0;
	s.last_periodic_ms = UINT64_MAX;
	s.stall_flips      = UINT64_MAX;
	s.last_stall_ms    = 0;
}

uint64_t NowMsForTest() {
	return ClockMs();
}

} // namespace CompileWatch
