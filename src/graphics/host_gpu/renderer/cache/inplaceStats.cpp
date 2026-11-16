#include "graphics/host_gpu/renderer/cache/inplaceStats.h"
#include "common/staticOnce.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <mutex>
#include <unordered_map>

#include <xxhash.h>

#include "kernel/memory.h"

namespace Libs::Graphics::InplaceStats {

namespace {

constexpr uint64_t kPageBits = 12;

struct Entry {
	uint32_t fault_frame  = 0; // last frame with a write fault (0: never)
	uint32_t upload_frame = 0;
	uint32_t stream_frame = 0;
	uint8_t  streak       = 0; // consecutive frames with a write fault, saturating
	uint64_t hash         = 0; // XXH3 of the page at its last upload (0: none)
};

struct PathTotals {
	uint64_t bytes = 0, unchanged = 0, no_fault = 0, hot = 0, hot_unchanged = 0;
};

struct KindTotals {
	uint64_t count = 0;
	uint64_t bytes = 0;
	uint64_t in_fault  = 0;
	uint64_t in_upload = 0;
	uint64_t in_stream = 0;
	uint64_t in_any    = 0;
};

struct Totals {
	uint64_t faults        = 0;
	uint64_t streak[3]     = {};
	uint64_t fault_upload  = 0;
	uint64_t fault_stream  = 0;
	uint64_t fault_neither = 0;
	uint64_t upload_bytes  = 0;
	uint64_t upload_copies = 0;
	uint64_t upload_pages  = 0;
	uint64_t stream_bytes  = 0;
	uint64_t stream_copies = 0;
	KindTotals cp[static_cast<int>(CpKind::Count)];
	PathTotals path[static_cast<int>(UploadPath::Count)];
};

// Plain counts are lock-free (lite mode runs in timing boots: a mutex in the fault handler
// perturbed Sky Garden's load on 6 cores); the page map takes the mutex (full mode only).
struct Fast {
	std::atomic<uint64_t> faults {0}, upload_bytes {0}, upload_copies {0}, stream_bytes {0}, stream_copies {0};
	std::atomic<uint64_t> cp_count[static_cast<int>(CpKind::Count)] {}, cp_bytes[static_cast<int>(CpKind::Count)] {};
};
Fast                                g_fast;
std::mutex                          g_mutex;
std::unordered_map<uint64_t, Entry> g_pages;
Totals                              g_totals;
std::atomic<uint32_t>               g_frame {1};
uint32_t                            g_period_frames = 0;
std::chrono::steady_clock::time_point g_period_start;
thread_local CpKind                 t_kind = CpKind::LabelOther;

bool Recent(uint32_t stamp, uint32_t frame) noexcept {
	return stamp != 0 && stamp + 1 >= frame;
}

template <typename Func>
void ForEachPage(uint64_t vaddr, uint64_t size, Func&& func) {
	if (size == 0) {
		return;
	}
	const uint64_t first = vaddr >> kPageBits;
	const uint64_t last  = (vaddr + size - 1) >> kPageBits;
	for (uint64_t page = first; page <= last; page++) {
		func(page);
	}
}

const char* KindName(int kind) {
	switch (static_cast<CpKind>(kind)) {
		case CpKind::WriteData: return "write-data";
		case CpKind::DmaFill: return "dma-fill";
		case CpKind::DmaCopy: return "dma-copy";
		case CpKind::ConstRam: return "const-ram";
		default: return "label/other";
	}
}

} // namespace

// 0 off, 1 full (per-page classification), 2 "lite" (byte and fault counts only, no page map).
int Mode() noexcept {
	static constexpr auto mode_init = [] {
		const auto* value = std::getenv("KYTY_INPLACE_STATS");
		if (value == nullptr) {
			return 0;
		}
		if (std::strcmp(value, "1") == 0) {
			return 1;
		}
		return std::strcmp(value, "lite") == 0 ? 2 : 0;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(mode_init())>> mode_once;
	const int& mode = mode_once.Get(mode_init);
	return mode;
}

bool Enabled() noexcept {
	return Mode() != 0;
}

bool Full() noexcept {
	return Mode() == 1;
}

void NoteFault(uint64_t vaddr) noexcept {
	if (!Enabled()) {
		return;
	}
	const uint32_t frame = g_frame.load(std::memory_order_relaxed);
	g_fast.faults.fetch_add(1, std::memory_order_relaxed);
	if (!Full()) {
		return;
	}
	std::scoped_lock lock {g_mutex};
	auto& entry = g_pages[vaddr >> kPageBits];
	if (entry.fault_frame != frame) {
		entry.streak      = (entry.fault_frame != 0 && entry.fault_frame + 1 == frame && entry.streak != UINT8_MAX)
		                        ? static_cast<uint8_t>(entry.streak + 1)
		                        : uint8_t {1};
		entry.fault_frame = frame;
	}
	g_totals.streak[entry.streak >= 3 ? 2 : entry.streak - 1]++;
	const bool uploaded = Recent(entry.upload_frame, frame);
	const bool streamed = Recent(entry.stream_frame, frame);
	if (uploaded) {
		g_totals.fault_upload++;
	}
	if (streamed) {
		g_totals.fault_stream++;
	}
	if (!uploaded && !streamed) {
		g_totals.fault_neither++;
	}
}

void NoteUpload(uint64_t vaddr, uint64_t size) noexcept {
	if (!Full()) {
		return;
	}
	const uint32_t frame = g_frame.load(std::memory_order_relaxed);
	std::scoped_lock lock {g_mutex};
	ForEachPage(vaddr, size, [&](uint64_t page) {
		g_pages[page].upload_frame = frame;
		g_totals.upload_pages++;
	});
}

void NoteUploadDetail(UploadPath path, bool hot, uint64_t vaddr, uint64_t size) noexcept {
	if (!Full() || size == 0) {
		return;
	}
	const uint32_t frame = g_frame.load(std::memory_order_relaxed);
	std::scoped_lock lock {g_mutex};
	auto& totals = g_totals.path[static_cast<int>(path)];
	const uint64_t end = vaddr + size;
	ForEachPage(vaddr, size, [&](uint64_t page) {
		const uint64_t lo    = std::max(vaddr, page << kPageBits);
		const uint64_t hi    = std::min(end, (page + 1) << kPageBits);
		const uint64_t bytes = hi - lo;
		auto&          entry = g_pages[page];
		const auto*    host  = Libs::LibKernel::Memory::GuestBackingAlias(page << kPageBits, uint64_t {1} << kPageBits);
		const uint64_t hash  = host != nullptr ? (XXH3_64bits(host, uint64_t {1} << kPageBits) | 1u) : 0;
		const bool     same  = hash != 0 && entry.hash == hash;
		entry.hash           = hash;
		totals.bytes += bytes;
		totals.unchanged += same ? bytes : 0;
		totals.no_fault += Recent(entry.fault_frame, frame) ? 0 : bytes;
		if (hot) {
			totals.hot += bytes;
			totals.hot_unchanged += same ? bytes : 0;
		}
	});
}

void NoteUploadBytes(uint64_t bytes, uint64_t copies) noexcept {
	if (!Enabled()) {
		return;
	}
	g_fast.upload_bytes.fetch_add(bytes, std::memory_order_relaxed);
	g_fast.upload_copies.fetch_add(copies, std::memory_order_relaxed);
}

void NoteStream(uint64_t vaddr, uint64_t size) noexcept {
	if (!Enabled()) {
		return;
	}
	const uint32_t frame = g_frame.load(std::memory_order_relaxed);
	g_fast.stream_bytes.fetch_add(size, std::memory_order_relaxed);
	g_fast.stream_copies.fetch_add(1, std::memory_order_relaxed);
	if (!Full()) {
		return;
	}
	std::scoped_lock lock {g_mutex};
	ForEachPage(vaddr, size, [&](uint64_t page) { g_pages[page].stream_frame = frame; });
}

void NoteCpWrite(CpKind kind, uint64_t vaddr, uint64_t size) noexcept {
	if (!Enabled() || size == 0 || vaddr == 0) {
		return;
	}
	const uint32_t frame = g_frame.load(std::memory_order_relaxed);
	g_fast.cp_count[static_cast<int>(kind)].fetch_add(1, std::memory_order_relaxed);
	g_fast.cp_bytes[static_cast<int>(kind)].fetch_add(size, std::memory_order_relaxed);
	if (!Full()) {
		return;
	}
	std::scoped_lock lock {g_mutex};
	auto& totals = g_totals.cp[static_cast<int>(kind)];
	bool in_fault = false, in_upload = false, in_stream = false;
	ForEachPage(vaddr, size, [&](uint64_t page) {
		const auto it = g_pages.find(page);
		if (it == g_pages.end()) {
			return;
		}
		in_fault |= Recent(it->second.fault_frame, frame);
		in_upload |= Recent(it->second.upload_frame, frame);
		in_stream |= Recent(it->second.stream_frame, frame);
	});
	totals.in_fault += in_fault ? 1 : 0;
	totals.in_upload += in_upload ? 1 : 0;
	totals.in_stream += in_stream ? 1 : 0;
	totals.in_any += (in_fault || in_upload || in_stream) ? 1 : 0;
}

void NoteCpWriteDefault(uint64_t vaddr, uint64_t size) noexcept {
	if (!Enabled()) {
		return;
	}
	NoteCpWrite(t_kind, vaddr, size);
}

ScopedCpKind::ScopedCpKind(CpKind kind) noexcept: m_previous(t_kind) {
	t_kind = kind;
}

ScopedCpKind::~ScopedCpKind() {
	t_kind = m_previous;
}

void AdvanceFrame() noexcept {
	if (!Enabled()) {
		return;
	}
	const uint32_t frame = g_frame.fetch_add(1, std::memory_order_relaxed) + 1;
	const auto     now   = std::chrono::steady_clock::now();
	std::scoped_lock lock {g_mutex};
	if (g_period_frames == 0 && g_period_start == std::chrono::steady_clock::time_point {}) {
		g_period_start = now;
	}
	g_period_frames++;
	// Entries older than a few frames classify nothing any more.
	if ((frame & 63u) == 0) {
		for (auto it = g_pages.begin(); it != g_pages.end();) {
			const auto& e      = it->second;
			const auto  newest = std::max(e.fault_frame, std::max(e.upload_frame, e.stream_frame));
			it                 = newest + 4 < frame ? g_pages.erase(it) : std::next(it);
		}
	}
	const double seconds = std::chrono::duration<double>(now - g_period_start).count();
	if (seconds < 10.0) {
		return;
	}
	g_totals.faults        = g_fast.faults.exchange(0, std::memory_order_relaxed);
	g_totals.upload_bytes  = g_fast.upload_bytes.exchange(0, std::memory_order_relaxed);
	g_totals.upload_copies = g_fast.upload_copies.exchange(0, std::memory_order_relaxed);
	g_totals.stream_bytes  = g_fast.stream_bytes.exchange(0, std::memory_order_relaxed);
	g_totals.stream_copies = g_fast.stream_copies.exchange(0, std::memory_order_relaxed);
	for (int k = 0; k < static_cast<int>(CpKind::Count); k++) {
		g_totals.cp[k].count = g_fast.cp_count[k].exchange(0, std::memory_order_relaxed);
		g_totals.cp[k].bytes = g_fast.cp_bytes[k].exchange(0, std::memory_order_relaxed);
	}
	const double f = g_period_frames;
	const auto&  t = g_totals;
	char line[4096];
	int  n = std::snprintf(
        line, sizeof(line),
        "InplaceStats %.0fs: %.0f frames | faults/f %.1f: streak1 %.1f streak2 %.1f streak3+ %.1f | uploaded "
        "this/prev frame %.1f streamed %.1f neither %.1f | upload %.2f MB/f %.0f copies/f %.0f pages/f | stream "
        "%.2f MB/f %.0f copies/f | pages tracked %zu |",
        seconds, f, t.faults / f, t.streak[0] / f, t.streak[1] / f, t.streak[2] / f, t.fault_upload / f,
        t.fault_stream / f, t.fault_neither / f, t.upload_bytes / f / 1e6, t.upload_copies / f, t.upload_pages / f,
        t.stream_bytes / f / 1e6, t.stream_copies / f, g_pages.size());
	for (int k = 0; k < static_cast<int>(CpKind::Count) && n > 0 && n < static_cast<int>(sizeof(line)); k++) {
		const auto& c = t.cp[k];
		n += std::snprintf(line + n, sizeof(line) - n,
		                   " cp %s %.1f/f %.0f B/f: on faulted %.1f uploaded %.1f streamed %.1f any %.1f;", KindName(k),
		                   c.count / f, c.bytes / f, c.in_fault / f, c.in_upload / f, c.in_stream / f, c.in_any / f);
	}
	if (Full()) {
		static const char* names[] = {"bda-sync", "written", "texel", "read", "bda-reach"};
		for (int k = 0; k < static_cast<int>(UploadPath::Count) && n > 0 && n < static_cast<int>(sizeof(line)); k++) {
			const auto& u = t.path[k];
			n += std::snprintf(line + n, sizeof(line) - n,
			                   " up %s %.2f MB/f: unchanged %.2f, pages without own fault %.2f, hot %.2f (unchanged %.2f);",
			                   names[k], u.bytes / f / 1e6, u.unchanged / f / 1e6, u.no_fault / f / 1e6, u.hot / f / 1e6,
			                   u.hot_unchanged / f / 1e6);
		}
	}
	std::printf("%s\n", line);
	std::fflush(stdout);
	g_totals        = Totals {};
	g_period_frames = 0;
	g_period_start  = now;
}

} // namespace Libs::Graphics::InplaceStats
