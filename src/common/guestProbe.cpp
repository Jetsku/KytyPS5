#include "common/guestProbe.h"

#include "common/logging/log.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <deque>
#include <vector>

#include <fmt/format.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tlhelp32.h>
#endif

namespace Common::GuestProbe {

#if defined(_WIN32)

namespace {

struct MemSpec {
	int      reg    = 0;
	uint64_t off    = 0;
	bool     deref  = false;
	uint64_t off2   = 0;
	std::vector<uint64_t> chain; // further '*<off>' levels after off2
	uint64_t length = 0x40;
	bool     condition = false; // '!' prefix: log only when these bytes are not all zero
	bool     write     = false; // '=' prefix: store the 8-byte value <len> at the address (experiments)
	bool     watch     = false; // '&' prefix: at the first matched hit, a hardware write watch (8 bytes)
	bool     equals    = false; // '?' prefix: a condition; the 8 bytes there must equal `value`
	uint64_t value     = 0;
};

// (*(u64*)at) + off2, then (*(u64*)that) + each chain offset; 0 when a level is unreadable.
uint64_t Deref(const MemSpec& m, uint64_t at);

struct Probe {
	uint64_t             va       = 0;
	uint64_t             address  = 0;
	uint32_t             max_hits = 8;
	uint32_t             hits     = 0;
	uint64_t             passes   = 0;
	uint8_t              original = 0;
	bool                 armed    = false;
	bool                 deferred = false; // '^' prefix: armed by ArmDeferred()
	// '~' prefix: never retires; keeps the last <hits> matched lines in a ring that ArmDeferred()
	// prints once (the hits leading up to the event that armed the deferred probes).
	bool                    ring = false;
	std::deque<std::string> ring_lines;
	std::vector<MemSpec> mem;
};

constexpr const char* kRegNames[16] = {"rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
                                       "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15"};

std::mutex         g_mutex;
std::vector<Probe> g_probes;
uint64_t           g_base = 0;
thread_local Probe* t_rearm = nullptr;

uint64_t RegValue(const CONTEXT& c, int reg) {
	switch (reg) {
		case 0: return c.Rax;
		case 1: return c.Rbx;
		case 2: return c.Rcx;
		case 3: return c.Rdx;
		case 4: return c.Rsi;
		case 5: return c.Rdi;
		case 6: return c.Rbp;
		case 7: return c.Rsp;
		case 8: return c.R8;
		case 9: return c.R9;
		case 10: return c.R10;
		case 11: return c.R11;
		case 12: return c.R12;
		case 13: return c.R13;
		case 14: return c.R14;
		default: return c.R15;
	}
}

bool Readable(uint64_t address, uint64_t size) {
	if (address < 0x10000 || size == 0) {
		return false;
	}
	uint64_t page = address & ~uint64_t {0xfff};
	while (page < address + size) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(page), &info, sizeof(info)) == 0 ||
		    info.State != MEM_COMMIT || (info.Protect & (PAGE_GUARD | PAGE_NOACCESS)) != 0 ||
		    (info.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_EXECUTE_READ |
		                     PAGE_EXECUTE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_WRITECOPY)) == 0) {
			return false;
		}
		page += 0x1000;
	}
	return true;
}

uint64_t Deref(const MemSpec& m, uint64_t at) {
	if (!Readable(at, 8)) {
		return 0;
	}
	at = *reinterpret_cast<const uint64_t*>(at) + m.off2;
	for (auto o: m.chain) {
		if (!Readable(at, 8)) {
			return 0;
		}
		at = *reinterpret_cast<const uint64_t*>(at) + o;
	}
	return at;
}

void WriteByte(uint64_t address, uint8_t value) {
	DWORD old = 0;
	VirtualProtect(reinterpret_cast<void*>(address), 1, PAGE_EXECUTE_READWRITE, &old);
	*reinterpret_cast<volatile uint8_t*>(address) = value;
	DWORD unused = 0;
	VirtualProtect(reinterpret_cast<void*>(address), 1, old, &unused);
	FlushInstructionCache(GetCurrentProcess(), reinterpret_cast<void*>(address), 1);
}

std::string Describe(const Probe& probe, const CONTEXT& c) {
	std::string line = fmt::format("GuestProbe: 0x{:x} hit {} tid {}:", probe.va, probe.hits,
	                               GetCurrentThreadId());
	for (int r = 0; r < 16; r++) {
		line += fmt::format(" {}={:x}", kRegNames[r], RegValue(c, r));
	}
	if (Readable(c.Rsp, 8)) {
		const uint64_t ret = *reinterpret_cast<const uint64_t*>(c.Rsp);
		line += fmt::format(" ret={:x}", ret >= g_base ? ret - g_base : ret);
	}
	// Low float of each xmm register (the game's float math keeps scalars there).
	line += " xmm";
	for (int r = 0; r < 16; r++) {
		float value = 0.0f;
		std::memcpy(&value, reinterpret_cast<const uint8_t*>(&c.Xmm0) + static_cast<size_t>(r) * 16u, 4);
		line += fmt::format("{}{:g}", r == 0 ? "=" : ",", value);
	}
	for (const auto& m: probe.mem) {
		if (m.watch) {
			line += fmt::format(" watch({}+{:x})", kRegNames[m.reg], m.off);
			continue;
		}
		if (m.write) {
			line += fmt::format(" write({}+{:x})={:x}", kRegNames[m.reg], m.off, m.length);
			continue;
		}
		uint64_t address = RegValue(c, m.reg) + m.off;
		std::string what  = fmt::format("{}+{:x}", kRegNames[m.reg], m.off);
		if (m.deref) {
			if (!Readable(address, 8)) {
				line += fmt::format(" [{}]=unreadable", what);
				continue;
			}
			address = Deref(m, address);
			what += fmt::format("*{:x}", m.off2);
			for (auto o: m.chain) {
				what += fmt::format("*{:x}", o);
			}
			if (address == 0) {
				line += fmt::format(" [{}]=unreadable", what);
				continue;
			}
		}
		line += fmt::format(" mem({}@{:x})=", what, address);
		if (!Readable(address, m.length)) {
			line += "unreadable";
			continue;
		}
		const auto* bytes = reinterpret_cast<const uint8_t*>(address);
		for (uint64_t i = 0; i < m.length; i++) {
			line += fmt::format("{:02x}", bytes[i]);
			if ((i & 7u) == 7u && i + 1 < m.length) {
				line += '.';
			}
		}
	}
	line += '\n';
	return line;
}

std::atomic<uint64_t> g_watch_address {0};
std::atomic<uint32_t> g_watch_logged {0};

// KYTY_LABEL_READ_WATCH (diagnostic): hardware read/write watches (DR0-DR3, 8 bytes) on up to four
// guest addresses; hits are counted per (watch, pc) and summarised by a background thread.
struct ReadWatchHit {
	std::atomic<uint64_t> key {0}; // (index + 1) << 56 | pc
	std::atomic<uint64_t> count {0};
	uint64_t              tid   = 0;
	uint64_t              value = 0;
	uint64_t              ret   = 0;
	uint64_t              printed = 0;
};
constexpr uint32_t    kReadWatchSlots = 512;
ReadWatchHit          g_read_hits[kReadWatchSlots];
std::atomic<bool>     g_read_watch_on {false};
uint64_t              g_read_watch[4] = {};
uint32_t              g_read_watch_count = 0;
std::atomic<uint64_t> g_read_watch_overflow {0};
uint64_t              g_size = 0;

void RecordReadWatch(uint32_t index, const CONTEXT& c) {
	const uint64_t key  = (uint64_t {index} + 1u) << 56u | (c.Rip & ((uint64_t {1} << 56u) - 1u));
	uint32_t       slot = static_cast<uint32_t>((c.Rip * 0x9E3779B97F4A7C15ull) >> 55u) % kReadWatchSlots;
	for (uint32_t probe = 0; probe < kReadWatchSlots; probe++, slot = (slot + 1) % kReadWatchSlots) {
		auto&    hit      = g_read_hits[slot];
		uint64_t expected = 0;
		if (hit.key.load(std::memory_order_acquire) == key ||
		    (hit.key.load(std::memory_order_relaxed) == 0 &&
		     hit.key.compare_exchange_strong(expected, key, std::memory_order_acq_rel))) {
			if (hit.count.fetch_add(1, std::memory_order_relaxed) == 0) {
				hit.tid   = GetCurrentThreadId();
				hit.value = *reinterpret_cast<const volatile uint64_t*>(g_read_watch[index]);
				uint64_t ret = 0;
				if (c.Rbp != 0 && Readable(c.Rbp + 8, 8)) {
					ret = *reinterpret_cast<const uint64_t*>(c.Rbp + 8);
				}
				hit.ret = ret;
			}
			return;
		}
		if (expected == key) {
			hit.count.fetch_add(1, std::memory_order_relaxed);
			return;
		}
	}
	g_read_watch_overflow.fetch_add(1, std::memory_order_relaxed);
}

LONG WINAPI Handler(PEXCEPTION_POINTERS e) {
	const auto code = e->ExceptionRecord->ExceptionCode;
	auto&      c    = *e->ContextRecord;
	if (code == EXCEPTION_SINGLE_STEP && g_read_watch_on.load(std::memory_order_relaxed) &&
	    (c.Dr6 & 0xfu) != 0) {
		for (uint32_t i = 0; i < g_read_watch_count; i++) {
			if ((c.Dr6 & (1u << i)) != 0) {
				RecordReadWatch(i, c);
			}
		}
		c.Dr6 &= ~uint64_t {0xf};
		if (t_rearm == nullptr) {
			return EXCEPTION_CONTINUE_EXECUTION;
		}
	}
	if (code == EXCEPTION_SINGLE_STEP && (c.Dr6 & 1u) != 0 && g_watch_address.load() != 0) {
		const auto watch = g_watch_address.load();
		c.Dr6 &= ~uint64_t {0xf};
		// Log each (pc, new value) pair once, so a per-frame writer does not use up the lines.
		const uint64_t watch_pc    = c.Rip >= g_base ? c.Rip - g_base : c.Rip;
		const uint64_t watch_value = Readable(watch, 8) ? *reinterpret_cast<const uint64_t*>(watch) : 0;
		bool           seen        = false;
		{
			static std::mutex                                 seen_mutex;
			static std::vector<std::pair<uint64_t, uint64_t>> seen_pairs;
			std::scoped_lock                                  seen_lock(seen_mutex);
			for (const auto& p: seen_pairs) {
				seen = seen || (p.first == watch_pc && p.second == watch_value);
			}
			if (!seen && seen_pairs.size() < 4096) {
				seen_pairs.emplace_back(watch_pc, watch_value);
			}
		}
		if (!seen && g_watch_logged.fetch_add(1) < 200) {
			std::string line = fmt::format("GuestWatch: t={} write to 0x{:x} = {:x} by pc {:x} (host {:x}) tid {}:",
			                               std::chrono::duration_cast<std::chrono::milliseconds>(
			                                   std::chrono::steady_clock::now().time_since_epoch())
			                                   .count(),
			                               watch, Readable(watch, 8) ? *reinterpret_cast<const uint64_t*>(watch) : 0,
			                               c.Rip >= g_base ? c.Rip - g_base : c.Rip, c.Rip,
			                               GetCurrentThreadId());
			for (int r = 0; r < 16; r++) {
				line += fmt::format(" {}={:x}", kRegNames[r], RegValue(c, r));
			}
			if (Readable(c.Rsp, 8)) {
				const uint64_t ret = *reinterpret_cast<const uint64_t*>(c.Rsp);
				line += fmt::format(" [rsp]={:x}", ret >= g_base ? ret - g_base : ret);
			}
			if (Readable(c.Rbp + 8, 8)) {
				const uint64_t ret = *reinterpret_cast<const uint64_t*>(c.Rbp + 8);
				line += fmt::format(" [rbp+8]={:x}", ret >= g_base ? ret - g_base : ret);
			}
			line += '\n';
			Log::WriteToConsoleAndLog(line);
		}
		if (t_rearm == nullptr) {
			return EXCEPTION_CONTINUE_EXECUTION;
		}
	}
	if (code == EXCEPTION_SINGLE_STEP) {
		Probe* probe = t_rearm;
		if (probe == nullptr) {
			return EXCEPTION_CONTINUE_SEARCH;
		}
		t_rearm = nullptr;
		std::scoped_lock lock(g_mutex);
		if (probe->ring || probe->hits < probe->max_hits) {
			WriteByte(probe->address, 0xcc);
		} else {
			probe->armed = false;
		}
		return EXCEPTION_CONTINUE_EXECUTION;
	}
	if (code != EXCEPTION_BREAKPOINT) {
		return EXCEPTION_CONTINUE_SEARCH;
	}
	const auto address = reinterpret_cast<uint64_t>(e->ExceptionRecord->ExceptionAddress);
	std::string line;
	uint64_t    new_watch = 0;
	{
		std::scoped_lock lock(g_mutex);
		Probe* probe = nullptr;
		for (auto& p: g_probes) {
			if (p.armed && p.address == address) {
				probe = &p;
			}
		}
		if (probe == nullptr) {
			// A thread that executed a probe's int3 just before another thread retired the probe
			// (its last hit restored the original byte and cleared `armed`) arrives here late. The
			// breakpoint is not this thread's to report: run the restored instruction.
			for (const auto& p: g_probes) {
				if (p.address == address && *reinterpret_cast<const volatile uint8_t*>(address) != 0xcc) {
					c.Rip = address;
					return EXCEPTION_CONTINUE_EXECUTION;
				}
			}
			return EXCEPTION_CONTINUE_SEARCH;
		}
		probe->passes++;
		bool matched = true;
		for (const auto& m: probe->mem) {
			if (m.equals) {
				uint64_t at = RegValue(c, m.reg) + m.off;
				if (m.deref) {
					at = Deref(m, at);
				}
				matched = matched && at != 0 && Readable(at, 8) &&
				          *reinterpret_cast<const uint64_t*>(at) == m.value;
				continue;
			}
			if (!m.condition) {
				continue;
			}
			uint64_t at = RegValue(c, m.reg) + m.off;
			if (m.deref) {
				at = Deref(m, at);
			}
			bool nonzero = false;
			if (at != 0 && Readable(at, m.length)) {
				for (uint64_t i = 0; i < m.length; i++) {
					nonzero = nonzero || reinterpret_cast<const uint8_t*>(at)[i] != 0;
				}
			}
			matched = matched && nonzero;
		}
		if (matched) {
			probe->hits++;
			if (probe->ring) {
				auto entry = Describe(*probe, c);
				if (entry.empty() || entry.back() != '\n') {
					entry += '\n';
				}
				probe->ring_lines.push_back(std::move(entry));
				if (probe->ring_lines.size() > probe->max_hits) {
					probe->ring_lines.pop_front();
				}
			} else if (probe->hits <= probe->max_hits) {
				line = Describe(*probe, c);
			}
			for (const auto& m: probe->mem) {
				if (m.watch && g_watch_address.load() == 0 && new_watch == 0) {
					uint64_t at = RegValue(c, m.reg) + m.off;
					if (m.deref) {
						at = Deref(m, at);
					}
					new_watch = at;
				}
				if (!m.write) {
					continue;
				}
				uint64_t at = RegValue(c, m.reg) + m.off;
				if (m.deref) {
					at = Deref(m, at);
				}
				if (at != 0 && Readable(at, 8)) {
					*reinterpret_cast<uint64_t*>(at) = m.length;
				}
			}
		}
		if (!probe->ring && (probe->passes % 1000u) == 0u) {
			line += fmt::format("GuestProbe: 0x{:x} passes {}, logged {}\n", probe->va,
			                    probe->passes, probe->hits);
		}
		WriteByte(address, probe->original);
		c.Rip = address;
		c.EFlags |= 0x100u;
		t_rearm = probe;
	}
	if (!line.empty()) {
		Log::WriteToConsoleAndLog(line);
	}
	if (new_watch != 0) {
		// The other threads through ArmWriteWatch, this one through its exception context.
		ArmWriteWatch(new_watch);
		if (g_watch_address.load() == new_watch) {
			c.Dr0 = new_watch;
			c.Dr7 = (c.Dr7 & ~uint64_t {0xf0003}) | 1u | (1u << 16u) | (2u << 18u);
		}
	}
	return EXCEPTION_CONTINUE_EXECUTION;
}

int ParseReg(std::string_view name) {
	for (int r = 0; r < 16; r++) {
		if (name == kRegNames[r]) {
			return r;
		}
	}
	return -1;
}

std::vector<std::string_view> Split(std::string_view text, char sep) {
	std::vector<std::string_view> parts;
	size_t                        start = 0;
	while (start <= text.size()) {
		const auto end = text.find(sep, start);
		const auto part = text.substr(start, end == std::string_view::npos ? text.npos : end - start);
		if (!part.empty()) {
			parts.push_back(part);
		}
		if (end == std::string_view::npos) {
			break;
		}
		start = end + 1;
	}
	return parts;
}

uint64_t Hex(std::string_view text) {
	return std::strtoull(std::string(text).c_str(), nullptr, 16);
}

std::once_flag g_handler_once;

void InstallHandler() {
	std::call_once(g_handler_once, [] { AddVectoredExceptionHandler(1, Handler); });
}

// Sets DR0-DR3 to the read watches on every thread of the process except the caller.
uint32_t ArmReadWatchesOnThreads() {
	uint64_t dr7 = 0;
	for (uint32_t i = 0; i < g_read_watch_count; i++) {
		// Li enable, RWi = 11 (read or write), LENi = 10 (8 bytes)
		dr7 |= (uint64_t {1} << (2u * i)) | (uint64_t {0xb} << (16u + 4u * i));
	}
	const DWORD self  = GetCurrentThreadId();
	uint32_t    armed = 0;
	HANDLE      snap  = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snap == INVALID_HANDLE_VALUE) {
		return 0;
	}
	THREADENTRY32 entry {};
	entry.dwSize = sizeof(entry);
	for (BOOL ok = Thread32First(snap, &entry); ok; ok = Thread32Next(snap, &entry)) {
		if (entry.th32OwnerProcessID != GetCurrentProcessId() || entry.th32ThreadID == self) {
			continue;
		}
		HANDLE thread = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_SUSPEND_RESUME,
		                           FALSE, entry.th32ThreadID);
		if (thread == nullptr) {
			continue;
		}
		if (SuspendThread(thread) != static_cast<DWORD>(-1)) {
			CONTEXT ctx {};
			ctx.ContextFlags = CONTEXT_DEBUG_REGISTERS;
			if (GetThreadContext(thread, &ctx) != 0) {
				ctx.Dr0 = g_read_watch[0];
				ctx.Dr1 = g_read_watch[1];
				ctx.Dr2 = g_read_watch[2];
				ctx.Dr3 = g_read_watch[3];
				ctx.Dr7 = (ctx.Dr7 & ~uint64_t {0xffff00ff}) | dr7;
				if (SetThreadContext(thread, &ctx) != 0) {
					armed++;
				}
			}
			ResumeThread(thread);
		}
		CloseHandle(thread);
	}
	CloseHandle(snap);
	return armed;
}

std::string ThreadName(uint64_t tid) {
	std::string name;
	HANDLE      thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, static_cast<DWORD>(tid));
	if (thread == nullptr) {
		return name;
	}
	PWSTR text = nullptr;
	if (SUCCEEDED(GetThreadDescription(thread, &text)) && text != nullptr) {
		for (const wchar_t* w = text; *w != 0; w++) {
			name += *w < 128 ? static_cast<char>(*w) : '?';
		}
		LocalFree(text);
	}
	CloseHandle(thread);
	return name;
}

uint64_t Relative(uint64_t pc) {
	return pc >= g_base && pc < g_base + g_size ? pc - g_base : pc;
}

// KYTY_LABEL_READ_WATCH=<hex addr>[,<hex addr>...] (at most 4): after
// KYTY_LABEL_READ_WATCH_DELAY_S seconds (default 30) hardware read/write watches on those guest
// addresses on every thread (re-armed every 10 s for new threads); every 10 s one
// "LabelWatch:" line per (address, pc) with new hits: count, thread, first value seen, [rbp+8].
void StartLabelReadWatch() {
	const char* env = std::getenv("KYTY_LABEL_READ_WATCH");
	if (env == nullptr || *env == '\0') {
		return;
	}
	for (auto part: Split(env, ',')) {
		if (g_read_watch_count < 4) {
			g_read_watch[g_read_watch_count++] = Hex(part) & ~uint64_t {7};
		}
	}
	const char* delay_env = std::getenv("KYTY_LABEL_READ_WATCH_DELAY_S");
	const auto  delay     = delay_env != nullptr ? std::strtoul(delay_env, nullptr, 10) : 30ul;
	InstallHandler();
	std::thread([delay] {
		Sleep(static_cast<DWORD>(delay * 1000u));
		g_read_watch_on.store(true);
		uint32_t round = 0;
		for (;;) {
			const auto armed = ArmReadWatchesOnThreads();
			if (round == 0) {
				std::string line = fmt::format("LabelWatch: armed on {} threads:", armed);
				for (uint32_t i = 0; i < g_read_watch_count; i++) {
					line += fmt::format(" dr{}=0x{:x}", i, g_read_watch[i]);
				}
				Log::WriteToConsoleAndLog(line + "\n");
			}
			round++;
			Sleep(10000);
			std::string text;
			for (auto& hit: g_read_hits) {
				const auto key   = hit.key.load(std::memory_order_acquire);
				const auto count = hit.count.load(std::memory_order_relaxed);
				if (key == 0 || count == hit.printed) {
					continue;
				}
				const uint32_t index = static_cast<uint32_t>(key >> 56u) - 1u;
				const uint64_t pc    = key & ((uint64_t {1} << 56u) - 1u);
				text += fmt::format("LabelWatch: t{} 0x{:x} pc {:x} n={} (+{}) tid {} '{}' first={:x} "
				                    "[rbp+8]={:x}\n",
				                    round * 10, g_read_watch[index], Relative(pc), count,
				                    count - hit.printed, hit.tid, ThreadName(hit.tid), hit.value,
				                    Relative(hit.ret));
				hit.printed = count;
			}
			if (const auto lost = g_read_watch_overflow.load(); lost != 0) {
				text += fmt::format("LabelWatch: {} hits not counted (table full)\n", lost);
			}
			if (!text.empty()) {
				Log::WriteToConsoleAndLog(text);
			}
		}
	}).detach();
}

} // namespace

void Arm(uint64_t base, uint64_t size) {
	if (g_base != 0) {
		return;
	}
	g_base          = base;
	g_size          = size;
	StartLabelReadWatch();
	const char* env = std::getenv("KYTY_GUEST_PROBE");
	if (env == nullptr || *env == '\0') {
		return;
	}
	std::scoped_lock lock(g_mutex);
	std::string text = env;
	if (text[0] == '@') {
		// @<file>: one probe per line (lines starting with '#' are comments).
		text.clear();
		if (std::FILE* file = std::fopen(env + 1, "rb"); file != nullptr) {
			bool comment = false;
			bool start   = true;
			for (int ch = std::fgetc(file); ch != EOF; ch = std::fgetc(file)) {
				if (ch == '\r' || ch == '\n') {
					text += ';';
					comment = false;
					start   = true;
					continue;
				}
				if (start && ch == '#') {
					comment = true;
				}
				start = false;
				if (!comment && ch != ' ' && ch != '\t') {
					text += static_cast<char>(ch);
				}
			}
			std::fclose(file);
		} else {
			Log::WriteToConsoleAndLog(fmt::format("GuestProbe: cannot open '{}'\n", env + 1));
		}
	}
	for (auto spec: Split(text, ';')) {
		const auto fields = Split(spec, ',');
		if (fields.empty()) {
			continue;
		}
		Probe probe;
		auto  head = fields[0];
		if (!head.empty() && head[0] == '^') {
			probe.deferred = true;
			head           = head.substr(1);
		}
		if (!head.empty() && head[0] == '~') {
			probe.ring = true;
			head       = head.substr(1);
		}
		if (const auto at = head.find('@'); at != head.npos) {
			probe.max_hits = static_cast<uint32_t>(std::strtoul(std::string(head.substr(at + 1)).c_str(), nullptr, 10));
			head           = head.substr(0, at);
		}
		probe.va = Hex(head);
		for (size_t i = 1; i < fields.size(); i++) {
			auto    f = fields[i];
			MemSpec m;
			if (!f.empty() && f[0] == '!') {
				m.condition = true;
				f           = f.substr(1);
			}
			if (!f.empty() && f[0] == '&') {
				m.watch = true;
				f       = f.substr(1);
			}
			if (!f.empty() && f[0] == '=') {
				m.write = true;
				f       = f.substr(1);
			}
			if (!f.empty() && f[0] == '?') {
				m.equals = true;
				m.length = 8;
				f        = f.substr(1);
				if (const auto eq = f.find('='); eq != f.npos) {
					m.value = Hex(f.substr(eq + 1));
					f       = f.substr(0, eq);
				}
			}
			if (const auto colon = f.find(':'); colon != f.npos) {
				m.length = Hex(f.substr(colon + 1));
				f        = f.substr(0, colon);
			}
			if (const auto star = f.find('*'); star != f.npos) {
				m.deref = true;
				auto rest = f.substr(star + 1);
				auto next = rest.find('*');
				m.off2    = Hex(rest.substr(0, next));
				while (next != rest.npos) {
					rest = rest.substr(next + 1);
					next = rest.find('*');
					m.chain.push_back(Hex(rest.substr(0, next)));
				}
				f       = f.substr(0, star);
			}
			const auto sign = f.find_first_of("+-");
			m.reg           = ParseReg(f.substr(0, sign));
			m.off           = sign == f.npos ? 0 : Hex(f.substr(sign + 1));
			if (sign != f.npos && f[sign] == '-') {
				m.off = uint64_t {0} - m.off;
			}
			if (m.reg < 0 || (!m.write && !m.watch && (m.length == 0 || m.length > 0x1000))) {
				Log::WriteToConsoleAndLog(fmt::format("GuestProbe: bad memory spec '{}'\n", fields[i]));
				continue;
			}
			probe.mem.push_back(m);
		}
		if (probe.va >= size || probe.max_hits == 0) {
			Log::WriteToConsoleAndLog(fmt::format("GuestProbe: bad probe '{}'\n", spec));
			continue;
		}
		probe.address = base + probe.va;
		g_probes.push_back(probe);
	}
	if (g_probes.empty()) {
		return;
	}
	InstallHandler();
	// Every probe's original byte first: a deferred probe at the address of an armed one would
	// otherwise record the armed probe's int3 and restore it when it retires.
	for (auto& p: g_probes) {
		p.original = *reinterpret_cast<const uint8_t*>(p.address);
	}
	for (auto& p: g_probes) {
		if (p.deferred) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "GuestProbe: deferred 0x{:x} ({} hits, {} dumps; armed by ArmDeferred)\n", p.va,
			    p.max_hits, p.mem.size()));
			continue;
		}
		WriteByte(p.address, 0xcc);
		p.armed = true;
		Log::WriteToConsoleAndLog(fmt::format(
		    "GuestProbe: armed 0x{:x} (host 0x{:x}, byte {:02x}, {} hits, {} dumps)\n", p.va,
		    p.address, p.original, p.max_hits, p.mem.size()));
	}
}

void ArmDeferred(const char* reason) {
	// KYTY_GUEST_WRITE_WATCH=<hex address> (diagnostic): a hardware write watch on those 8 bytes,
	// armed with the deferred probes (KYTY_GUEST_PROBE_ARM_LOAD_END).
	if (const char* watch = std::getenv("KYTY_GUEST_WRITE_WATCH"); watch != nullptr && *watch != '\0') {
		ArmWriteWatch(std::strtoull(watch, nullptr, 16) & ~uint64_t {7});
	}
	std::scoped_lock lock(g_mutex);
	for (auto& p: g_probes) {
		if (p.ring && p.armed) {
			std::string text = fmt::format("GuestProbe: ring 0x{:x}, last {} of {} hits ({}):\n", p.va,
			                               p.ring_lines.size(), p.hits, reason != nullptr ? reason : "");
			for (const auto& l: p.ring_lines) {
				text += l;
			}
			Log::WriteToConsoleAndLog(text);
			p.ring_lines.clear();
			p.max_hits = 0;
			p.ring     = false;
		}
	}
	for (auto& p: g_probes) {
		if (!p.deferred || p.armed || p.hits != 0) {
			continue;
		}
		WriteByte(p.address, 0xcc);
		p.armed = true;
		Log::WriteToConsoleAndLog(fmt::format("GuestProbe: armed deferred 0x{:x} ({})\n", p.va,
		                                      reason != nullptr ? reason : ""));
	}
}

void ArmWriteWatch(uint64_t address) {
	if (address == 0 || g_watch_address.exchange(address) != 0) {
		return;
	}
	InstallHandler();
	const DWORD self  = GetCurrentThreadId();
	uint32_t    armed = 0;
	HANDLE      snap  = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snap != INVALID_HANDLE_VALUE) {
		THREADENTRY32 entry {};
		entry.dwSize = sizeof(entry);
		for (BOOL ok = Thread32First(snap, &entry); ok; ok = Thread32Next(snap, &entry)) {
			if (entry.th32OwnerProcessID != GetCurrentProcessId() || entry.th32ThreadID == self) {
				continue;
			}
			HANDLE thread = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_SUSPEND_RESUME,
			                           FALSE, entry.th32ThreadID);
			if (thread == nullptr) {
				continue;
			}
			if (SuspendThread(thread) != static_cast<DWORD>(-1)) {
				CONTEXT ctx {};
				ctx.ContextFlags = CONTEXT_DEBUG_REGISTERS;
				if (GetThreadContext(thread, &ctx) != 0) {
					ctx.Dr0 = address;
					// L0, RW0 = 01 (write), LEN0 = 10 (8 bytes)
					ctx.Dr7 = (ctx.Dr7 & ~uint64_t {0xf0003}) | 1u | (1u << 16u) | (2u << 18u);
					if (SetThreadContext(thread, &ctx) != 0) {
						armed++;
					}
				}
				ResumeThread(thread);
			}
			CloseHandle(thread);
		}
		CloseHandle(snap);
	}
	Log::WriteToConsoleAndLog(fmt::format("GuestWatch: write watch on 0x{:x} armed on {} threads\n", address, armed));
}

#else

void Arm(uint64_t /*base*/, uint64_t /*size*/) {}
void ArmDeferred(const char* /*reason*/) {}
void ArmWriteWatch(uint64_t /*address*/) {}

#endif

} // namespace Common::GuestProbe
