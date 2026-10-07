#include "common/hangWatchdog.h"

#include "common/hostException.h"
#include "common/liveSwitch.h"
#include "kytyGitVersion.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <memory>
#include <mutex>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>
#else
#include <unistd.h>
#endif

namespace HangWatchdog {
std::atomic<bool> g_enabled {true};
// KYTY_HANG_WATCHDOG: "0" off, "auto" (value 2) on only for NVIDIA RTX 50 GPUs once the GPU is
// known (ResolveAutoForDevice), anything else or unset on.
constexpr int64_t        AutoMode = 2;
static std::atomic<bool> g_auto_on {false};
static int64_t ParseWatchdog(const char* value) {
	if (value != nullptr && (std::strcmp(value, "auto") == 0 || std::strcmp(value, "AUTO") == 0)) {
		return AutoMode;
	}
	return Live::ParseDefaultOn(value);
}
static Live::Switch g_switch("KYTY_HANG_WATCHDOG", ParseWatchdog, [](int64_t, int64_t value) {
	g_enabled.store(value == 1 || (value == AutoMode && g_auto_on.load()), std::memory_order_relaxed);
});
namespace {
constexpr size_t MaxThreads = 512, MaxDepth = 16, MaxQueues = 57, History = 32, MaxEvents = 1024;
struct Record {
	std::atomic<uint64_t>                version {0};
	std::atomic<const char*>             kind {nullptr};
	std::array<std::atomic<uint64_t>, 8> args {};
};
struct Thread {
	std::atomic<uint64_t>             tid {0}, guest {0}, queue {UINT32_MAX}, submission {0};
	std::array<std::atomic<char>, 64> name {};
	std::array<Record, MaxDepth>      scopes;
};
struct Queue {
	struct Packet {
		std::atomic_flag writing = ATOMIC_FLAG_INIT;
		Record           record;
	};
	Record                      state, wait;
	std::array<Packet, History> packets;
	// Queue 0 has concurrent parser and resolver publishers when the sequencer is enabled.
	std::atomic<uint64_t> next_packet {0};
};
struct Event {
	bool   used = false; // publisher mutex only; tombstones keep probing chains intact
	Record record;
	std::array<std::atomic<char>, 64> name {};
};
struct GuestModule {
	Record                            range;
	std::array<std::atomic<char>, 64> name {};
};
std::array<GuestModule, 64>         g_modules;
std::atomic<size_t>                 g_module_count {0};
Record                              g_fatal;
std::atomic_flag                    g_fatal_claimed = ATOMIC_FLAG_INIT;
std::array<std::atomic<char>, 1024> g_fatal_text {};
std::array<std::atomic<char>, 128>  g_fatal_file {};
std::array<Thread, MaxThreads>      g_threads;
std::array<Queue, MaxQueues>        g_queues;
std::array<Event, MaxEvents>        g_events;
constexpr size_t                    NativeHistory = 256, NativeSemaphores = 8;
struct NativeSubmit {
	std::atomic_flag                     writing = ATOMIC_FLAG_INIT;
	Record                               header;
	std::array<Record, NativeSemaphores> waits, signals;
};
std::array<NativeSubmit, NativeHistory> g_native;
std::atomic<uint64_t>                   g_native_sequence {0};
struct Timeline {
	std::atomic<uint64_t> semaphore {0}, current {0}, gpu {0}, dispatched {0};
};
std::array<Timeline, 64>    g_timelines;
std::atomic<uint32_t>       g_next_timeline {0};
std::atomic<size_t>         g_next_thread {0};
std::atomic<uint64_t>       g_overflow {0}, g_flips {0}, g_submissions {0}, g_time_ms {0};
thread_local Thread*        t_thread = nullptr;
thread_local size_t         t_depth  = 0;
std::mutex                  g_event_mutex; // ONLY event publishers use this, never the watchdog.
std::mutex                  g_sleep_mutex;
std::condition_variable_any g_sleep;
std::jthread                g_watchdog;
std::string                 g_directory;
std::atomic<bool>           g_terminal_requested {false};
std::atomic<uint32_t>       g_report_reason {0};
const auto                  g_start = std::chrono::steady_clock::now();

uint64_t NowMs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                 std::chrono::steady_clock::now() - g_start)
	                                 .count());
}
uint64_t Tid() {
#ifdef _WIN32
	return GetCurrentThreadId();
#else
	return std::hash<std::thread::id>()(std::this_thread::get_id());
#endif
}
uint64_t Pid() {
#ifdef _WIN32
	return GetCurrentProcessId();
#else
	return getpid();
#endif
}
Thread* CurrentThread() {
	if (t_thread) return t_thread;
	const auto index = g_next_thread.fetch_add(1, std::memory_order_relaxed);
	if (index >= MaxThreads) {
		++g_overflow;
		return nullptr;
	}
	t_thread = &g_threads[index];
	t_thread->tid.store(Tid(), std::memory_order_release);
	return t_thread;
}
template <size_t N>
void CopyName(std::array<std::atomic<char>, N>& dest, std::string_view name) {
	for (size_t i = 0; i < N; ++i)
		dest[i].store(i < std::min(name.size(), N - 1) ? name[i] : 0, std::memory_order_relaxed);
}
template <size_t N>
std::string ReadName(const std::array<std::atomic<char>, N>& source) {
	std::string name;
	for (const auto& c: source) {
		const char value = c.load(std::memory_order_relaxed);
		if (!value) break;
		name += value;
	}
	return name;
}
void Store(Record& record, const char* kind, const std::array<uint64_t, 8>& args) {
	const auto version = record.version.load(std::memory_order_relaxed);
	record.version.store(version + 1, std::memory_order_relaxed);
	std::atomic_thread_fence(std::memory_order_release);
	record.kind.store(kind, std::memory_order_relaxed);
	for (size_t i = 0; i < args.size(); ++i)
		record.args[i].store(args[i], std::memory_order_relaxed);
	record.version.store(version + 2, std::memory_order_release);
}
bool Read(const Record& record, const char*& kind, std::array<uint64_t, 8>& args) {
	// Bounded even if a writer is blocked in the middle of publishing. All fields are atomic:
	// checking an epoch never makes concurrent ordinary loads legal in the C++ memory model.
	for (int retry = 0; retry < 3; ++retry) {
		const auto before = record.version.load(std::memory_order_acquire);
		if (before & 1u) continue;
		kind = record.kind.load(std::memory_order_relaxed);
		for (size_t i = 0; i < args.size(); ++i)
			args[i] = record.args[i].load(std::memory_order_relaxed);
		std::atomic_thread_fence(std::memory_order_acquire);
		if (before == record.version.load(std::memory_order_relaxed)) return kind != nullptr;
	}
	return false;
}
std::string Snapshot() {
	std::string out = fmt::format(
	    "Kyty hang watchdog v1\nbuild={}\npid={} t_ms={} flips={} cp_submissions={} overflow={}\n",
	    KYTY_BUILD_LABEL, Pid(), NowMs(), g_flips.load(), g_submissions.load(), g_overflow.load());
	out += fmt::format("trigger={}\n", g_report_reason.load() == 2   ? "terminal-error"
	                                   : g_report_reason.load() == 1 ? "stopped-progress"
	                                                                 : "manual-snapshot");
	out += "Scope operands: address/resource, expected, observed, mask, aux, start_ms. Values are "
	       "last observations; no driver/guest reads are made here.\n";
	const char*             kind = nullptr;
	std::array<uint64_t, 8> a {};
	if (Read(g_fatal, kind, a))
		out += fmt::format("fatal-before-shutdown tid={} t_ms={} file='{}' line={} message='{}'\n",
		                   a[1], a[2], ReadName(g_fatal_file), a[0], ReadName(g_fatal_text));
	for (const auto& module: g_modules)
		if (Read(module.range, kind, a))
			out += fmt::format("guest-module name='{}' base=0x{:x} size=0x{:x}\n",
			                   ReadName(module.name), a[0], a[1]);
	for (const auto& thread: g_threads) {
		const auto tid = thread.tid.load(std::memory_order_acquire);
		if (!tid) continue;
		out += fmt::format("thread tid={} name='{}' guest=0x{:x} queue={} submission={}\n", tid,
		                   ReadName(thread.name), thread.guest.load(), thread.queue.load(),
		                   thread.submission.load());
		for (size_t depth = 0; depth < MaxDepth; ++depth) {
			if (Read(thread.scopes[depth], kind, a))
				out += fmt::format(
				    "  scope depth={} kind={} address=0x{:x} expected=0x{:x} observed=0x{:x} "
				    "mask=0x{:x} aux=0x{:x} start_ms={} queue={} submission={}\n",
				    depth, kind, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		}
	}
	for (size_t q = 0; q < MaxQueues; ++q) {
		if (Read(g_queues[q].state, kind, a))
			out += fmt::format("queue={} guest_queue={} state={} submission={} frame_fence={}\n", q,
			                   q ? q + 31 : 0, kind, a[0], a[1]);
		if (Read(g_queues[q].wait, kind, a)) {
			if (!std::strcmp(kind, "WAIT_FLIP_DONE"))
				out +=
				    fmt::format("  blocked=WAIT_FLIP_DONE handle={} buffer_index={} pending={}\n",
				                a[0], a[3], a[2]);
			else
				out += fmt::format("  blocked={} address=0x{:x} expected=0x{:x} observed=0x{:x} "
				                   "mask=0x{:x} compare={} size={}\n",
				                   kind, a[0], a[1], a[2], a[3], a[4], a[5]);
		}
		// Slots carry monotonic packet ids. Sorting a bounded copy never touches the CP ring.
		struct Packet {
			const char*             kind;
			std::array<uint64_t, 8> a;
		};
		std::array<Packet, History> packets {};
		size_t                      count = 0;
		for (const auto& p: g_queues[q].packets)
			if (Read(p.record, kind, a)) packets[count++] = {kind, a};
		std::sort(packets.begin(), packets.begin() + count,
		          [](const Packet& x, const Packet& y) { return x.a[7] < y.a[7]; });
		for (size_t i = 0; i < count; ++i) {
			const auto& p = packets[i];
			out +=
			    fmt::format("  packet={} type={} submission={} address=0x{:x} opcode=0x{:x} "
			                "args=0x{:x},0x{:x},0x{:x},0x{:x}\n",
			                p.a[7], p.kind, p.a[0], p.a[1], p.a[2], p.a[3], p.a[4], p.a[5], p.a[6]);
		}
	}
	for (const auto& event: g_events)
		if (Read(event.record, kind, a)) {
			if (static_cast<int16_t>(a[2]) == INT16_MIN)
				out += fmt::format(
				    "pending-label queue={} address=0x{:x} tick={} value=0x{:x} size={}\n", a[6],
				    a[0], a[1], a[4], a[5]);
			else
				out += fmt::format("equeue={} name='{}' ident=0x{:x} filter={} triggered={} "
				                   "data=0x{:x} udata=0x{:x}\n",
				                   a[0], ReadName(event.name), a[1], static_cast<int16_t>(a[2]),
				                   a[3], a[4], a[5]);
		}
	for (const auto& entry: g_native)
		if (Read(entry.header, kind, a)) {
			const auto sequence = a[0];
			out += fmt::format(
			    "native-submit sequence={} queue=0x{:x} tick={} first_command=0x{:x} waits={} "
			    "signals={} tid={} t_ms={} (about-to-call; not proof of return)\n",
			    a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
			for (const auto* operands: {&entry.waits, &entry.signals})
				for (const auto& operand: *operands) {
					if (Read(operand, kind, a) && a[3] == sequence)
						out += fmt::format("  {} semaphore=0x{:x} value={} stages=0x{:x}\n", kind,
						                   a[0], a[1], a[2]);
				}
		}
	for (const auto& timeline: g_timelines)
		if (const auto sem = timeline.semaphore.load())
			out += fmt::format("timeline semaphore=0x{:x} current={} gpu_last_query={} "
			                   "dispatched_last_observation={}\n",
			                   sem, timeline.current.load(), timeline.gpu.load(),
			                   timeline.dispatched.load());
	return out;
}

#ifdef _WIN32
// Copy each thread's stack while suspended, resume immediately, and only then unwind and print.
// No allocation, logging, loader or Vulkan calls while a target is suspended. Module offsets
// plus the matching executable/PDB let the next test resolve uninstrumented library waits too.
uint32_t Unwind(CONTEXT* context, uint64_t live, uint64_t copy, uint64_t size, uint64_t* frames) {
	uint32_t count = 0;
	Common::HostException::EnterProbe();
	__try {
		while (count < 48) {
			frames[count++]    = context->Rip;
			DWORD64 image_base = 0;
			auto*   fn         = RtlLookupFunctionEntry(context->Rip, &image_base, nullptr);
			if (!fn) {
				if (context->Rsp < copy || context->Rsp + 8 > copy + size) break;
				context->Rip = *reinterpret_cast<DWORD64*>(context->Rsp);
				context->Rsp += 8;
			} else {
				PVOID   data        = nullptr;
				DWORD64 establisher = 0;
				RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context->Rip, fn, context, &data,
				                 &establisher, nullptr);
			}
			for (auto* r: {&context->Rax, &context->Rbx, &context->Rcx, &context->Rdx,
			               &context->Rsi, &context->Rdi, &context->Rbp, &context->Rsp, &context->R8,
			               &context->R9, &context->R10, &context->R11, &context->R12, &context->R13,
			               &context->R14, &context->R15})
				if (*r >= live && *r < live + size) *r = *r - live + copy;
			if (!context->Rip || context->Rsp < copy || context->Rsp >= copy + size) break;
		}
	} __except (EXCEPTION_EXECUTE_HANDLER) {
	}
	Common::HostException::LeaveProbe();
	return count;
}
void WriteStacks(FILE* file) {
	const auto snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snapshot == INVALID_HANDLE_VALUE) return;
	struct Module {
		uint64_t base, size;
		char     name[MAX_PATH];
	};
	std::array<HMODULE, 512> handles {};
	std::array<Module, 512>  modules {};
	DWORD                    needed       = 0;
	size_t                   module_count = 0;
	if (K32EnumProcessModules(GetCurrentProcess(), handles.data(), sizeof(handles), &needed)) {
		for (size_t i = 0; i < std::min<size_t>(needed / sizeof(HMODULE), handles.size()); ++i) {
			MODULEINFO info {};
			if (K32GetModuleInformation(GetCurrentProcess(), handles[i], &info, sizeof(info))) {
				auto& m = modules[module_count++];
				m.base  = reinterpret_cast<uint64_t>(info.lpBaseOfDll);
				m.size  = info.SizeOfImage;
				K32GetModuleBaseNameA(GetCurrentProcess(), handles[i], m.name, MAX_PATH);
			}
		}
	}
	const char*             module_kind = nullptr;
	std::array<uint64_t, 8> range {};
	for (const auto& guest: g_modules)
		if (module_count < modules.size() && Read(guest.range, module_kind, range)) {
			auto& m         = modules[module_count++];
			m.base          = range[0];
			m.size          = range[1];
			const auto name = ReadName(guest.name);
			std::snprintf(m.name, sizeof(m.name), "%s", name.c_str());
		}
	for (size_t i = 0; i < module_count; ++i)
		std::fprintf(file, "native-module name='%s' base=0x%llx size=0x%llx\n", modules[i].name,
		             modules[i].base, modules[i].size);
	alignas(16) std::array<uint8_t, 65536> stack {};
	THREADENTRY32                          entry {};
	entry.dwSize = sizeof(entry);
	for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
		if (entry.th32OwnerProcessID != Pid() || entry.th32ThreadID == Tid()) continue;
		const auto thread =
		    OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE,
		               entry.th32ThreadID);
		if (!thread) continue;
		CONTEXT context {};
		context.ContextFlags  = CONTEXT_FULL;
		SIZE_T   copied       = 0;
		uint64_t live         = 0;
		bool     captured     = false;
		bool     have_context = false;
		if (SuspendThread(thread) != DWORD(-1)) {
			if (GetThreadContext(thread, &context)) {
				have_context = true;
				live         = context.Rsp;
				MEMORY_BASIC_INFORMATION memory {};
				if (VirtualQuery(reinterpret_cast<void*>(live), &memory, sizeof(memory))) {
					const auto size = std::min<uint64_t>(
					    stack.size(),
					    reinterpret_cast<uint64_t>(memory.BaseAddress) + memory.RegionSize - live);
					captured = ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<void*>(live),
					                             stack.data(), size, &copied) &&
					           copied != 0;
				}
			}
			ResumeThread(thread);
		}
		CloseHandle(thread);
		if (!have_context) {
			std::fprintf(file, "native tid=%lu context unavailable\n", entry.th32ThreadID);
			continue;
		}
		const auto rip = context.Rip;
		std::fprintf(file, "native tid=%lu rip=0x%llx rsp=0x%llx registers=%llx,%llx,%llx,%llx\n",
		             entry.th32ThreadID, rip, live, context.Rcx, context.Rdx, context.R8,
		             context.R9);
		std::fprintf(file,
		             "  registers rax=%llx rbx=%llx rcx=%llx rdx=%llx rsi=%llx rdi=%llx "
		             "rbp=%llx r8=%llx r9=%llx r10=%llx r11=%llx r12=%llx r13=%llx "
		             "r14=%llx r15=%llx eflags=%lx\n",
		             context.Rax, context.Rbx, context.Rcx, context.Rdx, context.Rsi,
		             context.Rdi, context.Rbp, context.R8, context.R9, context.R10,
		             context.R11, context.R12, context.R13, context.R14, context.R15,
		             context.EFlags);
		if (!captured) {
			std::fprintf(file, "  stack unavailable (register context retained)\n");
			continue;
		}
		const auto base = reinterpret_cast<uint64_t>(stack.data());
		for (auto* r:
		     {&context.Rax, &context.Rbx, &context.Rcx, &context.Rdx, &context.Rsi, &context.Rdi,
		      &context.Rbp, &context.Rsp, &context.R8, &context.R9, &context.R10, &context.R11,
		      &context.R12, &context.R13, &context.R14, &context.R15})
			if (*r >= live && *r < live + copied) *r = *r - live + base;
		uint64_t   frames[48] {};
		const auto count = Unwind(&context, live, base, copied, frames);
		for (uint32_t i = 0; i < count; ++i) {
			bool found = false;
			for (size_t m = 0; m < module_count; ++m)
				if (frames[i] >= modules[m].base && frames[i] < modules[m].base + modules[m].size) {
					std::fprintf(file, "  %s+0x%llx\n", modules[m].name,
					             frames[i] - modules[m].base);
					found = true;
					break;
				}
			if (!found) std::fprintf(file, "  guest/unknown 0x%llx\n", frames[i]);
		}
	}
	CloseHandle(snapshot);
}
#else
void WriteStacks(FILE*) {}
#endif
bool WriteSnapshot(const std::string& directory, bool stacks) {
	try {
		std::error_code error;
		const auto      directory_path = std::filesystem::u8path(directory);
		std::filesystem::create_directories(directory_path, error);
		if (error) return false;
		const auto path = directory_path / "watchdog.txt";
#ifdef _WIN32
		FILE* file = _wfopen(path.c_str(), L"wb");
#else
		FILE* file = std::fopen(path.c_str(), "wb");
#endif
		if (!file) return false;
		const std::unique_ptr<FILE, decltype(&std::fclose)> owner(file, &std::fclose);
		const auto                                          snapshot = Snapshot();
		if (std::fwrite(snapshot.data(), 1, snapshot.size(), file) != snapshot.size()) return false;
		if (std::fflush(file) != 0)
			return false; // Wait metadata survives even if native stack capture is interrupted.
		if (stacks) WriteStacks(file);
		return std::fflush(file) == 0 && !std::ferror(file);
	} catch (...) {
		// Diagnostic allocation/path failures must not terminate the emulated process.
		return false;
	}
}
} // namespace

bool StallDetector::Poll(uint64_t now, uint64_t f, uint64_t s, uint64_t timeout) {
	if (fired) return false;
	if (!armed) {
		armed         = f != 0 && s != 0;
		flips         = f;
		submissions   = s;
		idle_since_ms = now;
		return false;
	}
	if (f != flips || s != submissions) {
		flips         = f;
		submissions   = s;
		idle_since_ms = now;
		return false;
	}
	if (now >= idle_since_ms && now - idle_since_ms >= timeout) {
		fired = true;
		return true;
	}
	return false;
}
namespace {
std::string DefaultDirectory() {
	std::filesystem::path root = "_HangTrace";
#ifdef _WIN32
	wchar_t    exe[32768] {};
	const auto n = GetModuleFileNameW(nullptr, exe, std::size(exe));
	if (n && n < std::size(exe)) root = std::filesystem::path(exe).parent_path() / root;
#endif
	const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
	                       std::chrono::system_clock::now().time_since_epoch())
	                       .count();
	const auto path  = (root / fmt::format("watchdog-{}-pid{}", stamp, Pid())).u8string();
	return std::string(reinterpret_cast<const char*>(path.data()), path.size());
}
std::string g_trace_directory; // Initialize's argument, for a monitor that ResolveAutoForDevice starts
void StartMonitor(std::string_view trace_directory) {
	if (g_watchdog.joinable()) return;
	if (!trace_directory.empty())
		g_directory = trace_directory;
	else if (const auto* dir = std::getenv("KYTY_HANG_TRACE_DIR"); dir && *dir)
		g_directory = dir;
	else
		g_directory = DefaultDirectory();
	const auto* value = std::getenv("KYTY_HANG_WATCHDOG_MS");
	const auto  timeout =
	    value ? std::clamp<uint64_t>(std::strtoull(value, nullptr, 10), 1000, 600000) : 5000;
	g_watchdog = std::jthread([timeout](std::stop_token stop) {
		try {
			StallDetector    detector;
			std::unique_lock lock(g_sleep_mutex);
			for (;;) {
				const bool terminal = g_terminal_requested.exchange(false);
				if (stop.stop_requested() && !terminal) return;
				const auto now = NowMs();
				g_time_ms.store(now, std::memory_order_relaxed);
				if (!Enabled() && !terminal)
					detector = {};
				else if (terminal ||
				         detector.Poll(now, g_flips.load(), g_submissions.load(), timeout)) {
					g_report_reason.store(terminal ? 2 : 1);
					lock.unlock();
					if (!WriteSnapshot(g_directory, true)) {
						// An explicit unwritable trace path must not silently discard the
						// report.
						(void)WriteSnapshot(DefaultDirectory(), true);
					}
					return;
				}
				g_sleep.wait_for(lock, stop, std::chrono::seconds(1),
				                 [] { return g_terminal_requested.load(); });
			}
		} catch (...) {
			// A monitor failure must not change guest execution or terminate the process.
		}
	});
}
} // namespace
void Initialize(std::string_view trace_directory) {
	try {
		g_trace_directory = std::string(trace_directory);
		// Auto mode stays off until ResolveAutoForDevice knows the GPU.
		g_enabled.store(g_switch.Get() == 1, std::memory_order_relaxed);
		const auto* live = std::getenv("KYTY_LIVE_FILE");
		if (!Enabled() && (!live || !*live)) return;
		StartMonitor(trace_directory);
	} catch (...) {
		g_enabled.store(false, std::memory_order_relaxed);
	}
}
bool IsNvidiaBlackwell(uint32_t vendor_id, uint32_t device_id, std::string_view name) {
	if (vendor_id != 0x10deu) return false;
	// GB202 0x2b8x-0x2bbx, GB203 0x2c0x-0x2c3x, GB206/GB207 0x2d0x-0x2dbx, GB205 0x2f0x-0x2f3x.
	if (device_id >= 0x2b00u && device_id <= 0x2fffu) return true;
	// A board with a later device id: "RTX 5050" to "RTX 5090" in its name ("Quadro RTX 5000" is Turing).
	for (auto pos = name.find("RTX 50"); pos != std::string_view::npos; pos = name.find("RTX 50", pos + 1)) {
		if (pos + 8 > name.size()) break;
		const char tens = name[pos + 6], ones = name[pos + 7];
		const bool next_digit = pos + 8 < name.size() && name[pos + 8] >= '0' && name[pos + 8] <= '9';
		if (tens >= '5' && tens <= '9' && ones == '0' && !next_digit) return true;
	}
	return false;
}
AutoResult ResolveAutoForDevice(uint32_t vendor_id, uint32_t device_id, std::string_view name) {
	if (g_switch.Get() != AutoMode) return AutoResult::NotAuto;
	const bool on = IsNvidiaBlackwell(vendor_id, device_id, name);
	g_auto_on.store(on, std::memory_order_relaxed);
	if (!on) return AutoResult::Off;
	try {
		g_enabled.store(true, std::memory_order_relaxed);
		StartMonitor(g_trace_directory);
	} catch (...) {
		g_enabled.store(false, std::memory_order_relaxed);
		return AutoResult::Off;
	}
	return AutoResult::On;
}
void Shutdown() {
	if (g_watchdog.joinable()) {
		// Fatal emergency shutdown calls Profiler::Shutdown too. Preserve its error and active
		// scopes before stopping the monitor, even if there has not yet been a five-second stall.
		if (g_fatal_claimed.test(std::memory_order_acquire)) g_terminal_requested.store(true);
		g_watchdog.request_stop();
		g_sleep.notify_all();
		g_watchdog.join();
	}
}
void SetThreadName(std::string_view name) {
	if (Enabled())
		if (auto* t = CurrentThread()) CopyName(t->name, name);
}
void SetGuestThread(uint64_t guest, std::string_view name) {
	if (Enabled())
		if (auto* t = CurrentThread()) {
			t->guest.store(guest);
			if (!name.empty()) CopyName(t->name, name);
		}
}
void SetCpContext(uint32_t queue, uint64_t submission) {
	if (!Enabled()) return;
	if (auto* t = CurrentThread()) {
		t->queue.store(queue, std::memory_order_relaxed);
		t->submission.store(submission, std::memory_order_relaxed);
	}
}
uint32_t CurrentCpQueue() {
	return t_thread ? static_cast<uint32_t>(t_thread->queue.load(std::memory_order_relaxed))
	                : UINT32_MAX;
}
uint64_t CurrentCpSubmission() {
	return t_thread ? t_thread->submission.load(std::memory_order_relaxed) : 0;
}
void RegisterGuestCode(uint64_t base, uint64_t size, std::string_view name) {
	if (!Enabled() || !size) return;
	const auto index = g_module_count.fetch_add(1);
	if (index >= g_modules.size()) {
		++g_overflow;
		return;
	}
	CopyName(g_modules[index].name, name);
	Store(g_modules[index].range, "guest-module", {base, size});
}
void NoteFatal(std::string_view text, std::string_view file, uint32_t line) {
	if (!Enabled() || g_fatal_claimed.test_and_set(std::memory_order_acquire)) return;
	CopyName(g_fatal_text, text);
	const auto separator = file.find_last_of("/\\");
	CopyName(g_fatal_file, separator == std::string_view::npos ? file : file.substr(separator + 1));
	Store(g_fatal, "fatal", {line, Tid(), NowMs()});
}
void NoteFlip() {
	if (Enabled()) g_flips.fetch_add(1, std::memory_order_relaxed);
}
void NoteSubmission() {
	if (Enabled()) g_submissions.fetch_add(1, std::memory_order_relaxed);
}
Scope::Scope(const char* kind, uint64_t addr, uint64_t expected, uint64_t observed, uint64_t mask,
             uint64_t aux) {
	if (!Enabled()) return;
	auto* t = CurrentThread();
	if (!t || t_depth >= MaxDepth) {
		++g_overflow;
		return;
	}
	auto& record = t->scopes[t_depth++];
	m_record     = &record;
	Store(record, kind,
	      {addr, expected, observed, mask, aux, g_time_ms.load(std::memory_order_relaxed),
	       t->queue.load(), t->submission.load()});
}
Scope::~Scope() {
	if (m_record) {
		auto& r = *static_cast<Record*>(m_record);
		r.kind.store(nullptr, std::memory_order_release);
		--t_depth;
	}
}
void Scope::Observed(uint64_t value) {
	if (m_record) static_cast<Record*>(m_record)->args[2].store(value, std::memory_order_relaxed);
}
void NotePacket(uint32_t queue, uint64_t submission, uint64_t address, uint32_t opcode, uint64_t a,
                uint64_t b, uint64_t c, uint64_t d, const char* kind) {
	if (!Enabled() || queue >= MaxQueues) return;
	auto&      q  = g_queues[queue];
	const auto id = q.next_packet.fetch_add(1, std::memory_order_relaxed) + 1;
	auto&      p  = q.packets[id % History];
	if (p.writing.test_and_set(std::memory_order_acquire)) {
		++g_overflow;
		return; // A preempted diagnostic publisher must never park either CP thread.
	}
	if (p.record.args[7].load(std::memory_order_relaxed) >= id) {
		++g_overflow; // An older publisher resumed after this slot was already replaced.
	} else {
		Store(p.record, kind, {submission, address, opcode, a, b, c, d, id});
	}
	p.writing.clear(std::memory_order_release);
}
void NoteNativeSubmit(uint64_t queue, uint64_t tick, uint64_t command,
                      std::span<const SemaphoreValue> waits,
                      std::span<const SemaphoreValue> signals, uint32_t full_wait_count,
                      uint32_t full_signal_count) {
	if (!Enabled()) return;
	const auto sequence = g_native_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
	auto&      entry    = g_native[sequence % NativeHistory];
	if (entry.writing.test_and_set(std::memory_order_acquire)) {
		++g_overflow;
		return;
	}
	// Hide the slot while the complete bundle is replaced. No diagnostic publisher waits.
	entry.header.kind.store(nullptr, std::memory_order_release);
	for (size_t i = 0; i < NativeSemaphores; ++i) {
		if (i < waits.size())
			Store(entry.waits[i], "native-wait",
			      {waits[i].semaphore, waits[i].value, waits[i].stages, sequence});
		else
			entry.waits[i].kind.store(nullptr, std::memory_order_release);
		if (i < signals.size())
			Store(entry.signals[i], "native-signal",
			      {signals[i].semaphore, signals[i].value, signals[i].stages, sequence});
		else
			entry.signals[i].kind.store(nullptr, std::memory_order_release);
	}
	const auto wait_count   = std::max<size_t>(waits.size(), full_wait_count);
	const auto signal_count = std::max<size_t>(signals.size(), full_signal_count);
	if (wait_count > NativeSemaphores || signal_count > NativeSemaphores) ++g_overflow;
	Store(entry.header, "native-submit",
	      {sequence, queue, tick, command, wait_count, signal_count, Tid(), g_time_ms.load()});
	entry.writing.clear(std::memory_order_release);
}
void NoteQueue(uint32_t queue, uint64_t submission, const char* state, uint64_t fence) {
	if (Enabled() && queue < MaxQueues) Store(g_queues[queue].state, state, {submission, fence});
}
void NoteQueueWait(uint32_t queue, const char* kind, uint64_t address, uint64_t expected,
                   uint64_t observed, uint64_t mask, uint64_t compare, uint64_t size) {
	if (Enabled() && queue < MaxQueues)
		Store(g_queues[queue].wait, kind, {address, expected, observed, mask, compare, size});
}
void ClearQueueWait(uint32_t queue) {
	if (Enabled() && queue < MaxQueues) Store(g_queues[queue].wait, nullptr, {});
}
uint32_t RegisterTimeline(uint64_t semaphore) {
	const auto slot = g_next_timeline.fetch_add(1);
	if (slot < g_timelines.size())
		g_timelines[slot].semaphore.store(semaphore);
	else
		++g_overflow;
	return slot;
}
void UpdateTimeline(uint32_t slot, uint64_t current, uint64_t gpu, uint64_t dispatched) {
	if (!Enabled() || slot >= g_timelines.size()) return;
	auto& t = g_timelines[slot];
	t.current.store(current, std::memory_order_relaxed);
	t.gpu.store(gpu, std::memory_order_relaxed);
	t.dispatched.store(dispatched, std::memory_order_relaxed);
}
void NoteEvent(uint64_t queue, std::string_view name, uint64_t ident, int16_t filter,
               bool triggered, uint64_t data, uint64_t udata, bool deleted) {
	if (!Enabled()) return;
	std::lock_guard lock(g_event_mutex);
	Event*          target = nullptr;
	const auto      hash   = (queue >> 4) ^ ident ^ (ident >> 32) ^ static_cast<uint16_t>(filter);
	for (size_t probe = 0; probe < MaxEvents; ++probe) {
		auto&      e    = g_events[(hash + probe) % MaxEvents];
		const auto kind = e.record.kind.load(std::memory_order_relaxed);
		if (!kind && !target) target = &e;
		if (!e.used) break;
		if (kind && e.record.args[0].load() == queue && e.record.args[1].load() == ident &&
		    static_cast<int16_t>(e.record.args[2].load()) == filter) {
			target = &e;
			break;
		}
	}
	if (!target) {
		++g_overflow;
		return;
	}
	if (deleted && !target->record.kind.load()) return;
	if (!target->record.kind.load()) CopyName(target->name, name);
	target->used        = true;
	const auto cp_queue = t_thread ? t_thread->queue.load() : UINT32_MAX;
	Store(target->record, deleted ? nullptr : "event",
	      {queue, ident, static_cast<uint64_t>(filter), triggered, data, udata, cp_queue});
}
void DebugDelay(const char* site, uint64_t key) {
	static const auto* selected = std::getenv("KYTY_HANG_DELAY_SITE");
	if (!selected || std::strcmp(selected, site)) return;
	static const uint64_t hash = [] {
		const auto* p = std::getenv("KYTY_HANG_DELAY_KEY");
		return p ? std::strtoull(p, nullptr, 0) : 0;
	}();
	static const uint64_t after = [] {
		const auto* p = std::getenv("KYTY_HANG_DELAY_AFTER_MS");
		return p ? std::strtoull(p, nullptr, 10) : 30000;
	}();
	static const uint64_t delay = [] {
		const auto* p = std::getenv("KYTY_HANG_DELAY_MS");
		return p ? std::min<uint64_t>(std::strtoull(p, nullptr, 10), 30000) : 6000;
	}();
	if ((std::getenv("KYTY_HANG_DELAY_KEY") && key != hash) || NowMs() < after) return;
	static std::atomic<bool> used {false};
	if (used.exchange(true)) return;
	const char* kind = !std::strcmp(site, "queue")         ? "debug-delay-queue"
	                   : !std::strcmp(site, "compile")     ? "debug-delay-compile"
	                   : !std::strcmp(site, "label")       ? "debug-delay-label"
	                   : !std::strcmp(site, "readback")    ? "debug-delay-readback"
	                   : !std::strcmp(site, "eager-issue") ? "debug-delay-eager-issue"
	                                                       : "debug-delay-submit";
	Scope       scope(kind, key, delay);
	std::this_thread::sleep_for(std::chrono::milliseconds(delay));
}
std::string SnapshotForTest() {
	return Snapshot();
}
bool WriteSnapshotForTest(const std::string& directory) {
	return WriteSnapshot(directory, false);
}
} // namespace HangWatchdog
