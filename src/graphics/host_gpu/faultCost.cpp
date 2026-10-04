#include "graphics/host_gpu/faultCost.h"

#include "common/common.h"
#include "common/liveSwitch.h"
#include "common/platform/uffdWriteWatch.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>
#include <psapi.h>
#undef min
#undef max
#else
#include <csignal>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/utsname.h>
#include <unistd.h>
#if defined(__linux__) && __has_include(<linux/userfaultfd.h>)
#include <linux/userfaultfd.h>
#include <sys/ioctl.h>
#define KYTY_FAULT_COST_UFFD 1
#endif
#endif

namespace Libs::Graphics::FaultCost {
namespace {

constexpr uint64_t PageBytes = 4096;

int64_t ParseCount(const char* value) {
	if (value == nullptr) {
		return 0;
	}
	char*      end    = nullptr;
	const auto parsed = std::strtoull(value, &end, 10);
	if (end == value) {
		return 0;
	}
	return static_cast<int64_t>(std::min<unsigned long long>(parsed, 1'000'000ull));
}

Live::Switch g_sim_fault_us("KYTY_SIM_FAULT_US", ParseCount);
Live::Switch g_sim_protect_us("KYTY_SIM_PROTECT_US", ParseCount);
Live::Switch g_sim_protect_page_ns("KYTY_SIM_PROTECT_PAGE_NS", ParseCount);
Live::Switch g_sim_unprotect_us("KYTY_SIM_UNPROTECT_US", ParseCount);
Live::Switch g_sim_unprotect_page_ns("KYTY_SIM_UNPROTECT_PAGE_NS", ParseCount);
Live::Switch g_sim_serial("KYTY_SIM_SERIAL", Live::ParseDefaultOn);

void Pause() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
	_mm_pause();
#endif
}

void SpinNs(uint64_t ns) noexcept {
	if (ns == 0) {
		return;
	}
	const auto end = NowNs() + ns;
	while (NowNs() < end) {
		Pause();
	}
}

double Median(std::vector<double> values) {
	if (values.empty()) {
		return 0;
	}
	std::sort(values.begin(), values.end());
	const auto middle = values.size() / 2;
	return values.size() % 2 != 0 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}

// ---- Live numbers ----------------------------------------------------------------------------

struct LiveCounters {
	std::atomic<uint64_t> bda_passes[4] {};
	std::atomic<uint64_t> bda_bytes[4] {};
	std::atomic<uint64_t> faults {0};
	std::atomic<uint64_t> fault_ns {0};
	std::atomic<uint64_t> protect_calls {0};
	std::atomic<uint64_t> protect_pages {0};
	std::atomic<uint64_t> protect_ns {0};
	std::atomic<uint64_t> unprotect_calls {0};
	std::atomic<uint64_t> unprotect_pages {0};
	std::atomic<uint64_t> unprotect_ns {0};
};
LiveCounters g_live;

struct LiveSnapshot {
	uint64_t bda_passes[4] {};
	uint64_t bda_bytes[4] {};
	uint64_t faults = 0, fault_ns = 0;
	uint64_t protect_calls = 0, protect_pages = 0, protect_ns = 0;
	uint64_t unprotect_calls = 0, unprotect_pages = 0, unprotect_ns = 0;
	uint64_t frames = 0, time_ns = 0;
};

LiveSnapshot TakeLive(uint64_t frames, uint64_t now) {
	LiveSnapshot s;
	for (int i = 0; i < 4; i++) {
		s.bda_passes[i] = g_live.bda_passes[i].load(std::memory_order_relaxed);
		s.bda_bytes[i]  = g_live.bda_bytes[i].load(std::memory_order_relaxed);
	}
	s.faults          = g_live.faults.load(std::memory_order_relaxed);
	s.fault_ns        = g_live.fault_ns.load(std::memory_order_relaxed);
	s.protect_calls   = g_live.protect_calls.load(std::memory_order_relaxed);
	s.protect_pages   = g_live.protect_pages.load(std::memory_order_relaxed);
	s.protect_ns      = g_live.protect_ns.load(std::memory_order_relaxed);
	s.unprotect_calls = g_live.unprotect_calls.load(std::memory_order_relaxed);
	s.unprotect_pages = g_live.unprotect_pages.load(std::memory_order_relaxed);
	s.unprotect_ns    = g_live.unprotect_ns.load(std::memory_order_relaxed);
	s.frames          = frames;
	s.time_ns         = now;
	return s;
}

std::atomic<uint32_t> g_frame {0};
LiveSnapshot          g_log_base;     // command processor thread only
LiveSnapshot          g_model_base;   // command processor thread only
uint64_t              g_log_period_ns = 0;
bool                  g_log_period_read = false;

// Model inputs (relaxed doubles stored as bit patterns are overkill: a torn read is impossible for
// a 64-bit aligned atomic<double>).
std::atomic<double> g_live_fault_us {-1.0};
std::atomic<double> g_live_protect_call_us {-1.0};
std::atomic<double> g_live_tighten_fixed_us {-1.0};

// SlowLevel(): the tracker runs on the command processor thread (and once at startup, before it).
SlowLevelTracker g_slow_tracker;
std::atomic<int> g_slow_level {0};

Benchmark g_benchmark;
Benchmark g_benchmark_uffd; // Linux with KYTY_UFFD_WP: the write-protection actually in use
std::once_flag g_benchmark_once;

// ---- Simulation lock ---------------------------------------------------------------------------

std::mutex g_sim_protect_mutex;
thread_local bool t_sim_protect_locked = false;

// ---- Fault map ---------------------------------------------------------------------------------

bool MapEnabledInit() {
	const auto* value = std::getenv("KYTY_FAULT_MAP");
	return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}
const bool g_map_enabled = MapEnabledInit();

uint64_t MapPeriodNs() {
	const auto* value   = std::getenv("KYTY_FAULT_MAP_SECONDS");
	auto        seconds = value != nullptr ? std::strtoull(value, nullptr, 10) : 10ull;
	if (seconds == 0) {
		seconds = 10;
	}
	return seconds * 1'000'000'000ull;
}

thread_local uint64_t t_fault_rip = 0;
thread_local bool     t_in_fault  = false;

enum ThreadClass : int { Cp = 0, Guest = 1, Host = 2, ThreadClasses = 3 };
const char* const kThreadNames[ThreadClasses] = {"cp", "guest", "host"};

std::atomic<ThreadDescriber> g_describer {nullptr};

int CurrentThreadClass(char* name, uint64_t size) noexcept {
	const auto describer = g_describer.load(std::memory_order_acquire);
	const int  thread    = describer != nullptr ? std::clamp(describer(name, size), 0, 2) : Host;
	if (name != nullptr && name[0] == '\0') {
		std::snprintf(name, size, "%s", kThreadNames[thread]);
	}
	return thread;
}

uint64_t g_exe_begin = 0;
uint64_t g_exe_end   = 0;

void InitExeRange() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	auto* base = reinterpret_cast<const uint8_t*>(GetModuleHandleW(nullptr));
	if (base == nullptr) {
		return;
	}
	const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
	const auto* nt  = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
	g_exe_begin     = reinterpret_cast<uint64_t>(base);
	g_exe_end       = g_exe_begin + nt->OptionalHeader.SizeOfImage;
#endif
}

struct BlockStats {
	uint64_t faults[ThreadClasses] {};
	uint64_t ns         = 0;
	uint64_t page_mask  = 0; // pages of the 64 KiB block that faulted (bit per 4 KiB)
	uint32_t frames     = 0; // frames with a fault here
	uint32_t last_frame = UINT32_MAX;
	uint64_t rip        = 0;
	char     thread[24] {};
};

struct ProtectStats {
	uint64_t calls = 0, pages = 0, ns = 0;
};

// [thread class][0 read-only, 1 no-access, 2 read-write][0 outside, 1 inside a fault]
using ProtectTable = std::array<std::array<std::array<ProtectStats, 2>, 3>, ThreadClasses>;

struct MapWindow {
	std::unordered_map<uint64_t, BlockStats> blocks; // key: address >> 16
	uint64_t faults[ThreadClasses] {};
	uint64_t fault_ns[ThreadClasses] {};
	uint64_t part_ns[ThreadClasses][static_cast<int>(Part::Count)] {};
	uint64_t read_faults = 0;
	// Frames since the same page last faulted: 0 (same frame), 1, 2, 3, 4-7, 8+, first time.
	uint64_t period[7] {};
	// Time since the same page last faulted: <20 us, 20-200 us, 0.2-2 ms, 2-20 ms, longer.
	uint64_t since[5] {};
	uint64_t found_dirty = 0;
	uint64_t emulator_rip = 0;
	ProtectTable protect {};
	uint32_t first_frame = 0;
};

std::mutex                                g_map_mutex;
MapWindow                                 g_map;
struct PageLast {
	uint32_t frame = 0;
	uint64_t ns    = 0;
};
std::unordered_map<uint64_t, PageLast>    g_page_last_frame; // page -> its last fault
uint64_t                                  g_map_last_dump = 0;
uint64_t                                  g_map_period_ns = 0;

void DumpMap(const MapWindow& window, uint32_t frame, double seconds) {
	const auto frames = std::max<uint32_t>(frame - window.first_frame, 1);
	const auto per    = [frames](uint64_t value) { return static_cast<double>(value) / frames; };
	uint64_t   total  = 0;
	for (int c = 0; c < ThreadClasses; c++) {
		total += window.faults[c];
	}
	std::printf("FaultMap %.1f s, %u frames (%.1f fps): %.1f faults/frame (cp %.1f, guest %.1f, host %.1f; "
	            "%.1f read), %.1f%% at emulator code\n",
	            seconds, frames, frames / std::max(seconds, 0.001), per(total), per(window.faults[Cp]),
	            per(window.faults[Guest]), per(window.faults[Host]), per(window.read_faults),
	            total != 0 ? 100.0 * static_cast<double>(window.emulator_rip) / static_cast<double>(total) : 0.0);
	for (int c = 0; c < ThreadClasses; c++) {
		if (window.faults[c] == 0) {
			continue;
		}
		const auto n = static_cast<double>(window.faults[c]);
		std::printf("FaultMap   %s faults: %.2f ms/frame in handler, %.1f us/fault (buffer %.1f, texture %.1f, "
		            "protection update %.1f)\n",
		            kThreadNames[c], per(window.fault_ns[c]) / 1e6, static_cast<double>(window.fault_ns[c]) / n / 1e3,
		            static_cast<double>(window.part_ns[c][0]) / n / 1e3,
		            static_cast<double>(window.part_ns[c][1]) / n / 1e3,
		            static_cast<double>(window.part_ns[c][2]) / n / 1e3);
	}
	std::printf("FaultMap   same page faulted again after: same frame %.1f, 1 frame %.1f, 2 %.1f, 3 %.1f, 4-7 %.1f, "
	            "8+ %.1f, first time %.1f (per frame)\n",
	            per(window.period[0]), per(window.period[1]), per(window.period[2]), per(window.period[3]),
	            per(window.period[4]), per(window.period[5]), per(window.period[6]));
	std::printf("FaultMap   ... after: <20 us %.1f, 20-200 us %.1f, 0.2-2 ms %.1f, 2-20 ms %.1f, longer %.1f; page "
	            "already CPU-dirty (another thread's fault in flight, or an image) %.1f (per frame)\n",
	            per(window.since[0]), per(window.since[1]), per(window.since[2]), per(window.since[3]),
	            per(window.since[4]), per(window.found_dirty));
	static const char* const kinds[3] = {"read-only", "no-access", "read-write"};
	for (int c = 0; c < ThreadClasses; c++) {
		for (int k = 0; k < 3; k++) {
			for (int f = 0; f < 2; f++) {
				const auto& s = window.protect[c][k][f];
				if (s.calls == 0) {
					continue;
				}
				std::printf("FaultMap   protect %s %s%s: %.1f calls/frame, %.1f pages/frame, %.3f ms/frame, "
				            "%.2f us/call\n",
				            kThreadNames[c], kinds[k], f != 0 ? " (in fault)" : "", per(s.calls), per(s.pages),
				            per(s.ns) / 1e6, static_cast<double>(s.ns) / static_cast<double>(s.calls) / 1e3);
			}
		}
	}
	// Top 4 MiB regions, then top 64 KiB blocks.
	std::unordered_map<uint64_t, std::array<uint64_t, 4>> regions; // faults by class + ns
	for (const auto& [key, block]: window.blocks) {
		auto& r = regions[key >> 6u];
		for (int c = 0; c < ThreadClasses; c++) {
			r[c] += block.faults[c];
		}
		r[3] += block.ns;
	}
	std::vector<std::pair<uint64_t, std::array<uint64_t, 4>>> region_list(regions.begin(), regions.end());
	std::sort(region_list.begin(), region_list.end(), [](const auto& a, const auto& b) {
		return a.second[0] + a.second[1] + a.second[2] > b.second[0] + b.second[1] + b.second[2];
	});
	const size_t region_count = std::min<size_t>(region_list.size(), 12);
	for (size_t i = 0; i < region_count; i++) {
		const auto& [key, r] = region_list[i];
		std::printf("FaultMap   region 0x%010" PRIx64 " (4 MiB): %.1f faults/frame (cp %.1f, guest %.1f, host %.1f), "
		            "%.1f us/fault\n",
		            key << 22u, per(r[0] + r[1] + r[2]), per(r[0]), per(r[1]), per(r[2]),
		            static_cast<double>(r[3]) / std::max<double>(static_cast<double>(r[0] + r[1] + r[2]), 1.0) / 1e3);
	}
	std::vector<std::pair<uint64_t, const BlockStats*>> blocks;
	blocks.reserve(window.blocks.size());
	for (const auto& [key, block]: window.blocks) {
		blocks.emplace_back(key, &block);
	}
	std::sort(blocks.begin(), blocks.end(), [](const auto& a, const auto& b) {
		const auto sum = [](const BlockStats& s) { return s.faults[0] + s.faults[1] + s.faults[2]; };
		return sum(*a.second) > sum(*b.second);
	});
	const size_t block_count = std::min<size_t>(blocks.size(), 24);
	for (size_t i = 0; i < block_count; i++) {
		const auto& [key, s] = blocks[i];
		const auto sum       = s->faults[0] + s->faults[1] + s->faults[2];
		const bool emulator  = s->rip >= g_exe_begin && s->rip < g_exe_end;
		std::printf("FaultMap   block 0x%010" PRIx64 " (64 KiB): %.2f faults/frame (cp %.2f, guest %.2f, host %.2f), "
		            "%d pages, in %u of %u frames, %.1f us/fault, last rip %s0x%" PRIx64 " thread '%s'\n",
		            key << 16u, per(sum), per(s->faults[0]), per(s->faults[1]), per(s->faults[2]),
		            __builtin_popcountll(s->page_mask), s->frames, frames,
		            static_cast<double>(s->ns) / std::max<double>(static_cast<double>(sum), 1.0) / 1e3,
		            emulator ? "kyty+" : "", emulator ? s->rip - g_exe_begin : s->rip, s->thread);
	}
	std::printf("FaultMap   %zu blocks, %zu regions with faults\n", window.blocks.size(), regions.size());
	std::fflush(stdout);
}

void MaybeDumpMap(uint32_t frame) {
	if (!g_map_enabled) {
		return;
	}
	const auto now = NowNs();
	MapWindow  window;
	double     seconds = 0;
	{
		std::scoped_lock lock(g_map_mutex);
		if (g_map_period_ns == 0) {
			g_map_period_ns = MapPeriodNs();
			g_map_last_dump = now;
			g_map.first_frame = frame;
			return;
		}
		if (now - g_map_last_dump < g_map_period_ns) {
			return;
		}
		seconds         = static_cast<double>(now - g_map_last_dump) / 1e9;
		g_map_last_dump = now;
		window          = std::move(g_map);
		g_map           = MapWindow {};
		g_map.first_frame = frame;
		// Forget pages that have not faulted for a while (keeps the map small).
		for (auto it = g_page_last_frame.begin(); it != g_page_last_frame.end();) {
			it = frame - it->second.frame > 64 ? g_page_last_frame.erase(it) : std::next(it);
		}
	}
	DumpMap(window, frame, seconds);
}

// ---- Platform facts ----------------------------------------------------------------------------

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

using NtQuerySystemInformationFn = LONG(WINAPI*)(ULONG, PVOID, ULONG, PULONG);
using RtlGetVersionFn            = LONG(WINAPI*)(PRTL_OSVERSIONINFOW);

DWORD ReadRegistryDword(const wchar_t* key, const wchar_t* value, DWORD fallback) {
	DWORD data = 0;
	DWORD size = sizeof(data);
	if (RegGetValueW(HKEY_LOCAL_MACHINE, key, value, RRF_RT_REG_DWORD, nullptr, &data, &size) != ERROR_SUCCESS) {
		return fallback;
	}
	return data;
}

std::string DescribeEntry(HMODULE ntdll, const char* name, bool syscall_stub) {
	const auto* code = reinterpret_cast<const uint8_t*>(GetProcAddress(ntdll, name));
	if (code == nullptr) {
		return std::string(name) + " missing";
	}
	bool hooked = false;
	if (syscall_stub) {
		// mov r10, rcx; mov eax, <service>
		hooked = !(code[0] == 0x4c && code[1] == 0x8b && code[2] == 0xd1 && code[3] == 0xb8);
	} else {
		hooked = code[0] == 0xe9 || (code[0] == 0xff && code[1] == 0x25) || code[0] == 0x68 ||
		         (code[0] == 0x48 && code[1] == 0xb8);
	}
	char text[96];
	std::snprintf(text, sizeof(text), "%s %s (%02x %02x %02x %02x)", name, hooked ? "HOOKED" : "ok", code[0],
	              code[1], code[2], code[3]);
	return text;
}

void LogPlatform() {
	auto* ntdll = GetModuleHandleW(L"ntdll.dll");
	char  os[64] = "Windows";
	if (ntdll != nullptr) {
		if (auto get_version = reinterpret_cast<RtlGetVersionFn>(GetProcAddress(ntdll, "RtlGetVersion"));
		    get_version != nullptr) {
			RTL_OSVERSIONINFOW info {};
			info.dwOSVersionInfoSize = sizeof(info);
			if (get_version(&info) == 0) {
				std::snprintf(os, sizeof(os), "Windows %lu.%lu.%lu", info.dwMajorVersion, info.dwMinorVersion,
				              info.dwBuildNumber);
			}
		}
	}
	int regs[4] {};
	__cpuid(regs, 1);
	const bool hypervisor = (static_cast<uint32_t>(regs[2]) & (1u << 31u)) != 0;
	char       vendor[13] {};
	if (hypervisor) {
		__cpuid(regs, 0x40000000);
		std::memcpy(vendor + 0, &regs[1], 4);
		std::memcpy(vendor + 4, &regs[2], 4);
		std::memcpy(vendor + 8, &regs[3], 4);
	}
	// SystemCodeIntegrityInformation (103): CODEINTEGRITY_OPTION_HVCI_KMCI_ENABLED (0x400) is memory
	// integrity running; 0x2000 is isolated user mode (VBS's secure kernel) running.
	uint32_t ci_options = 0;
	bool     ci_valid   = false;
	if (ntdll != nullptr) {
		if (auto query = reinterpret_cast<NtQuerySystemInformationFn>(GetProcAddress(ntdll, "NtQuerySystemInformation"));
		    query != nullptr) {
			struct {
				ULONG length;
				ULONG options;
			} info {sizeof(info), 0};
			ULONG returned = 0;
			ci_valid       = query(103, &info, sizeof(info), &returned) == 0;
			ci_options     = info.options;
		}
	}
	const auto vbs_config =
	    ReadRegistryDword(L"SYSTEM\\CurrentControlSet\\Control\\DeviceGuard", L"EnableVirtualizationBasedSecurity", 0xffffffffu);
	const auto hvci_config = ReadRegistryDword(
	    L"SYSTEM\\CurrentControlSet\\Control\\DeviceGuard\\Scenarios\\HypervisorEnforcedCodeIntegrity", L"Enabled",
	    0xffffffffu);
	LARGE_INTEGER frequency {};
	QueryPerformanceFrequency(&frequency);
	std::printf("Kyty platform: %s; hypervisor %s%s%s; memory integrity (HVCI) %s, VBS secure kernel %s "
	            "(code integrity options 0x%08x%s); registry: VBS %s, HVCI %s; QPC %.3f MHz\n",
	            os, hypervisor ? "present (" : "absent", vendor, hypervisor ? ")" : "",
	            !ci_valid ? "unknown" : (ci_options & 0x400u) != 0 ? "RUNNING" : "off",
	            !ci_valid ? "unknown" : (ci_options & 0x2000u) != 0 ? "running" : "off", ci_options,
	            ci_valid ? "" : ", query failed",
	            vbs_config == 0xffffffffu ? "unset" : vbs_config != 0 ? "enabled" : "disabled",
	            hvci_config == 0xffffffffu ? "unset" : hvci_config != 0 ? "enabled" : "disabled",
	            static_cast<double>(frequency.QuadPart) / 1e6);
	if (ntdll != nullptr) {
		std::printf("Kyty platform: ntdll entry points: %s; %s; %s\n",
		            DescribeEntry(ntdll, "NtProtectVirtualMemory", true).c_str(),
		            DescribeEntry(ntdll, "NtContinue", true).c_str(),
		            DescribeEntry(ntdll, "KiUserExceptionDispatcher", false).c_str());
	}
	// Modules loaded from outside Windows and the emulator's folder: antivirus hooks, overlays.
	wchar_t windows_dir[MAX_PATH] {};
	GetWindowsDirectoryW(windows_dir, MAX_PATH);
	wchar_t exe_path[MAX_PATH] {};
	GetModuleFileNameW(nullptr, exe_path, MAX_PATH);
	std::wstring exe_dir(exe_path);
	exe_dir = exe_dir.substr(0, exe_dir.find_last_of(L"\\/") + 1);
	const auto starts_with = [](const std::wstring& text, const std::wstring& prefix) {
		return text.size() >= prefix.size() && _wcsnicmp(text.c_str(), prefix.c_str(), prefix.size()) == 0;
	};
	HMODULE modules[1024];
	DWORD   needed = 0;
	std::string list;
	int         others = 0;
	if (K32EnumProcessModules(GetCurrentProcess(), modules, sizeof(modules), &needed) != 0) {
		const auto count = std::min<size_t>(needed / sizeof(HMODULE), 1024);
		for (size_t i = 0; i < count; i++) {
			wchar_t path[MAX_PATH] {};
			if (GetModuleFileNameW(modules[i], path, MAX_PATH) == 0) {
				continue;
			}
			const std::wstring name(path);
			if (starts_with(name, windows_dir) || starts_with(name, exe_dir)) {
				continue;
			}
			others++;
			if (list.size() < 400) {
				const auto file = name.substr(name.find_last_of(L"\\/") + 1);
				char       narrow[MAX_PATH] {};
				WideCharToMultiByte(CP_UTF8, 0, file.c_str(), -1, narrow, sizeof(narrow), nullptr, nullptr);
				list += list.empty() ? "" : ", ";
				list += narrow;
			}
		}
	}
	std::printf("Kyty platform: %d module(s) from outside Windows and the emulator folder%s%s\n", others,
	            others != 0 ? ": " : "", list.c_str());
	std::fflush(stdout);
}

#else

std::string ReadFirstLine(const char* path) {
	std::FILE* file = std::fopen(path, "r");
	if (file == nullptr) {
		return "?";
	}
	char line[256] {};
	if (std::fgets(line, sizeof(line), file) == nullptr) {
		line[0] = '\0';
	}
	std::fclose(file);
	std::string text(line);
	while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
		text.pop_back();
	}
	return text;
}

size_t CountMappings() {
	std::FILE* file = std::fopen("/proc/self/maps", "r");
	if (file == nullptr) {
		return 0;
	}
	size_t lines = 0;
	int    c     = 0;
	while ((c = std::fgetc(file)) != EOF) {
		lines += c == '\n' ? 1 : 0;
	}
	std::fclose(file);
	return lines;
}

#if defined(KYTY_FAULT_COST_UFFD)
#ifndef UFFD_USER_MODE_ONLY
#define UFFD_USER_MODE_ONLY 1
#endif
uint64_t ProbeUffdFeatures(int* error) {
	*error = 0;
	int fd = static_cast<int>(syscall(__NR_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY));
	if (fd < 0) {
		fd = static_cast<int>(syscall(__NR_userfaultfd, O_CLOEXEC | O_NONBLOCK));
	}
	if (fd < 0) {
		*error = errno;
		return 0;
	}
	uffdio_api api {};
	api.api      = UFFD_API;
	api.features = 0;
	uint64_t features = 0;
	if (ioctl(fd, UFFDIO_API, &api) == 0) {
		features = api.features;
	} else {
		*error = errno;
	}
	close(fd);
	return features;
}
#endif

void LogPlatform() {
	utsname name {};
	uname(&name);
	const auto max_map_count = ReadFirstLine("/proc/sys/vm/max_map_count");
	const auto unprivileged  = ReadFirstLine("/proc/sys/vm/unprivileged_userfaultfd");
	std::printf("Kyty platform: %s %s %s; vm.max_map_count %s; %zu mappings now; unprivileged_userfaultfd %s\n",
	            name.sysname, name.release, name.machine, max_map_count.c_str(), CountMappings(),
	            unprivileged.c_str());
#if defined(KYTY_FAULT_COST_UFFD)
	int        error    = 0;
	const auto features = ProbeUffdFeatures(&error);
	// Bits from linux/userfaultfd.h (spelled out: older headers lack the newer names).
	const auto has = [features](int bit) { return (features & (1ull << bit)) != 0; };
	std::printf("Kyty platform: userfaultfd %s (features 0x%" PRIx64 "): SIGBUS %s, write-protect on shmem %s, "
	            "WP_UNPOPULATED %s, WP_ASYNC %s\n",
	            features != 0 ? "available" : error != 0 ? std::strerror(error) : "unavailable", features,
	            has(7) ? "yes" : "no", has(12) ? "yes" : "no", has(13) ? "yes" : "no", has(15) ? "yes" : "no");
#else
	std::printf("Kyty platform: userfaultfd not compiled in\n");
#endif
	std::fflush(stdout);
}

#endif

// ---- Startup benchmark -------------------------------------------------------------------------

constexpr size_t BenchPages = 64;

std::atomic<uint64_t> g_bench_begin {0};
std::atomic<uint64_t> g_bench_end {0};
std::atomic<uint64_t> g_bench_handler_ns {0};
// Linux, KYTY_UFFD_WP: the second benchmark round uses userfaultfd write-protection.
std::atomic<bool>     g_bench_uffd {false};

bool BenchProtect(uint8_t* address, size_t bytes, bool writable) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	DWORD old = 0;
	return VirtualProtect(address, bytes, writable ? PAGE_READWRITE : PAGE_READONLY, &old) != 0;
#else
	if (g_bench_uffd.load(std::memory_order_relaxed)) {
		return Common::UffdWriteWatch::WriteProtect(reinterpret_cast<uint64_t>(address), bytes, !writable);
	}
	return mprotect(address, bytes, writable ? PROT_READ | PROT_WRITE : PROT_READ) == 0;
#endif
}

bool BenchResolve(uint64_t address) {
	const auto begin = g_bench_begin.load(std::memory_order_acquire);
	const auto end   = g_bench_end.load(std::memory_order_acquire);
	if (address < begin || address >= end) {
		return false;
	}
	const auto start = NowNs();
	const bool ok    = BenchProtect(reinterpret_cast<uint8_t*>(address & ~(PageBytes - 1)), PageBytes, true);
	g_bench_handler_ns.fetch_add(NowNs() - start, std::memory_order_relaxed);
	return ok;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

LONG CALLBACK BenchHandler(PEXCEPTION_POINTERS exception) {
	if (exception->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION ||
	    exception->ExceptionRecord->NumberParameters < 2) {
		return EXCEPTION_CONTINUE_SEARCH;
	}
	return BenchResolve(exception->ExceptionRecord->ExceptionInformation[1]) ? EXCEPTION_CONTINUE_EXECUTION
	                                                                       : EXCEPTION_CONTINUE_SEARCH;
}

struct BenchArea {
	HANDLE   mapping = nullptr;
	uint8_t* base    = nullptr;
	bool Create() {
		mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
		                             static_cast<DWORD>(BenchPages * PageBytes), nullptr);
		if (mapping == nullptr) {
			return false;
		}
		base = static_cast<uint8_t*>(MapViewOfFile(mapping, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, BenchPages * PageBytes));
		return base != nullptr;
	}
	~BenchArea() {
		if (base != nullptr) {
			UnmapViewOfFile(base);
		}
		if (mapping != nullptr) {
			CloseHandle(mapping);
		}
	}
};

struct BenchHandlerScope {
	void* handle = nullptr;
	BenchHandlerScope() { handle = AddVectoredExceptionHandler(1, BenchHandler); }
	~BenchHandlerScope() {
		if (handle != nullptr) {
			RemoveVectoredExceptionHandler(handle);
		}
	}
	[[nodiscard]] bool Ok() const { return handle != nullptr; }
};

#else

struct sigaction g_previous_segv {};
struct sigaction g_previous_bus {};

void ChainSignal(int signal_number, siginfo_t* info, void* context) {
	const auto& previous = signal_number == SIGBUS ? g_previous_bus : g_previous_segv;
	if ((previous.sa_flags & SA_SIGINFO) != 0 && previous.sa_sigaction != nullptr) {
		previous.sa_sigaction(signal_number, info, context);
		return;
	}
	if ((previous.sa_flags & SA_SIGINFO) == 0 && previous.sa_handler != SIG_DFL && previous.sa_handler != SIG_IGN) {
		previous.sa_handler(signal_number);
		return;
	}
	sigaction(signal_number, &previous, nullptr); // the retry takes the default action
}

void BenchSignal(int signal_number, siginfo_t* info, void* context) {
	if (BenchResolve(reinterpret_cast<uint64_t>(info->si_addr))) {
		return;
	}
	ChainSignal(signal_number, info, context);
}

struct BenchArea {
	int      fd   = -1;
	uint8_t* base = nullptr;
	bool Create() {
#if defined(__linux__)
		fd = static_cast<int>(syscall(SYS_memfd_create, "KytyFaultBench", 0));
#endif
		if (fd < 0) {
			auto* ptr = mmap(nullptr, BenchPages * PageBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
			base      = ptr == MAP_FAILED ? nullptr : static_cast<uint8_t*>(ptr);
			return base != nullptr;
		}
		if (ftruncate(fd, static_cast<off_t>(BenchPages * PageBytes)) != 0) {
			return false;
		}
		auto* ptr = mmap(nullptr, BenchPages * PageBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		base      = ptr == MAP_FAILED ? nullptr : static_cast<uint8_t*>(ptr);
		return base != nullptr;
	}
	~BenchArea() {
		if (base != nullptr) {
			munmap(base, BenchPages * PageBytes);
		}
		if (fd >= 0) {
			close(fd);
		}
	}
};

struct BenchHandlerScope {
	bool ok = false;
	BenchHandlerScope() {
		struct sigaction action {};
		action.sa_sigaction = BenchSignal;
		sigemptyset(&action.sa_mask);
		action.sa_flags = SA_SIGINFO | SA_ONSTACK;
		ok = sigaction(SIGSEGV, &action, &g_previous_segv) == 0 && sigaction(SIGBUS, &action, &g_previous_bus) == 0;
	}
	~BenchHandlerScope() {
		sigaction(SIGSEGV, &g_previous_segv, nullptr);
		sigaction(SIGBUS, &g_previous_bus, nullptr);
	}
	[[nodiscard]] bool Ok() const { return ok; }
};

#endif

Benchmark Measure(bool uffd) {
	Benchmark result;
	struct UffdRound {
		explicit UffdRound(bool on) { g_bench_uffd.store(on, std::memory_order_relaxed); }
		~UffdRound() { g_bench_uffd.store(false, std::memory_order_relaxed); }
		UffdRound(const UffdRound&)            = delete;
		UffdRound& operator=(const UffdRound&) = delete;
	};
	{
		std::vector<double> clock;
		for (int round = 0; round < 8; round++) {
			const auto start = NowNs();
			uint64_t   sink  = 0;
			for (int i = 0; i < 256; i++) {
				sink += NowNs();
			}
			clock.push_back(static_cast<double>(NowNs() - start) / 257.0);
			if (sink == 0) {
				std::printf("\n");
			}
		}
		result.clock_ns = Median(clock);
	}
	BenchArea area;
	if (!area.Create()) {
		return result;
	}
	for (size_t page = 0; page < BenchPages; page++) {
		area.base[page * PageBytes] = 1;
	}
	if (uffd && !Common::UffdWriteWatch::Register(reinterpret_cast<uint64_t>(area.base), BenchPages * PageBytes)) {
		return result;
	}
	const UffdRound round_mode(uffd);
	std::vector<double> ro1, rw1, ro64, rw64, faults, handler;
	for (int round = 0; round < 2; round++) {
		for (size_t page = 0; page < BenchPages; page++) {
			auto*      address = area.base + page * PageBytes;
			const auto start   = NowNs();
			if (!BenchProtect(address, PageBytes, false)) {
				return result;
			}
			ro1.push_back(static_cast<double>(NowNs() - start));
		}
		for (size_t page = 0; page < BenchPages; page++) {
			auto*      address = area.base + page * PageBytes;
			const auto start   = NowNs();
			if (!BenchProtect(address, PageBytes, true)) {
				return result;
			}
			rw1.push_back(static_cast<double>(NowNs() - start));
		}
	}
	for (int round = 0; round < 16; round++) {
		auto start = NowNs();
		if (!BenchProtect(area.base, BenchPages * PageBytes, false)) {
			return result;
		}
		ro64.push_back(static_cast<double>(NowNs() - start));
		start = NowNs();
		if (!BenchProtect(area.base, BenchPages * PageBytes, true)) {
			return result;
		}
		rw64.push_back(static_cast<double>(NowNs() - start));
	}
	{
		const BenchHandlerScope handler_scope;
		if (!handler_scope.Ok()) {
			return result;
		}
		g_bench_begin.store(reinterpret_cast<uint64_t>(area.base), std::memory_order_release);
		g_bench_end.store(reinterpret_cast<uint64_t>(area.base) + BenchPages * PageBytes, std::memory_order_release);
		for (int round = 0; round < 2; round++) {
			for (size_t page = 0; page < BenchPages; page++) {
				auto* address = area.base + page * PageBytes;
				if (!BenchProtect(address, PageBytes, false)) {
					g_bench_end.store(0, std::memory_order_release);
					return result;
				}
				const auto before = g_bench_handler_ns.load(std::memory_order_relaxed);
				const auto start  = NowNs();
				*reinterpret_cast<volatile uint32_t*>(address) = static_cast<uint32_t>(page);
				faults.push_back(static_cast<double>(NowNs() - start));
				handler.push_back(static_cast<double>(g_bench_handler_ns.load(std::memory_order_relaxed) - before));
			}
		}
		g_bench_end.store(0, std::memory_order_release);
		g_bench_begin.store(0, std::memory_order_release);
	}
	const auto ro1_us  = Median(ro1) / 1e3;
	const auto rw1_us  = Median(rw1) / 1e3;
	const auto ro64_us = Median(ro64) / 1e3;
	const auto rw64_us = Median(rw64) / 1e3;
	result.protect_call_us   = ro1_us;
	result.unprotect_call_us = rw1_us;
	result.protect_page_us   = std::max(0.0, (ro64_us - ro1_us) / static_cast<double>(BenchPages - 1));
	result.unprotect_page_us = std::max(0.0, (rw64_us - rw1_us) / static_cast<double>(BenchPages - 1));
	result.fault_us          = Median(faults) / 1e3;
	result.fault_handler_us  = Median(handler) / 1e3;
	result.valid             = true;
	return result;
}

} // namespace

uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
	        .count());
}

void RunStartupBenchmark() {
	std::call_once(g_benchmark_once, [] {
		InitExeRange();
		LogPlatform();
		const auto start = NowNs();
		g_benchmark      = Measure(false);
		const auto spent = static_cast<double>(NowNs() - start) / 1e6;
		if (!g_benchmark.valid) {
			std::printf("Kyty fault cost: startup benchmark failed (%.1f ms)\n", spent);
			std::fflush(stdout);
			return;
		}
		const auto& b = g_benchmark;
		// A Sky Garden frame of Astro Bot: ~1,500 write faults, ~100 tightening calls over ~5,000
		// pages, ~700 loosening calls (most inside the faults).
		const double frame_ms = (1500 * (b.fault_us + b.unprotect_page_us * 2) + 100 * b.protect_call_us +
		                         5000 * b.protect_page_us) /
		                        1e3;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		const char* call = "VirtualProtect";
#else
		const char* call = "mprotect";
#endif
		std::printf("Kyty fault cost: write fault round trip %.2f us (%.2f us of it the %s in the handler); %s "
		            "read-only %.2f us/call + %.3f us/page, read-write %.2f us/call + %.3f us/page; clock read %.0f ns; "
		            "a frame with 1,500 faults would spend ~%.1f ms on them (benchmark %.1f ms)\n",
		            b.fault_us, b.fault_handler_us, call, call, b.protect_call_us, b.protect_page_us,
		            b.unprotect_call_us, b.unprotect_page_us, b.clock_ns, frame_ms, spent);
		g_slow_tracker.Seed(b.protect_call_us, b.fault_us);
		const char* why = "startup benchmark";
#if defined(__linux__)
		if (!Common::UffdWriteWatch::Enabled()) {
			// mprotect convoys on the mmap lock (SlowLevel in faultCost.h).
			g_slow_tracker.Raise(2);
			why = "Linux mprotect tracking";
		}
#endif
		if (g_slow_tracker.Level() != 0) {
			g_slow_level.store(g_slow_tracker.Level(), std::memory_order_relaxed);
			std::printf("Kyty fault cost: write tracking is slow on this PC (%s): level %d\n", why,
			            g_slow_tracker.Level());
		}
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
		if (Common::UffdWriteWatch::Enabled()) {
			g_benchmark_uffd = Measure(true);
			const auto& u    = g_benchmark_uffd;
			if (u.valid) {
				std::printf("Kyty fault cost (userfaultfd write-protection, KYTY_UFFD_WP): write fault round trip %.2f us "
				            "(%.2f us of it the release in the handler); protect %.2f us/call + %.3f us/page, release "
				            "%.2f us/call + %.3f us/page\n",
				            u.fault_us, u.fault_handler_us, u.protect_call_us, u.protect_page_us, u.unprotect_call_us,
				            u.unprotect_page_us);
			} else {
				std::printf("Kyty fault cost (userfaultfd write-protection): benchmark failed\n");
			}
		}
#endif
		std::fflush(stdout);
	});
}

const Benchmark& StartupBenchmark() {
	return g_benchmark;
}

CostModel Model() {
	CostModel model;
	const auto& b = g_benchmark_uffd.valid ? g_benchmark_uffd : g_benchmark;
	if (b.valid) {
		model.fault_us        = b.fault_us;
		model.protect_call_us = (b.protect_call_us + b.unprotect_call_us) / 2;
		model.protect_page_us = (b.protect_page_us + b.unprotect_page_us) / 2;
	}
	const auto live_fault = g_live_fault_us.load(std::memory_order_relaxed);
	if (live_fault >= 0) {
		// The handler's live time plus what the benchmark saw outside its handler.
		const auto outside = b.valid ? std::max(0.0, b.fault_us - b.fault_handler_us) : 2.0;
		model.fault_us     = outside + live_fault;
	}
	const auto live_protect = g_live_protect_call_us.load(std::memory_order_relaxed);
	if (live_protect >= 0) {
		model.protect_call_us = live_protect;
	}
	model.tighten_fixed_us = g_live_tighten_fixed_us.load(std::memory_order_relaxed);
	return model;
}

int SlowLevel() noexcept {
	return g_slow_level.load(std::memory_order_relaxed);
}

void SlowLevelTracker::Seed(double tighten_call_us, double fault_round_trip_us) noexcept {
	const int level = tighten_call_us > 5.0 || fault_round_trip_us > 20.0  ? 2
	                  : tighten_call_us > 2.0 || fault_round_trip_us > 10.0 ? 1
	                                                                         : 0;
	m_level = std::max(m_level, level);
}

int SlowLevelTracker::Update(double tighten_fixed_us) noexcept {
	const int measured = tighten_fixed_us > 16.0 ? 2 : tighten_fixed_us > 10.0 ? 1 : 0;
	for (int level = 1; level <= 2; level++) {
		m_streak[level] = measured >= level ? m_streak[level] + 1 : 0;
		if (m_streak[level] >= PeriodsNeeded && m_level < level) {
			m_level = level;
		}
	}
	return m_level;
}

void NoteFault(uint64_t handler_ns) noexcept {
	g_live.faults.fetch_add(1, std::memory_order_relaxed);
	g_live.fault_ns.fetch_add(handler_ns, std::memory_order_relaxed);
}

std::atomic<PeriodicReporter> g_periodic_reporter {nullptr};

void SetPeriodicReporter(PeriodicReporter reporter) noexcept {
	g_periodic_reporter.store(reporter, std::memory_order_release);
}

void NoteBdaPass(int kind, uint64_t bytes) noexcept {
	if (kind >= 0 && kind < 4) {
		g_live.bda_passes[kind].fetch_add(1, std::memory_order_relaxed);
		g_live.bda_bytes[kind].fetch_add(bytes, std::memory_order_relaxed);
	}
}

void NoteProtect(bool unprotect, uint64_t pages, uint64_t ns) noexcept {
	if (unprotect) {
		g_live.unprotect_calls.fetch_add(1, std::memory_order_relaxed);
		g_live.unprotect_pages.fetch_add(pages, std::memory_order_relaxed);
		g_live.unprotect_ns.fetch_add(ns, std::memory_order_relaxed);
	} else {
		g_live.protect_calls.fetch_add(1, std::memory_order_relaxed);
		g_live.protect_pages.fetch_add(pages, std::memory_order_relaxed);
		g_live.protect_ns.fetch_add(ns, std::memory_order_relaxed);
	}
}

uint32_t Frame() noexcept {
	return g_frame.load(std::memory_order_relaxed);
}

void AdvanceFrame() noexcept {
	const auto frame = g_frame.fetch_add(1, std::memory_order_relaxed) + 1;
	const auto now   = NowNs();
	if (!g_log_period_read) {
		g_log_period_read = true;
		const auto* value = std::getenv("KYTY_FAULT_COST_LOG");
		const auto  secs  = value != nullptr ? std::strtoull(value, nullptr, 10) : 60ull;
		g_log_period_ns   = secs * 1'000'000'000ull;
		g_log_base        = TakeLive(frame, now);
		g_model_base      = g_log_base;
	}
	// The model's live inputs: every ~2 s, from at least 64 faults / protection calls.
	if (now - g_model_base.time_ns >= 2'000'000'000ull) {
		const auto current = TakeLive(frame, now);
		const auto faults  = current.faults - g_model_base.faults;
		const auto calls   = current.protect_calls + current.unprotect_calls - g_model_base.protect_calls -
		                   g_model_base.unprotect_calls;
		if (faults >= 64) {
			const auto value = static_cast<double>(current.fault_ns - g_model_base.fault_ns) / static_cast<double>(faults) / 1e3;
			const auto prev  = g_live_fault_us.load(std::memory_order_relaxed);
			g_live_fault_us.store(prev < 0 ? value : prev * 0.5 + value * 0.5, std::memory_order_relaxed);
		}
		if (calls >= 64) {
			const auto ns = current.protect_ns + current.unprotect_ns - g_model_base.protect_ns - g_model_base.unprotect_ns;
			const auto value = static_cast<double>(ns) / static_cast<double>(calls) / 1e3;
			const auto prev  = g_live_protect_call_us.load(std::memory_order_relaxed);
			g_live_protect_call_us.store(prev < 0 ? value : prev * 0.5 + value * 0.5, std::memory_order_relaxed);
		}
		// SlowLevel(): the tightening calls' cost beyond their pages, which the fault-ahead window
		// does not change (the simulation's waits included, by design).
		if (const auto tighten = current.protect_calls - g_model_base.protect_calls; tighten >= 32) {
			const auto& b        = g_benchmark_uffd.valid ? g_benchmark_uffd : g_benchmark;
			const auto  page_us  = b.valid ? b.protect_page_us : 0.02;
			const auto  calls_d  = static_cast<double>(tighten);
			const auto  per_call = static_cast<double>(current.protect_ns - g_model_base.protect_ns) / calls_d / 1e3;
			const auto  pages    = static_cast<double>(current.protect_pages - g_model_base.protect_pages) / calls_d;
			const auto  fixed    = std::max(0.0, per_call - page_us * pages);
			g_live_tighten_fixed_us.store(fixed, std::memory_order_relaxed);
			const int before = g_slow_tracker.Level();
			const int level  = g_slow_tracker.Update(fixed);
			if (level != before) {
				g_slow_level.store(level, std::memory_order_relaxed);
				std::printf("Kyty fault cost: write tracking is slow on this PC (tightening protection calls %.1f us "
				            "each beyond their pages for %d periods in a row): level %d\n",
				            fixed, SlowLevelTracker::PeriodsNeeded, level);
				std::fflush(stdout);
			}
		}
		g_model_base = current;
	}
	if (g_log_period_ns != 0 && now - g_log_base.time_ns >= g_log_period_ns) {
		const auto c       = TakeLive(frame, now);
		const auto& p      = g_log_base;
		const auto seconds = static_cast<double>(c.time_ns - p.time_ns) / 1e9;
		const auto frames  = std::max<uint64_t>(c.frames - p.frames, 1);
		const auto avg     = [](uint64_t ns, uint64_t n) { return n != 0 ? static_cast<double>(ns) / static_cast<double>(n) / 1e3 : 0.0; };
		const auto faults  = c.faults - p.faults;
		const auto pcalls  = c.protect_calls - p.protect_calls;
		const auto ucalls  = c.unprotect_calls - p.unprotect_calls;
		const auto model   = Model();
		{
			const auto f = static_cast<double>(frames);
			std::printf("Kyty BDA passes: last %.0f s, per frame: idle %.1f, hot runs %.1f (%.2f MB), dirty log %.1f "
			            "(%.2f MB), full scan %.2f (%.2f MB)\n",
			            seconds, static_cast<double>(c.bda_passes[0] - p.bda_passes[0]) / f,
			            static_cast<double>(c.bda_passes[1] - p.bda_passes[1]) / f,
			            static_cast<double>(c.bda_bytes[1] - p.bda_bytes[1]) / f / 1e6,
			            static_cast<double>(c.bda_passes[2] - p.bda_passes[2]) / f,
			            static_cast<double>(c.bda_bytes[2] - p.bda_bytes[2]) / f / 1e6,
			            static_cast<double>(c.bda_passes[3] - p.bda_passes[3]) / f,
			            static_cast<double>(c.bda_bytes[3] - p.bda_bytes[3]) / f / 1e6);
		}
		std::printf("Kyty fault cost: last %.0f s, %.1f fps: %.0f faults/frame at %.1f us in the handler "
		            "(%.2f ms/frame); protect %.0f calls/frame at %.1f us (%.0f pages/call); unprotect %.0f "
		            "calls/frame at %.1f us (%.0f pages/call); model %.1f us/fault, %.1f us/call, tightening %.1f us "
		            "beyond its pages, slow level %d\n",
		            seconds, static_cast<double>(frames) / seconds, static_cast<double>(faults) / static_cast<double>(frames),
		            avg(c.fault_ns - p.fault_ns, faults),
		            static_cast<double>(c.fault_ns - p.fault_ns) / static_cast<double>(frames) / 1e6,
		            static_cast<double>(pcalls) / static_cast<double>(frames), avg(c.protect_ns - p.protect_ns, pcalls),
		            pcalls != 0 ? static_cast<double>(c.protect_pages - p.protect_pages) / static_cast<double>(pcalls) : 0.0,
		            static_cast<double>(ucalls) / static_cast<double>(frames), avg(c.unprotect_ns - p.unprotect_ns, ucalls),
		            ucalls != 0 ? static_cast<double>(c.unprotect_pages - p.unprotect_pages) / static_cast<double>(ucalls) : 0.0,
		            model.fault_us, model.protect_call_us, model.tighten_fixed_us, SlowLevel());
		if (const auto reporter = g_periodic_reporter.load(std::memory_order_acquire); reporter != nullptr) {
			reporter(seconds, frames);
		}
		std::fflush(stdout);
		g_log_base = c;
	}
	MaybeDumpMap(frame);
}

bool SimEnabled() noexcept {
	return g_sim_fault_us.Get() != 0 || g_sim_protect_us.Get() != 0 || g_sim_protect_page_ns.Get() != 0 ||
	       g_sim_unprotect_us.Get() != 0 || g_sim_unprotect_page_ns.Get() != 0;
}

void SimulateFault() noexcept {
	const auto us = g_sim_fault_us.Get();
	if (us == 0) {
		return;
	}
	if (g_sim_serial.On() && !t_sim_protect_locked) {
		// Linux: the fault's kernel part waits for the mmap lock that every mprotect holds for
		// writing, so faults and protection calls of all threads go one at a time.
		std::scoped_lock lock(g_sim_protect_mutex);
		SpinNs(static_cast<uint64_t>(us) * 1000u);
		return;
	}
	SpinNs(static_cast<uint64_t>(us) * 1000u);
}

void SimProtectBegin(bool unprotect, uint64_t pages) noexcept {
	const auto call_us = unprotect ? g_sim_unprotect_us.Get() : g_sim_protect_us.Get();
	const auto page_ns = unprotect ? g_sim_unprotect_page_ns.Get() : g_sim_protect_page_ns.Get();
	if (call_us == 0 && page_ns == 0) {
		return;
	}
	g_sim_protect_mutex.lock();
	t_sim_protect_locked = true;
	SpinNs(static_cast<uint64_t>(call_us) * 1000u + static_cast<uint64_t>(page_ns) * pages);
}

void SimProtectEnd() noexcept {
	if (t_sim_protect_locked) {
		t_sim_protect_locked = false;
		g_sim_protect_mutex.unlock();
	}
}

bool MapEnabled() noexcept {
	return g_map_enabled;
}

void SetThreadDescriber(ThreadDescriber describer) noexcept {
	g_describer.store(describer, std::memory_order_release);
}

void SetFaultInstruction(uint64_t rip) noexcept {
	t_fault_rip = rip;
}

void MapInFault(bool inside) noexcept {
	t_in_fault = inside;
}

void MapFault(const FaultRecord& record) noexcept {
	if (!g_map_enabled) {
		return;
	}
	char       name[24] {};
	const int  thread = CurrentThreadClass(name, sizeof(name));
	const auto frame  = Frame();
	const auto page   = record.address & ~(PageBytes - 1);
	std::scoped_lock lock(g_map_mutex);
	auto&            window = g_map;
	window.faults[thread]++;
	window.fault_ns[thread] += record.handler_ns;
	for (int part = 0; part < static_cast<int>(Part::Count); part++) {
		window.part_ns[thread][part] += record.part_ns[part];
	}
	if (!record.write) {
		window.read_faults++;
	}
	if (t_fault_rip >= g_exe_begin && t_fault_rip < g_exe_end) {
		window.emulator_rip++;
	}
	const auto now = NowNs();
	if (record.found_dirty) {
		window.found_dirty++;
	}
	auto [it, inserted] = g_page_last_frame.try_emplace(page, PageLast {frame, now});
	if (inserted) {
		window.period[6]++;
	} else {
		const auto d = frame - it->second.frame;
		window.period[d == 0 ? 0 : d == 1 ? 1 : d == 2 ? 2 : d == 3 ? 3 : d < 8 ? 4 : 5]++;
		const auto us = (now - it->second.ns) / 1000u;
		window.since[us < 20 ? 0 : us < 200 ? 1 : us < 2000 ? 2 : us < 20000 ? 3 : 4]++;
		it->second = PageLast {frame, now};
	}
	auto& block = window.blocks[record.address >> 16u];
	block.faults[thread]++;
	block.ns += record.handler_ns;
	block.page_mask |= 1ull << ((record.address >> 12u) & 15u);
	if (block.last_frame != frame) {
		block.last_frame = frame;
		block.frames++;
	}
	block.rip = t_fault_rip;
	std::memcpy(block.thread, name, sizeof(block.thread));
}

void MapProtect(bool unprotect, bool no_access, uint64_t pages, uint64_t ns) noexcept {
	if (!g_map_enabled) {
		return;
	}
	const int        thread = CurrentThreadClass(nullptr, 0);
	const int        kind   = unprotect ? 2 : no_access ? 1 : 0;
	std::scoped_lock lock(g_map_mutex);
	auto&            stats = g_map.protect[thread][kind][t_in_fault ? 1 : 0];
	stats.calls++;
	stats.pages += pages;
	stats.ns += ns;
}

} // namespace Libs::Graphics::FaultCost
