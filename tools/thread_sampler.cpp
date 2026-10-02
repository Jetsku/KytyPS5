// Poor man's sampling profiler for a few threads of one Windows process. No admin rights needed.
//
//   thread_sampler.exe <pid> <seconds> <hz> <out.csv> <tid> [tid ...]
//
// Every 1/hz seconds it suspends each listed thread in turn, reads its registers, walks the call
// stack and resumes it, so the cost per sample is a few tens of microseconds per thread.
//
// Output:
//   <out.csv>           tid,rip,module,offset,stack   (stack = caller return addresses, hex, ';'-separated)
//   <out.csv>.modules   base,size,name              (loaded modules, to resolve the addresses afterwards)
//
// Stack walking: code inside a loaded module (the emulator, Windows DLLs) is unwound with the PE unwind
// tables through DbgHelp's StackWalk64. Code outside any module is guest code, which uses frame pointers
// (push rbp; mov rbp,rsp), so it is unwound along the rbp chain; the walk switches between the two as the
// stack alternates between guest and host frames. tools/analyze_samples.py turns the CSV into a profile.
//
// Build (from a VS developer prompt):  clang-cl /EHsc /O1 tools\thread_sampler.cpp /Fe:thread_sampler.exe
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dbghelp.h>
#include <psapi.h>
#include <timeapi.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "psapi.lib")
#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "dbghelp.lib")

namespace {

constexpr int MAX_DEPTH = 18;

struct Module {
	uint64_t    base = 0;
	uint64_t    size = 0;
	std::string name;
};

std::vector<Module> LoadModules(HANDLE process) {
	std::vector<Module>  result;
	std::vector<HMODULE> modules(1024);
	DWORD                needed = 0;
	if (!EnumProcessModulesEx(process, modules.data(), static_cast<DWORD>(modules.size() * sizeof(HMODULE)), &needed,
	                          LIST_MODULES_ALL)) {
		return result;
	}
	modules.resize(needed / sizeof(HMODULE));
	for (auto module: modules) {
		MODULEINFO info {};
		char       path[MAX_PATH] = {};
		if (!GetModuleInformation(process, module, &info, sizeof(info))) continue;
		GetModuleFileNameExA(process, module, path, MAX_PATH);
		SymLoadModuleEx(process, nullptr, path, nullptr, reinterpret_cast<DWORD64>(info.lpBaseOfDll), info.SizeOfImage,
		                nullptr, 0);
		std::string name(path);
		const auto  slash = name.find_last_of("\\/");
		if (slash != std::string::npos) name = name.substr(slash + 1);
		result.push_back({reinterpret_cast<uint64_t>(info.lpBaseOfDll), info.SizeOfImage, name});
	}
	return result;
}

const Module* Find(const std::vector<Module>& modules, uint64_t address) {
	for (const auto& module: modules) {
		if (address >= module.base && address < module.base + module.size) return &module;
	}
	return nullptr;
}

// The thread is suspended only long enough to read its registers and copy the top of its stack; the
// stack is unwound afterwards, from this copy, while the thread runs again. Unwinding with DbgHelp takes
// milliseconds, so doing it with the thread suspended slowed the sampled game down noticeably.
struct StackCopy {
	uint64_t             base = 0;
	std::vector<uint8_t> data;
};
StackCopy g_copy;

BOOL CALLBACK ReadMemory(HANDLE process, DWORD64 address, PVOID buffer, DWORD size, LPDWORD read) {
	if (address >= g_copy.base && address + size <= g_copy.base + g_copy.data.size()) {
		std::memcpy(buffer, g_copy.data.data() + (address - g_copy.base), size);
		if (read != nullptr) *read = size;
		return TRUE;
	}
	SIZE_T     got = 0;
	const BOOL ok  = ReadProcessMemory(process, reinterpret_cast<LPCVOID>(address), buffer, size, &got);
	if (read != nullptr) *read = static_cast<DWORD>(got);
	return ok;
}

bool ReadU64(HANDLE process, uint64_t address, uint64_t* value) {
	DWORD read = 0;
	return address != 0 && ReadMemory(process, address, value, sizeof(*value), &read) && read == sizeof(*value);
}

// Caller addresses of the thread's current position. `context` is a suspended thread's context.
std::vector<uint64_t> WalkStack(HANDLE process, HANDLE thread, const std::vector<Module>& modules, CONTEXT context) {
	std::vector<uint64_t> callers;
	STACKFRAME64          frame {};
	frame.AddrPC.Offset    = context.Rip;
	frame.AddrPC.Mode      = AddrModeFlat;
	frame.AddrFrame.Offset = context.Rbp;
	frame.AddrFrame.Mode   = AddrModeFlat;
	frame.AddrStack.Offset = context.Rsp;
	frame.AddrStack.Mode   = AddrModeFlat;

	uint64_t rip = context.Rip;
	uint64_t rbp = context.Rbp;
	bool     first_step = true;
	for (int depth = 0; depth < MAX_DEPTH; depth++) {
		if (Find(modules, rip) != nullptr) {
			// Module code: one StackWalk64 step. The first call returns the current frame itself.
			if (first_step) {
				if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, process, thread, &frame, &context, ReadMemory,
				                 SymFunctionTableAccess64, SymGetModuleBase64, nullptr)) {
					break;
				}
				first_step = false;
			}
			if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, process, thread, &frame, &context, ReadMemory,
			                 SymFunctionTableAccess64, SymGetModuleBase64, nullptr)) {
				break;
			}
			rip = frame.AddrPC.Offset;
			rbp = context.Rbp;
			if (rip == 0) break;
			callers.push_back(rip);
		} else {
			// Guest code: rbp chain, [rbp] = caller's rbp, [rbp+8] = return address.
			uint64_t next_rbp = 0;
			uint64_t ret      = 0;
			if (rbp == 0 || !ReadU64(process, rbp, &next_rbp) || !ReadU64(process, rbp + 8, &ret) || ret == 0) break;
			if (next_rbp != 0 && next_rbp <= rbp) break; // the chain must grow towards the stack base
			const uint64_t callee_rbp = rbp;
			rip = ret;
			rbp = next_rbp;
			callers.push_back(rip);
			// Keep the StackWalk64 state consistent for a later switch back into a module: after the
			// return the stack pointer is the callee's frame pointer plus the saved rbp and return slots.
			frame.AddrPC.Offset    = rip;
			frame.AddrFrame.Offset = rbp;
			frame.AddrStack.Offset = callee_rbp + 16;
			context.Rip            = rip;
			context.Rbp            = rbp;
			context.Rsp            = callee_rbp + 16;
			first_step             = false;
			if (rbp == 0) break;
		}
	}
	return callers;
}

uint64_t g_fault_count = 0;

// A bad sample (an access violation while unwinding, seen as a crash inside memmove after a few seconds
// at 250 Hz) must cost one sample, not the whole run: the sampled thread was resumed before the walk, so
// nothing is left suspended. The first few faults are described on stderr to find the cause.
LONG ReportWalkFault(EXCEPTION_POINTERS* info, uint64_t rip_in_exe) {
	g_fault_count++;
	if (g_fault_count <= 5) {
		const auto* record = info->ExceptionRecord;
		std::fprintf(stderr, "thread_sampler: exception 0x%lx at 0x%llx (exe +0x%llx), access %llu address 0x%llx\n",
		             record->ExceptionCode, static_cast<unsigned long long>(reinterpret_cast<uint64_t>(record->ExceptionAddress)),
		             static_cast<unsigned long long>(reinterpret_cast<uint64_t>(record->ExceptionAddress) - rip_in_exe),
		             record->NumberParameters > 0 ? static_cast<unsigned long long>(record->ExceptionInformation[0]) : 0ull,
		             record->NumberParameters > 1 ? static_cast<unsigned long long>(record->ExceptionInformation[1]) : 0ull);
	}
	return EXCEPTION_EXECUTE_HANDLER;
}

// No object with a destructor may live in a function that uses __try, hence the out parameter.
bool SafeWalkStack(HANDLE process, HANDLE thread, const std::vector<Module>* modules, const CONTEXT* context,
                   std::vector<uint64_t>* callers) {
	const uint64_t exe_base = reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr));
	__try {
		*callers = WalkStack(process, thread, *modules, *context);
		return true;
	} __except (ReportWalkFault(GetExceptionInformation(), exe_base)) {
		return false;
	}
}

} // namespace

int wmain(int argc, wchar_t** argv) {
	if (argc < 6) {
		std::fprintf(stderr, "usage: thread_sampler <pid> <seconds> <hz> <out.csv> <tid> [tid ...]\n");
		return 2;
	}
	const DWORD  pid     = static_cast<DWORD>(std::wcstoul(argv[1], nullptr, 10));
	const double seconds = std::wcstod(argv[2], nullptr);
	const double hz      = std::wcstod(argv[3], nullptr);
	if (seconds <= 0 || hz <= 0) return 2;

	HANDLE process = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid);
	if (process == nullptr) {
		std::fprintf(stderr, "cannot open process %lu (error %lu)\n", pid, GetLastError());
		return 1;
	}
	SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_NO_PROMPTS);
	SymInitialize(process, nullptr, FALSE);
	const auto modules = LoadModules(process);

	std::vector<DWORD>  tids;
	std::vector<HANDLE> handles;
	for (int i = 5; i < argc; i++) {
		const DWORD tid    = static_cast<DWORD>(std::wcstoul(argv[i], nullptr, 10));
		HANDLE      handle = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, tid);
		if (handle == nullptr) {
			std::fprintf(stderr, "cannot open thread %lu (error %lu), skipped\n", tid, GetLastError());
			continue;
		}
		tids.push_back(tid);
		handles.push_back(handle);
	}
	if (handles.empty()) return 1;

	FILE* out = nullptr;
	if (_wfopen_s(&out, argv[4], L"w") != 0 || out == nullptr) {
		std::fprintf(stderr, "cannot write output\n");
		return 1;
	}
	std::fprintf(out, "tid,rip,module,offset,stack\n");
	{
		const std::wstring modules_path = std::wstring(argv[4]) + L".modules";
		FILE*              mf           = nullptr;
		if (_wfopen_s(&mf, modules_path.c_str(), L"w") == 0 && mf != nullptr) {
			std::fprintf(mf, "base,size,name\n");
			for (const auto& m: modules) {
				std::fprintf(mf, "0x%llx,0x%llx,%s\n", static_cast<unsigned long long>(m.base),
				             static_cast<unsigned long long>(m.size), m.name.c_str());
			}
			std::fclose(mf);
		}
	}

	timeBeginPeriod(1);
	const auto interval = std::chrono::duration<double>(1.0 / hz);
	const auto end      = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
	auto       next     = std::chrono::steady_clock::now();
	uint64_t   samples  = 0;
	while (std::chrono::steady_clock::now() < end) {
		for (size_t i = 0; i < handles.size(); i++) {
			if (SuspendThread(handles[i]) == static_cast<DWORD>(-1)) continue;
			CONTEXT context {};
			context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
			const BOOL ok        = GetThreadContext(handles[i], &context);
			if (ok) {
				g_copy.base = context.Rsp;
				g_copy.data.resize(32768);
				SIZE_T got = 0;
				if (!ReadProcessMemory(process, reinterpret_cast<LPCVOID>(context.Rsp), g_copy.data.data(), g_copy.data.size(), &got) ||
				    got == 0) {
					g_copy.data.resize(4096);
					got = 0;
					ReadProcessMemory(process, reinterpret_cast<LPCVOID>(context.Rsp), g_copy.data.data(), g_copy.data.size(), &got);
				}
				g_copy.data.resize(got);
			}
			ResumeThread(handles[i]);
			if (!ok) continue;
			std::vector<uint64_t> callers;
			if (!SafeWalkStack(process, handles[i], &modules, &context, &callers)) callers.clear();
			const uint64_t rip    = context.Rip;
			const Module*  module = Find(modules, rip);
			std::fprintf(out, "%lu,0x%llx,%s,0x%llx,", tids[i], static_cast<unsigned long long>(rip),
			             module != nullptr ? module->name.c_str() : "-",
			             static_cast<unsigned long long>(module != nullptr ? rip - module->base : 0));
			for (size_t k = 0; k < callers.size(); k++) {
				std::fprintf(out, "%s0x%llx", k ? ";" : "", static_cast<unsigned long long>(callers[k]));
			}
			std::fprintf(out, "\n");
			samples++;
		}
		next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(interval);
		std::this_thread::sleep_until(next);
	}
	timeEndPeriod(1);
	std::fclose(out);
	for (auto handle: handles) CloseHandle(handle);
	SymCleanup(process);
	CloseHandle(process);
	std::printf("thread_sampler: %llu samples of %zu threads (%llu unwinds faulted, stacks dropped)\n",
	            static_cast<unsigned long long>(samples), handles.size(), static_cast<unsigned long long>(g_fault_count));
	return 0;
}
