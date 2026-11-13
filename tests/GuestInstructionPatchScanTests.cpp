// The load-time instruction patcher scans a module's functions on several threads, then patches the
// functions with rewrite sites in address order. Its output must be the same as a single-threaded
// pass: same code bytes, same trampolines, same counters.
#include "loader/guestInstructionPatcher.h"

#include <sys/mman.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "GuestInstructionPatchScanTests: %s\n", message);
		g_failures++;
	}
}

constexpr size_t FunctionSize    = 32;
constexpr size_t FunctionCount   = 2048;
constexpr size_t CodeSize        = FunctionSize * FunctionCount;
constexpr size_t TrampolineSize  = 1024 * 1024;

// 2048 small functions; every 7th uses VRSQRTPS, every 11th EXTRQ, every 13th INSERTQ.
std::vector<uint8_t> MakeCode() {
	std::vector<uint8_t> code(CodeSize, 0xcc);
	for (size_t i = 0; i < FunctionCount; ++i) {
		uint8_t*             f = code.data() + i * FunctionSize;
		std::vector<uint8_t> body;
		if (i % 7 == 0) {
			body = {0xc5, 0xf8, 0x52, 0xc1, 0x01, 0xc8, 0xc3};  // vrsqrtps xmm0, xmm1; add eax, ecx; ret
		} else if (i % 11 == 0) {
			body = {0x66, 0x0f, 0x78, 0xc0, 0x04, 0x08, 0xc3};  // extrq xmm0, 4, 8; ret
		} else if (i % 13 == 0) {
			body = {0x90, 0xf2, 0x0f, 0x79, 0xc1, 0xc3};        // nop; insertq xmm0, xmm1; ret
		} else {
			body = {0xb8, static_cast<uint8_t>(i), static_cast<uint8_t>(i >> 8), 0x00, 0x00, 0xc3};  // mov eax, i; ret
		}
		std::memcpy(f, body.data(), body.size());
	}
	return code;
}

struct Run {
	std::vector<uint8_t>                code;
	std::vector<uint8_t>                trampolines;
	Loader::GuestInstructionPatchResult result {};
};

Run Patch(uint8_t* area, const std::vector<uint8_t>& code, size_t threads) {
	std::memset(area, 0xcc, CodeSize + TrampolineSize);
	std::memcpy(area, code.data(), code.size());
	Loader::RegisterGuestInstructionPatchModule(area, CodeSize, area + CodeSize, TrampolineSize);
	std::vector<uintptr_t> starts;
	for (size_t i = 0; i < FunctionCount; ++i) {
		starts.push_back(reinterpret_cast<uintptr_t>(area) + i * FunctionSize);
	}
	Loader::SetGuestInstructionPatchScanThreads(threads);
	const Loader::GuestInstructionHostFeatures intel {false, false, false};
	Run run;
	run.result = Loader::PatchGuestInstructions(reinterpret_cast<uintptr_t>(area), CodeSize, starts, false, true, intel);
	Loader::SetGuestInstructionPatchScanThreads(0);
	Loader::UnregisterGuestInstructionPatchModule(area);
	run.code.assign(area, area + CodeSize);
	run.trampolines.assign(area + CodeSize, area + CodeSize + TrampolineSize);
	return run;
}

bool SameCounts(const Loader::InstructionPatchCounts& a, const Loader::InstructionPatchCounts& b) {
	return a.found == b.found && a.native == b.native && a.trapped == b.trapped &&
	       std::memcmp(a.rejected, b.rejected, sizeof(a.rejected)) == 0;
}

bool SameResult(const Loader::GuestInstructionPatchResult& a, const Loader::GuestInstructionPatchResult& b) {
	return a.function_count == b.function_count && a.instruction_count == b.instruction_count &&
	       a.red_zone_function_count == b.red_zone_function_count &&
	       a.memory_instruction_count == b.memory_instruction_count &&
	       a.short_memory_instruction_count == b.short_memory_instruction_count &&
	       a.patched_memory_instruction_count == b.patched_memory_instruction_count &&
	       a.stack_dependent_memory_instruction_count == b.stack_dependent_memory_instruction_count &&
	       a.control_flow_memory_instruction_count == b.control_flow_memory_instruction_count &&
	       a.unrelocatable_memory_instruction_count == b.unrelocatable_memory_instruction_count &&
	       a.indirect_red_zone_function_count == b.indirect_red_zone_function_count &&
	       SameCounts(a.reciprocal_sqrt, b.reciprocal_sqrt) && SameCounts(a.extrq, b.extrq) &&
	       SameCounts(a.insertq, b.insertq) && SameCounts(a.rdpid, b.rdpid) && SameCounts(a.clwb, b.clwb);
}

} // namespace

int main() {
	auto* area = static_cast<uint8_t*>(mmap(nullptr, CodeSize + TrampolineSize, PROT_READ | PROT_WRITE | PROT_EXEC,
	                                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
	Check(area != MAP_FAILED, "allocate the patch area");
	if (area == MAP_FAILED) {
		return 1;
	}
	const auto code   = MakeCode();
	const Run  serial = Patch(area, code, 1);
	Check(serial.result.function_count == FunctionCount, "every function is scanned");
	Check(serial.result.reciprocal_sqrt.found == (FunctionCount + 6) / 7, "every VRSQRTPS is found");
	Check(serial.result.extrq.found != 0 && serial.result.insertq.found != 0, "EXTRQ and INSERTQ are found");
	Check(serial.code != code, "the serial pass patches the code");
	for (const size_t threads: std::array<size_t, 3> {2, 5, 16}) {
		const Run parallel = Patch(area, code, threads);
		Check(parallel.code == serial.code, "parallel scan: same patched code");
		Check(parallel.trampolines == serial.trampolines, "parallel scan: same trampolines");
		Check(SameResult(parallel.result, serial.result), "parallel scan: same counters");
	}
	munmap(area, CodeSize + TrampolineSize);
	if (g_failures != 0) {
		std::fprintf(stderr, "GuestInstructionPatchScanTests: %d checks failed\n", g_failures);
		return 1;
	}
	std::printf("GuestInstructionPatchScanTests: all passed (%zu functions, VRSQRTPS %llu native %llu, EXTRQ %llu, INSERTQ %llu)\n",
	            FunctionCount, static_cast<unsigned long long>(serial.result.reciprocal_sqrt.found),
	            static_cast<unsigned long long>(serial.result.reciprocal_sqrt.native),
	            static_cast<unsigned long long>(serial.result.extrq.found),
	            static_cast<unsigned long long>(serial.result.insertq.found));
	return 0;
}
