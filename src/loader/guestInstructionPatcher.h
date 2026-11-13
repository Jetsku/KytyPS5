// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#ifndef KYTY_LOADER_GUEST_INSTRUCTION_PATCHER_H_
#define KYTY_LOADER_GUEST_INSTRUCTION_PATCHER_H_

#include "common/common.h"

#include <span>
#include <vector>

namespace Loader {

struct GuestInstructionHostFeatures {
	bool sse4a = false;
	bool rdpid = false;
	bool clwb  = false;
};

GuestInstructionHostFeatures GetGuestInstructionHostFeatures();

// Why a found instruction has no native trampoline (it traps, or is left unpatched).
enum class PatchRejection : uint8_t {
	SpanBranchTarget,       // a neighbour needed for the 5-byte jump is a branch target
	SpanTerminator,         // a jump, ret or similar ends the span before 5 bytes
	SpanUndecoded,          // a neighbour is outside the decoded function (padding, data)
	IndirectBranchFunction, // unresolved indirect jump: neighbours cannot be relocated safely
	EncodeFailed,           // a relocated neighbour could not be re-encoded
	TrampolineExhausted,    // trampoline space ran out; the site traps inside its trampoline
	NoRelaySlot,            // short-jump relay: no host or relay slot within 127 bytes
	Count
};
constexpr size_t PatchRejectionCount = static_cast<size_t>(PatchRejection::Count);
inline const char* PatchRejectionName(PatchRejection reason) {
	switch (reason) {
		case PatchRejection::SpanBranchTarget: return "branch-target";
		case PatchRejection::SpanTerminator: return "terminator";
		case PatchRejection::SpanUndecoded: return "undecoded";
		case PatchRejection::IndirectBranchFunction: return "indirect-branch";
		case PatchRejection::EncodeFailed: return "encode-failed";
		case PatchRejection::TrampolineExhausted: return "trampoline-full";
		case PatchRejection::NoRelaySlot: return "no-relay";
		default: return "?";
	}
}

struct InstructionPatchCounts {
	uint64_t found   = 0;
	uint64_t native  = 0;
	uint64_t trapped = 0;
	uint64_t rejected[PatchRejectionCount] {};

	uint64_t Skipped() const { return found - native - trapped; }

	InstructionPatchCounts& operator+=(const InstructionPatchCounts& other) {
		found += other.found;
		native += other.native;
		trapped += other.trapped;
		for (size_t i = 0; i < PatchRejectionCount; ++i) {
			rejected[i] += other.rejected[i];
		}
		return *this;
	}
};

struct GuestInstructionPatchResult {
	uint64_t               function_count                           = 0;
	uint64_t               instruction_count                        = 0;
	uint64_t               red_zone_function_count                  = 0;
	uint64_t               memory_instruction_count                 = 0;
	uint64_t               short_memory_instruction_count           = 0;
	uint64_t               patched_memory_instruction_count         = 0;
	uint64_t               stack_dependent_memory_instruction_count = 0;
	uint64_t               control_flow_memory_instruction_count    = 0;
	uint64_t               unrelocatable_memory_instruction_count   = 0;
	uint64_t               indirect_red_zone_function_count         = 0;
	InstructionPatchCounts reciprocal_sqrt;
	InstructionPatchCounts extrq;
	InstructionPatchCounts insertq;
	InstructionPatchCounts rdpid;
	InstructionPatchCounts clwb;
};

void RegisterGuestInstructionPatchModule(void* module_ptr, uint64_t module_size,
                                         void* trampoline_area_ptr, uint64_t trampoline_area_size);
void UnregisterGuestInstructionPatchModule(void* module_ptr);

// Tests: the number of threads that scan a module's functions (0: automatic). The patched code is the
// same for every thread count.
void SetGuestInstructionPatchScanThreads(size_t threads);

// Whether [begin, end) holds the bytes every instruction form the AMD instruction patch rewrites
// contains: VRSQRTPS xmm, xmm (C5 [L=0 pp=00] 52 /r, C4 [map 0F] [L=0 pp=00] 52 /r, register
// ModRM), EXTRQ/INSERTQ on registers (0F 78, 0F 79), RDPID (0F C7 /7, register) and CLWB
// (0F AE /6, memory). A function without them has no rewrite site. Exposed for tests.
bool MayContainAmdInstruction(const uint8_t* begin, const uint8_t* end);

// Apply enabled instruction fixes using native trampolines or safe trap fallbacks.
GuestInstructionPatchResult PatchGuestInstructions(
    uint64_t segment_addr, uint64_t segment_size, std::span<const uintptr_t> function_starts,
    bool protect_memory, bool emulate_amd,
    GuestInstructionHostFeatures host_features = GetGuestInstructionHostFeatures());

bool DecodeEhFrameFunctionStarts(uint64_t eh_frame_header_addr, uint64_t eh_frame_header_size,
                                 std::vector<uintptr_t>* function_starts);

} // namespace Loader

#endif /* KYTY_LOADER_GUEST_INSTRUCTION_PATCHER_H_ */
