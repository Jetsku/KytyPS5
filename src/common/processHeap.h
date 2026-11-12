#ifndef EMULATOR_SRC_COMMON_PROCESSHEAP_H_
#define EMULATOR_SRC_COMMON_PROCESSHEAP_H_

// Which heap implementation backs the process heap (and so the UCRT's malloc and operator new):
// the Segment Heap (KYTY_SEGMENT_HEAP, the executable's manifest) or the NT heap. Read from the
// heap's signature word at +0x10 (x64): 0xDDEEDDEE for the Segment Heap, the heap segment's
// 0xFFEEFFEE for the NT heap (both checked on Windows 11 by process_heap_tests). Only reads the
// heap's own header; never allocates.

#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__))
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <cstdint>
#include <cstring>

namespace Common {

enum class ProcessHeapKind { Unknown, Nt, Segment };

[[nodiscard]] inline ProcessHeapKind GetProcessHeapKind() noexcept {
#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__))
	const auto* heap = static_cast<const unsigned char*>(static_cast<const void*>(GetProcessHeap()));
	if (heap == nullptr) {
		return ProcessHeapKind::Unknown;
	}
	uint32_t signature = 0;
	std::memcpy(&signature, heap + 0x10, sizeof(signature));
	if (signature == 0xDDEEDDEEu) {
		return ProcessHeapKind::Segment;
	}
	if (signature == 0xFFEEFFEEu) {
		return ProcessHeapKind::Nt;
	}
#endif
	return ProcessHeapKind::Unknown;
}

[[nodiscard]] inline const char* ProcessHeapKindName(ProcessHeapKind kind) noexcept {
	switch (kind) {
		case ProcessHeapKind::Nt: return "NT heap";
		case ProcessHeapKind::Segment: return "Segment Heap";
		default: return "unknown";
	}
}

} // namespace Common

#endif // EMULATOR_SRC_COMMON_PROCESSHEAP_H_
