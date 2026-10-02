#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_VRAMSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_VRAMSTATS_H_

// KYTY_VRAM_STATS=1 (default off; diagnostics only, it changes no decision): GPU memory by owner.
// Native allocations are counted here by kind and memory (device-local or not) when they are
// created and released, and every Buffer is listed with its size and guest address. Every
// KYTY_VRAM_STATS_SECONDS (default 10) the GPU thread writes a report (RenderContext::
// RunGarbageCollector) to KYTY_VRAM_STATS_FILE (default kyty-vram-stats-<pid>.log in the working
// directory): VMA heap usage and budget (VK_EXT_memory_budget), VMA's own block and allocation
// bytes per heap, these totals, the native image and tiler scratch pools, the texture cache by
// binding kind, frame age and residency (largest images, render-target sizes, aliases at one guest
// address) and the buffer cache (largest buffers, ages). With it off, Note() is one branch on a
// cached flag.

#include <cstdint>
#include <vector>

namespace Libs::Graphics::VramStats {

enum class Kind : uint8_t {
	Image,        // live native images (the native image pool is reported separately)
	GuestBuffer,  // buffers that mirror guest memory (BufferCache)
	RingBuffer,   // upload/download/stream rings (MemoryUsage other than DeviceLocal)
	OtherBuffer,  // device-local buffers without a guest address (page tables, scratch rings, ...)
	TilerScratch, // detile/tile scratch buffers, in use or pooled
	Count
};

[[nodiscard]] bool Enabled() noexcept;
// bytes > 0: an allocation, bytes < 0: its release (same kind and memory).
void Note(Kind kind, bool device_local, int64_t bytes) noexcept;

struct KindTotals {
	int64_t bytes[2] {}; // [0]: not device-local (system memory), [1]: device-local
	int64_t count[2] {};
};
[[nodiscard]] KindTotals Totals(Kind kind) noexcept;
[[nodiscard]] const char* KindName(Kind kind) noexcept;

// Buffers (streamBuffer.cpp registers each one while it exists).
struct BufferEntry {
	uint64_t bytes        = 0;
	uint64_t cpu_address  = 0;
	uint8_t  usage        = 0; // MemoryUsage
	bool     device_local = false;
	bool     host_cached  = false;
};
void RegisterBuffer(const void* key, const BufferEntry& entry) noexcept;
void UnregisterBuffer(const void* key) noexcept;
[[nodiscard]] std::vector<BufferEntry> BufferSnapshot();

// GPU thread: true at most once per report interval (the first call starts the clock).
[[nodiscard]] bool ReportDue() noexcept;
// Seconds since the first ReportDue call.
[[nodiscard]] double Seconds() noexcept;
// One report line ("VRAM " prefixed, newline added) and the flush at the end of a report.
void Line(const char* format, ...) noexcept;
void Flush() noexcept;

[[nodiscard]] inline double ToMiB(uint64_t bytes) noexcept {
	return static_cast<double>(bytes) / (1024.0 * 1024.0);
}
[[nodiscard]] inline double ToMiB(int64_t bytes) noexcept {
	return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

} // namespace Libs::Graphics::VramStats

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_VRAMSTATS_H_
