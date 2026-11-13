#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_BDALIVEWINDOWS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_BDALIVEWINDOWS_H_

#include <cstdint>
#include <functional>
#include <map>

// KYTY_BDA_LIVE_PAGES (bufferCacheLive.cpp): bookkeeping of the imported windows of the guest
// backing alias. A window is [base + k * window_bytes, + window_bytes) of the backing (the last one
// is shorter), imported once on first use and kept for the process lifetime (never released while
// the GPU can read it; the owner frees the imports at teardown). Translate gives the device address
// of host bytes [host, host + bytes) when they lie in one window that is imported (or importable
// under the cap); 0 otherwise. No Vulkan here: the import is a callback (tests fake it).
namespace Libs::Graphics {

class BdaLiveWindows final {
public:
	// Imports [host, host + bytes) and returns its device address, or 0 on failure.
	using ImportFn = std::function<uint64_t(uint64_t host, uint64_t bytes)>;

	// Host-pointer imports need 4 KiB alignment of the pointer and the size (the 3090 driver's
	// minImportedHostPointerAlignment; checked here for every window).
	static constexpr uint64_t IMPORT_ALIGNMENT = 4096;

	BdaLiveWindows(uint64_t backing_base, uint64_t backing_size, uint64_t window_bytes, uint64_t cap_bytes,
	               ImportFn import);

	// False when the backing or the window size breaks the import rules: nothing is ever imported.
	[[nodiscard]] bool Valid() const noexcept { return m_valid; }
	[[nodiscard]] uint64_t Translate(uint64_t host, uint64_t bytes);

	[[nodiscard]] uint64_t WindowBytes() const noexcept { return m_window_bytes; }
	[[nodiscard]] uint64_t Windows() const noexcept { return m_imported; }
	[[nodiscard]] uint64_t PinnedBytes() const noexcept { return m_pinned; }
	[[nodiscard]] uint64_t Failures() const noexcept { return m_failures; }
	[[nodiscard]] uint64_t CapRejects() const noexcept { return m_cap_rejects; }
	[[nodiscard]] uint64_t RangeRejects() const noexcept { return m_range_rejects; }

private:
	struct Window {
		uint64_t device = 0; // 0: failed (never retried)
		uint64_t size   = 0;
	};
	uint64_t                     m_base;
	uint64_t                     m_size;
	uint64_t                     m_window_bytes;
	uint64_t                     m_cap;
	ImportFn                     m_import;
	bool                         m_valid         = false;
	std::map<uint64_t, Window>   m_windows;      // by window index
	uint64_t                     m_imported      = 0;
	uint64_t                     m_pinned        = 0;
	uint64_t                     m_failures      = 0;
	uint64_t                     m_cap_rejects   = 0;
	uint64_t                     m_range_rejects = 0;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_BDALIVEWINDOWS_H_ */
