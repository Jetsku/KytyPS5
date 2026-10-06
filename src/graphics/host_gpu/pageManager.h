#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_

#include "common/common.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <memory>

namespace Libs::Graphics {

enum class PageFaultAccess { Read, Write, Execute, Unknown };

class PageManager final {
public:
	PageManager();
	// The owner must stop all PageManager callers before destruction.
	~PageManager();

	KYTY_CLASS_NO_COPY(PageManager);

	[[nodiscard]] uint64_t GetPageSize() const;

	template <bool track>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size);
	template <bool track, bool is_read = false>
	void UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask);

	// Pages of [vaddr, vaddr + size) that resource tracking watches: for writes only (the host
	// page is read-only) and for every access (no access). Any thread; a snapshot.
	struct WatchedPages {
		uint64_t write  = 0;
		uint64_t access = 0;
	};
	[[nodiscard]] WatchedPages CountWatchedPages(uint64_t vaddr, uint64_t size);

	// Deferred write-unprotect (KYTY_DEFER_UNPROTECT, default on). While a scope is alive on a
	// thread, a released write watcher that makes pages writable only updates the watcher counts,
	// under the region's count lock and so in order with every other change; the host protection
	// of those pages is set when the thread's outermost scope ends, from the counts as they are
	// then. A caller opens the scope before it takes its own lock (a memory-tracker region, the
	// texture cache), so the host call happens after that lock is released, and the release
	// itself never waits for another thread's host call.
	// Everything else stays synchronous: watches (every change that tightens protection),
	// read-watcher changes, and calls outside a scope. Host calls are serialized per region, and
	// each is decided from the counts under that serialization; the only change that can overtake
	// one is a deferred release, which only loosens. So a page's host protection is never looser
	// than its counts ask once a call returns: a page whose release is still pending only faults
	// once more, and the last update always writes the latest state whatever order threads run in.
	// KYTY_DEFER_UNPROTECT=0 releases synchronously as before; =verify also checks after every
	// change, with the region's host calls quiesced, that no page is looser than its counts, as
	// recorded or (Windows) on the host, and that the host holds what was just set.
	class DeferUnprotectScope final {
	public:
		DeferUnprotectScope() noexcept;
		~DeferUnprotectScope();
		DeferUnprotectScope(const DeferUnprotectScope&)            = delete;
		DeferUnprotectScope& operator=(const DeferUnprotectScope&) = delete;
	};
	[[nodiscard]] static bool InDeferUnprotectScope() noexcept;

	// Sets the host protection of [vaddr, vaddr + size) to what the watcher counts ask now. Inside
	// a scope, unless `now`, it joins the scope's pending spans. For a faulting page: when its
	// write watcher was released by another thread whose host update is still pending, the
	// faulting thread must not wait for that thread to run.
	void Reconcile(uint64_t vaddr, uint64_t size, bool now);

	// The host protection of [vaddr, vaddr + size) was replaced behind the tracker (a view mapped
	// again with its own access by a partial unmap, a guest mprotect that allows writes): records
	// it as read-write and protects the watched pages again, so CPU writes to them fault (and reach
	// the GPU) once more. Unwatched pages keep the protection they were given. Any thread, outside
	// the guest address-space lock.
	void ResyncHostProtection(uint64_t vaddr, uint64_t size);

	enum class DeferMode { Off, On, Verify };
	[[nodiscard]] static DeferMode GetDeferMode();
	static void                    SetDeferModeForTests(DeferMode mode);
	struct DeferStats {
		uint64_t spans             = 0; // releases whose host update was deferred
		uint64_t calls             = 0; // host protection calls made by deferred updates
		uint64_t settled           = 0; // spans that needed no call (already current)
		uint64_t overflows         = 0; // spans applied at once: the thread's batch was full
		uint64_t verify_checks     = 0;
		uint64_t verify_mismatches = 0;
	};
	[[nodiscard]] static DeferStats GetDeferStats();

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
