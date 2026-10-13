#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_

#include "common/common.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <memory>
#include <vector>

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

	// Deferred write protections (KYTY_DEFER_PROTECTS, default off; the idea of peepostarexx's
	// DEFER_PROTECTS, re-implemented on the watcher counts). While a DeferProtectsScope is active on
	// a thread, a new write watcher that makes pages read-only only updates the counts, under the
	// region's count lock; the pages are marked pending and their spans join the thread's pending
	// list instead of calling the host. Whoever takes the list (TakePendingProtects) must apply it
	// (ApplyProtects, any thread) before it reads the bytes of those pages for the copy the
	// protection guards: a guest write landing before the protection is then in the copy, and every
	// later write faults. Applying is ApplySpan: it sets the host protection from the counts as they
	// are then, under the region's host lock, so it needs no ordering with other changes of the
	// same pages (each change sets what the counts ask for the pages it changes; a pending page only
	// stays looser until it is applied). BufferCache: read uploads whose guest bytes the upload DMA
	// worker copies hand the list to the worker with the copy; every other path applies it on the
	// calling thread before its own copy.
	struct PendingProtect {
		PageManager* manager = nullptr;
		void*        region  = nullptr; // Impl::Region
		uint64_t     base    = 0;
		uint32_t     first   = 0;
		uint32_t     last    = 0;
	};
	using PendingProtects = std::vector<PendingProtect>;
	class DeferProtectsScope final {
	public:
		explicit DeferProtectsScope(bool enable) noexcept;
		// Applies whatever is still pending on this thread (nothing read those pages).
		~DeferProtectsScope();
		DeferProtectsScope(const DeferProtectsScope&)            = delete;
		DeferProtectsScope& operator=(const DeferProtectsScope&) = delete;
		// Stops deferring (the collection is over); the pending list stays.
		static void Stop() noexcept;

	private:
		bool m_enabled = false;
	};
	[[nodiscard]] static bool HasPendingProtects() noexcept;
	// Moves the thread's pending list out.
	[[nodiscard]] static PendingProtects TakePendingProtects() noexcept;
	// Applies the thread's pending list now (before this thread reads those pages).
	static void ApplyPendingProtects() noexcept;
	// Applies a taken list; `worker`: counted as made off the calling (command-processor) thread.
	static void ApplyProtects(const PendingProtects& list, bool worker) noexcept;
	// KYTY_DEFER_PROTECTS: 0 off, 1 on (uploads the DMA worker copies anyway), "all": read uploads
	// below the DMA minimum go to the worker too; "verify" / "verify-all": the same, and every
	// applied span is checked, with the region's host calls quiesced, for pages still looser than
	// their counts on the host (a guest write there would escape tracking), plus the end-of-scope
	// leftovers. Live switch. Bits: 1 on, 2 all, 4 verify.
	[[nodiscard]] static int DeferProtectsMode() noexcept;
	struct DeferProtectsStats {
		uint64_t spans          = 0; // spans put on pending lists
		uint64_t pages          = 0; // pages whose protection was deferred
		uint64_t worker_spans   = 0; // spans applied off the deferring thread
		uint64_t worker_calls   = 0; // host calls those made
		uint64_t worker_ns      = 0;
		uint64_t local_spans    = 0; // spans applied on the deferring thread (no worker copy)
		uint64_t local_calls    = 0;
		uint64_t leftover_spans = 0; // applied at the scope end
		uint64_t pending_pages  = 0; // pages pending now
		uint64_t verify_checks  = 0;
		uint64_t verify_mismatches = 0;
		// Pages a deferred tighten made while another thread's host update of the region was in
		// flight (pending whatever `applied` said), and pages an update recorded looser than their
		// counts without being pending (0 unless KYTY_DEFER_PROTECTS_RACE=old).
		uint64_t race_pages = 0;
		uint64_t lost_pages = 0;
	};
	[[nodiscard]] static DeferProtectsStats GetDeferProtectsStats() noexcept;
	// Host protection calls made by the calling thread (cumulative): count, pages, ns.
	struct ThreadProtectTotals {
		uint64_t calls = 0;
		uint64_t pages = 0;
		uint64_t ns    = 0;
	};
	[[nodiscard]] static ThreadProtectTotals ThreadProtects() noexcept;
	// KYTY_DEFER_PROTECTS_STATS=1: the command processor's flip prints a cumulative line about
	// every 250 ms (its own protection calls and the deferral counters), for per-slot A/B deltas.
	static void NoteCpFlip() noexcept;

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
