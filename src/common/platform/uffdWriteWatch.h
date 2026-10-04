#ifndef EMULATOR_SRC_COMMON_PLATFORM_UFFDWRITEWATCH_H_
#define EMULATOR_SRC_COMMON_PLATFORM_UFFDWRITEWATCH_H_

#include <cstdint>

// Guest write tracking with userfaultfd write-protection (Linux), KYTY_UFFD_WP.
//
// The GPU caches write-protect the guest pages they mirror. On Linux that used mprotect, and every
// mprotect call takes the process's mmap lock for writing and splits or merges VMAs: each page
// fault of every thread waits behind it, and the tracking faults themselves need one more mprotect
// to resolve. At Astro Bot's ~1,300 tracking faults and ~900 protection calls per frame that convoy
// made one fault cost ~80 us on a Linux PC (Windows: ~5 us).
//
// With userfaultfd write-protection (UFFDIO_REGISTER_MODE_WP) the protection is a bit in each page
// table entry: UFFDIO_WRITEPROTECT takes the mmap lock only for reading and changes no VMA, and a
// write to a protected page raises SIGBUS in the writing thread (UFFD_FEATURE_SIGBUS), which the
// tracker resolves exactly like the SIGSEGV of a read-only page (the error code says "write").
// No-access protection (GPU-owned pages) still uses mprotect.
//
// Measured in WSL2 (15 writer threads, 4,800 tracked pages written per frame): a tracking fault
// costs 1.4-3 us instead of 26-97 us with mprotect, but on memfd views every first write after a
// release also takes a minor (write-notify) fault, ~4,700 per frame, so the total (6-7 ms of fault
// time per frame) loses to mprotect with the larger fault-ahead windows (0.9-2 ms).
// KYTY_UFFD_WP=1 turns it on (default off until it has run in the game on Linux; 0 = mprotect).
// Requirements, probed once at the first use and logged ("Kyty platform: userfaultfd ..."):
//  - the userfaultfd syscall. When vm.unprivileged_userfaultfd is 0, an unprivileged process gets
//    a user-mode-only descriptor (UFFD_USER_MODE_ONLY, Linux 5.11+). Kernel-mode writes to a
//    protected page (a read() into it) then fail with EFAULT, exactly as with mprotect;
//  - SIGBUS mode (UFFD_FEATURE_SIGBUS, Linux 4.14+);
//  - write-protection of shared memory (UFFD_FEATURE_WP_HUGETLBFS_SHMEM, Linux 5.19+): guest direct
//    memory is a memfd mapped MAP_SHARED;
//  - for private (anonymous) guest memory also UFFD_FEATURE_WP_UNPOPULATED (Linux 6.4+), without
//    which a page never touched would not fault; such mappings keep mprotect otherwise.
// Every failure falls back to mprotect (logged once).
namespace Common::UffdWriteWatch {

struct Support {
	bool     probed        = false;
	bool     requested     = false; // KYTY_UFFD_WP=1
	bool     shared        = false; // shared-memory mappings can be tracked
	bool     private_anon  = false; // private anonymous mappings can be tracked
	bool     user_mode     = false; // the descriptor is user-mode only
	uint64_t features      = 0;     // what the kernel offers (UFFDIO_API)
	int      error         = 0;     // errno of the failed step, if any
	int      fd            = -1;
};

// Probes once (thread-safe) and logs the result when KYTY_UFFD_WP=1.
[[nodiscard]] const Support& Get();
// KYTY_UFFD_WP=1 and at least shared-memory tracking works.
[[nodiscard]] bool Enabled();

// UFFDIO_REGISTER with UFFDIO_REGISTER_MODE_WP over [vaddr, vaddr + size) (whole mappings).
[[nodiscard]] bool Register(uint64_t vaddr, uint64_t size);
// UFFDIO_WRITEPROTECT: protect (true) or release (false) [vaddr, vaddr + size). Retries EAGAIN.
// `quiet`: a failure is expected to be retried in parts (not logged or counted).
[[nodiscard]] bool WriteProtect(uint64_t vaddr, uint64_t size, bool protect, bool quiet = false);

// Counters for logs and tests.
struct Stats {
	uint64_t registers       = 0;
	uint64_t protects        = 0;
	uint64_t releases        = 0;
	uint64_t failures        = 0;
};
[[nodiscard]] Stats GetStats();

namespace Testing {
// Probes again with KYTY_UFFD_WP forced on (tests).
void ForceEnable();
} // namespace Testing

} // namespace Common::UffdWriteWatch

#endif // EMULATOR_SRC_COMMON_PLATFORM_UFFDWRITEWATCH_H_
