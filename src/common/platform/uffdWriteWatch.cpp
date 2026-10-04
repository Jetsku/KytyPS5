#include "common/platform/uffdWriteWatch.h"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#if defined(__linux__) && !defined(__APPLE__) && __has_include(<linux/userfaultfd.h>)
#define KYTY_UFFD_WRITE_WATCH 1
#include <fcntl.h>
#include <linux/userfaultfd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Common::UffdWriteWatch {
namespace {

std::mutex          g_mutex;
Support             g_support;
std::atomic<bool>   g_ready {false};
bool                g_force = false;
std::atomic<uint64_t> g_registers {0};
std::atomic<uint64_t> g_protects {0};
std::atomic<uint64_t> g_releases {0};
std::atomic<uint64_t> g_failures {0};
std::atomic<uint32_t> g_failure_logs {0};

bool Requested() {
	if (g_force) {
		return true;
	}
	const auto* value = std::getenv("KYTY_UFFD_WP");
	return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

void LogFailure(const char* what, uint64_t vaddr, uint64_t size, int error) {
	g_failures.fetch_add(1, std::memory_order_relaxed);
	if (g_failure_logs.fetch_add(1, std::memory_order_relaxed) < 8) {
		std::fprintf(stderr, "UffdWriteWatch: %s failed at 0x%llx+0x%llx: %s (falling back to mprotect)\n", what,
		             static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size),
		             std::strerror(error));
		std::fflush(stderr);
	}
}

#if defined(KYTY_UFFD_WRITE_WATCH)
#ifndef UFFD_USER_MODE_ONLY
#define UFFD_USER_MODE_ONLY 1
#endif
// Feature bits by number: older headers lack the newer names.
constexpr uint64_t FeatureSigbus         = 1ull << 7u;
constexpr uint64_t FeatureWpShmem        = 1ull << 12u;
constexpr uint64_t FeatureWpUnpopulated  = 1ull << 13u;

void Probe(Support& s) {
	int fd = static_cast<int>(syscall(__NR_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY));
	s.user_mode = fd >= 0;
	if (fd < 0) {
		fd = static_cast<int>(syscall(__NR_userfaultfd, O_CLOEXEC | O_NONBLOCK));
	}
	if (fd < 0) {
		s.error = errno;
		return;
	}
	// The first UFFDIO_API call fixes the features of a descriptor: ask what the kernel offers on a
	// throwaway descriptor, then enable what is needed on the real one.
	uffdio_api api {};
	api.api = UFFD_API;
	if (ioctl(fd, UFFDIO_API, &api) != 0) {
		s.error = errno;
		close(fd);
		return;
	}
	close(fd);
	s.features = api.features;
	if ((s.features & FeatureSigbus) == 0 || (s.features & FeatureWpShmem) == 0) {
		s.error = ENOTSUP;
		return;
	}
	fd = static_cast<int>(syscall(__NR_userfaultfd, O_CLOEXEC | O_NONBLOCK | (s.user_mode ? UFFD_USER_MODE_ONLY : 0)));
	if (fd < 0) {
		s.error = errno;
		return;
	}
	uffdio_api enable {};
	enable.api      = UFFD_API;
	enable.features = FeatureSigbus | FeatureWpShmem | (s.features & FeatureWpUnpopulated);
	if (ioctl(fd, UFFDIO_API, &enable) != 0) {
		s.error = errno;
		close(fd);
		return;
	}
	s.fd           = fd;
	s.shared       = true;
	s.private_anon = (s.features & FeatureWpUnpopulated) != 0;
}
#endif

void EnsureProbed() {
	if (g_ready.load(std::memory_order_acquire)) {
		return;
	}
	std::scoped_lock lock(g_mutex);
	if (g_ready.load(std::memory_order_relaxed)) {
		return;
	}
	Support s;
	s.probed    = true;
	s.requested = Requested();
#if defined(KYTY_UFFD_WRITE_WATCH)
	if (s.requested) {
		Probe(s);
		std::printf("Kyty platform: guest write tracking with userfaultfd write-protection (KYTY_UFFD_WP=1): %s "
		            "(kernel features 0x%llx, %s descriptor%s%s)\n",
		            s.shared ? "on" : "unavailable, using mprotect", static_cast<unsigned long long>(s.features),
		            s.user_mode ? "user-mode-only" : "full", s.private_anon ? ", private memory too" : ", shared memory only",
		            s.error != 0 ? (std::string(", error: ") + std::strerror(s.error)).c_str() : "");
		std::fflush(stdout);
	}
#endif
	g_support = s;
	g_ready.store(true, std::memory_order_release);
}

} // namespace

const Support& Get() {
	EnsureProbed();
	return g_support;
}

bool Enabled() {
	const auto& s = Get();
	return s.requested && s.shared && s.fd >= 0;
}

bool Register(uint64_t vaddr, uint64_t size) {
#if defined(KYTY_UFFD_WRITE_WATCH)
	const auto& s = Get();
	if (s.fd < 0) {
		return false;
	}
	uffdio_register reg {};
	reg.range.start = vaddr;
	reg.range.len   = size;
	reg.mode        = UFFDIO_REGISTER_MODE_WP;
	if (ioctl(s.fd, UFFDIO_REGISTER, &reg) != 0) {
		LogFailure("UFFDIO_REGISTER", vaddr, size, errno);
		return false;
	}
	g_registers.fetch_add(1, std::memory_order_relaxed);
	return true;
#else
	(void)vaddr;
	(void)size;
	return false;
#endif
}

bool WriteProtect(uint64_t vaddr, uint64_t size, bool protect, bool quiet) {
#if defined(KYTY_UFFD_WRITE_WATCH)
	const auto& s = Get();
	if (s.fd < 0) {
		return false;
	}
	uffdio_writeprotect wp {};
	wp.range.start = vaddr;
	wp.range.len   = size;
	wp.mode        = protect ? UFFDIO_WRITEPROTECT_MODE_WP : 0;
	for (int attempt = 0; attempt < 1000; attempt++) {
		if (ioctl(s.fd, UFFDIO_WRITEPROTECT, &wp) == 0) {
			(protect ? g_protects : g_releases).fetch_add(1, std::memory_order_relaxed);
			return true;
		}
		if (errno != EAGAIN) {
			break;
		}
	}
	if (!quiet) {
		LogFailure(protect ? "UFFDIO_WRITEPROTECT(protect)" : "UFFDIO_WRITEPROTECT(release)", vaddr, size, errno);
	}
	return false;
#else
	(void)vaddr;
	(void)size;
	(void)protect;
	(void)quiet;
	return false;
#endif
}

Stats GetStats() {
	Stats stats;
	stats.registers = g_registers.load(std::memory_order_relaxed);
	stats.protects  = g_protects.load(std::memory_order_relaxed);
	stats.releases  = g_releases.load(std::memory_order_relaxed);
	stats.failures  = g_failures.load(std::memory_order_relaxed);
	return stats;
}

namespace Testing {
void ForceEnable() {
	std::scoped_lock lock(g_mutex);
	g_force = true;
	g_ready.store(false, std::memory_order_release);
}
} // namespace Testing

} // namespace Common::UffdWriteWatch
