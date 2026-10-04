#ifndef EMULATOR_SRC_COMMON_RAMSTATS_H_
#define EMULATOR_SRC_COMMON_RAMSTATS_H_

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Optional address tags for a read-only external RAM sampler. Extents are allocation capacities,
// not resident byte counts. Guest aliases and host-visible device memory must not be double-counted.
namespace Common::RamStats {

inline bool Enabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_RAM_STATS");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

inline void Range(const char* owner, const void* base, uint64_t bytes) {
	if (Enabled() && base != nullptr && bytes != 0) {
		std::printf("RAM range: %s base=0x%016" PRIx64 " bytes=%" PRIu64 "\n", owner,
		            reinterpret_cast<uint64_t>(base), bytes);
	}
}

} // namespace Common::RamStats

#endif
