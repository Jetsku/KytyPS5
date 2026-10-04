#pragma once

#include <cstdint>
#include <span>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

// The compiler must consume this owned snapshot, not the guest span used to identify it.
template <typename Reader>
bool CopyVerifiedShaderCode(std::span<const uint32_t> guest, uint64_t expected_hash,
                            std::vector<uint32_t>& owned, Reader&& read) {
	if (guest.empty()) return false;
	owned.resize(guest.size());
	return read(guest, std::span<uint32_t>(owned)) &&
	       XXH3_64bits(owned.data(), owned.size() * sizeof(uint32_t)) == expected_hash;
}

} // namespace Libs::Graphics
