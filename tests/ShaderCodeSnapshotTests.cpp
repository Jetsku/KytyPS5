#include "graphics/host_gpu/renderer/pipeline/shaderCodeSnapshot.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

int main() {
	using Libs::Graphics::CopyVerifiedShaderCode;
	auto require = [](bool value) { if (!value) std::abort(); };
	std::vector<uint32_t> guest {1, 2, 3, 4};
	const auto original = guest;
	const auto hash = XXH3_64bits(guest.data(), guest.size() * sizeof(uint32_t));
	std::vector<uint32_t> owned;
	auto copy = [](auto source, auto target) {
		std::copy(source.begin(), source.end(), target.begin()); return true;
	};
	// A write after the read cannot change the compiler's code or its source identity.
	require(CopyVerifiedShaderCode(guest, hash, owned, [&](auto source, auto target) {
		copy(source, target); guest[0] = 99; return true;
	}));
	require(owned == original && owned != guest);
	// A write before the read must not publish the new bytes under the old source key.
	require(!CopyVerifiedShaderCode(guest, hash, owned, copy));
	guest = original;
	require(!CopyVerifiedShaderCode(guest, hash, owned, [](auto, auto) { return false; }));
	require(!CopyVerifiedShaderCode(guest, hash ^ 1, owned, copy));
	require(!CopyVerifiedShaderCode({}, hash, owned, copy));
	std::puts("shader code snapshots: stable ownership and stale-key rejection passed");
}
