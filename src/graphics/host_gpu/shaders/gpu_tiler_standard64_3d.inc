#include "gpu_tiler_standard4_3d.inc"
uvec3 standard64_3d_extent() {
	if (ELEMENT_BYTES == 1) return uvec3(64, 32, 32);
	if (ELEMENT_BYTES == 2) return uvec3(32, 32, 32);
	if (ELEMENT_BYTES == 4) return uvec3(32, 32, 16);
	if (ELEMENT_BYTES == 8) return uvec3(32, 16, 16);
	return uvec3(16, 16, 16);
}
uint standard64_3d_offset(uvec3 p) {
	// Scalar selectors avoid vector specialization-constant Select on AMD's Windows compiler.
	uint sx = ELEMENT_BYTES == 1 ? 4u : ELEMENT_BYTES == 16 ? 2u : 3u;
	uint sz = ELEMENT_BYTES <= 2 ? 4u : 3u;
	uint sy = ELEMENT_BYTES <= 4 ? 4u : 3u;
	uint sw = ELEMENT_BYTES == 1 ? 5u : ELEMENT_BYTES == 16 ? 3u : 4u;
	return standard3d_offset(p) ^ (((p.x >> sx) & 1u) << 12) ^
	       (((p.z >> sz) & 1u) << 13) ^ (((p.y >> sy) & 1u) << 14) ^
	       (((p.x >> sw) & 1u) << 15);
}
