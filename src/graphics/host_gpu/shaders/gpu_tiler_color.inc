// ColorTransform values match imageInfo.h. Preserve the floating-point bit fields exactly
// (transform 3 narrows one mantissa bit, as imageInfo.h's NarrowSmallFloat).

uint narrow_small_float(uint bits, uint mantissa_bits) {
	uint narrow = bits >> 1u;
	uint mask   = (1u << mantissa_bits) - 1u;
	if ((bits >> mantissa_bits) == 0x1fu && (bits & mask) != 0u && (narrow & (mask >> 1u)) == 0u) {
		narrow |= 1u;
	}
	return narrow;
}

uvec2 transform_color(uvec2 value, uint transform, bool to_host) {
	switch (transform) {
		case 1u:
			return uvec2((value.y & 0xffffu) | (value.x & 0xffff0000u),
			             (value.x & 0xffffu) | (value.y & 0xffff0000u));
		case 2u: {
			uint packed = value.x;
			if (to_host) {
				value.x = (packed >> 21u) | (((packed >> 10u) & 0x7ffu) << 11u) |
				          ((packed & 0x3ffu) << 22u);
			} else {
				value.x = (packed >> 22u) | (((packed >> 11u) & 0x7ffu) << 10u) |
				          ((packed & 0x7ffu) << 21u);
			}
			break;
		}
		case 3u: {
			// KYTY_10_11_11_RED: each component keeps its place (Guest10_11_11ToHost,
			// Host10_11_11ToGuest).
			uint packed = value.x;
			if (to_host) {
				value.x = ((packed & 0x3ffu) << 1u) | (((packed >> 10u) & 0x7ffu) << 11u) |
				          (narrow_small_float(packed >> 21u, 6u) << 22u);
			} else {
				value.x = narrow_small_float(packed & 0x7ffu, 6u) |
				          (((packed >> 11u) & 0x7ffu) << 10u) | (((packed >> 22u) << 1u) << 21u);
			}
			break;
		}
		case 4u:
			// KYTY_2_10_10_10_ROTATE (upstream 963aaeb3b): the guest's 2-bit component moves to the
			// host's alpha bits at the top.
			if (to_host) {
				value.x = (value.x >> 2u) | (value.x << 30u);
			} else {
				value.x = (value.x << 2u) | (value.x >> 30u);
			}
			break;
	}
	return value;
}
