#version 450

// Presents one mip level of a prepared frame to a smaller swapchain image (see PresentFilter),
// or converts an HDR10 frame for an SDR window.
// Each target pixel averages a two-texel box of the source on every downscaled axis, which
// cancels one-texel dither patterns at any phase; an axis that is not downscaled interpolates
// linearly like a blit.

layout(location = 0) out vec4 color;
layout(binding = 0) uniform sampler2D source; // linear, clamp to edge
layout(push_constant) uniform Downscale {
	ivec2 size;  // extent of the source level
	vec2  scale; // source texels per target pixel
	int   level;
	uint  flags;      // bit 0: the source is HDR10 (PQ, BT.2020) and the target SDR; bit 1: one
	                  // linear tap per pixel, as a blit (a downscale beyond 2:1, which the box cannot cover)
	float white_nits; // SDR reference white for the PQ conversion
	float peak_nits;  // highlight luminance mapped to SDR white (soft shoulder)
};

// HDR10 -> SDR for a game that outputs PQ while the window is SDR: the ST 2084 EOTF, the
// BT.2020 -> BT.709 primaries, an extended Reinhard shoulder on luminance (white_nits maps to 1,
// peak_nits rolls off to 1) and the sRGB encoding of the UNORM swapchain.
vec3 PqToSdr(vec3 pq) {
	const float m1 = 2610.0 / 16384.0;
	const float m2 = 2523.0 / 32.0;
	const float c1 = 3424.0 / 4096.0;
	const float c2 = 2413.0 / 128.0;
	const float c3 = 2392.0 / 128.0;
	vec3  e      = pow(clamp(pq, 0.0, 1.0), vec3(1.0 / m2));
	vec3  nits   = 10000.0 * pow(max(e - c1, 0.0) / (c2 - c3 * e), vec3(1.0 / m1));
	const mat3 to_rec709 = mat3(1.660491, -0.124550, -0.018151,
	                            -0.587641, 1.132900, -0.100579,
	                            -0.072850, -0.008349, 1.118730);
	vec3  rgb    = max(to_rec709 * (nits / white_nits), 0.0);
	float l      = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
	float w      = max(peak_nits / white_nits, 1.0);
	float mapped = l * (1.0 + l / (w * w)) / (1.0 + l);
	rgb          = l > 0.0 ? rgb * (mapped / l) : rgb;
	rgb          = clamp(rgb, 0.0, 1.0);
	return mix(rgb * 12.92, 1.055 * pow(rgb, vec3(1.0 / 2.4)) - 0.055,
	           greaterThan(rgb, vec3(0.0031308)));
}

// The box [c - 1, c + 1] (source texels) covers texels t, t + 1 and t + 2 with weights
// (1 - f) / 2, 1 / 2 and f / 2, where t + f = c - 1. One linear tap at a reads the first two,
// one at b the third. Positions are in texels, weights sum to one.
void Taps(float c, float s, out float a, out float wa, out float b, out float wb) {
	if (s > 1.0) {
		const float t = floor(c - 1.0);
		const float f = c - 1.0 - t;
		a             = t + 0.5 + 1.0 / (2.0 - f);
		wa            = 1.0 - 0.5 * f;
		b             = t + 2.5;
		wb            = 0.5 * f;
	} else {
		a  = c;
		wa = 1.0;
		b  = c;
		wb = 0.0;
	}
}

void main() {
	const vec2 c = gl_FragCoord.xy * scale;
	float      ax, wax, bx, wbx, ay, way, by, wby;
	Taps(c.x, scale.x, ax, wax, bx, wbx);
	Taps(c.y, scale.y, ay, way, by, wby);
	const vec2  texel = 1.0 / vec2(size);
	const float lod   = float(level);
	if ((flags & 2u) != 0u) {
		color = textureLod(source, c * texel, lod);
	} else {
		color = (textureLod(source, vec2(ax, ay) * texel, lod) * wax +
		         textureLod(source, vec2(bx, ay) * texel, lod) * wbx) *
		            way +
		        (textureLod(source, vec2(ax, by) * texel, lod) * wax +
		         textureLod(source, vec2(bx, by) * texel, lod) * wbx) *
		            wby;
	}
	if ((flags & 1u) != 0u) {
		color.rgb = PqToSdr(color.rgb);
		color.a   = 1.0;
	}
}
