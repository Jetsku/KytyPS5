#ifndef EMULATOR_INCLUDE_EMULATOR_AUDIO_OBJECTS_H_
#define EMULATOR_INCLUDE_EMULATOR_AUDIO_OBJECTS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

// AudioOut2 object ports (port type 0x1xx): mono sound sources that the console's 3D audio
// renderer places with the attributes the game sets on each port. The emulator has no HRTF; it
// pans every object into the context's main bed (2, 8 or 12 channels) before that bed is played.
//
// Attribute ids (from Astro Bot's calls, 2026-10-07; they differ from PS4 Audio3d's): 0 PCM,
// 1 gain (float, 1.0 by default), 2 priority (uint32, 100), 3 position (3 floats, (0, 0, 1) =
// straight ahead), 4 spread (float, 0), 5 passthrough (uint32: 1 left, 2 right; Astro's stereo
// pair on ports 3/4), 8 ambisonics (uint32, 0x40 | ACN; Astro feeds a third-order scene, ACN 0-15).
namespace Libs::Audio::Objects {

constexpr uint32_t ATTRIBUTE_PCM         = 0;
constexpr uint32_t ATTRIBUTE_GAIN        = 1;
constexpr uint32_t ATTRIBUTE_PRIORITY    = 2;
constexpr uint32_t ATTRIBUTE_POSITION    = 3;
constexpr uint32_t ATTRIBUTE_SPREAD      = 4;
constexpr uint32_t ATTRIBUTE_PASSTHROUGH = 5;
constexpr uint32_t ATTRIBUTE_AMBISONICS  = 8;

constexpr uint32_t PASSTHROUGH_NONE  = 0;
constexpr uint32_t PASSTHROUGH_LEFT  = 1;
constexpr uint32_t PASSTHROUGH_RIGHT = 2;

// An ambisonics channel is tagged 0x40 | ACN (SN3D); other values mean "a plain object".
constexpr uint32_t AMBISONICS_CHANNEL = 0x40;

constexpr uint32_t MAX_CHANNELS = 12;
constexpr float    HALF_POWER   = 0.70710677f;
constexpr float    PI           = 3.14159265f;

struct Params {
	float    gain         = 1.0f;
	bool     has_position = false;
	float    x            = 0.0f; // right
	float    y            = 0.0f; // up
	float    z            = 0.0f; // front
	float    spread       = 0.0f; // radians
	uint32_t passthrough  = PASSTHROUGH_NONE;
	uint32_t ambisonics   = 0xffffffffu;
};

// Applies one attribute to *params; returns false for attributes that do not affect the mix
// (PCM, priority, unknown ids) or values of an unexpected size.
inline bool ApplyAttribute(Params* params, uint32_t id, const void* value, uint64_t size) {
	if (params == nullptr || value == nullptr) {
		return false;
	}
	switch (id) {
		case ATTRIBUTE_POSITION: {
			if (size < 3 * sizeof(float)) {
				return false;
			}
			float xyz[3] {};
			std::memcpy(xyz, value, sizeof(xyz));
			if (!std::isfinite(xyz[0]) || !std::isfinite(xyz[1]) || !std::isfinite(xyz[2])) {
				return false;
			}
			params->x            = xyz[0];
			params->y            = xyz[1];
			params->z            = xyz[2];
			params->has_position = true;
			return true;
		}
		case ATTRIBUTE_SPREAD:
		case ATTRIBUTE_GAIN: {
			if (size != sizeof(float)) {
				return false;
			}
			float v = 0.0f;
			std::memcpy(&v, value, sizeof(float));
			if (!std::isfinite(v) || v < 0.0f) {
				return false;
			}
			(id == ATTRIBUTE_GAIN ? params->gain : params->spread) = v;
			return true;
		}
		case ATTRIBUTE_PASSTHROUGH:
		case ATTRIBUTE_AMBISONICS: {
			if (size != sizeof(uint32_t)) {
				return false;
			}
			uint32_t v = 0;
			std::memcpy(&v, value, sizeof(uint32_t));
			(id == ATTRIBUTE_PASSTHROUGH ? params->passthrough : params->ambisonics) = v;
			return true;
		}
		default: return false;
	}
}

inline bool IsAmbisonicsChannel(uint32_t ambisonics) {
	return (ambisonics & ~0x3fu) == AMBISONICS_CHANNEL;
}

// Per-channel gains for one object in a bed of `channels` channels (2, 8 or 12; a mono bed gets
// the plain gain). Returns false when the object adds nothing (silent, or an ambisonics order the
// stereo decode ignores).
//
// - Plain objects: constant-power pan from the azimuth (x right, z front; y is ignored). On a bed
//   with surrounds, objects behind the listener move to the side/back pairs, split evenly over
//   channels 4/5 and 6/7, which both 8-channel orders use as left/right pairs; downmixed to stereo
//   (fronts 1, the rest 0.707) that keeps the object's level. Spread widens towards the centre
//   (pi radians and more: fully diffuse). No position: centre (0.707 to each front channel).
// - Passthrough left/right: that front channel only.
// - Ambisonics W/Y (ACN 0/1): first-order decode with two cardioids facing left and right; higher
//   orders are skipped (summing them into the fronts cancels most directions).
inline bool SpeakerGains(const Params& params, uint32_t channels, float* gains) {
	std::fill(gains, gains + std::min(channels, MAX_CHANNELS), 0.0f);
	if (channels == 0 || channels > MAX_CHANNELS || params.gain <= 0.0f) {
		return false;
	}
	if (channels == 1) {
		gains[0] = params.gain;
		return true;
	}
	float left  = HALF_POWER;
	float right = HALF_POWER;
	float front = 1.0f;
	float back  = 0.0f;
	if (IsAmbisonicsChannel(params.ambisonics)) {
		const auto acn = params.ambisonics & 0x3fu;
		if (acn == 1) {
			right = -HALF_POWER;
		} else if (acn != 0) {
			return false;
		}
	} else if (params.passthrough == PASSTHROUGH_LEFT || params.passthrough == PASSTHROUGH_RIGHT) {
		left  = params.passthrough == PASSTHROUGH_LEFT ? 1.0f : 0.0f;
		right = 1.0f - left;
	} else if (params.has_position) {
		const float horizontal = std::sqrt(params.x * params.x + params.z * params.z);
		if (horizontal > 1e-6f) {
			const float width = std::clamp(1.0f - params.spread / PI, 0.0f, 1.0f);
			const float side  = std::clamp(params.x / horizontal, -1.0f, 1.0f) * width;
			const float ahead = std::clamp(params.z / horizontal, -1.0f, 1.0f) * width;
			const float pan   = (side + 1.0f) * (PI / 4.0f); // 0 = left, pi/2 = right
			left              = std::cos(pan);
			right             = std::sin(pan);
			if (channels >= 8) {
				front = std::sqrt((1.0f + ahead) / 2.0f);
				back  = std::sqrt((1.0f - ahead) / 2.0f);
			}
		}
	}
	gains[0] = left * front * params.gain;
	gains[1] = right * front * params.gain;
	if (channels >= 8 && back > 0.0f) {
		const float pair = back * HALF_POWER * params.gain;
		gains[4]         = left * pair;
		gains[5]         = right * pair;
		gains[6]         = left * pair;
		gains[7]         = right * pair;
	}
	return true;
}

// Adds `gain` x mono into an interleaved float bed through per-channel `gains`.
inline void MixInto(float* bed, uint32_t channels, uint32_t frames, const float* mono,
                    const float* gains, float gain) {
	float g[MAX_CHANNELS] {};
	bool  any = false;
	for (uint32_t ch = 0; ch < channels && ch < MAX_CHANNELS; ch++) {
		g[ch] = gains[ch] * gain;
		any   = any || g[ch] != 0.0f;
	}
	if (!any) {
		return;
	}
	for (uint32_t frame = 0; frame < frames; frame++) {
		const float sample = mono[frame];
		if (!std::isfinite(sample)) {
			continue;
		}
		float* out = bed + static_cast<size_t>(frame) * channels;
		for (uint32_t ch = 0; ch < channels && ch < MAX_CHANNELS; ch++) {
			out[ch] += sample * g[ch];
		}
	}
}

} // namespace Libs::Audio::Objects

#endif // EMULATOR_INCLUDE_EMULATOR_AUDIO_OBJECTS_H_
