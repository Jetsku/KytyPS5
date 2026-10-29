#include "libs/wheelFfb.h"

#include <algorithm>
#include <cmath>

namespace Libs::Wheel {

using Config::Wheel::FfbDirection;

namespace {

int16_t ToLevel(float value) {
	return static_cast<int16_t>(std::lround(std::clamp(value, -1.0f, 1.0f) * 32767.0f));
}

uint16_t ToUnsigned(float value) {
	return static_cast<uint16_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 65535.0f));
}

// The restoring force of a condition at a position (spring) or speed (damper).
float Condition(const Effect& e, float metric) {
	const float d = metric - e.center;
	if (std::abs(d) <= e.dead_band) {
		return 0.0f;
	}
	if (d > 0.0f) {
		return std::clamp(-e.right_coeff * (d - e.dead_band), -e.right_sat, e.right_sat);
	}
	return std::clamp(-e.left_coeff * (d + e.dead_band), -e.left_sat, e.left_sat);
}

// A choice because some Windows DirectInput drivers ignore SDL_HAPTIC_STEERING_AXIS (SDL issue
// 7941; RPCS3 #18776 added the same choice for a MOZA R3).
SDL_HapticDirection MakeSteeringDirection(FfbDirection direction) {
	SDL_HapticDirection dir {};
	switch (direction) {
		case FfbDirection::SteeringAxis: dir.type = SDL_HAPTIC_STEERING_AXIS; break;
		case FfbDirection::Cartesian:
			dir.type   = SDL_HAPTIC_CARTESIAN;
			dir.dir[0] = 1;
			break;
		case FfbDirection::Polar: dir.type = SDL_HAPTIC_POLAR; break;
	}
	return dir;
}

} // namespace

bool IsCondition(EffectType type) {
	return type == EffectType::Spring || type == EffectType::Damper ||
	       type == EffectType::Friction || type == EffectType::Inertia;
}

uint32_t SdlFeatureFor(EffectType type) {
	switch (type) {
		case EffectType::Constant: return SDL_HAPTIC_CONSTANT;
		case EffectType::Spring: return SDL_HAPTIC_SPRING;
		case EffectType::Damper: return SDL_HAPTIC_DAMPER;
		case EffectType::Friction: return SDL_HAPTIC_FRICTION;
		case EffectType::Inertia: return SDL_HAPTIC_INERTIA;
		case EffectType::None: break;
	}
	return 0;
}

bool ToSdlEffect(const Effect& e, const FfbSettings& s, SDL_HapticEffect* out) {
	*out             = {};
	const float gain = std::clamp(s.gain, 0.0f, 1.0f);

	if (e.type == EffectType::Constant) {
		auto& c     = out->constant;
		c.type      = SDL_HAPTIC_CONSTANT;
		c.direction = MakeSteeringDirection(s.direction);
		c.length    = SDL_HAPTIC_INFINITY;
		c.level     = ToLevel((s.reverse ? -e.level : e.level) * gain);
		return true;
	}
	if (!IsCondition(e.type)) {
		return false;
	}
	auto& c     = out->condition;
	c.type      = static_cast<uint16_t>(SdlFeatureFor(e.type));
	c.direction = MakeSteeringDirection(s.direction);
	c.length    = SDL_HAPTIC_INFINITY;
	// Scaling the coefficients and the saturations together scales every force by the gain.
	const bool m     = s.mirror_conditions;
	c.right_coeff[0] = ToLevel((m ? e.left_coeff : e.right_coeff) * gain);
	c.left_coeff[0]  = ToLevel((m ? e.right_coeff : e.left_coeff) * gain);
	c.right_sat[0]   = ToUnsigned((m ? e.left_sat : e.right_sat) * gain);
	c.left_sat[0]    = ToUnsigned((m ? e.right_sat : e.left_sat) * gain);
	c.center[0]      = ToLevel(m ? -e.center : e.center);
	c.deadband[0]    = ToUnsigned(e.dead_band);
	return true;
}

float RenderEffect(const Effect& e, const FfbSettings& s, float position, float velocity) {
	const float gain = std::clamp(s.gain, 0.0f, 1.0f);
	switch (e.type) {
		case EffectType::Constant:
			return std::clamp((s.reverse ? -e.level : e.level) * gain, -1.0f, 1.0f);
		case EffectType::Spring:
		case EffectType::Damper:
		case EffectType::Friction: {
			// A force pushing the rim right, from its own position and speed: it follows the host
			// wheel's sense alone, never the game's convention. A condition always resists, so the
			// coefficients count by size. Damper and friction stop at half force, as the speed is a
			// difference of positions and stays noisy.
			Effect r      = e;
			r.left_coeff  = std::abs(e.left_coeff);
			r.right_coeff = std::abs(e.right_coeff);
			float force   = 0.0f;
			if (e.type == EffectType::Spring) {
				force = Condition(r, position);
			} else if (e.type == EffectType::Damper) {
				force = std::clamp(Condition(r, velocity), -0.5f, 0.5f);
			} else {
				// Against the motion, smoothed near standstill.
				const float k     = std::clamp(velocity * 20.0f, -1.0f, 1.0f);
				const float coeff = k >= 0.0f ? r.right_coeff : r.left_coeff;
				const float sat   = std::min(k >= 0.0f ? r.right_sat : r.left_sat, 0.5f);
				force             = std::clamp(-coeff * k, -sat, sat);
			}
			return std::clamp(force * (s.device_reverse ? -1.0f : 1.0f) * gain, -1.0f, 1.0f);
		}
		case EffectType::Inertia:
			// It would need the acceleration, a second difference of positions: too noisy.
			return 0.0f;
		case EffectType::None: break;
	}
	return 0.0f;
}

float RenderSoftLock(float position, float velocity, float lock) {
	const float over = std::abs(position) - lock;
	if (lock >= 1.0f || over <= 0.0f) {
		return 0.0f;
	}
	// Full force 3 % of the travel past the stop, damped so the rim does not bounce off it.
	constexpr float Stiffness = 1.0f / 0.03f;
	constexpr float Damping   = 0.3f;
	const float     dir       = position > 0.0f ? 1.0f : -1.0f;
	float           force     = -dir * over * Stiffness;
	if (velocity * dir > 0.0f) {
		force -= velocity * Damping;
	}
	return std::clamp(force, -1.0f, 1.0f);
}

} // namespace Libs::Wheel
