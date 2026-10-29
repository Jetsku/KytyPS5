#ifndef EMULATOR_SRC_LIBS_WHEEL_FFB_H_
#define EMULATOR_SRC_LIBS_WHEEL_FFB_H_

#include "common/common.h"
#include "common/wheelConfig.h"

#include <SDL3/SDL_haptic.h>

namespace Libs::Wheel {

enum class EffectType { None, Constant, Spring, Damper, Friction, Inertia };

// A force-feedback effect as the emulated wheel receives it, independent of the game's protocol.
// Levels, coefficients and saturations are fractions of full force, positions fractions of half
// the travel (-1 full left, 1 full right). Effects play until replaced or stopped.
struct Effect {
	EffectType type = EffectType::None;

	// Constant
	float level = 0.0f;

	// Spring, damper, friction, inertia
	float center      = 0.0f;
	float dead_band   = 0.0f;
	float left_coeff  = 0.0f;
	float right_coeff = 0.0f;
	float left_sat    = 1.0f;
	float right_sat   = 1.0f;

	bool operator==(const Effect& other) const = default;
};

struct FfbSettings {
	float                       gain      = 1.0f;
	Config::Wheel::FfbDirection direction = Config::Wheel::FfbDirection::SteeringAxis;
	// Mirrors the game's forces: its convention combined with the host wheel's sense.
	bool reverse = false;
	// Mirrors forces computed here from the rim's position as the steering binding reads it: the
	// wheel's measured sense combined with the binding's reverse.
	bool device_reverse = false;
	// The device plays conditions on its raw axis: mirrored when the steering binding reverses it.
	bool mirror_conditions = false;
};

bool IsCondition(EffectType type);

// The SDL feature bit an effect type needs.
uint32_t SdlFeatureFor(EffectType type);

// The effect as SDL plays it, strength and direction applied; false for EffectType::None.
bool ToSdlEffect(const Effect& effect, const FfbSettings& settings, SDL_HapticEffect* out);

// The force (-1..1) an effect asks for at a rim position and speed (half travels a second), for
// effects the device cannot play itself; strength and sense applied.
float RenderEffect(const Effect& effect, const FfbSettings& settings, float position,
                   float velocity);

// The soft end stop: nothing inside the lock, a stiff damped spring beyond it. lock is where the
// game's lock is, as a fraction of half the travel.
float RenderSoftLock(float position, float velocity, float lock);

} // namespace Libs::Wheel

#endif /* EMULATOR_SRC_LIBS_WHEEL_FFB_H_ */
