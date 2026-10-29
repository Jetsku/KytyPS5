#ifndef EMULATOR_SRC_LIBS_WHEEL_HOST_H_
#define EMULATOR_SRC_LIBS_WHEEL_HOST_H_

#include "common/common.h"
#include "common/wheelConfig.h"
#include "libs/wheelFfb.h"

// The host side of the emulated racing wheel: the SDL joysticks bound in wheel.json give the
// input, and the game's force feedback is played on an SDL haptic device. The guest side (what
// the game is shown) reads Sample() and posts effects; it never calls SDL itself.
namespace Libs::Wheel {

void Initialize();
void Shutdown();
void EmergencyShutdown();

struct Lifecycle {
	static constexpr const char* name               = "Wheel";
	static constexpr auto        initialize         = Libs::Wheel::Initialize;
	static constexpr auto        shutdown           = Libs::Wheel::Shutdown;
	static constexpr auto        emergency_shutdown = Libs::Wheel::EmergencyShutdown;
};

struct Input {
	// The device bound to steering is connected. While it is not, everything else is neutral.
	bool present = false;
	// -1 full left .. 1 full right, at the game's lock when the game set one.
	float steering = 0.0f;
	// The rim's position over the whole physical travel, before dead zone and curve.
	float steering_raw = 0.0f;
	// 0 released .. 1 fully pressed.
	float throttle  = 0.0f;
	float brake     = 0.0f;
	float clutch    = 0.0f;
	float handbrake = 0.0f;
	// Bit (control - FIRST_BUTTON) per button control.
	uint64_t buttons = 0;
	// 0 neutral, 1-7, -1 reverse.
	int gear = 0;

	[[nodiscard]] bool Pressed(Config::Wheel::Control c) const {
		return (buttons >> (static_cast<int>(c) - Config::Wheel::FIRST_BUTTON) & 1u) != 0;
	}
};

constexpr int MAX_EFFECT_SLOTS = 16;

bool Enabled();
bool Present();
// Whether a host device of this vendor and product is bound to the wheel; such a device is not
// also used as a gamepad.
bool IsBoundDevice(uint16_t vendor_id, uint16_t product_id);

// Thread-safe.
Input Sample();

// The lock the game asks for, in degrees lock to lock.
void SetGameRotation(uint32_t degrees);
// The rotation set in the wheel's own software; 0 when the wheel is disabled.
uint32_t GetPhysicalRotation();

// Applied on the wheel thread; the latest request for a slot wins.
void SetEffect(int slot, const Effect& effect);
void PlayEffect(int slot, bool play);
void StopAllEffects();

} // namespace Libs::Wheel

#endif /* EMULATOR_SRC_LIBS_WHEEL_HOST_H_ */
