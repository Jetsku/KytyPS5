#ifndef KYTY_COMMON_WHEEL_CONFIG_H_
#define KYTY_COMMON_WHEEL_CONFIG_H_

#include "common/common.h"

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

// The racing wheel's settings: which host joystick input drives each control of the emulated
// wheel, and how its force feedback is played. Stored as wheel.json.
namespace Config::Wheel {

constexpr int CONFIG_VERSION = 1;

// The wheels the game can be shown.
constexpr std::string_view MODEL_GT_DD_PRO = "gt_dd_pro";

// The emulated wheel's controls: its axes, the PlayStation buttons it reports (the shift paddles
// are L1 and R1) and the H-shifter's gears. Buttons follow the axes; a button's bit in
// Input::buttons is its index minus FIRST_BUTTON.
enum class Control {
	Steering,
	Throttle,
	Brake,
	Clutch,
	Handbrake,

	Cross,
	Circle,
	Square,
	Triangle,
	Up,
	Down,
	Left,
	Right,
	L1,
	R1,
	L2,
	R2,
	L3,
	R3,
	Options,
	TouchPad,
	Gear1,
	Gear2,
	Gear3,
	Gear4,
	Gear5,
	Gear6,
	Gear7,
	GearReverse,

	Count
};

constexpr int CONTROL_COUNT = static_cast<int>(Control::Count);
constexpr int FIRST_BUTTON  = static_cast<int>(Control::Cross);

constexpr bool IsAxisControl(Control c) {
	return static_cast<int>(c) < FIRST_BUTTON;
}

enum class BindingType { None, Axis, Button, Hat };
enum class HatDirection { None, Up, Down, Left, Right };
enum class FfbDirection { SteeringAxis, Cartesian, Polar };
enum class ForceSense { Unknown, Normal, Reversed };

// A host joystick, recognised across sessions by its vendor and product and its numbers of axes,
// hats and buttons; packed as RPCS3's emulated Logitech G27 packs it, so its mappings import.
struct DeviceId {
	uint16_t vendor_id   = 0;
	uint16_t product_id  = 0;
	uint16_t num_axes    = 0;
	uint16_t num_hats    = 0;
	uint16_t num_buttons = 0;

	[[nodiscard]] uint64_t AsU64() const;
	[[nodiscard]] bool     IsNone() const { return AsU64() == 0; }
	static DeviceId        FromU64(uint64_t value);

	bool operator==(const DeviceId& other) const { return AsU64() == other.AsU64(); }
};

struct Binding {
	DeviceId     device;
	BindingType  type  = BindingType::None;
	int          index = 0;
	HatDirection hat   = HatDirection::None;
	// Axis: the host value negated. Button and hat: pressed when released.
	bool reverse = false;
	// A button or hat driving an axis moves it to the positive end instead of the negative one.
	bool positive = false;
	// Axes only. Steering: a centred dead zone, as a fraction of half the travel. Pedals: travel
	// ignored at the released end, as a fraction of the whole travel.
	float dead_zone = 0.0f;
	// Axes only: output = input ^ curve after the dead zone; 1 is linear.
	float curve = 1.0f;

	[[nodiscard]] bool IsBound() const { return type != BindingType::None && !device.IsNone(); }
};

struct WheelConfig {
	// The wheel the game is shown; one model today, the place for more.
	std::string                        model {MODEL_GT_DD_PRO};
	std::array<Binding, CONTROL_COUNT> bindings {};
	// The joystick that plays force feedback; none: the steering axis's joystick.
	DeviceId ffb_device;
	// Force-feedback strength, in percent of what the game asks for.
	uint32_t     ffb_gain      = 50;
	FfbDirection ffb_direction = FfbDirection::SteeringAxis;
	// The game's forces are mirrored.
	bool reverse_effects = false;
	// Which way a positive force moves the steering axis as the device reports it (before the
	// binding's reverse), measured by the launcher's force test. Forces computed here (the end
	// stop, rendered damper and friction) need it and are off while it is unknown; once known,
	// the game's forces are mirrored where a positive force turns the wheel towards the high end
	// of the steering axis, combined with reverse_effects.
	ForceSense force_sense = ForceSense::Unknown;
	// Damper and friction computed here and played as a constant force, for wheel drivers that
	// mishandle those effects (on a MOZA R3, RyGT found a damper took away all force feedback).
	bool software_conditions = false;
	// The rotation set in the wheel's own software. When the game sets a car's lock, steering is
	// scaled so that lock is reached at game / physical of the travel, with a soft end stop there.
	uint32_t physical_rotation_deg = 900;
	bool     soft_lock             = true;
	// Keep reading the wheel while the game window has no focus. SDL applies this to every
	// controller, the pads too.
	bool background_input = false;

	[[nodiscard]] const Binding& Get(Control c) const { return bindings[static_cast<int>(c)]; }
	Binding&                     Get(Control c) { return bindings[static_cast<int>(c)]; }
};

std::string_view       ControlName(Control c);
std::optional<Control> ControlFromName(std::string_view name);
std::string_view       BindingTypeName(BindingType type);
std::string_view       HatDirectionName(HatDirection hat);
std::string_view       FfbDirectionName(FfbDirection direction);
std::string_view       ForceSenseName(ForceSense sense);

// --wheel and --wheel-config. Kept here rather than in ConfigOptions: emulatorConfig.h is part of
// the shader cache's source key, which the wheel must not change.
void                  SetEnabled(bool enabled);
bool                  IsEnabled();
void                  SetConfigPath(const std::filesystem::path& path);
std::filesystem::path GetConfigPath();

std::string ToJson(const WheelConfig& config);
// Unknown keys are ignored and missing ones keep their defaults; nothing when the text is not a
// JSON object or comes from a newer version.
std::optional<WheelConfig> FromJson(std::string_view text, std::string* error = nullptr);

std::optional<WheelConfig> Load(const std::filesystem::path& path, std::string* error = nullptr);
bool                       Save(const std::filesystem::path& path, const WheelConfig& config,
                                std::string* error = nullptr);

// RPCS3's emulated Logitech G27 mapping (config/LogitechG27.yml): its bindings, force-feedback
// device and direction encoding.
std::optional<WheelConfig> ImportRpcs3G27(std::string_view yaml, std::string* error = nullptr);

} // namespace Config::Wheel

#endif /* KYTY_COMMON_WHEEL_CONFIG_H_ */
