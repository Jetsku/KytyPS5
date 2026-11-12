#include "libs/wheelHost.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/threads.h"

#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fmt/format.h>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace Libs::Wheel {

using Config::Wheel::Binding;
using Config::Wheel::BindingType;
using Config::Wheel::Control;
using Config::Wheel::DeviceId;
using Config::Wheel::ForceSense;
using Config::Wheel::HatDirection;
using Config::Wheel::WheelConfig;

namespace {

constexpr uint64_t RefreshIntervalNs = 1'000'000'000;
constexpr uint32_t IdleWaitMicros    = 100'000;
constexpr uint32_t RenderWaitMicros  = 2'000;
// No input read for this long: the game is paused, stuck or gone, so no force is left on.
constexpr uint64_t WatchdogNs = 1'000'000'000;
// An axis counts as alive once its readings have spanned this much.
constexpr int      AxisActivity      = 2000;
constexpr float    RenderUpdateDelta = 0.002f;
constexpr uint64_t RenderRefreshNs   = 50'000'000;
// A constant force the game has not updated for this long is held off: long enough not to cut a
// force the game holds steady, short enough for a frozen game.
constexpr uint64_t StaleForceNs = 2'000'000'000;
// The rim's speed is measured over at least this long, and capped (half travels a second).
constexpr uint64_t VelocityStepNs = 4'000'000;
constexpr float    MaxVelocity    = 8.0f;

struct HostDevice {
	SDL_JoystickID id       = 0;
	SDL_Joystick*  joystick = nullptr;
	DeviceId       device;
	std::string    name;
	// The range each axis has covered since it was opened; DirectInput can report zeros until
	// its first poll, so the value at open is no reference.
	std::vector<int16_t> axis_min;
	std::vector<int16_t> axis_max;
	uint64_t             active_axes = 0;
};

struct SlotRequest {
	Effect   effect;
	bool     playing    = false;
	uint64_t generation = 0;
};

// What ApplyFfb takes from the shared state, so the haptic work runs without m_mutex.
struct FfbFrame {
	std::array<SlotRequest, MAX_EFFECT_SLOTS> requests {};
	FfbSettings                               settings;
	bool                                      present             = false;
	bool                                      sense_known         = false;
	bool                                      software_conditions = false;
	// Where the game's lock is, as a fraction of half the travel; 1 or more: no end stop.
	float lock_at  = 1.0f;
	float position = 0.0f;
};

struct SlotApplied {
	Effect             effect;
	bool               playing    = false;
	uint64_t           generation = 0;
	SDL_HapticEffectID sdl_id     = -1;
	bool               software   = false;
};

int16_t ReadAxisRaw(HostDevice& d, int index) {
	if (index < 0 || static_cast<size_t>(index) >= d.axis_min.size()) {
		return 0;
	}
	int16_t v = SDL_GetJoystickAxis(d.joystick, index);
	// Many drivers report -32767 as their minimum; treat it as the real one, as RPCS3 does.
	if (v == -32767) {
		v = -32768;
	}
	d.axis_min[index] = std::min(d.axis_min[index], v);
	d.axis_max[index] = std::max(d.axis_max[index], v);
	if (index < 64 && d.axis_max[index] - d.axis_min[index] > AxisActivity) {
		d.active_axes |= uint64_t {1} << index;
	}
	return v;
}

bool AxisActive(const HostDevice& d, int index) {
	return index >= 0 && index < 64 && (d.active_axes >> index & 1u) != 0;
}

int16_t Negate(int16_t v) {
	return v == -32768 ? int16_t {32767} : static_cast<int16_t>(-v);
}

bool HatPressed(Uint8 hat, HatDirection dir) {
	switch (dir) {
		case HatDirection::Up: return (hat & SDL_HAT_UP) != 0;
		case HatDirection::Down: return (hat & SDL_HAT_DOWN) != 0;
		case HatDirection::Left: return (hat & SDL_HAT_LEFT) != 0;
		case HatDirection::Right: return (hat & SDL_HAT_RIGHT) != 0;
		case HatDirection::None: break;
	}
	return false;
}

float Curve(float value, float curve) {
	return curve == 1.0f ? value : std::pow(value, curve);
}

// True the first time for a type: a warning is logged once per type.
bool Warn(uint32_t* warned, EffectType type) {
	const uint32_t bit   = 1u << static_cast<int>(type);
	const bool     first = (*warned & bit) == 0;
	*warned |= bit;
	return first;
}

const char* EffectTypeName(EffectType type) {
	switch (type) {
		case EffectType::Constant: return "constant force";
		case EffectType::Spring: return "spring";
		case EffectType::Damper: return "damper";
		case EffectType::Friction: return "friction";
		case EffectType::Inertia: return "inertia";
		case EffectType::None: break;
	}
	return "no effect";
}

class WheelHost {
public:
	void Start(const std::filesystem::path& config_path);
	void Stop();
	void Emergency();

	Input    Sample();
	void     SetGameRotation(uint32_t degrees) { m_game_rotation = degrees; }
	uint32_t GetPhysicalRotation() {
		Common::LockGuard lock(m_mutex);
		return m_config.physical_rotation_deg;
	}
	void Request(int slot, const Effect* effect, std::optional<bool> play);
	void RequestStopAll();
	bool Present();
	bool IsBoundDevice(uint16_t vendor_id, uint16_t product_id);

private:
	// The wheel thread's steps.
	void ReloadConfig(bool force);
	void RefreshDevices();
	void RefreshHaptic();
	void ApplyFfb(uint64_t now);
	void Loop();
	void CloseAll();

	const HostDevice* FindDevice(const DeviceId& id, int axis) const;
	HostDevice*       FindDeviceMutable(const DeviceId& id, int axis) {
		return const_cast<HostDevice*>(FindDevice(id, axis));
	}
	bool    IsBound(uint16_t vendor_id, uint16_t product_id) const;
	int16_t ReadAsAxis(const Binding& b, int16_t neutral);
	bool    ReadAsButton(const Binding& b);
	float   ReadSteeringRaw();
	Input   ComputeInput();
	bool    ApplyFfbLocked(uint64_t now, const FfbFrame& frame, std::string* notes);
	void    DestroySlot(SlotApplied* a);
	void    StopRender();
	void    CloseHaptic(const char* why);

	Common::Mutex   m_mutex;
	Common::CondVar m_cond;

	// Under m_mutex.
	std::vector<HostDevice>                   m_devices;
	WheelConfig                               m_config;
	std::filesystem::path                     m_config_path;
	std::filesystem::file_time_type           m_config_time {};
	bool                                      m_config_loaded = false;
	bool                                      m_present       = false;
	std::array<SlotRequest, MAX_EFFECT_SLOTS> m_requests {};
	uint64_t                                  m_generation = 0;
	bool                                      m_ffb_dirty  = false;

	std::atomic<uint32_t> m_game_rotation {0};
	std::atomic<uint64_t> m_last_sample_ns {0};
	// The last force request from the game: a force it stopped updating is not kept on.
	std::atomic<uint64_t> m_last_request_ns {0};
	std::atomic_bool      m_render_active {false};
	std::atomic_bool      m_quit {false};
	std::atomic_bool      m_thread_done {false};
	std::atomic<int>      m_wheel_thread {-1};

	// Wheel thread only. m_haptic_mutex is held around the haptic calls, so that an emergency stop
	// from another thread never runs alongside them.
	Common::Mutex                             m_haptic_mutex;
	SDL_Haptic*                               m_haptic          = nullptr;
	SDL_Joystick*                             m_haptic_joystick = nullptr;
	DeviceId                                  m_haptic_device;
	bool                                      m_stale    = false;
	uint32_t                                  m_features = 0;
	std::array<SlotApplied, MAX_EFFECT_SLOTS> m_applied {};
	FfbSettings                               m_settings;
	bool                                      m_software_conditions = false;
	bool                                      m_sense_known         = false;
	bool                                      m_paused              = false;
	SDL_HapticEffectID                        m_render_id           = -1;
	float                                     m_render_force        = 0.0f;
	uint64_t                                  m_render_time         = 0;
	bool                                      m_render_running      = false;
	float                                     m_prev_position       = 0.0f;
	float                                     m_velocity            = 0.0f;
	uint64_t                                  m_prev_time           = 0;
	uint32_t                                  m_warned_unsupported  = 0;
	uint32_t                                  m_warned_upload       = 0;
	bool                                      m_warned_no_haptic    = false;

	std::unique_ptr<std::thread> m_thread;
};

WheelHost* g_host = nullptr;

const HostDevice* WheelHost::FindDevice(const DeviceId& id, int axis) const {
	const HostDevice* first = nullptr;
	for (const auto& d: m_devices) {
		if (d.device == id) {
			// Some drivers (Fanatec's) expose two identical devices of which one is dead: prefer
			// the one whose axis has moved.
			if (axis < 0 || AxisActive(d, axis)) {
				return &d;
			}
			if (first == nullptr) {
				first = &d;
			}
		}
	}
	if (first != nullptr) {
		return first;
	}
	// A firmware update can change the button or axis count: accept the one device with the
	// same vendor and product.
	const HostDevice* only = nullptr;
	for (const auto& d: m_devices) {
		if (d.device.vendor_id == id.vendor_id && d.device.product_id == id.product_id) {
			if (only != nullptr) {
				return nullptr;
			}
			only = &d;
		}
	}
	return only;
}

int16_t WheelHost::ReadAsAxis(const Binding& b, int16_t neutral) {
	if (!b.IsBound()) {
		return neutral;
	}
	if (b.type == BindingType::Axis) {
		// Read the axis on every identical device first, so the activity that picks the live one
		// of two twins is seen on both.
		for (auto& d: m_devices) {
			if (d.device == b.device) {
				ReadAxisRaw(d, b.index);
			}
		}
	}
	auto* d = FindDeviceMutable(b.device, b.type == BindingType::Axis ? b.index : -1);
	if (d == nullptr) {
		return neutral;
	}
	switch (b.type) {
		case BindingType::Axis: {
			const int16_t v = ReadAxisRaw(*d, b.index);
			return b.reverse ? Negate(v) : v;
		}
		case BindingType::Button:
		case BindingType::Hat: {
			bool pressed = b.type == BindingType::Button
			                   ? SDL_GetJoystickButton(d->joystick, b.index)
			                   : HatPressed(SDL_GetJoystickHat(d->joystick, b.index), b.hat);
			pressed ^= b.reverse;
			if (!pressed) {
				return neutral;
			}
			return b.positive ? int16_t {32767} : int16_t {-32768};
		}
		case BindingType::None: break;
	}
	return neutral;
}

bool WheelHost::ReadAsButton(const Binding& b) {
	if (!b.IsBound()) {
		return false;
	}
	bool any = false;
	for (auto& d: m_devices) {
		if (!(d.device == b.device)) {
			continue;
		}
		any          = true;
		bool pressed = false;
		switch (b.type) {
			case BindingType::Axis: {
				int16_t v = ReadAxisRaw(d, b.index);
				v         = b.reverse ? Negate(v) : v;
				pressed   = b.positive ? v > 16383 : v < -16384;
				break;
			}
			case BindingType::Button:
				pressed = SDL_GetJoystickButton(d.joystick, b.index) != b.reverse;
				break;
			case BindingType::Hat:
				pressed = HatPressed(SDL_GetJoystickHat(d.joystick, b.index), b.hat) != b.reverse;
				break;
			case BindingType::None: break;
		}
		// Identical devices are OR-ed, so a dead twin never hides a press.
		if (pressed) {
			return true;
		}
	}
	if (!any) {
		if (auto* d = FindDeviceMutable(b.device, -1); d != nullptr) {
			Binding exact = b;
			exact.device  = d->device;
			return ReadAsButton(exact);
		}
	}
	return false;
}

float WheelHost::ReadSteeringRaw() {
	const int16_t raw = ReadAsAxis(m_config.Get(Control::Steering), 0);
	return raw >= 0 ? raw / 32767.0f : raw / 32768.0f;
}

Input WheelHost::ComputeInput() {
	Input       in;
	const auto& steering = m_config.Get(Control::Steering);
	in.present = steering.IsBound() &&
	             FindDevice(steering.device,
	                        steering.type == BindingType::Axis ? steering.index : -1) != nullptr;
	if (!in.present) {
		return in;
	}

	{
		float pos       = ReadSteeringRaw();
		in.steering_raw = pos;
		const float mag = std::abs(pos);
		const float dz  = steering.dead_zone;
		float       out = mag <= dz ? 0.0f : (mag - dz) / (1.0f - dz);
		out             = Curve(std::min(out, 1.0f), steering.curve);
		pos             = pos < 0.0f ? -out : out;

		in.steering = pos;
		if (const uint32_t game = m_game_rotation; game != 0) {
			const float physical = static_cast<float>(m_config.physical_rotation_deg);
			in.steering = std::clamp(pos * physical / static_cast<float>(game), -1.0f, 1.0f);
		}
	}

	auto pedal = [&](Control c) {
		const auto& b = m_config.Get(c);
		// The released pedal reads +32767 after reverse, as in RPCS3's emulated G27, so its
		// mappings import unchanged.
		const int16_t raw     = ReadAsAxis(b, 32767);
		float         pressed = (32767.0f - static_cast<float>(raw)) / 65535.0f;
		pressed = b.dead_zone >= 1.0f ? 0.0f : (pressed - b.dead_zone) / (1.0f - b.dead_zone);
		return Curve(std::clamp(pressed, 0.0f, 1.0f), b.curve);
	};
	in.throttle  = pedal(Control::Throttle);
	in.brake     = pedal(Control::Brake);
	in.clutch    = pedal(Control::Clutch);
	in.handbrake = pedal(Control::Handbrake);

	for (int i = Config::Wheel::FIRST_BUTTON; i < Config::Wheel::CONTROL_COUNT; i++) {
		if (ReadAsButton(m_config.bindings[i])) {
			in.buttons |= uint64_t {1} << (i - Config::Wheel::FIRST_BUTTON);
		}
	}
	constexpr std::array<Control, 8> Gears = {Control::Gear1, Control::Gear2,      Control::Gear3,
	                                          Control::Gear4, Control::Gear5,      Control::Gear6,
	                                          Control::Gear7, Control::GearReverse};
	for (size_t g = 0; g < Gears.size(); g++) {
		if (in.Pressed(Gears[g])) {
			in.gear = g == 7 ? -1 : static_cast<int>(g) + 1;
			break;
		}
	}
	return in;
}

Input WheelHost::Sample() {
	m_last_sample_ns = SDL_GetTicksNS();
	Common::LockGuard lock(m_mutex);
	return ComputeInput();
}

bool WheelHost::Present() {
	Common::LockGuard lock(m_mutex);
	return m_present;
}

bool WheelHost::IsBoundDevice(uint16_t vendor_id, uint16_t product_id) {
	Common::LockGuard lock(m_mutex);
	return IsBound(vendor_id, product_id);
}

bool WheelHost::IsBound(uint16_t vendor_id, uint16_t product_id) const {
	auto matches = [&](const DeviceId& d) {
		return !d.IsNone() && d.vendor_id == vendor_id && d.product_id == product_id;
	};
	if (matches(m_config.ffb_device)) {
		return true;
	}
	return std::any_of(m_config.bindings.begin(), m_config.bindings.end(),
	                   [&](const Binding& b) { return b.IsBound() && matches(b.device); });
}

void WheelHost::Request(int slot, const Effect* effect, std::optional<bool> play) {
	if (slot < 0 || slot >= MAX_EFFECT_SLOTS) {
		return;
	}
	{
		Common::LockGuard lock(m_mutex);
		auto&             r = m_requests[slot];
		if (effect != nullptr) {
			if (effect->type == EffectType::None) {
				r.effect  = {};
				r.playing = false;
			} else {
				r.effect = *effect;
			}
		}
		if (play.has_value()) {
			r.playing = *play && r.effect.type != EffectType::None;
		}
		r.generation = ++m_generation;
		m_ffb_dirty  = true;
	}
	m_last_request_ns = SDL_GetTicksNS();
	m_cond.Signal();
}

void WheelHost::RequestStopAll() {
	{
		Common::LockGuard lock(m_mutex);
		for (auto& r: m_requests) {
			if (r.playing) {
				r.playing    = false;
				r.generation = ++m_generation;
			}
		}
		m_ffb_dirty = true;
	}
	m_cond.Signal();
}

void WheelHost::ReloadConfig(bool force) {
	std::error_code ec;
	const auto      time = std::filesystem::last_write_time(m_config_path, ec);
	if (ec) {
		if (force || m_config_loaded) {
			LOGF("Wheel: no config at %s; nothing is bound\n",
			     Common::PathToString(m_config_path).c_str());
			Common::LockGuard lock(m_mutex);
			m_config    = {};
			m_ffb_dirty = true;
		}
		m_config_loaded = false;
		return;
	}
	if (!force && m_config_loaded && time == m_config_time) {
		return;
	}
	m_config_time = time;
	std::string error;
	auto        config = Config::Wheel::Load(m_config_path, &error);
	if (!config) {
		LOGF_COLOR(Log::Color::BrightYellow,
		           "Wheel: %s not loaded (%s); the previous mapping stays\n",
		           Common::PathToString(m_config_path).c_str(), error.c_str());
		return;
	}
	SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, config->background_input ? "1" : "0");
	{
		Common::LockGuard lock(m_mutex);
		m_config    = *config;
		m_ffb_dirty = true;
	}
	m_config_loaded = true;
	LOGF("Wheel: config %s loaded: model %s, gain %u%%, direction %s, force sense %s, rotation %u "
	     "deg%s\n",
	     Common::PathToString(m_config_path).c_str(), config->model.c_str(), config->ffb_gain,
	     std::string(Config::Wheel::FfbDirectionName(config->ffb_direction)).c_str(),
	     std::string(Config::Wheel::ForceSenseName(config->force_sense)).c_str(),
	     config->physical_rotation_deg, config->soft_lock ? " with soft lock" : "");
}

void WheelHost::RefreshDevices() {
	int             count = 0;
	SDL_JoystickID* ids   = SDL_GetJoysticks(&count);

	std::vector<HostDevice> keep;
	std::vector<HostDevice> old;
	{
		Common::LockGuard lock(m_mutex);
		old = m_devices;
	}
	for (int i = 0; i < count; i++) {
		// Only the devices the wheel uses are opened; the others are left to the window's pads.
		bool bound = false;
		{
			Common::LockGuard lock(m_mutex);
			bound =
			    IsBound(SDL_GetJoystickVendorForID(ids[i]), SDL_GetJoystickProductForID(ids[i]));
		}
		if (!bound) {
			continue;
		}
		auto it = std::find_if(old.begin(), old.end(),
		                       [&](const HostDevice& d) { return d.id == ids[i]; });
		if (it != old.end()) {
			keep.push_back(*it);
			old.erase(it);
			continue;
		}
		SDL_Joystick* joystick = SDL_OpenJoystick(ids[i]);
		if (joystick == nullptr) {
			continue;
		}
		HostDevice d;
		d.id                = ids[i];
		d.joystick          = joystick;
		d.device.vendor_id  = SDL_GetJoystickVendor(joystick);
		d.device.product_id = SDL_GetJoystickProduct(joystick);
		d.device.num_axes   = static_cast<uint16_t>(std::max(SDL_GetNumJoystickAxes(joystick), 0));
		d.device.num_hats   = static_cast<uint16_t>(std::max(SDL_GetNumJoystickHats(joystick), 0));
		d.device.num_buttons =
		    static_cast<uint16_t>(std::max(SDL_GetNumJoystickButtons(joystick), 0));
		const char* name = SDL_GetJoystickName(joystick);
		d.name           = name != nullptr ? name : "";
		for (int a = 0; a < d.device.num_axes; a++) {
			const int16_t v = SDL_GetJoystickAxis(joystick, a);
			d.axis_min.push_back(v);
			d.axis_max.push_back(v);
		}
		LOGF("Wheel: device %s (%04X:%04X, %u axes, %u hats, %u buttons) id 0x%016" PRIX64 "\n",
		     d.name.c_str(), d.device.vendor_id, d.device.product_id, d.device.num_axes,
		     d.device.num_hats, d.device.num_buttons, d.device.AsU64());
		keep.push_back(std::move(d));
	}
	SDL_free(ids);

	bool changed = false;
	bool present = false;
	{
		Common::LockGuard lock(m_mutex);
		// Keep the activity seen by guest reads since the copy was taken.
		for (auto& d: keep) {
			for (const auto& cur: m_devices) {
				if (cur.id == d.id) {
					d.active_axes |= cur.active_axes;
					d.axis_min = cur.axis_min;
					d.axis_max = cur.axis_max;
				}
			}
		}
		m_devices         = std::move(keep);
		const auto& steer = m_config.Get(Control::Steering);
		present =
		    steer.IsBound() &&
		    FindDevice(steer.device, steer.type == BindingType::Axis ? steer.index : -1) != nullptr;
		changed   = present != m_present;
		m_present = present;
	}
	// SDL's DirectInput haptic borrows its joystick's device: close it before the joystick.
	for (const auto& d: old) {
		if (m_haptic != nullptr && d.joystick == m_haptic_joystick) {
			CloseHaptic("its device was removed or unbound");
		}
	}
	for (auto& d: old) {
		LOGF("Wheel: device %s removed or unbound\n", d.name.c_str());
		SDL_CloseJoystick(d.joystick);
	}
	if (changed) {
		LOGF("Wheel: steering device %s\n",
		     present ? "connected: the wheel is attached" : "absent: the wheel is detached");
	}
}

void WheelHost::RefreshHaptic() {
	DeviceId      target;
	SDL_Joystick* joystick = nullptr;
	std::string   name;
	{
		Common::LockGuard lock(m_mutex);
		target = m_config.ffb_device.IsNone() ? m_config.Get(Control::Steering).device
		                                      : m_config.ffb_device;
		if (!target.IsNone()) {
			const HostDevice* found = nullptr;
			// Of two identical devices, the one with force feedback.
			for (const auto& d: m_devices) {
				if (d.device == target && SDL_IsJoystickHaptic(d.joystick)) {
					found = &d;
					break;
				}
			}
			found = found != nullptr ? found : FindDevice(target, -1);
			if (found != nullptr) {
				joystick = found->joystick;
				name     = found->name;
			}
		}
	}
	if (m_haptic != nullptr &&
	    (joystick == nullptr || !(target == m_haptic_device) || joystick != m_haptic_joystick)) {
		CloseHaptic("the force-feedback device changed");
	}
	if (m_haptic == nullptr && joystick != nullptr && !m_quit) {
		if (!SDL_IsJoystickHaptic(joystick)) {
			if (!std::exchange(m_warned_no_haptic, true)) {
				LOGF_COLOR(Log::Color::BrightYellow,
				           "Wheel: %s has no force feedback through SDL (on Windows it must be a "
				           "DirectInput device)\n",
				           name.c_str());
			}
		} else {
			Common::LockGuard lock(m_haptic_mutex);
			// After a quit nothing is opened again: the emergency stop may have closed it.
			m_haptic = m_quit ? nullptr : SDL_OpenHapticFromJoystick(joystick);
			if (m_quit) {
				return;
			}
			if (m_haptic == nullptr) {
				LOGF_COLOR(Log::Color::BrightYellow,
				           "Wheel: cannot open force feedback on %s: %s\n", name.c_str(),
				           SDL_GetError());
			} else {
				m_haptic_device   = target;
				m_haptic_joystick = joystick;
				m_features        = SDL_GetHapticFeatures(m_haptic);
				// The game centres the wheel itself; the device's own spring would add to it.
				if ((m_features & SDL_HAPTIC_AUTOCENTER) != 0) {
					SDL_SetHapticAutocenter(m_haptic, 0);
				}
				// Strength is applied to every effect here, not by the device.
				if ((m_features & SDL_HAPTIC_GAIN) != 0) {
					SDL_SetHapticGain(m_haptic, 100);
				}
				for (auto& a: m_applied) {
					a.generation = 0;
				}
				LOGF("Wheel: force feedback on %s, features 0x%08x\n", name.c_str(), m_features);
			}
		}
	}
	Common::LockGuard lock(m_mutex);
	m_ffb_dirty |= m_haptic != nullptr;
}

void WheelHost::CloseHaptic(const char* why) {
	Common::LockGuard lock(m_haptic_mutex);
	if (m_haptic == nullptr) {
		return;
	}
	SDL_StopHapticEffects(m_haptic);
	SDL_CloseHaptic(m_haptic);
	m_haptic          = nullptr;
	m_haptic_joystick = nullptr;
	m_features        = 0;
	m_applied         = {};
	m_render_id       = -1;
	m_render_running  = false;
	LOGF("Wheel: force-feedback device closed: %s\n", why);
}

void WheelHost::DestroySlot(SlotApplied* a) {
	if (a->sdl_id >= 0 && m_haptic != nullptr) {
		SDL_DestroyHapticEffect(m_haptic, a->sdl_id);
	}
	a->sdl_id   = -1;
	a->software = false;
}

void WheelHost::StopRender() {
	if (m_render_running && m_haptic != nullptr && m_render_id >= 0) {
		SDL_StopHapticEffect(m_haptic, m_render_id);
	}
	m_render_running = false;
	m_render_force   = 0.0f;
}

void WheelHost::ApplyFfb(uint64_t now) {
	FfbFrame frame;
	{
		Common::LockGuard lock(m_mutex);
		frame.requests                   = m_requests;
		m_ffb_dirty                      = false;
		const auto& steering             = m_config.Get(Control::Steering);
		frame.sense_known                = m_config.force_sense != ForceSense::Unknown;
		frame.software_conditions        = m_config.software_conditions && frame.sense_known;
		frame.settings.gain              = static_cast<float>(m_config.ffb_gain) / 100.0f;
		frame.settings.direction         = m_config.ffb_direction;
		frame.settings.mirror_conditions = steering.reverse;
		// The sense is measured on the device's own axis; forces computed here start from the
		// position the binding reads, so its reverse counts too.
		frame.settings.device_reverse =
		    frame.sense_known && (m_config.force_sense == ForceSense::Reversed) != steering.reverse;
		// A positive force from the game turns the wheel towards the low end of the steering it
		// reads: a wheel whose sense measured reversed needed the game's forces as they come
		// (mirrored, they pushed away from the centre). So they are mirrored only where a positive
		// force turns the other way. While the sense is unknown they go through as they come, as
		// RPCS3 sends them.
		frame.settings.reverse =
		    m_config.reverse_effects != (frame.sense_known && !frame.settings.device_reverse);
		frame.present          = m_present;
		if (const uint32_t game = m_game_rotation;
		    game != 0 && m_config.soft_lock && frame.sense_known) {
			frame.lock_at =
			    static_cast<float>(game) / static_cast<float>(m_config.physical_rotation_deg);
		}
		// Read outside the haptic lock, so that a guest thread holding m_mutex never keeps the
		// emergency stop out of it.
		frame.position = ReadSteeringRaw();
	}
	bool        render = false;
	std::string notes;
	if (m_haptic != nullptr && !m_quit) {
		Common::LockGuard haptic_lock(m_haptic_mutex);
		if (m_haptic != nullptr) {
			render = ApplyFfbLocked(now, frame, &notes);
		}
	}
	m_render_active = render;
	if (!notes.empty()) {
		LOGF("%s", notes.c_str());
	}
}

bool WheelHost::ApplyFfbLocked(uint64_t now, const FfbFrame& frame, std::string* notes) {
	auto note = [&](const std::string& text) { *notes += "Wheel: " + text + "\n"; };

	// No force before the game reads the wheel, and none once it stops reading it.
	const uint64_t last   = m_last_sample_ns;
	const bool     idle   = last == 0 || (now > last && now - last > WatchdogNs);
	const bool     paused = idle || !frame.present;
	if (paused) {
		if (!m_paused) {
			note(fmt::format("{}: forces stopped", !frame.present
			                                           ? "wheel absent"
			                                           : "the game is not reading the wheel"));
			SDL_StopHapticEffects(m_haptic);
			// Everything is applied again when the game reads the wheel again.
			for (auto& a: m_applied) {
				a.playing    = false;
				a.generation = 0;
			}
			m_render_running = false;
			m_paused         = true;
		}
		return false;
	}
	if (m_paused) {
		note("the game reads the wheel again: forces resumed");
		m_paused = false;
	}

	const auto& settings = frame.settings;
	const bool  settings_changed =
	    settings.gain != m_settings.gain || settings.direction != m_settings.direction ||
	    settings.reverse != m_settings.reverse ||
	    settings.device_reverse != m_settings.device_reverse ||
	    settings.mirror_conditions != m_settings.mirror_conditions ||
	    frame.software_conditions != m_software_conditions || frame.sense_known != m_sense_known;

	m_settings            = settings;
	m_software_conditions = frame.software_conditions;
	m_sense_known         = frame.sense_known;

	// A constant force the game has stopped updating (a frozen game whose input reads go on) is
	// not kept on; conditions follow the rim and stay.
	const uint64_t last_request = m_last_request_ns;
	const bool stale = last_request != 0 && now > last_request && now - last_request > StaleForceNs;
	if (stale != m_stale) {
		note(stale ? "no force update from the game: its forces are held off"
		           : "force updates again");
		m_stale = stale;
	}

	int software = 0;
	for (int i = 0; i < MAX_EFFECT_SLOTS; i++) {
		const auto& r    = frame.requests[i];
		auto&       a    = m_applied[i];
		const auto  type = r.effect.type;
		const bool  want = r.playing && !(stale && !IsCondition(type));
		if (r.generation != a.generation || settings_changed || want != a.playing) {
			a.generation = r.generation;
			if (type == EffectType::None) {
				DestroySlot(&a);
				a            = {};
				a.generation = r.generation;
				continue;
			}
			// Computed here: a constant always; damper and friction when the device cannot play
			// them or software_conditions asks, springs when the device cannot play them, all only
			// with a known sense. Inertia would need the rim's acceleration, too noisy to compute.
			const bool renderable =
			    type == EffectType::Constant || (frame.sense_known && type != EffectType::Inertia);
			const bool forced_software =
			    frame.software_conditions &&
			    (type == EffectType::Damper || type == EffectType::Friction);
			SDL_HapticEffect sdl {};
			const bool supported = ToSdlEffect(r.effect, settings, &sdl) && !forced_software &&
			                       (m_features & SdlFeatureFor(type)) != 0 &&
			                       SDL_HapticEffectSupported(m_haptic, &sdl);
			bool       recreated = false;
			if (supported) {
				if (a.sdl_id >= 0 && a.effect.type == type) {
					if ((!(a.effect == r.effect) || settings_changed) &&
					    !SDL_UpdateHapticEffect(m_haptic, a.sdl_id, &sdl)) {
						DestroySlot(&a);
					}
				} else {
					DestroySlot(&a);
				}
				if (a.sdl_id < 0) {
					a.sdl_id  = SDL_CreateHapticEffect(m_haptic, &sdl);
					recreated = a.sdl_id >= 0;
					if (a.sdl_id < 0 && Warn(&m_warned_upload, type)) {
						note(fmt::format("{} upload failed: {}", EffectTypeName(type),
						                 SDL_GetError()));
					}
				}
			} else {
				DestroySlot(&a);
				if (!forced_software && Warn(&m_warned_unsupported, type)) {
					note(fmt::format("the device does not play {}; {}", EffectTypeName(type),
					                 renderable ? "computed here as a constant force"
					                 : type == EffectType::Inertia
					                     ? "left out"
					                     : "left out until the launcher's force test has "
					                       "measured the wheel's force direction"));
				}
			}
			a.software = a.sdl_id < 0 && renderable;
			a.effect   = r.effect;
			if (a.sdl_id >= 0) {
				if (want && (!a.playing || recreated)) {
					SDL_RunHapticEffect(m_haptic, a.sdl_id, 1);
				} else if (!want && a.playing) {
					SDL_StopHapticEffect(m_haptic, a.sdl_id);
				}
			}
			a.playing = want;
		}
		if (a.playing && a.software) {
			software++;
		}
	}

	// Computed here: effects the device cannot play, and the soft end stop.
	const bool can_render = (m_features & SDL_HAPTIC_CONSTANT) != 0;
	const bool render     = can_render && (software > 0 || frame.lock_at < 1.0f);
	float      force      = 0.0f;
	if (render) {
		// The rim's real position, before dead zone and curve.
		const float pos = frame.position;
		// Speed over at least 4 ms: SDL refreshes positions every 1 ms and this runs on every
		// force request, so shorter steps turn one position step into a huge speed.
		if (m_prev_time == 0) {
			m_prev_position = pos;
			m_prev_time     = now;
		} else if (now - m_prev_time >= VelocityStepNs) {
			const float dt  = static_cast<float>(now - m_prev_time) / 1e9f;
			const float vel = std::clamp((pos - m_prev_position) / dt, -MaxVelocity, MaxVelocity);
			m_velocity += (vel - m_velocity) * 0.5f;
			m_prev_position = pos;
			m_prev_time     = now;
		}
		for (const auto& a: m_applied) {
			if (a.playing && a.software) {
				force += RenderEffect(a.effect, settings, pos, m_velocity);
			}
		}
		force += RenderSoftLock(pos, m_velocity, frame.lock_at) * settings.gain *
		         (settings.device_reverse ? -1.0f : 1.0f);
		force = std::clamp(force, -1.0f, 1.0f);

		Effect constant;
		constant.type  = EffectType::Constant;
		constant.level = force;
		SDL_HapticEffect sdl {};
		// Already scaled and signed above.
		FfbSettings as_is;
		as_is.direction = settings.direction;
		ToSdlEffect(constant, as_is, &sdl);
		if (m_render_id < 0) {
			m_render_id    = SDL_CreateHapticEffect(m_haptic, &sdl);
			m_render_force = force;
			m_render_time  = now;
		} else if (std::abs(force - m_render_force) > RenderUpdateDelta ||
		           now - m_render_time > RenderRefreshNs) {
			SDL_UpdateHapticEffect(m_haptic, m_render_id, &sdl);
			m_render_force = force;
			m_render_time  = now;
		}
		if (m_render_id >= 0 && !m_render_running) {
			SDL_RunHapticEffect(m_haptic, m_render_id, 1);
			m_render_running = true;
		}
	} else {
		StopRender();
		m_prev_time = 0;
	}

	return render;
}

void WheelHost::Loop() {
	m_wheel_thread        = Common::Thread::GetThreadIdUnique();
	uint64_t next_refresh = 0;
	while (!m_quit) {
		const uint64_t now = SDL_GetTicksNS();
		if (now >= next_refresh) {
			ReloadConfig(false);
			RefreshDevices();
			RefreshHaptic();
			next_refresh = now + RefreshIntervalNs;
		}
		ApplyFfb(now);

		Common::LockGuard lock(m_mutex);
		if (!m_quit && !m_ffb_dirty) {
			const uint32_t wait = m_render_active ? RenderWaitMicros : IdleWaitMicros;
			m_cond.WaitFor(&m_mutex, wait);
		}
	}
	CloseAll();
	m_thread_done = true;
}

void WheelHost::CloseAll() {
	CloseHaptic("shutdown");
	std::vector<HostDevice> devices;
	{
		Common::LockGuard lock(m_mutex);
		devices.swap(m_devices);
		m_present = false;
	}
	for (auto& d: devices) {
		SDL_CloseJoystick(d.joystick);
	}
}

void WheelHost::Start(const std::filesystem::path& config_path) {
	m_config_path = config_path;
	ReloadConfig(true);
	m_thread = std::make_unique<std::thread>([this] { Loop(); });
}

void WheelHost::Stop() {
	m_quit = true;
	m_cond.Signal();
	if (m_thread) {
		m_thread->join();
		m_thread.reset();
	}
}

void WheelHost::Emergency() {
	m_quit = true;
	m_cond.Signal();
	// Failing on the wheel thread itself: nothing else uses the haptic, stop it here.
	if (Common::Thread::GetThreadIdUnique() == m_wheel_thread.load()) {
		if (m_haptic != nullptr) {
			SDL_StopHapticEffects(m_haptic);
			SDL_CloseHaptic(m_haptic);
			m_haptic = nullptr;
		}
		return;
	}
	// The wheel thread sees the quit within a wait and stops and closes the device itself.
	for (int i = 0; i < 300 && m_thread && !m_thread_done; i++) {
		Common::Thread::SleepMicro(1000);
	}
	if (m_thread_done || !m_thread) {
		if (!m_thread) {
			CloseHaptic("emergency");
		}
		return;
	}
	// The wheel thread is stuck: stop the forces only if it is not inside a haptic call, never
	// alongside it (SDL's haptic API is not thread-safe). If it is, the device is released
	// when the process ends.
	if (m_haptic_mutex.TryLock()) {
		if (m_haptic != nullptr) {
			SDL_StopHapticEffects(m_haptic);
			SDL_CloseHaptic(m_haptic);
			m_haptic = nullptr;
		}
		m_haptic_mutex.Unlock();
	}
}

// The fork's LastWords final action is not in this line: an unhandled host exception (chained to
// the filter before it) and process exit stop the forces instead. Installed only with the wheel.
#if defined(_WIN32)
LPTOP_LEVEL_EXCEPTION_FILTER g_previous_filter = nullptr;

LONG WINAPI WheelCrashFilter(EXCEPTION_POINTERS* info) {
	EmergencyShutdown();
	return g_previous_filter != nullptr ? g_previous_filter(info) : EXCEPTION_CONTINUE_SEARCH;
}
#endif

void InstallFinalAction() {
	static std::atomic<bool> installed {false};
	if (installed.exchange(true)) {
		return;
	}
	std::atexit(&EmergencyShutdown);
#if defined(_WIN32)
	g_previous_filter = SetUnhandledExceptionFilter(&WheelCrashFilter);
#endif
}

} // namespace

void Initialize() {
	if (!Config::Wheel::IsEnabled()) {
		return;
	}
	EXIT_IF(!Common::Thread::IsMainThread());
	if (!SDL_InitSubSystem(SDL_INIT_JOYSTICK | SDL_INIT_HAPTIC)) {
		LOGF_COLOR(Log::Color::BrightYellow,
		           "Wheel: SDL joystick/haptic init failed: %s; no wheel\n", SDL_GetError());
		return;
	}
	g_host = new WheelHost;
	g_host->Start(Config::Wheel::GetConfigPath());
	LOGF("Wheel: shown to the game as a Fanatec GT DD Pro\n");
	// A crash or process exit stops the forces.
	InstallFinalAction();
}

void Shutdown() {
	if (g_host == nullptr) {
		return;
	}
	g_host->Stop();
	delete g_host;
	g_host = nullptr;
	SDL_QuitSubSystem(SDL_INIT_JOYSTICK | SDL_INIT_HAPTIC);
}

void EmergencyShutdown() {
	if (g_host != nullptr) {
		g_host->Emergency();
	}
}

bool Enabled() {
	return g_host != nullptr;
}

bool Present() {
	return g_host != nullptr && g_host->Present();
}

bool IsBoundDevice(uint16_t vendor_id, uint16_t product_id) {
	return g_host != nullptr && g_host->IsBoundDevice(vendor_id, product_id);
}

Input Sample() {
	return g_host != nullptr ? g_host->Sample() : Input {};
}

void SetGameRotation(uint32_t degrees) {
	if (g_host != nullptr) {
		g_host->SetGameRotation(degrees);
	}
}

uint32_t GetPhysicalRotation() {
	return g_host != nullptr ? g_host->GetPhysicalRotation() : 0;
}

void SetEffect(int slot, const Effect& effect) {
	if (g_host != nullptr) {
		g_host->Request(slot, &effect, std::nullopt);
	}
}

void PlayEffect(int slot, bool play) {
	if (g_host != nullptr) {
		g_host->Request(slot, nullptr, play);
	}
}

void StopAllEffects() {
	if (g_host != nullptr) {
		g_host->RequestStopAll();
	}
}

} // namespace Libs::Wheel
