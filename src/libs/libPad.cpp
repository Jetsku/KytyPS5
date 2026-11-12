#include "common/abi.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "kernel/pthread.h"
#include "libs/controller.h"
#include "libs/errno.h"
#include "libs/libs.h"
#include "libs/padData.h"
#include "libs/wheelFanatec.h"
#include "libs/wheelHost.h"
#include "loader/symbolDatabase.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace Libs {

LIB_VERSION("Pad", 1, "Pad", 1, 1);

// The emulated racing wheel (--wheel): one special device, a Fanatec GT DD Pro, on its own handle.
// The pad handles stay as they are.
namespace PadWheel {

constexpr int HANDLE = 0x10;

constexpr int PAD_ERROR_INVALID_ARG          = -2137915391; /* 0x80920001 */
constexpr int PAD_ERROR_INVALID_HANDLE       = -2137915389; /* 0x80920003 */
constexpr int PAD_ERROR_ALREADY_OPENED       = -2137915388; /* 0x80920004 */
constexpr int PAD_ERROR_DEVICE_NOT_CONNECTED = -2137915385; /* 0x80920007 */

constexpr int PAD_DEVICE_CLASS_STEERING_WHEEL = 6;

std::atomic_bool g_opened = false;

bool IsHandle(int handle) {
	return handle == HANDLE && Wheel::Enabled() && g_opened;
}

std::string Hex(const uint8_t* buf, uint32_t length) {
	std::string s;
	for (uint32_t i = 0; i < length; i++) {
		s += fmt::format("{:02x}{}", buf[i], i % 8 == 7 ? "  " : " ");
	}
	return s;
}

// Report traffic is logged in full for its first calls per report id, then at powers of two, so
// a session's log shows every kind of report the game sends without flooding.
bool ShouldLog(std::array<std::atomic<uint32_t>, 256>& counts, uint8_t id) {
	const uint32_t n = ++counts[id];
	return n <= 32 || (n & (n - 1)) == 0;
}

std::array<std::atomic<uint32_t>, 256> g_output_counts {};
std::array<std::atomic<uint32_t>, 256> g_get_counts {};
std::array<std::atomic<uint32_t>, 256> g_set_counts {};

// What the game sends, by kind (a command's first two bytes). Each kind is logged in full for its
// first reports wherever in the session it first comes, and every 10 s a line counts the kinds,
// with the range of the constant force and the rumble calls on the wheel's handle, so a moment of
// the game (a crash, a kerb) can be matched with what it sent.
constexpr uint64_t SummaryPeriodUs = 10'000'000;
constexpr uint32_t LoggedPerKind   = 8;
constexpr size_t   SummaryKinds    = 16;

struct CommandStats {
	Common::Mutex                mutex;
	std::map<uint16_t, uint32_t> session;
	std::map<uint16_t, uint32_t> window;
	uint64_t                     window_start_us   = 0;
	uint32_t                     constants         = 0;
	float                        constant_min      = 0.0f;
	float                        constant_max      = 0.0f;
	float                        constant_last     = 0.0f;
	float                        constant_step     = 0.0f;
	bool                         constant_any      = false;
	uint32_t                     vibrations        = 0;
	uint32_t                     vibrations_on     = 0;
	uint32_t                     vibrations_logged = 0;
	uint8_t                      vibration_large   = 0;
	uint8_t                      vibration_small   = 0;
};

CommandStats g_stats;

uint16_t CommandKind(const uint8_t* buf, uint32_t length) {
	const uint8_t first  = length > 1 ? buf[1] : 0;
	const uint8_t second = length > 2 ? buf[2] : 0;
	// A slot's stop carries no effect type.
	return static_cast<uint16_t>(first << 8u |
	                             ((first & 0xF0) != 0xF0 && (first & 0x0F) == 0x03 ? 0 : second));
}

// Under g_stats.mutex. Empty when nothing came since the last line.
std::string TakeSummaryLocked(uint64_t now_us) {
	auto&          s       = g_stats;
	const uint64_t elapsed = now_us > s.window_start_us ? now_us - s.window_start_us : 0;
	std::string    line;
	if (!s.window.empty() || s.vibrations != 0) {
		std::vector<std::pair<uint16_t, uint32_t>> kinds(s.window.begin(), s.window.end());
		std::sort(kinds.begin(), kinds.end(),
		          [](const auto& a, const auto& b) { return a.second > b.second; });
		line = fmt::format("Wheel: the game sent in {:.1f} s:", static_cast<double>(elapsed) / 1e6);
		for (size_t i = 0; i < kinds.size() && i < SummaryKinds; i++) {
			line += fmt::format(" {:02x} {:02x} x{}", kinds[i].first >> 8u, kinds[i].first & 0xFFu,
			                    kinds[i].second);
		}
		if (kinds.size() > SummaryKinds) {
			line += fmt::format(" (+{} more kinds)", kinds.size() - SummaryKinds);
		}
		if (s.constants != 0) {
			line += fmt::format("; constant force {:+.2f} to {:+.2f}, largest step {:.2f}",
			                    s.constant_min, s.constant_max, s.constant_step);
		}
		line += fmt::format("; rumble calls {}", s.vibrations);
		if (s.vibrations_on != 0) {
			line += fmt::format(" ({} on, largest motors {}/{})", s.vibrations_on,
			                    s.vibration_large, s.vibration_small);
		}
		line += "\n";
	}
	s.window.clear();
	s.window_start_us = now_us;
	s.constants       = 0;
	s.constant_step   = 0.0f;
	s.vibrations      = 0;
	s.vibrations_on   = 0;
	s.vibration_large = 0;
	s.vibration_small = 0;
	return line;
}

// Returns whether this report is to be logged in full.
bool NoteCommand(const uint8_t* buf, uint32_t length, const Wheel::Fanatec::Command& c) {
	const uint64_t now  = LibKernel::KernelGetProcessTime();
	const uint16_t kind = CommandKind(buf, length);
	std::string    line;
	bool           full = false;
	{
		Common::LockGuard lock(g_stats.mutex);
		auto&             s = g_stats;
		if (s.window_start_us == 0) {
			s.window_start_us = now;
		}
		full = ++s.session[kind] <= LoggedPerKind;
		s.window[kind]++;
		if (c.kind == Wheel::Fanatec::Command::Kind::SetAndPlay &&
		    c.effect.type == Wheel::EffectType::Constant) {
			const float level = c.effect.level;
			if (s.constants == 0) {
				s.constant_min = s.constant_max = level;
			}
			if (s.constant_any) {
				s.constant_step = std::max(s.constant_step, std::abs(level - s.constant_last));
			}
			s.constant_any  = true;
			s.constant_min  = std::min(s.constant_min, level);
			s.constant_max  = std::max(s.constant_max, level);
			s.constant_last = level;
			s.constants++;
		}
		if (now - s.window_start_us >= SummaryPeriodUs) {
			line = TakeSummaryLocked(now);
		}
	}
	if (!line.empty()) {
		LOGF("%s", line.c_str());
	}
	return full;
}

void NoteVibration(const uint8_t* motors) {
	const uint64_t now = LibKernel::KernelGetProcessTime();
	std::string    line;
	bool           full = false;
	{
		Common::LockGuard lock(g_stats.mutex);
		auto&             s = g_stats;
		if (s.window_start_us == 0) {
			s.window_start_us = now;
		}
		s.vibrations++;
		if (motors[0] != 0 || motors[1] != 0) {
			s.vibrations_on++;
			// The first calls in full, and every call that starts a motor's strongest yet.
			full = s.vibrations_logged < LoggedPerKind || motors[0] > s.vibration_large ||
			       motors[1] > s.vibration_small;
		} else {
			full = s.vibrations_logged < LoggedPerKind;
		}
		s.vibration_large = std::max(s.vibration_large, motors[0]);
		s.vibration_small = std::max(s.vibration_small, motors[1]);
		if (full) {
			s.vibrations_logged++;
		}
		if (now - s.window_start_us >= SummaryPeriodUs) {
			line = TakeSummaryLocked(now);
		}
	}
	if (full) {
		LOGF("Pad: wheel: SetVibration(large %u, small %u): no rumble on a wheel, left out\n",
		     motors[0], motors[1]);
	}
	if (!line.empty()) {
		LOGF("%s", line.c_str());
	}
}

void FlushSummary() {
	std::string line;
	{
		Common::LockGuard lock(g_stats.mutex);
		line = TakeSummaryLocked(LibKernel::KernelGetProcessTime());
	}
	if (!line.empty()) {
		LOGF("%s", line.c_str());
	}
}

int OpenExt(int user_id, int type, int index, const uint16_t* param) {
	if (param == nullptr) {
		return PAD_ERROR_INVALID_ARG;
	}
	const bool wanted  = Wheel::Fanatec::Matches(param[0], param[1], param[2]);
	const bool present = Wheel::Present();
	int        result  = PAD_ERROR_DEVICE_NOT_CONNECTED;
	if (wanted && present) {
		// GT7's Fanatec code probes again while it has the wheel; the open wheel still counts.
		result = g_opened.exchange(true) ? PAD_ERROR_ALREADY_OPENED : HANDLE;
	}
	static std::atomic<uint32_t> logged = 0;
	if (const uint32_t n = ++logged; n <= 32 || (n & (n - 1)) == 0 || result == HANDLE) {
		LOGF("Pad: wheel: OpenExt(user %d, type %d, index %d, %04x:%04x-%04x) #%u -> 0x%08x%s\n",
		     user_id, type, index, param[0], param[1], param[2], n, static_cast<uint32_t>(result),
		     wanted ? (present ? "" : " (wheel absent)") : "");
	}
	return result;
}

int Close() {
	g_opened = false;
	Wheel::StopAllEffects();
	FlushSummary();
	LOGF("Pad: wheel: closed\n");
	return OK;
}

int ReadOne(Controller::PadData* data) {
	const auto in = Wheel::Sample();
	Wheel::Fanatec::FillPadData(in, data);
	data->timestamp = LibKernel::KernelGetProcessTime();
	return OK;
}

int GetFeatureReport(uint8_t report_id, uint8_t* buf, uint32_t length) {
	if (buf == nullptr || length == 0) {
		return PAD_ERROR_INVALID_ARG;
	}
	const bool answered = Wheel::Fanatec::FeatureReport(report_id, buf, length);
	if (ShouldLog(g_get_counts, report_id)) {
		LOGF("Pad: wheel: GetFeatureReport(0x%02x, %u) #%u -> %s\n", report_id, length,
		     g_get_counts[report_id].load(),
		     answered ? Hex(buf, std::min<uint32_t>(length, 16)).c_str() : "not answered");
	}
	return answered ? OK : PAD_ERROR_INVALID_ARG;
}

int SetFeatureReport(uint8_t report_id, const uint8_t* buf, uint32_t length) {
	if (buf == nullptr || length == 0) {
		return PAD_ERROR_INVALID_ARG;
	}
	if (ShouldLog(g_set_counts, report_id)) {
		LOGF("Pad: wheel: SetFeatureReport(0x%02x, %u) #%u: %s\n", report_id, length,
		     g_set_counts[report_id].load(), Hex(buf, std::min<uint32_t>(length, 64)).c_str());
	}
	return OK;
}

int OutputReport(uint8_t type, const uint8_t* buf, uint32_t length) {
	if (buf == nullptr || length == 0) {
		return PAD_ERROR_INVALID_ARG;
	}
	using Kind   = Wheel::Fanatec::Command::Kind;
	const auto c = Wheel::Fanatec::DecodeOutputReport(buf, length);
	switch (c.kind) {
		case Kind::SetAndPlay:
			Wheel::SetEffect(c.slot, c.effect);
			Wheel::PlayEffect(c.slot, true);
			break;
		case Kind::Stop: Wheel::PlayEffect(c.slot, false); break;
		case Kind::StopAll: Wheel::StopAllEffects(); break;
		case Kind::Rotation:
			LOGF("Pad: wheel: the game sets the rotation to %u degrees\n", c.degrees);
			Wheel::SetGameRotation(c.degrees);
			break;
		case Kind::Ignored:
		case Kind::Unknown: break;
	}
	const bool first_of_kind = NoteCommand(buf, length, c);
	if (ShouldLog(g_output_counts, type) || first_of_kind) {
		LOGF("Pad: wheel: OutputReport(0x%02x, %u) #%u%s: %s\n", type, length,
		     g_output_counts[type].load(), c.kind == Kind::Unknown ? " (not decoded)" : "",
		     Hex(buf, std::min<uint32_t>(length, 64)).c_str());
	}
	return OK;
}

} // namespace PadWheel

static int KYTY_SYSV_ABI PadSetVibrationMode(int handle, int mode) {
	PRINT_NAME();

	LOGF("\t handle      = %d\n"
	     "\t mode        = %d\n",
	     handle, mode);

	return Controller::PadCheckHandle(handle);
}

static int KYTY_SYSV_ABI PadSetVibrationTriggerEffectWeakWhileEmbeddedMicInUse(bool enabled) {
	PRINT_NAME();

	LOGF("\t enabled = %d\n", enabled ? 1 : 0);

	return OK;
}

struct PadDeviceClassExtendedInformation {
	int32_t device_class;
	uint8_t reserved[4];
	uint8_t class_data[12];
};

struct PadDeviceClassData {
	int32_t device_class;
	bool    data_valid;
	uint8_t reserved[3];
	union {
		struct {
			float    steering_wheel_angle;
			uint16_t steering_wheel;
			uint16_t accelerator_pedal;
			uint16_t brake_pedal;
			uint16_t clutch_pedal;
			uint16_t hand_brake;
			uint8_t  gear;
			uint8_t  reserved[1];
		} steering_wheel;
		struct {
			uint8_t data_len;
			uint8_t reserved[3];
			uint8_t data[12];
		} others;
		uint8_t raw[16];
	} class_data;
};

static_assert(sizeof(PadDeviceClassData) == 24);

static int KYTY_SYSV_ABI
PadDeviceClassGetExtendedInformation(int handle, PadDeviceClassExtendedInformation* info) {
	PRINT_NAME();

	constexpr int pad_error_invalid_arg = -2137915391; /* 0x80920001 */

	if (PadWheel::IsHandle(handle)) {
		if (info == nullptr) {
			return pad_error_invalid_arg;
		}
		std::memset(info, 0, sizeof(*info));
		if (!Wheel::Present()) {
			info->device_class = -1; // no device (AnyPS5 #9878, measured on a PS4)
			return OK;
		}
		info->device_class = PadWheel::PAD_DEVICE_CLASS_STEERING_WHEEL;
		// steeringWheel: u8 capability, u8 reserved, u16 maxPhysicalWheelAngle.
		const auto angle = static_cast<uint16_t>(Wheel::GetPhysicalRotation());
		std::memcpy(info->class_data + 2, &angle, 2);
		return OK;
	}
	if (const auto error = Controller::PadCheckHandle(handle); error != OK) {
		return error;
	}
	if (info == nullptr) {
		return pad_error_invalid_arg;
	}

	std::memset(info, 0, sizeof(*info));

	return OK;
}

static int KYTY_SYSV_ABI PadDeviceClassParseData(int handle, const Controller::PadData* data,
                                                 PadDeviceClassData* class_data) {
	PRINT_NAME();

	constexpr int pad_error_invalid_arg = -2137915391; /* 0x80920001 */

	LOGF("\t handle     = %d\n"
	     "\t data       = 0x%016" PRIx64 "\n"
	     "\t class_data = 0x%016" PRIx64 "\n",
	     handle, reinterpret_cast<uint64_t>(data), reinterpret_cast<uint64_t>(class_data));

	if (PadWheel::IsHandle(handle)) {
		if (data == nullptr || class_data == nullptr) {
			return pad_error_invalid_arg;
		}
		// From the live wheel, not the record: the record's device-unique block is the Fanatec
		// layout. Steering 0..0xFFFF centred at 0x8000, pedals 0 released to 0xFFFF pressed (as
		// DriveClub's consumer reads them, driveclub-recomp runtime_wheel.inc), angle in degrees
		// from the centre, gear one bit per gear with reverse bit 7 (GT7 1.710 eboot 0xbcda06:
		// tzcnt).
		const auto in = Wheel::Sample();
		std::memset(class_data, 0, sizeof(*class_data));
		if (!in.present) {
			class_data->device_class = -1;
			return OK;
		}
		class_data->device_class = PadWheel::PAD_DEVICE_CLASS_STEERING_WHEEL;
		class_data->data_valid   = true;
		auto& w                  = class_data->class_data.steering_wheel;
		auto  to_u16 = [](float v) { return static_cast<uint16_t>(std::lround(v * 65535.0f)); };
		w.steering_wheel_angle =
		    in.steering_raw * static_cast<float>(Wheel::GetPhysicalRotation()) * 0.5f;
		w.steering_wheel    = static_cast<uint16_t>(std::lround((in.steering + 1.0f) * 32767.5f));
		w.accelerator_pedal = to_u16(in.throttle);
		w.brake_pedal       = to_u16(in.brake);
		w.clutch_pedal      = to_u16(in.clutch);
		w.hand_brake        = to_u16(in.handbrake);
		w.gear = static_cast<uint8_t>(in.gear < 0 ? 0x80 : in.gear == 0 ? 0 : 1u << (in.gear - 1));
		return OK;
	}
	if (const auto error = Controller::PadCheckHandle(handle); error != OK) {
		return error;
	}
	if (data == nullptr || class_data == nullptr) {
		return pad_error_invalid_arg;
	}

	std::memset(class_data, 0, sizeof(*class_data));
	class_data->device_class = 0; // SCE_PAD_DEVICE_CLASS_STANDARD
	class_data->data_valid   = data->connected;

	if (data->device_unique_data_len > 0) {
		const auto data_len =
		    (data->device_unique_data_len > 12 ? 12 : data->device_unique_data_len);
		class_data->device_class = -1; // SCE_PAD_DEVICE_CLASS_INVALID/unsupported device payload
		class_data->class_data.others.data_len = data_len;
		std::memcpy(class_data->class_data.others.data, data->device_unique_data, data_len);
	}

	return OK;
}

static int KYTY_SYSV_ABI PadSetTiltCorrectionState(int handle, bool enabled) {
	PRINT_NAME();

	LOGF("\t handle  = %d\n"
	     "\t enabled = %d\n",
	     handle, enabled ? 1 : 0);

	return Controller::PadCheckHandle(handle);
}

static int KYTY_SYSV_ABI PadClose(int handle) {
	PRINT_NAME();

	LOGF("\t handle = %d\n", handle);

	if (PadWheel::IsHandle(handle)) {
		return PadWheel::Close();
	}
	return Controller::PadClose(handle);
}

// A wheel has no rumble, light bar, adaptive triggers or motion sensors: on its handle these
// succeed and do nothing. Rumble calls are counted, to see whether the game sends any.
static int KYTY_SYSV_ABI PadSetVibrationWheelAware(int                                  handle,
                                                   const Controller::PadVibrationParam* param) {
	if (!PadWheel::IsHandle(handle)) {
		return Controller::PadSetVibration(handle, param);
	}
	if (param != nullptr) {
		// The large motor's byte, then the small one's.
		PadWheel::NoteVibration(reinterpret_cast<const uint8_t*>(param));
	}
	return OK;
}

static int KYTY_SYSV_ABI
PadSetTriggerEffectWheelAware(int handle, const Controller::PadTriggerEffectParam* param) {
	return PadWheel::IsHandle(handle) ? OK : Controller::PadSetTriggerEffect(handle, param);
}

static int KYTY_SYSV_ABI PadSetLightBarWheelAware(int                                 handle,
                                                  const Controller::PadLightBarParam* param) {
	return PadWheel::IsHandle(handle) ? OK : Controller::PadSetLightBar(handle, param);
}

static int KYTY_SYSV_ABI PadResetLightBarWheelAware(int handle) {
	return PadWheel::IsHandle(handle) ? OK : Controller::PadResetLightBar(handle);
}

static int KYTY_SYSV_ABI PadSetMotionSensorStateWheelAware(int handle, bool enable) {
	return PadWheel::IsHandle(handle) ? OK : Controller::PadSetMotionSensorState(handle, enable);
}

static int KYTY_SYSV_ABI PadSetAngularVelocityDeadbandStateWheelAware(int handle, bool enable) {
	return PadWheel::IsHandle(handle)
	           ? OK
	           : Controller::PadSetAngularVelocityDeadbandState(handle, enable);
}

static int KYTY_SYSV_ABI PadResetOrientationWheelAware(int handle) {
	return PadWheel::IsHandle(handle) ? OK : Controller::PadResetOrientation(handle);
}

static int KYTY_SYSV_ABI PadIsRemoteControllerWheelAware(int handle, bool* is_remote) {
	if (PadWheel::IsHandle(handle)) {
		if (is_remote == nullptr) {
			return PadWheel::PAD_ERROR_INVALID_ARG;
		}
		*is_remote = false;
		return OK;
	}
	return Controller::PadIsRemoteController(handle, is_remote);
}

// PPSA02385
static int KYTY_SYSV_ABI PadOpenExtStub(int user_id, int type, int index, const void* param) {
	if (Wheel::Enabled()) {
		// Logged by the wheel at a throttled rate: GT7 probes about 9 times a second.
		return PadWheel::OpenExt(user_id, type, index, static_cast<const uint16_t*>(param));
	}
	// Upstream 6a75ca42f: without an extended device, report "not connected" instead of handle 0.
	return Controller::PadOpenExt(user_id, type, index, param);
}

static int KYTY_SYSV_ABI PadReadWheelAware(int handle, Controller::PadData* data, int num) {
	if (PadWheel::IsHandle(handle)) {
		if (data == nullptr || num <= 0) {
			return PadWheel::PAD_ERROR_INVALID_ARG;
		}
		PadWheel::ReadOne(data);
		return 1;
	}
	return Controller::PadRead(handle, data, num);
}

static int KYTY_SYSV_ABI PadReadStateWheelAware(int handle, Controller::PadData* data) {
	if (PadWheel::IsHandle(handle)) {
		if (data == nullptr) {
			return PadWheel::PAD_ERROR_INVALID_ARG;
		}
		return PadWheel::ReadOne(data);
	}
	return Controller::PadReadState(handle, data);
}

// The three report calls have the same shape at every GT7 call site (PPSA01317 1.710 eboot
// 0x15b513c, 0x15ac9a1, 0x15c71a7): edi handle, esi report id (the buffer's first byte for output
// reports), rdx buffer, ecx length; the same order as the OpenOrbis header and RyGT's GT Sport
// logs.
static int KYTY_SYSV_ABI PadOutputReport(int handle, uint32_t type, const uint8_t* buf,
                                         uint32_t length) {
	if (!Wheel::Enabled()) {
		return OK; // as the unresolved-import stub answered before these were bound
	}
	if (!PadWheel::IsHandle(handle)) {
		return PadWheel::PAD_ERROR_INVALID_HANDLE;
	}
	return PadWheel::OutputReport(static_cast<uint8_t>(type), buf, length);
}

static int KYTY_SYSV_ABI PadGetFeatureReport(int handle, uint32_t report_id, uint8_t* buf,
                                             uint32_t length) {
	if (!Wheel::Enabled()) {
		return OK; // as the unresolved-import stub answered before these were bound
	}
	if (!PadWheel::IsHandle(handle)) {
		return PadWheel::PAD_ERROR_INVALID_HANDLE;
	}
	return PadWheel::GetFeatureReport(static_cast<uint8_t>(report_id), buf, length);
}

static int KYTY_SYSV_ABI PadSetFeatureReport(int handle, uint32_t report_id, const uint8_t* buf,
                                             uint32_t length) {
	if (!Wheel::Enabled()) {
		return OK; // as the unresolved-import stub answered before these were bound
	}
	if (!PadWheel::IsHandle(handle)) {
		return PadWheel::PAD_ERROR_INVALID_HANDLE;
	}
	return PadWheel::SetFeatureReport(static_cast<uint8_t>(report_id), buf, length);
}

// ScePadExtControllerInformation: the standard controller's information, then the special
// controller's type and capabilities.
struct PadExtControllerInformation {
	float    touch_pixel_density;
	uint16_t touch_resolution_x;
	uint16_t touch_resolution_y;
	uint8_t  stick_dead_zone_left;
	uint8_t  stick_dead_zone_right;
	uint8_t  connection_type;
	uint8_t  connected_count;
	bool     connected;
	int32_t  device_class;
	uint8_t  reserve[8];
	uint16_t pad_type1;
	uint16_t pad_type2;
	uint8_t  capability;
	union {
		uint8_t quantity_of_selector_switch;
		int32_t max_physical_wheel_angle;
		uint8_t data[8];
	};
};
static_assert(sizeof(PadExtControllerInformation) == 44);

// KYTY_PAD_EXT_INFO=1 (default off): the special controllers (wheels, sticks) a game opens with
// PadOpenExt get information that says not connected (all zeros, OK), as on a console with no
// wheel. Before (and by default) nothing was written and the handle check's result returned, so
// the caller kept whatever its buffer held. Gran Turismo 7 checks for a racing wheel this way.
static bool PadExtInfoEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PAD_EXT_INFO");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

namespace PadWheel {

// The wheel's information. The touch values are the ones controller.cpp reports for a DualSense.
// An absent wheel reads all zero, as a PS4 reads a port with nothing attached (AnyPS5 #9878).
PadExtControllerInformation ControllerInformation() {
	PadExtControllerInformation info {};
	if (!Wheel::Present()) {
		return info;
	}
	info.touch_pixel_density = 44.86f;
	info.touch_resolution_x  = 1920;
	info.touch_resolution_y  = 943;
	info.connection_type     = 0; // local
	info.connected_count     = 1;
	info.connected           = true;
	info.device_class        = PAD_DEVICE_CLASS_STEERING_WHEEL;
	// The wheel's USB identity (GT7's wheel code picks the model from it), no capability flags,
	// the maximum wheel angle.
	info.pad_type1                = Wheel::Fanatec::VENDOR_ID;
	info.pad_type2                = Wheel::Fanatec::PRODUCT_ID;
	info.max_physical_wheel_angle = static_cast<int32_t>(Wheel::GetPhysicalRotation());
	return info;
}

} // namespace PadWheel

static int KYTY_SYSV_ABI
PadGetControllerInformationWheelAware(int handle, Controller::PadControllerInformation* info) {
	if (PadWheel::IsHandle(handle)) {
		if (info == nullptr) {
			return PadWheel::PAD_ERROR_INVALID_ARG;
		}
		// ScePadControllerInformation is the extended information's first 0x1C bytes.
		const auto wheel = PadWheel::ControllerInformation();
		std::memcpy(info, &wheel, 0x1C);
		return OK;
	}
	return Controller::PadGetControllerInformation(handle, info);
}

// PPSA02385
static int KYTY_SYSV_ABI PadGetExtControllerInformationStub(int handle, void* info) {
	PRINT_NAME();

	LOGF("\t handle = %d\n"
	     "\t info   = 0x%016" PRIx64 "\n",
	     handle, reinterpret_cast<uint64_t>(info));

	if (PadExtInfoEnabled()) {
		if (info == nullptr) {
			return -2137915391; /* PAD_ERROR_INVALID_ARG, 0x80920001 */
		}
		std::memset(info, 0, sizeof(PadExtControllerInformation));
		return OK;
	}
	if (PadWheel::IsHandle(handle)) {
		*static_cast<PadExtControllerInformation*>(info) = PadWheel::ControllerInformation();
		return OK;
	}
	return Controller::PadCheckHandle(handle);
}

static int KYTY_SYSV_ABI PadUnknownN3kSX62fgNo(uint64_t arg0, uint64_t arg1, uint64_t arg2,
                                               uint64_t arg3, uint64_t arg4, uint64_t arg5) {
	PRINT_NAME();

	LOGF("\t argc-style args = (0x%016" PRIx64 ", 0x%016" PRIx64 ", 0x%016" PRIx64 ", 0x%016" PRIx64
	     ", 0x%016" PRIx64 ", 0x%016" PRIx64 ")\n",
	     arg0, arg1, arg2, arg3, arg4, arg5);

	if (arg0 >= 0x10000 && arg3 > 0 && arg3 <= 0x1000) {
		std::memset(reinterpret_cast<void*>(arg0), 0, static_cast<size_t>(arg3));
	}

	return OK;
}

LIB_DEFINE(InitPad_1) {
	PRINT_NAME_ENABLE(true);

	LIB_FUNC("hv1luiJrqQM", Controller::PadInit);
	LIB_FUNC("xk0AcarP3V4", Controller::PadOpen);
	LIB_FUNC("WFIiSfXGUq8", PadOpenExtStub);
	LIB_FUNC("u1GRHp+oWoY", Controller::PadGetHandle);
	LIB_FUNC("AcslpN1jHR8", PadDeviceClassGetExtendedInformation);
	LIB_FUNC("IHPqcbc0zCA", PadDeviceClassParseData);
	LIB_FUNC("clVvL4ZDntw", PadSetMotionSensorStateWheelAware);
	LIB_FUNC("r44mAxdSG+U", PadSetAngularVelocityDeadbandStateWheelAware);
	LIB_FUNC("rIZnR6eSpvk", PadResetOrientationWheelAware);
	LIB_FUNC("vDLMoJLde8I", PadSetTiltCorrectionState);
	LIB_FUNC("gjP9-KQzoUk", PadGetControllerInformationWheelAware);
	LIB_FUNC("fCWdlnmB1Ks", PadIsRemoteControllerWheelAware);
	LIB_FUNC("hGbf2QTBmqc", PadGetExtControllerInformationStub);
	LIB_FUNC("2JgFB2n9oUM", PadSetTriggerEffectWheelAware);
	LIB_FUNC("YndgXqQVV7c", PadReadStateWheelAware);
	LIB_FUNC("q1cHNfGycLI", PadReadWheelAware);
	LIB_FUNC("DrUu8cPrje8", PadOutputReport);
	LIB_FUNC("PZSoY8j0Pko", PadGetFeatureReport);
	LIB_FUNC("xqgVCEflEDY", PadSetFeatureReport);
	LIB_FUNC("yFVnOdGxvZY", PadSetVibrationWheelAware);
	LIB_FUNC("W2G-yoyMF5U", PadSetVibrationMode);
	LIB_FUNC("Yq0zOH7YNOM", PadSetVibrationTriggerEffectWeakWhileEmbeddedMicInUse);
	LIB_FUNC("znaWI0gpuo8", Controller::PadGetTriggerEffectState);
	LIB_FUNC("DscD1i9HX1w", PadResetLightBarWheelAware);
	LIB_FUNC("RR4novUEENY", PadSetLightBarWheelAware);
	LIB_FUNC("6ncge5+l5Qs", PadClose);
	LIB_FUNC("n3kSX62fgNo", PadUnknownN3kSX62fgNo);
	LIB_FUNC("iA-DdobUen8", Controller::PadVrControllerReadState);
	LIB_FUNC("v8P+9PRqg10", Controller::PadVrControllerSetTriggerEffects);
}

namespace LibMouse {

LIB_VERSION("Mouse", 1, "Mouse", 1, 1);

namespace Mouse {

constexpr int MOUSE_ERROR_INVALID_ARG    = -2132869119; /* 0x80DF0001 */
constexpr int MOUSE_ERROR_INVALID_HANDLE = -2132869117; /* 0x80DF0003 */

struct MouseData {
	uint64_t timestamp;
	bool     connected;
	uint8_t  padding[3];
	uint32_t buttons;
	int32_t  x_axis;
	int32_t  y_axis;
	int32_t  wheel;
	int32_t  tilt;
	uint8_t  reserved[8];
};

static int KYTY_SYSV_ABI MouseInit() {
	PRINT_NAME();

	return OK;
}

static int KYTY_SYSV_ABI MouseOpen(int user_id, int32_t type, int32_t index, const void* param) {
	PRINT_NAME();

	LOGF("\t user_id = %d\n"
	     "\t type    = %" PRId32 "\n"
	     "\t index   = %" PRId32 "\n"
	     "\t param   = 0x%016" PRIx64 "\n",
	     user_id, type, index, reinterpret_cast<uint64_t>(param));

	if (type != 0 || index < 0 || index >= 2) {
		return MOUSE_ERROR_INVALID_ARG;
	}

	return 1;
}

static int KYTY_SYSV_ABI MouseClose(int32_t handle) {
	PRINT_NAME();

	LOGF("\t handle = %" PRId32 "\n", handle);

	if (handle <= 0) {
		return MOUSE_ERROR_INVALID_HANDLE;
	}

	return OK;
}

static int KYTY_SYSV_ABI MouseRead(int32_t handle, MouseData* data, int32_t num) {
	PRINT_NAME();

	LOGF("\t handle = %" PRId32 "\n"
	     "\t data   = 0x%016" PRIx64 "\n"
	     "\t num    = %" PRId32 "\n",
	     handle, reinterpret_cast<uint64_t>(data), num);

	if (handle <= 0) {
		return MOUSE_ERROR_INVALID_HANDLE;
	}
	if (data == nullptr || num <= 0) {
		return MOUSE_ERROR_INVALID_ARG;
	}

	std::memset(data, 0, sizeof(MouseData) * static_cast<size_t>(num));

	return 0;
}

} // namespace Mouse

LIB_DEFINE(InitMouse_1) {
	LIB_FUNC("cAnT0Rw-IwU", Mouse::MouseClose);
	LIB_FUNC("Qs0wWulgl7U", Mouse::MouseInit);
	LIB_FUNC("RaqxZIf6DvE", Mouse::MouseOpen);
	LIB_FUNC("x8qnXqh-tiM", Mouse::MouseRead);
}

} // namespace LibMouse

namespace LibKeyboard {

LIB_VERSION("Keyboard", 1, "Keyboard", 1, 1);

namespace Keyboard {

constexpr int KEYBOARD_ERROR_INVALID_ARG    = -2133196799; /* 0x80da0001 */
constexpr int KEYBOARD_ERROR_INVALID_HANDLE = -2133196797; /* 0x80da0003 */
constexpr int KEYBOARD_MAX_KEYCODES         = 16;
constexpr int KEYBOARD_MAX_DATA_NUM         = 16;
constexpr int KEYBOARD_MAPPING_101          = 0;
constexpr int KEYBOARD_MAPPING_106          = 1;

struct KeyboardData {
	uint64_t timestamp;
	bool     intercepted;
	uint8_t  reserve1[7];
	bool     connected;
	int32_t  length;
	uint32_t led;
	uint32_t modifier_key;
	uint16_t key_code[KEYBOARD_MAX_KEYCODES];
	uint8_t  reserve2[32];
};

struct KeyboardCharData {
	bool     processed;
	int32_t  length;
	uint16_t char_code;
	uint8_t  reserve[8];
};

static int KYTY_SYSV_ABI KeyboardInit() {
	PRINT_NAME();

	return OK;
}

static int KYTY_SYSV_ABI KeyboardOpen(int user_id, int32_t type, int32_t index, const void* param) {
	PRINT_NAME();

	LOGF("\t user_id = %d\n"
	     "\t type    = %" PRId32 "\n"
	     "\t index   = %" PRId32 "\n"
	     "\t param   = 0x%016" PRIx64 "\n",
	     user_id, type, index, reinterpret_cast<uint64_t>(param));

	if (type != 0 || index < 0 || index >= 2) {
		return KEYBOARD_ERROR_INVALID_ARG;
	}

	return 1;
}

static int KYTY_SYSV_ABI KeyboardClose(int32_t handle) {
	PRINT_NAME();

	LOGF("\t handle = %" PRId32 "\n", handle);

	if (handle <= 0) {
		return KEYBOARD_ERROR_INVALID_HANDLE;
	}

	return OK;
}

static int KYTY_SYSV_ABI KeyboardRead(int32_t handle, KeyboardData* data, int32_t num) {
	PRINT_NAME();

	LOGF("\t handle = %" PRId32 "\n"
	     "\t data   = 0x%016" PRIx64 "\n"
	     "\t num    = %" PRId32 "\n",
	     handle, reinterpret_cast<uint64_t>(data), num);

	if (handle <= 0) {
		return KEYBOARD_ERROR_INVALID_HANDLE;
	}
	if (data == nullptr || num <= 0 || num > KEYBOARD_MAX_DATA_NUM) {
		return KEYBOARD_ERROR_INVALID_ARG;
	}

	std::memset(data, 0, sizeof(KeyboardData) * static_cast<size_t>(num));

	return 0;
}

static int KYTY_SYSV_ABI KeyboardReadState(int32_t handle, KeyboardData* data) {
	PRINT_NAME();

	LOGF("\t handle = %" PRId32 "\n"
	     "\t data   = 0x%016" PRIx64 "\n",
	     handle, reinterpret_cast<uint64_t>(data));

	if (handle <= 0) {
		return KEYBOARD_ERROR_INVALID_HANDLE;
	}
	if (data == nullptr) {
		return KEYBOARD_ERROR_INVALID_ARG;
	}

	std::memset(data, 0, sizeof(KeyboardData));

	return 0;
}

static int KYTY_SYSV_ABI KeyboardGetKey2Char(int32_t handle, int32_t arrange, uint32_t led,
                                             uint32_t modifier_key, uint16_t key_code,
                                             KeyboardCharData* char_data) {
	PRINT_NAME();

	LOGF("\t handle       = %" PRId32 "\n"
	     "\t arrange      = %" PRId32 "\n"
	     "\t led          = %" PRIu32 "\n"
	     "\t modifier_key = %" PRIu32 "\n"
	     "\t key_code     = %" PRIu16 "\n"
	     "\t char_data    = 0x%016" PRIx64 "\n",
	     handle, arrange, led, modifier_key, key_code, reinterpret_cast<uint64_t>(char_data));

	if (handle <= 0) {
		return KEYBOARD_ERROR_INVALID_HANDLE;
	}
	if (char_data == nullptr ||
	    (arrange != KEYBOARD_MAPPING_101 && arrange != KEYBOARD_MAPPING_106)) {
		return KEYBOARD_ERROR_INVALID_ARG;
	}

	std::memset(char_data, 0, sizeof(KeyboardCharData));

	return OK;
}

} // namespace Keyboard

LIB_DEFINE(InitKeyboard_1) {
	LIB_FUNC("wadT3QBCGY0", Keyboard::KeyboardInit);
	LIB_FUNC("HJ+KnEHcaxI", Keyboard::KeyboardOpen);
	LIB_FUNC("0LWei+c7RNc", Keyboard::KeyboardClose);
	LIB_FUNC("6HpE68bzX6M", Keyboard::KeyboardReadState);
	LIB_FUNC("xybbGMCr738", Keyboard::KeyboardRead);
	LIB_FUNC("yO9JwdRhtSA", Keyboard::KeyboardGetKey2Char);
}

} // namespace LibKeyboard

} // namespace Libs
