#include "libs/wheelFanatec.h"

#include "libs/controller.h"
#include "libs/padData.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace Libs::Wheel::Fanatec {

using Config::Wheel::Control;

bool Matches(uint16_t vendor_id, uint16_t product_first, uint16_t product_last) {
	// An inclusive range; no last id asks for the first one alone.
	const uint16_t last = product_last != 0 ? product_last : product_first;
	return vendor_id == VENDOR_ID && PRODUCT_ID >= product_first && PRODUCT_ID <= last;
}

void FillPadData(const Input& in, Controller::PadData* data) {
	struct ButtonBit {
		Control  control;
		uint32_t bit;
	};
	static constexpr std::array<ButtonBit, 16> Buttons = {{
	    {Control::Cross, Controller::PAD_BUTTON_CROSS},
	    {Control::Circle, Controller::PAD_BUTTON_CIRCLE},
	    {Control::Square, Controller::PAD_BUTTON_SQUARE},
	    {Control::Triangle, Controller::PAD_BUTTON_TRIANGLE},
	    {Control::Up, Controller::PAD_BUTTON_UP},
	    {Control::Down, Controller::PAD_BUTTON_DOWN},
	    {Control::Left, Controller::PAD_BUTTON_LEFT},
	    {Control::Right, Controller::PAD_BUTTON_RIGHT},
	    {Control::L1, Controller::PAD_BUTTON_L1},
	    {Control::R1, Controller::PAD_BUTTON_R1},
	    {Control::L2, Controller::PAD_BUTTON_L2},
	    {Control::R2, Controller::PAD_BUTTON_R2},
	    {Control::L3, Controller::PAD_BUTTON_L3},
	    {Control::R3, Controller::PAD_BUTTON_R3},
	    {Control::Options, Controller::PAD_BUTTON_OPTIONS},
	    {Control::TouchPad, Controller::PAD_BUTTON_TOUCH_PAD},
	}};

	std::memset(data, 0, sizeof(*data));
	if (in.present) {
		for (const auto& b: Buttons) {
			if (in.Pressed(b.control)) {
				data->buttons |= b.bit;
			}
		}
	}
	data->left_stick_x    = 0x80;
	data->left_stick_y    = 0x80;
	data->right_stick_x   = 0x80;
	data->right_stick_y   = 0x80;
	data->orientation_w   = 1.0f;
	data->connected       = in.present;
	data->connected_count = in.present ? 1 : 0;

	// Device-unique block, as GT7 reads it: u16 steering centred at 0x8000; u16 accelerator (2),
	// brake (4), clutch (6) and handbrake (9) from 0xFFFF released to 0 pressed; the H-shifter's
	// gear in byte 8, one bit per gear and bit 7 for reverse.
	auto& d                      = data->device_unique_data;
	data->device_unique_data_len = 12;
	auto put                     = [&](int at, uint16_t v) {
		d[at]     = static_cast<uint8_t>(v & 0xFF);
		d[at + 1] = static_cast<uint8_t>(v >> 8);
	};
	auto pedal = [&](float pressed) {
		const float p = in.present ? std::clamp(pressed, 0.0f, 1.0f) : 0.0f;
		return static_cast<uint16_t>(0xFFFF - std::lround(p * 65535.0f));
	};
	const float steering = in.present ? in.steering : 0.0f;
	put(0,
	    static_cast<uint16_t>(std::lround((std::clamp(steering, -1.0f, 1.0f) + 1.0f) * 32767.5f)));
	put(2, pedal(in.throttle));
	put(4, pedal(in.brake));
	put(6, pedal(in.clutch));
	put(9, pedal(in.handbrake));
	if (in.present && in.gear != 0) {
		d[8] = in.gear < 0 ? 0x80 : static_cast<uint8_t>(1u << (in.gear - 1));
	}
}

bool FeatureReport(uint8_t report_id, uint8_t* buf, uint32_t length) {
	if (buf == nullptr || length == 0) {
		return false;
	}
	uint8_t report[64] {};
	report[0] = report_id;
	switch (report_id) {
		case 0x31:
			// The base's setup, with the values RyGT answers GT Sport with: byte 2 a status (1),
			// byte 3 the rim, byte 6 the pedals on the base (1: ClubSport with a clutch; 0 makes
			// the game ignore the pedals), bytes 7-8 the firmware, big endian (16-bit constant
			// forces need 674 or later).
			report[2] = 0x01;
			report[3] = 0x01;
			report[6] = 0x01;
			report[7] = 0x02;
			report[8] = 0xB0;
			break;
		case 0x33:
			// The rim's extra inputs, all at rest; unanswered, the game's input update fails.
			break;
		default: return false;
	}
	std::memcpy(buf, report, std::min<uint32_t>(length, sizeof(report)));
	return true;
}

Command DecodeOutputReport(const uint8_t* buf, uint32_t length) {
	Command cmd;
	if (buf == nullptr || length < 8) {
		return cmd;
	}
	const uint8_t* c = buf + 1;

	if ((c[0] & 0xF0) == 0xF0) {
		// Not a slot: 0xF3 stops every effect, 0xF8 0x81 sets the rotation lock to lock in degrees.
		// The others (autocentre, display, LEDs) have nothing to drive.
		cmd.kind = Command::Kind::Ignored;
		if (c[0] == 0xF3) {
			cmd.kind = Command::Kind::StopAll;
		} else if (c[0] == 0xF8 && c[1] == 0x81) {
			cmd.degrees = c[2] | static_cast<uint32_t>(c[3]) << 8u;
			cmd.kind    = cmd.degrees >= 90 && cmd.degrees <= 2880 ? Command::Kind::Rotation
			                                                       : Command::Kind::Unknown;
		}
		return cmd;
	}
	// Slot commands: the slot in the high nibble, 1 play (with the effect's new parameters) or 3
	// stop in the low one. Slots 3 and 4 are inertia and friction, as in hid-fanatecff.
	cmd.slot = c[0] >> 4u;
	switch (c[0] & 0x0F) {
		case 0x01: break;
		case 0x03: cmd.kind = Command::Kind::Stop; return cmd;
		default: return cmd;
	}
	auto coeff = [](uint8_t k) { return static_cast<float>(k & 0x0F) / 15.0f; };
	auto clip  = [](uint8_t v) { return v == 0 ? 1.0f : static_cast<float>(v) / 255.0f; };

	Effect& e = cmd.effect;
	switch (c[1]) {
		case 0x08:
			// Constant: 16-bit around 0x8000 when byte 6 is 1, else 8-bit around 0x80.
			e.type  = EffectType::Constant;
			e.level = c[6] == 0x01 ? static_cast<float>((c[2] | c[3] << 8) - 0x8000) / 32767.0f
			                       : static_cast<float>(c[2] - 0x80) / 127.0f;
			e.level = std::clamp(e.level, -1.0f, 1.0f);
			break;
		case 0x0B: {
			// Spring: the dead band's edges (0x80 the centre), the stiffness nibbles, the clip (0
			// is full force), as RyGT reads them.
			const float left  = (static_cast<float>(c[2]) - 128.0f) / 128.0f;
			const float right = (static_cast<float>(c[3]) - 128.0f) / 128.0f;
			e.type            = EffectType::Spring;
			e.center          = (left + right) * 0.5f;
			e.dead_band       = std::max(right - left, 0.0f) * 0.5f;
			e.left_coeff      = coeff(c[4]);
			e.right_coeff     = coeff(c[4] >> 4u);
			e.left_sat = e.right_sat = clip(c[6]);
			break;
		}
		case 0x0C:
			// Damper, inertia or friction by slot: the coefficients' low nibbles, the clip. The
			// sign bytes are not applied, as a negative damper would push with the motion.
			e.type        = cmd.slot == 4   ? EffectType::Friction
			                : cmd.slot == 3 ? EffectType::Inertia
			                                : EffectType::Damper;
			e.left_coeff  = coeff(c[2]);
			e.right_coeff = coeff(c[4]);
			e.left_sat = e.right_sat = clip(c[6]);
			break;
		default: return cmd;
	}
	cmd.kind = Command::Kind::SetAndPlay;
	return cmd;
}

} // namespace Libs::Wheel::Fanatec
