#ifndef EMULATOR_SRC_LIBS_WHEEL_FANATEC_H_
#define EMULATOR_SRC_LIBS_WHEEL_FANATEC_H_

#include "common/common.h"
#include "libs/wheelHost.h"

namespace Libs::Controller {
struct PadData;
} // namespace Libs::Controller

// The emulated wheel as a Fanatec Gran Turismo DD Pro in its PlayStation mode, through libScePad:
// the game opens it with scePadOpenExt by Fanatec's vendor id and a product range, reads it from
// the pad record's device-unique block, asks the base's setup in feature reports and sends force
// feedback as output reports carrying Fanatec's slot commands (the protocol of the Linux
// hid-fanatecff driver).
namespace Libs::Wheel::Fanatec {

constexpr uint16_t VENDOR_ID = 0x0EB7;
// The GT DD Pro in its PlayStation mode ("FANATEC WHEEL(PS5 / PS4 MODE)").
constexpr uint16_t PRODUCT_ID = 0x0E07;

// Whether a scePadOpenExt request (vendor, first and last product id) asks for this wheel.
bool Matches(uint16_t vendor_id, uint16_t product_first, uint16_t product_last);

// The wheel's pad record: buttons, centred sticks and the device-unique block.
void FillPadData(const Input& input, Controller::PadData* data);

// The base's answer to a feature-report request; false for a report it does not answer.
bool FeatureReport(uint8_t report_id, uint8_t* buf, uint32_t length);

struct Command {
	enum class Kind { SetAndPlay, Stop, StopAll, Rotation, Ignored, Unknown };

	Kind     kind = Kind::Unknown;
	int      slot = 0;
	Effect   effect;
	uint32_t degrees = 0;
};

// One output report: the report id, then a 7-byte command.
Command DecodeOutputReport(const uint8_t* buf, uint32_t length);

} // namespace Libs::Wheel::Fanatec

#endif /* EMULATOR_SRC_LIBS_WHEEL_FANATEC_H_ */
