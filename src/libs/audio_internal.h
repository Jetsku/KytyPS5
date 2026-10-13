#ifndef EMULATOR_INCLUDE_EMULATOR_AUDIO_INTERNAL_H_
#define EMULATOR_INCLUDE_EMULATOR_AUDIO_INTERNAL_H_

#include "common/common.h"

namespace Libs::Audio::AudioInternal {

enum class Format {
	Unknown,
	Signed16bitMono,
	Signed16bitStereo,
	Signed16bit8Ch,
	FloatMono,
	FloatStereo,
	Float8Ch,
	Signed16bit8ChStd,
	Float8ChStd,
	Float12Ch,
};

struct OutputParam {
	int          handle = 0;
	const void*  data   = nullptr;
	const float* gains  = nullptr;
};

static constexpr int OUT_PORTS_MAX = 32;

int      AudioOutOpen(int type, uint32_t samples_num, uint32_t freq, Format format);
void     AudioOutClose(int handle);
bool     AudioOutHasDevice(int handle);
uint32_t AudioOutOutputs(const OutputParam* params, uint32_t num, bool blocking = true);
// The host gain of AudioOut2 3D object ports relative to the bed they are mixed into.
float    AudioOutObjectsGain();
// False when 3D audio objects are switched off: object PCM is dropped, not mixed.
bool     AudioOutObjectsEnabled();
// KYTY_AUDIO_TRACE BUZZ detector: which AudioOut2 port types it watches, and how many identical
// blocks make one second.
bool     AudioOut2BuzzWatchesPortType(uint16_t port_type);
uint32_t AudioOut2BuzzThresholdBlocks(uint32_t sampling_freq, uint32_t frames_per_block);

} // namespace Libs::Audio::AudioInternal

#endif // EMULATOR_INCLUDE_EMULATOR_AUDIO_INTERNAL_H_
