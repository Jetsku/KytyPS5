#ifndef EMULATOR_INCLUDE_EMULATOR_AUDIO_MIX_H_
#define EMULATOR_INCLUDE_EMULATOR_AUDIO_MIX_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// Host-side mix levels for the guest's audio output ports, and a level meter for them.
//
// The console mixes each AudioOut port type into its own destination: MAIN/AUX/AUDIO3D/VOICE/
// PERSONAL go to the TV, BGM goes to the TV as a separately ducked music bus, and PADSPK goes to
// the DualSense speaker. Without a DualSense the emulator plays PADSPK on the TV output too, where
// it lands at full scale on top of the main mix; the console's tiny controller speaker is far
// quieter than that. These settings let the user balance the categories that the guest keeps
// apart.
namespace Libs::Audio::Mix {

// Port types of sceAudioOutOpen (AudioOut2 port types are mapped onto these).
constexpr int PORT_TYPE_MAIN      = 0;
constexpr int PORT_TYPE_BGM       = 1;
constexpr int PORT_TYPE_VOICE     = 2;
constexpr int PORT_TYPE_PERSONAL  = 3;
constexpr int PORT_TYPE_PADSPK    = 4;
constexpr int PORT_TYPE_VIBRATION = 10;
constexpr int PORT_TYPE_AUDIO3D   = 126;
constexpr int PORT_TYPE_AUX       = 127;

enum class Category { Main, Music, PadSpeakerOnMain, Controller };

// Percentages; 100 is unity gain.
struct Settings {
	static constexpr uint32_t MAX_PERCENT = 200;

	uint32_t master       = 100;
	uint32_t main         = 100;
	uint32_t music        = 100;
	uint32_t pad_on_main  = 100;
	// AudioOut2 3D object ports, mixed into the main bed (so main and master apply on top).
	uint32_t objects      = 100;
};

// Which mix category a port's samples belong to. A pad speaker port counts as "on main" only when
// no DualSense took the samples; the controller itself has its own speaker volume.
inline Category CategoryOf(int port_type, bool on_main_output) {
	switch (port_type) {
		case PORT_TYPE_BGM: return Category::Music;
		case PORT_TYPE_PADSPK: return on_main_output ? Category::PadSpeakerOnMain : Category::Controller;
		case PORT_TYPE_VIBRATION: return Category::Controller;
		default: return Category::Main;
	}
}

inline float PercentToGain(uint32_t percent) {
	return static_cast<float>(std::min(percent, Settings::MAX_PERCENT)) / 100.0f;
}

// The host gain for samples of `port_type` that play on the main (TV) output, or 1 for samples
// that go to a DualSense (its speaker and vibration have their own controller settings).
inline float MainOutputGain(const Settings& settings, int port_type, bool on_main_output) {
	if (!on_main_output) {
		return 1.0f;
	}
	const float master = PercentToGain(settings.master);
	switch (CategoryOf(port_type, on_main_output)) {
		case Category::Music: return master * PercentToGain(settings.music);
		case Category::PadSpeakerOnMain: return master * PercentToGain(settings.pad_on_main);
		case Category::Controller: return 1.0f;
		case Category::Main:
		default: return master * PercentToGain(settings.main);
	}
}

// The gain of AudioOut2 object ports relative to the main bed they are mixed into. The bed then
// gets MainOutputGain (master x main), so "game sound" and master cover the objects as well.
inline float ObjectsGain(const Settings& settings) {
	return PercentToGain(settings.objects);
}

// Parses an environment value 0..MAX_PERCENT; returns false (leaving *percent alone) otherwise.
inline bool ParsePercent(const char* text, uint32_t* percent) {
	if (text == nullptr || text[0] == '\0') {
		return false;
	}
	char*      end   = nullptr;
	const auto value = std::strtoul(text, &end, 10);
	if (end == text || *end != '\0' || value > Settings::MAX_PERCENT) {
		return false;
	}
	*percent = static_cast<uint32_t>(value);
	return true;
}

// Environment overrides: KYTY_AUDIO_MASTER_VOLUME, KYTY_AUDIO_MAIN_VOLUME,
// KYTY_AUDIO_MUSIC_VOLUME, KYTY_AUDIO_PAD_SPEAKER_ON_MAIN_VOLUME and KYTY_AUDIO_OBJECTS_VOLUME, in
// percent (0..200).
template <class GetEnv>
inline Settings ApplyEnvironment(Settings settings, GetEnv get_env) {
	ParsePercent(get_env("KYTY_AUDIO_MASTER_VOLUME"), &settings.master);
	ParsePercent(get_env("KYTY_AUDIO_MAIN_VOLUME"), &settings.main);
	ParsePercent(get_env("KYTY_AUDIO_MUSIC_VOLUME"), &settings.music);
	ParsePercent(get_env("KYTY_AUDIO_PAD_SPEAKER_ON_MAIN_VOLUME"), &settings.pad_on_main);
	ParsePercent(get_env("KYTY_AUDIO_OBJECTS_VOLUME"), &settings.objects);
	return settings;
}

// Accumulates RMS and peak over interleaved PCM (int16 or float), scaled to full scale = 1.
class LevelMeter {
public:
	void Add(const void* data, uint32_t frames, uint32_t channels, bool is_float) {
		if (data == nullptr) {
			return;
		}
		const auto count = static_cast<uint64_t>(frames) * channels;
		if (is_float) {
			const auto* samples = static_cast<const float*>(data);
			for (uint64_t i = 0; i < count; i++) {
				Sample(samples[i]);
			}
		} else {
			const auto* samples = static_cast<const int16_t*>(data);
			for (uint64_t i = 0; i < count; i++) {
				Sample(static_cast<float>(samples[i]) / 32768.0f);
			}
		}
		m_samples += count;
	}

	[[nodiscard]] uint64_t Samples() const { return m_samples; }
	[[nodiscard]] double   Rms() const {
		return m_samples != 0 ? std::sqrt(m_sum_squares / static_cast<double>(m_samples)) : 0.0;
	}
	[[nodiscard]] double Peak() const { return m_peak; }

	static double ToDb(double level) { return level > 1e-9 ? 20.0 * std::log10(level) : -180.0; }

	void Reset() { *this = {}; }

private:
	void Sample(float value) {
		if (!std::isfinite(value)) {
			return;
		}
		const double v = value;
		m_sum_squares += v * v;
		m_peak = std::max(m_peak, std::abs(v));
	}

	double   m_sum_squares = 0.0;
	double   m_peak        = 0.0;
	uint64_t m_samples     = 0;
};

} // namespace Libs::Audio::Mix

#endif // EMULATOR_INCLUDE_EMULATOR_AUDIO_MIX_H_
