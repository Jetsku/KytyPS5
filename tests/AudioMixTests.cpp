// Host mix categories, gains, environment overrides and the level meter (libs/audioMix.h).
#include "libs/audioMix.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

namespace {
namespace Mix = Libs::Audio::Mix;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "AudioMixTests: %s\n", message);
		std::abort();
	}
}

bool Near(double a, double b) {
	return std::abs(a - b) < 1e-6;
}

void TestCategories() {
	Check(Mix::CategoryOf(Mix::PORT_TYPE_MAIN, true) == Mix::Category::Main, "main");
	Check(Mix::CategoryOf(Mix::PORT_TYPE_VOICE, true) == Mix::Category::Main, "voice");
	Check(Mix::CategoryOf(Mix::PORT_TYPE_PERSONAL, true) == Mix::Category::Main, "personal");
	Check(Mix::CategoryOf(Mix::PORT_TYPE_AUX, true) == Mix::Category::Main, "aux");
	Check(Mix::CategoryOf(Mix::PORT_TYPE_AUDIO3D, true) == Mix::Category::Main, "audio3d");
	Check(Mix::CategoryOf(Mix::PORT_TYPE_BGM, true) == Mix::Category::Music, "bgm");
	Check(Mix::CategoryOf(Mix::PORT_TYPE_PADSPK, true) == Mix::Category::PadSpeakerOnMain,
	      "pad speaker on the main output");
	Check(Mix::CategoryOf(Mix::PORT_TYPE_PADSPK, false) == Mix::Category::Controller,
	      "pad speaker on a DualSense");
	Check(Mix::CategoryOf(Mix::PORT_TYPE_VIBRATION, true) == Mix::Category::Controller,
	      "vibration");
}

void TestGains() {
	const Mix::Settings unity;
	for (const int type: {Mix::PORT_TYPE_MAIN, Mix::PORT_TYPE_BGM, Mix::PORT_TYPE_VOICE,
	                      Mix::PORT_TYPE_PADSPK, Mix::PORT_TYPE_AUX, Mix::PORT_TYPE_AUDIO3D}) {
		Check(Mix::MainOutputGain(unity, type, true) == 1.0f, "default settings are not unity");
	}

	Mix::Settings settings;
	settings.master      = 80;
	settings.main        = 50;
	settings.music       = 150;
	settings.pad_on_main = 30;
	Check(Near(Mix::MainOutputGain(settings, Mix::PORT_TYPE_MAIN, true), 0.4), "main gain");
	Check(Near(Mix::MainOutputGain(settings, Mix::PORT_TYPE_AUX, true), 0.4), "aux gain");
	Check(Near(Mix::MainOutputGain(settings, Mix::PORT_TYPE_BGM, true), 1.2), "music gain");
	Check(Near(Mix::MainOutputGain(settings, Mix::PORT_TYPE_PADSPK, true), 0.24),
	      "pad speaker on main gain");
	// Samples a DualSense plays are scaled by the controller settings only.
	Check(Mix::MainOutputGain(settings, Mix::PORT_TYPE_PADSPK, false) == 1.0f,
	      "DualSense speaker took a main output gain");
	Check(Mix::MainOutputGain(settings, Mix::PORT_TYPE_MAIN, false) == 1.0f,
	      "off-main samples took a gain");

	settings.master = 0;
	Check(Mix::MainOutputGain(settings, Mix::PORT_TYPE_BGM, true) == 0.0f, "master 0 is not mute");
	settings.master = 1000; // Clamped to MAX_PERCENT.
	settings.music  = 100;
	Check(Near(Mix::MainOutputGain(settings, Mix::PORT_TYPE_BGM, true), 2.0),
	      "gain not clamped to 200%");
}

void TestEnvironment() {
	uint32_t value = 7;
	Check(!Mix::ParsePercent(nullptr, &value) && value == 7, "null accepted");
	Check(!Mix::ParsePercent("", &value) && value == 7, "empty accepted");
	Check(!Mix::ParsePercent("abc", &value) && value == 7, "text accepted");
	Check(!Mix::ParsePercent("50%", &value) && value == 7, "suffix accepted");
	Check(!Mix::ParsePercent("201", &value) && value == 7, "201 accepted");
	Check(Mix::ParsePercent("0", &value) && value == 0, "0 rejected");
	Check(Mix::ParsePercent("200", &value) && value == 200, "200 rejected");

	std::map<std::string, const char*> env = {{"KYTY_AUDIO_MUSIC_VOLUME", "140"},
	                                          {"KYTY_AUDIO_PAD_SPEAKER_ON_MAIN_VOLUME", "20"},
	                                          {"KYTY_AUDIO_MAIN_VOLUME", "bad"}};
	Mix::Settings base;
	base.main   = 90;
	base.master = 70;
	const auto settings = Mix::ApplyEnvironment(base, [&env](const char* name) -> const char* {
		const auto it = env.find(name);
		return it != env.end() ? it->second : nullptr;
	});
	Check(settings.music == 140, "music override ignored");
	Check(settings.pad_on_main == 20, "pad speaker override ignored");
	Check(settings.main == 90, "invalid override replaced the configured value");
	Check(settings.master == 70, "unset override replaced the configured value");
}

void TestLevelMeter() {
	Mix::LevelMeter meter;
	Check(meter.Rms() == 0.0 && meter.Peak() == 0.0 && meter.Samples() == 0, "empty meter");
	Check(Mix::LevelMeter::ToDb(0.0) == -180.0, "silence dB");

	std::array<float, 8> square {0.5f, -0.5f, 0.5f, -0.5f, 0.5f, -0.5f, 0.5f, -0.5f};
	meter.Add(square.data(), 4, 2, true);
	Check(meter.Samples() == 8, "float sample count");
	Check(Near(meter.Rms(), 0.5) && Near(meter.Peak(), 0.5), "float square wave level");
	Check(Near(Mix::LevelMeter::ToDb(meter.Rms()), 20.0 * std::log10(0.5)), "dB conversion");

	meter.Reset();
	std::array<int16_t, 4> pcm {16384, -16384, 0, 0};
	meter.Add(pcm.data(), 4, 1, false);
	Check(Near(meter.Peak(), 0.5), "int16 peak");
	Check(Near(meter.Rms(), std::sqrt(0.125)), "int16 RMS");

	// Non-finite guest samples are skipped rather than poisoning the RMS.
	meter.Reset();
	const float bad[2] = {NAN, 0.25f};
	meter.Add(bad, 2, 1, true);
	Check(std::isfinite(meter.Rms()) && Near(meter.Peak(), 0.25), "NaN poisoned the meter");
	meter.Add(nullptr, 16, 2, true);
	Check(meter.Samples() == 2, "null block counted");
}
} // namespace

int main() {
	TestCategories();
	TestGains();
	TestEnvironment();
	TestLevelMeter();
	std::puts("AudioMixTests: all cases passed");
}
