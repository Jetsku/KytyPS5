// Host mix categories, gains, environment overrides and the level meter (libs/audioMix.h).
#include "libs/audioMix.h"
#include "libs/audioObjects.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

namespace {
namespace Mix     = Libs::Audio::Mix;
namespace Objects = Libs::Audio::Objects;

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
	                                          {"KYTY_AUDIO_MAIN_VOLUME", "bad"},
	                                          {"KYTY_AUDIO_OBJECTS_VOLUME", "60"}};
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
	Check(settings.objects == 60, "objects override ignored");
	Check(Near(Mix::ObjectsGain(settings), 0.6), "objects gain");
	Check(Near(Mix::ObjectsGain(Mix::Settings {}), 1.0), "objects default is unity");
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

bool NearF(float a, float b) {
	return std::abs(a - b) < 1e-4f;
}

void TestObjectAttributes() {
	Objects::Params params;
	const float     position[3] {1.0f, 2.0f, -3.0f};
	Check(Objects::ApplyAttribute(&params, Objects::ATTRIBUTE_POSITION, position, sizeof(position)),
	      "position rejected");
	Check(params.has_position && params.x == 1.0f && params.y == 2.0f && params.z == -3.0f,
	      "position not stored");
	const float gain = 0.5f;
	Check(Objects::ApplyAttribute(&params, Objects::ATTRIBUTE_GAIN, &gain, sizeof(gain)) &&
	          params.gain == 0.5f,
	      "gain not stored");
	const float bad = -1.0f;
	Check(!Objects::ApplyAttribute(&params, Objects::ATTRIBUTE_GAIN, &bad, sizeof(bad)) &&
	          params.gain == 0.5f,
	      "negative gain accepted");
	const double wrong_size = 0.25;
	Check(!Objects::ApplyAttribute(&params, Objects::ATTRIBUTE_GAIN, &wrong_size, sizeof(wrong_size)),
	      "8-byte gain accepted");
	const uint32_t priority = 3;
	Check(!Objects::ApplyAttribute(&params, Objects::ATTRIBUTE_PRIORITY, &priority, sizeof(priority)),
	      "priority changed the placement");
	const uint32_t acn = Objects::AMBISONICS_CHANNEL | 1;
	Check(Objects::ApplyAttribute(&params, Objects::ATTRIBUTE_AMBISONICS, &acn, sizeof(acn)) &&
	          params.ambisonics == acn,
	      "ambisonics not stored");
}

void TestObjectPanning() {
	float           g[Objects::MAX_CHANNELS] {};
	Objects::Params centre;
	Check(Objects::SpeakerGains(centre, 2, g) && NearF(g[0], Objects::HALF_POWER) &&
	          NearF(g[1], Objects::HALF_POWER),
	      "an object without position is not centred");

	Objects::Params right;
	right.has_position = true;
	right.x            = 5.0f;
	right.gain         = 0.5f;
	Check(Objects::SpeakerGains(right, 2, g) && NearF(g[0], 0.0f) && NearF(g[1], 0.5f),
	      "an object on the right is not panned right with its gain");

	Objects::Params front_left;
	front_left.has_position = true;
	front_left.x            = -1.0f;
	front_left.z            = 1.0f;
	Check(Objects::SpeakerGains(front_left, 2, g) && g[0] > g[1] &&
	          NearF(g[0] * g[0] + g[1] * g[1], 1.0f),
	      "front-left pan is not constant power");

	// Behind on a 7.1 bed: fronts silent, power split over both surround pairs; downmixed with
	// 0.707 per surround it is as loud as the same object in front.
	Objects::Params behind;
	behind.has_position = true;
	behind.z            = -2.0f;
	Check(Objects::SpeakerGains(behind, 8, g), "behind object dropped");
	Check(NearF(g[0], 0.0f) && NearF(g[1], 0.0f) && NearF(g[2], 0.0f) && NearF(g[3], 0.0f),
	      "behind object reached the front or LFE");
	const float downmixed_left = Objects::HALF_POWER * (g[4] + g[6]);
	Check(NearF(downmixed_left, Objects::HALF_POWER), "behind object changes level in a downmix");

	Objects::Params diffuse = right;
	diffuse.spread          = Objects::PI;
	Check(Objects::SpeakerGains(diffuse, 2, g) && NearF(g[0], g[1]), "full spread is not centred");

	Objects::Params passthrough;
	passthrough.passthrough = Objects::PASSTHROUGH_LEFT;
	Check(Objects::SpeakerGains(passthrough, 8, g) && NearF(g[0], 1.0f) && NearF(g[1], 0.0f),
	      "left passthrough");

	Objects::Params ambi_y;
	ambi_y.ambisonics = Objects::AMBISONICS_CHANNEL | 1;
	Check(Objects::SpeakerGains(ambi_y, 2, g) && NearF(g[0], Objects::HALF_POWER) &&
	          NearF(g[1], -Objects::HALF_POWER),
	      "ambisonics Y decode");
	Objects::Params ambi_high;
	ambi_high.ambisonics = Objects::AMBISONICS_CHANNEL | 4;
	Check(!Objects::SpeakerGains(ambi_high, 2, g), "higher-order ambisonics not skipped");

	Objects::Params silent;
	silent.gain = 0.0f;
	Check(!Objects::SpeakerGains(silent, 2, g), "silent object mixed");

	float       bed[2 * 4] {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
	const float mono[4] {1.0f, -1.0f, 0.5f, 0.0f};
	Objects::SpeakerGains(right, 2, g);
	Objects::MixInto(bed, 2, 4, mono, g, 2.0f);
	Check(NearF(bed[0], 0.1f) && NearF(bed[1], 1.1f) && NearF(bed[3], -0.9f) && NearF(bed[5], 0.6f),
	      "MixInto with the objects gain");
}

int main() {
	TestCategories();
	TestGains();
	TestEnvironment();
	TestLevelMeter();
	TestObjectAttributes();
	TestObjectPanning();
	std::puts("AudioMixTests: all cases passed");
}
