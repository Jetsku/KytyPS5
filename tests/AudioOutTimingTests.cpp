// Exercise production output scheduling with a deterministic clock and consuming SDL queues.
#include "common/threads.h"
#include "libs/dualSenseHaptics.h"

#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

namespace {
void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "AudioOutTimingTests: %s\n", message);
		std::abort();
	}
}

struct Stream {
	SDL_AudioSpec spec;
	uint64_t      frames = 0; // Millionths of a frame retain exact simulated consumption.
	uint64_t      updated;
	std::vector<uint8_t> pcm;
};

uint64_t                             now       = 1000000;
uint32_t                             oversleep = 0, processing = 0;
std::vector<uint32_t>                sleeps;
std::vector<std::unique_ptr<Stream>> streams;
bool                                 fail_open = false, fail_put = false, stalled = false;
bool                                 pad_connected = false, pad_bluetooth = false;
uint64_t                             pad_queue_us = 0;
const float*                         queued_pad_gains = nullptr;
int                                  clears = 0;
float                                last_put_sample = 0.0f; // First sample of the last queued block.

Stream& GetStream(SDL_AudioStream* stream) {
	return *reinterpret_cast<Stream*>(stream);
}

int64_t device_ppm = 0; // How much faster (or slower) than the process clock the device plays.

void Drain(Stream& stream) {
	if (!stalled) {
		const auto elapsed  = (now - stream.updated) * stream.spec.freq;
		const auto consumed = static_cast<uint64_t>(static_cast<int64_t>(elapsed) +
		                                            static_cast<int64_t>(elapsed) * device_ppm / 1000000);
		stream.frames -= std::min(stream.frames, consumed);
	}
	stream.updated = now;
}
} // namespace

namespace Fake {
bool InitSubSystem(SDL_InitFlags) {
	return true;
}
void QuitSubSystem(SDL_InitFlags) {}
bool ResumeAudioStreamDevice(SDL_AudioStream*) {
	return true;
}

SDL_AudioStream* OpenAudioDeviceStream(SDL_AudioDeviceID, const SDL_AudioSpec* spec,
                                       SDL_AudioStreamCallback, void*) {
	if (fail_open) {
		return nullptr;
	}

	streams.push_back(std::make_unique<Stream>(Stream {*spec, 0, now, {}}));
	return reinterpret_cast<SDL_AudioStream*>(streams.back().get());
}

void DestroyAudioStream(SDL_AudioStream* stream) {
	const auto it = std::find_if(streams.begin(), streams.end(), [stream](const auto& value) {
		return value.get() == &GetStream(stream);
	});
	Check(it != streams.end(), "unknown stream destroyed");
	streams.erase(it);
}

int GetAudioStreamQueued(SDL_AudioStream* stream) {
	auto& state = GetStream(stream);
	Drain(state);
	return static_cast<int>(state.frames / 1000000 * SDL_AUDIO_BYTESIZE(state.spec.format) *
	                        state.spec.channels);
}

bool PutAudioStreamData(SDL_AudioStream* stream, const void* data, int bytes) {
	now += processing;
	if (fail_put) {
		return false;
	}
	if (data != nullptr && bytes >= static_cast<int>(sizeof(float))) {
		last_put_sample = *static_cast<const float*>(data);
	}
	auto& state = GetStream(stream);
	const auto* pcm = static_cast<const uint8_t*>(data);
	state.pcm.assign(pcm, pcm + bytes);
	Drain(state);
	state.frames += static_cast<uint64_t>(bytes) * 1000000 /
	                (SDL_AUDIO_BYTESIZE(state.spec.format) * state.spec.channels);
	return true;
}

bool ClearAudioStream(SDL_AudioStream* stream) {
	auto& state   = GetStream(stream);
	state.frames  = 0;
	state.updated = now;
	clears++;
	return true;
}
} // namespace Fake

namespace Common {
class TimingThread: public Thread {
public:
	static void SleepMicro(uint32_t micros) {
		sleeps.push_back(micros);
		now += micros + oversleep;
	}
};
} // namespace Common

#define Thread                      TimingThread
#define SDL_InitSubSystem           Fake::InitSubSystem
#define SDL_QuitSubSystem           Fake::QuitSubSystem
#define SDL_OpenAudioDeviceStream   Fake::OpenAudioDeviceStream
#define SDL_ResumeAudioStreamDevice Fake::ResumeAudioStreamDevice
#define SDL_DestroyAudioStream      Fake::DestroyAudioStream
#define SDL_GetAudioStreamQueued    Fake::GetAudioStreamQueued
#define SDL_PutAudioStreamData      Fake::PutAudioStreamData
#define SDL_ClearAudioStream        Fake::ClearAudioStream
#include "libs/audio.cpp"
#undef Thread
#undef SDL_InitSubSystem
#undef SDL_QuitSubSystem
#undef SDL_OpenAudioDeviceStream
#undef SDL_ResumeAudioStreamDevice
#undef SDL_DestroyAudioStream
#undef SDL_GetAudioStreamQueued
#undef SDL_PutAudioStreamData
#undef SDL_ClearAudioStream

namespace Libs::Controller {
int GetActiveControllerId() {
	return 0;
}
float GetSettingScale(Setting) {
	return 1.0f;
}

namespace DualSenseHaptics {
Stream* Open(uint32_t, bool) {
	static int pad_stream;
	return pad_connected ? reinterpret_cast<Stream*>(&pad_stream) : nullptr;
}
void     Close(Stream*) {}
bool UsesBluetooth(const Stream* stream) {
	return stream != nullptr && pad_bluetooth;
}
uint64_t Queue(Stream* stream, int, const void*, uint32_t, uint32_t, bool, const int*, float, const float* gains) {
	queued_pad_gains = gains;
	return stream != nullptr ? pad_queue_us : 0;
}
} // namespace DualSenseHaptics
} // namespace Libs::Controller

namespace Libs::LibKernel {
uint64_t KYTY_SYSV_ABI KernelGetProcessTime() {
	return now;
}
} // namespace Libs::LibKernel

namespace Loader::Timer {
double GetTimeMs() {
	return static_cast<double>(now) / 1000;
}
} // namespace Loader::Timer

namespace {
using Audio = Libs::Audio::Audio;

struct Fixture {
	Audio                   audio;
	std::array<float, 2048> pcm {};

	Fixture() {
		Check(streams.empty(), "stream leaked between tests");
		now        = 1000000;
		processing = oversleep = 0;
		clears                 = 0;
		fail_open = fail_put = stalled = false;
		pad_connected = pad_bluetooth = false;
		pad_queue_us = 0;
		queued_pad_gains = nullptr;
		sleeps.clear();
	}

	Audio::Id Open(uint32_t frames = 256, int type = 0) {
		const auto id = audio.AudioOutOpen(type, frames, 48000, Audio::Format::FloatStereo);
		Check(id.IsValid(), "port open failed");
		return id;
	}

	void Output(Audio::Id port, bool blocking = true) {
		Audio::OutputParam param {port, pcm.data()};
		audio.AudioOutOutputs(&param, 1, blocking);
	}

	void Prime(Audio::Id port) {
		const auto start = now;
		for (int i = 0; i < 8; i++) {
			Output(port);
		}
		Check(now == start, "priming was paced before the 40 ms cushion filled");
		Output(port);
		Check(now - start >= 5333 && now - start <= 5334,
		      "priming accumulated a multi-block deadline");
		Drain(*streams.front());
		Check(streams.front()->frames >= uint64_t {7 * 256} * 1000000,
		      "first paced output consumed the priming cushion");
	}
};

void TestPrimingAndSteadyCadence() {
	Fixture    f;
	const auto port = f.Open();
	f.Prime(port);
	processing = 125;
	oversleep  = 175;
	for (int i = 0; i < 10; i++) {
		f.Output(port);
	}
	const auto start = now;
	for (int i = 0; i < 3000; i++) {
		f.Output(port);
	}
	Check(now - start >= 15999999 && now - start <= 16000001,
	      "processing, sleep overshoot, or fractional periods accumulated timing drift");
}

void TestUnderrunAndStalledQueueRecovery() {
	Fixture    f;
	const auto port = f.Open();
	f.Prime(port);
	now += 1000000;
	f.Prime(port);

	// A stopped device must clear its stale queue and restart priming, not catch up old deadlines.
	stalled = true;
	for (int i = 0; i < 3 && clears == 0; i++) {
		f.Output(port);
	}
	Check(clears == 1, "stalled playback queue was not cleared");
	stalled          = false;
	const auto start = now;
	for (int i = 0; i < 7; i++) {
		f.Output(port);
	}
	Check(now == start, "queue clear did not restart unpaced priming");
	f.Output(port);
	Check(now - start >= 5333 && now - start <= 5334,
	      "queue clear retained an old pacing deadline");
}

void TestFallbackClockAndLongGap() {
	Fixture f;
	fail_open       = true;
	const auto port = f.Open();
	f.Output(port);
	const auto start = now;
	oversleep        = 175;
	for (int i = 0; i < 3000; i++) {
		now += 125;
		f.Output(port);
	}
	Check(now - start >= 16000174 && now - start <= 16000176,
	      "device-free output accumulated timing drift");
	now += 1000000;
	f.Output(port);
	const auto resumed = now;
	f.Output(port);
	Check(now - resumed >= 5333 && now - resumed <= 5509,
	      "long interruption caused a burst of device-free catch-up submissions");
}

void TestAsyncDoesNotAccumulateDeadlines() {
	Fixture    f;
	const auto port  = f.Open();
	const auto start = now;
	for (int i = 0; i < 8; i++) {
		f.Output(port, false);
	}
	Check(now == start, "asynchronous output slept");
	f.Output(port);
	Check(now - start <= 5334, "asynchronous submissions accumulated a blocking deadline");
	for (int i = 0; i < 3; i++) {
		f.Output(port);
	}
	now += 1000000;
	f.Output(port, false);
	const auto resumed = now;
	for (int i = 0; i < 7; i++) {
		f.Output(port);
	}
	Check(now == resumed, "async-to-blocking transition did not reset an emptied queue");
}

void TestSynchronizedBatchAndInactivePorts() {
	Fixture            f;
	const auto         vibration  = f.Open(256, 10);
	const auto         main       = f.Open();
	const auto         background = f.Open();
	Audio::OutputParam batch[] {
	    {vibration, f.pcm.data()}, {main, f.pcm.data()}, {background, f.pcm.data()}};
	const auto start = now;
	for (int i = 0; i < 8; i++) {
		f.audio.AudioOutOutputs(batch, 3);
	}
	Check(now == start, "vibration fallback slowed a device-backed batch's priming");
	for (int i = 0; i < 3; i++) {
		sleeps.clear();
		f.audio.AudioOutOutputs(batch, 3);
		Check(sleeps.size() == 1 && sleeps.front() >= 5333 && sleeps.front() <= 5334,
		      "synchronized streams accumulated separate pacing waits");
	}

	Audio::OutputParam inactive[] {{main, nullptr}, {vibration, f.pcm.data()}};
	now += 1000000;
	f.audio.AudioOutOutputs(inactive, 2);
	const auto resumed = now;
	f.audio.AudioOutOutputs(inactive, 2);
	Check(now - resumed >= 5333 && now - resumed <= 5334,
	      "a device without PCM disabled fallback pacing");
}

void TestFallbackUsesEachPortsPeriod() {
	Fixture f;
	fail_open                     = true;
	const auto         short_port = f.Open(128);
	const auto         long_port  = f.Open(256);
	Audio::OutputParam batch[] {{short_port, f.pcm.data()}, {long_port, f.pcm.data()}};
	f.audio.AudioOutOutputs(batch, 2);
	const auto start = now;
	f.audio.AudioOutOutputs(batch, 2);
	Check(now - start >= 5333 && now - start <= 5334,
	      "fallback batch used the first port's period for every port");
}

void TestFailedQueueUsesFallbackClock() {
	Fixture    f;
	const auto port = f.Open();
	fail_put        = true;
	f.Output(port);
	const auto start = now;
	for (int i = 0; i < 3; i++) {
		f.Output(port);
	}
	Check(now - start == 16000, "repeated queue failures reset fallback pacing");
}

void TestControllerSpeakerPacing() {
	Fixture f;
	fail_open     = true; // No PC audio device paces the pad speaker.
	pad_connected = true;
	pad_queue_us  = 80000;
	const auto speaker = f.Open(256, 4);

	pad_bluetooth = true;
	f.Output(speaker);
	const auto bluetooth_start = now;
	f.Output(speaker);
	Check(now - bluetooth_start >= 5333 && now - bluetooth_start <= 5334,
	      "Bluetooth speaker waited on its HID queue instead of the sample clock");

	pad_bluetooth = false;
	const auto usb_start = now;
	f.Output(speaker);
	Check(now - usb_start >= 5333 && now - usb_start <= 5334,
	      "stalled USB speaker waited longer than one audio block");
}

void TestMixGains() {
	namespace Mix = Libs::Audio::Mix;
	Fixture f;
	f.pcm.fill(0.5f);

	// Unity settings leave every port's samples alone.
	const auto main  = f.Open(256, Mix::PORT_TYPE_MAIN);
	const auto bgm   = f.Open(256, Mix::PORT_TYPE_BGM);
	const auto pad   = f.Open(256, Mix::PORT_TYPE_PADSPK);
	const auto voice = f.Open(256, Mix::PORT_TYPE_VOICE);
	for (const auto port: {main, bgm, pad, voice}) {
		last_put_sample = 0.0f;
		f.Output(port, false);
		Check(last_put_sample == 0.5f, "unity mix changed the samples");
	}

	Mix::Settings settings;
	settings.master      = 50;
	settings.main        = 100;
	settings.music       = 200;
	settings.pad_on_main = 25;
	f.audio.SetMixSettings(settings);
	const auto played = [&f](Audio::Id port) {
		last_put_sample = -1.0f;
		f.Output(port, false);
		return last_put_sample;
	};
	Check(played(main) == 0.25f, "main port did not get master x main gain");
	Check(played(voice) == 0.25f, "voice port is not in the main category");
	Check(played(bgm) == 0.5f, "BGM port did not get master x music gain");
	Check(played(pad) == 0.0625f, "pad speaker on the main output did not get its gain");

	// A DualSense that takes the pad speaker gets the samples instead; the main output gets none.
	f.audio.AudioOutClose(pad);
	pad_connected       = true;
	pad_queue_us        = 1000;
	const auto pad_ds   = f.Open(256, Mix::PORT_TYPE_PADSPK);
	Check(played(pad_ds) == -1.0f, "pad speaker played on the main output with a DualSense");
}

void TestInvalidBatchSize() {
	Libs::Audio::AudioOut::AudioOutOutputParam param {};
	for (const auto count: {0u, 33u}) {
		Check(Libs::Audio::AudioOut::AudioOutOutputs(&param, count) ==
		          Libs::Audio::AUDIO_OUT_ERROR_INVALID_SIZE,
		      "invalid batch size was not rejected before accessing ports");
	}
}

void TestZeroOutputFrequency() {
	Fixture f;
	Check(Libs::Audio::AudioOut::AudioOutOpen(0, 0, 0, 256, 0, 4) ==
	          Libs::Audio::AUDIO_OUT_ERROR_INVALID_SAMPLE_FREQ,
	      "public output open accepted zero frequency");
	Check(!f.audio.AudioOutOpen(10, 256, 0, Audio::Format::FloatStereo).IsValid(),
	      "internal output open accepted zero frequency");
	Check(f.Open().ToInt() == 1, "rejected output port occupied a handle");
}

template <typename T>
void CheckOutputSamples(std::initializer_list<T> expected, const char* message) {
	const auto& pcm = streams.back()->pcm;
	Check(pcm.size() == expected.size() * sizeof(T) &&
	          std::memcmp(pcm.data(), expected.begin(), pcm.size()) == 0, message);
}

void TestChannelGains() {
	Fixture f;
	const std::array pcm {0.25f, -0.5f, 0.75f, -1.0f};
	const auto stereo = f.audio.AudioOutOpen(0, 2, 48000, Audio::Format::FloatStereo);
	std::array gains {2.0f, 0.5f};
	Audio::OutputParam output {stereo, pcm.data(), gains.data()};
	for (int push = 0; push < 2; push++) {
		f.audio.AudioOutOutputs(&output, 1, false);
		CheckOutputSamples({0.5f, -0.25f, 1.5f, -0.5f}, "float channel gains were not applied independently");
	}
	const int volume[] {16384, 32768};
	f.audio.AudioOutSetVolume(stereo, 3, volume);
	f.audio.AudioOutOutputs(&output, 1, false);
	CheckOutputSamples({0.25f, -0.25f, 0.75f, -0.5f}, "channel gain did not combine with port volume");
	gains = {0.0f, 0.0f};
	f.audio.AudioOutOutputs(&output, 1, false);
	CheckOutputSamples({0.0f, -0.0f, 0.0f, -0.0f}, "zero channel gains did not mute");
	output.gains = nullptr;
	f.audio.AudioOutOutputs(&output, 1, false);
	CheckOutputSamples({0.125f, -0.5f, 0.375f, -1.0f}, "channel scaling changed original float PCM");

	const std::array<int16_t, 2> integer_pcm {20000, -20000};
	const auto integer = f.audio.AudioOutOpen(0, 1, 48000, Audio::Format::Signed16bitStereo);
	output = {integer, integer_pcm.data(), gains.data()};
	for (const auto gain: {2.0f, std::numeric_limits<float>::max()}) {
		gains = {gain, gain};
		f.audio.AudioOutOutputs(&output, 1, false);
		CheckOutputSamples<int16_t>({32767, -32768}, "integer gain overflowed instead of saturating");
	}
	gains = {0.5f, 0.25f};
	f.audio.AudioOutOutputs(&output, 1, false);
	CheckOutputSamples<int16_t>({10000, -5000}, "integer channel gains changed original PCM");

	const std::array surround_pcm {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
	for (const auto format: {Audio::Format::Float8Ch, Audio::Format::Float12Ch}) {
		const auto surround = f.audio.AudioOutOpen(0, 1, 48000, format);
		output = {surround, surround_pcm.data(), surround_pcm.data()};
		f.audio.AudioOutOutputs(&output, 1, false);
		if (format == Audio::Format::Float8Ch) {
			CheckOutputSamples({1.0f, 4.0f, 9.0f, 16.0f, 49.0f, 64.0f, 25.0f, 36.0f},
			                   "channel reordering used destination channel gains");
		} else {
			CheckOutputSamples({82.0f, 104.0f, 9.0f, 16.0f, 170.0f, 208.0f, 25.0f, 36.0f},
			                   "height downmix lost per-channel gains");
		}
	}
	pad_connected = true;
	const auto speaker = f.Open(2, 4);
	gains = {0.75f, 0.25f};
	output = {speaker, pcm.data(), gains.data()};
	f.audio.AudioOutOutputs(&output, 1, false);
	Check(queued_pad_gains != nullptr && queued_pad_gains[0] == gains[0] && queued_pad_gains[1] == gains[1],
	      "connected controller did not receive channel gains");
}

// A device that plays faster than the process clock drains the cushion until the queue runs dry
// (one gap per drain, then a re-prime); with the queue catch-up the device sets the pace instead.
// Returns how many outputs found less than one block queued after the first priming.
int RunFastDevice(bool catchup, int64_t ppm, int seconds) {
	Fixture f;
	Libs::Audio::g_queue_catchup = catchup ? 1 : 0;
	device_ppm                   = ppm;
	const auto port              = f.Open();
	f.Prime(port);
	processing = 125;
	oversleep  = 175;
	int        low   = 0;
	const auto start = now;
	while (now - start < static_cast<uint64_t>(seconds) * 1000000) {
		Drain(*streams.front());
		low += streams.front()->frames < uint64_t {256} * 1000000 ? 1 : 0;
		f.Output(port);
	}
	Drain(*streams.front());
	const auto queued_ms = streams.front()->frames / 1000000 / 48;
	Check(queued_ms <= 60, "the queue grew past its target");
	device_ppm                   = 0;
	Libs::Audio::g_queue_catchup = 0;
	f.audio.AudioOutClose(port);
	return low;
}

// A port that joins a batch the first port already paces starts under one block and stays there;
// KYTY_AUDIO_PAD_PRIME tops it up with silence once.
uint64_t LateJoinerQueuedFrames(bool pad_prime) {
	Fixture f;
	Libs::Audio::g_pad_prime = pad_prime ? 1 : 0;
	const auto bed           = f.Open(512);
	for (int i = 0; i < 8; i++) {
		f.Output(bed);
	}
	const auto         speaker = f.Open(512, 4);
	Audio::OutputParam batch[] {{bed, f.pcm.data()}, {speaker, f.pcm.data()}};
	uint64_t           min_frames = ~uint64_t {0};
	for (int i = 0; i < 400; i++) {
		f.audio.AudioOutOutputs(batch, 2);
		if (i >= 10) {
			Drain(*streams[1]);
			min_frames = std::min(min_frames, streams[1]->frames / 1000000);
		}
	}
	Libs::Audio::g_pad_prime = 0;
	f.audio.AudioOutClose(speaker);
	f.audio.AudioOutClose(bed);
	return min_frames;
}

// The AudioOut2 shape on a device faster than the process clock: a bed and a late pad speaker
// port in one batch. With catch-up and pad prime neither queue may run dry.
void TestFastDeviceBatch() {
	Fixture f;
	Libs::Audio::g_queue_catchup = 1;
	Libs::Audio::g_pad_prime     = 1;
	device_ppm                   = 500;
	processing                   = 125;
	oversleep                    = 175;
	const auto bed               = f.Open(512);
	for (int i = 0; i < 8; i++) {
		f.Output(bed);
	}
	const auto         speaker = f.Open(512, 4);
	Audio::OutputParam batch[] {{bed, f.pcm.data()}, {speaker, f.pcm.data()}};
	uint64_t           min_bed = ~uint64_t {0}, min_speaker = ~uint64_t {0}, max_bed = 0;
	const auto         start = now;
	for (int i = 0; now - start < 300000000; i++) {
		if (i >= 10) {
			Drain(*streams[0]);
			Drain(*streams[1]);
			min_bed     = std::min(min_bed, streams[0]->frames / 1000000);
			max_bed     = std::max(max_bed, streams[0]->frames / 1000000);
			min_speaker = std::min(min_speaker, streams[1]->frames / 1000000);
		}
		f.audio.AudioOutOutputs(batch, 2);
	}
	device_ppm                   = 0;
	Libs::Audio::g_queue_catchup = 0;
	Libs::Audio::g_pad_prime     = 0;
	f.audio.AudioOutClose(speaker);
	f.audio.AudioOutClose(bed);
	Check(min_bed >= 512, "the bed ran under one block on a fast device");
	Check(min_speaker >= 512, "the late pad speaker port ran under one block on a fast device");
	Check(max_bed <= 3 * 1024, "the bed grew past its target");
}

void TestLateJoinerPadPrime() {
	Check(LateJoinerQueuedFrames(false) < 512, "a late joiner unexpectedly built its own cushion");
	Check(LateJoinerQueuedFrames(true) >= 1024, "pad prime did not give the late joiner a cushion");
}

void TestFastDeviceDrift() {
	Check(RunFastDevice(false, 500, 300) > 0, "without catch-up a 500 ppm fast device never ran dry");
	Check(RunFastDevice(true, 500, 300) == 0, "catch-up let a 500 ppm fast device run dry");
	Check(RunFastDevice(true, 20000, 60) == 0, "catch-up let a 2% fast device run dry");
	Check(RunFastDevice(true, -500, 300) == 0, "a slow device ran dry with catch-up");
	Check(RunFastDevice(false, -500, 300) == 0, "a slow device ran dry");
}
} // namespace

int main() {
	TestFastDeviceDrift();
	TestLateJoinerPadPrime();
	TestFastDeviceBatch();
	// The scheduling cases hold with the queue catch-up and pad prime off and on (the default).
	for (const int fixes: {0, 1}) {
		Libs::Audio::g_queue_catchup = fixes;
		Libs::Audio::g_pad_prime     = fixes;
		TestPrimingAndSteadyCadence();
		TestUnderrunAndStalledQueueRecovery();
		TestFallbackClockAndLongGap();
		TestAsyncDoesNotAccumulateDeadlines();
		TestSynchronizedBatchAndInactivePorts();
		TestFallbackUsesEachPortsPeriod();
		TestFailedQueueUsesFallbackClock();
		TestControllerSpeakerPacing();
		TestMixGains();
		TestInvalidBatchSize();
		TestZeroOutputFrequency();
		TestChannelGains();
	}
	Check(streams.empty(), "output stream leaked");
	std::puts("AudioOutTimingTests: all cases passed");
}
