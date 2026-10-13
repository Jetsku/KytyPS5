#include "libs/audio.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "kernel/pthread.h"
#include "libs/audioDiag.h"
#include "libs/audioMix.h"
#include "libs/audio_internal.h"
#include "libs/controller.h"
#include "libs/dualSenseHaptics.h"
#include "libs/errno.h"
#include "libs/libs.h"

#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <magic_enum.hpp>
#include <vector>

namespace Libs::Audio {

namespace {

constexpr int AUDIO_OUT_PORT_TYPE_MAIN      = 0;
constexpr int AUDIO_OUT_PORT_TYPE_BGM       = 1;
constexpr int AUDIO_OUT_PORT_TYPE_VOICE     = 2;
constexpr int AUDIO_OUT_PORT_TYPE_PERSONAL  = 3;
constexpr int AUDIO_OUT_PORT_TYPE_PADSPK    = 4;
constexpr int AUDIO_OUT_PORT_TYPE_VIBRATION = 10;
constexpr int AUDIO_OUT_PORT_TYPE_AUDIO3D   = 126;
constexpr int AUDIO_OUT_PORT_TYPE_AUX       = 127;

constexpr uint32_t AUDIO_OUT_PARAM_FORMAT_MASK = 0x000000ffu;
constexpr uint64_t AUDIO_OUT_TARGET_LATENCY_US = 40000;

constexpr int      AUDIO_IN_SILENT_STATE_DEVICE_NONE = 0x1;
constexpr uint32_t AUDIO_IN_GRAIN_MAX_ASYNC          = 384;

struct OutputClock {
	uint64_t next_time = 0;
	uint64_t remainder = 0;

	uint64_t Advance(uint64_t now, uint32_t frames, uint32_t rate) {
		// Preserve small scheduling errors; discard timing debt after a discontinuity.
		if (next_time == 0 || (now > next_time && now - next_time > 100000)) {
			next_time = now;
			remainder = 0;
		}
		const auto deadline = next_time;
		const auto duration = 1000000ULL * frames + remainder;
		next_time += duration / rate;
		remainder = duration % rate;
		return deadline;
	}
};

// KYTY_AUDIO_CLOCK_SKEW_PPM (testing): the sample clock paces this many ppm slower (positive) or
// faster than the process clock, imitating a playback device whose crystal runs faster (or
// slower) than the host's TSC. 0 by default.
static int64_t audio_out_clock_skew_ppm() {
	static const int64_t ppm = [] {
		const char* value = std::getenv("KYTY_AUDIO_CLOCK_SKEW_PPM");
		return value != nullptr ? std::clamp<int64_t>(std::strtoll(value, nullptr, 10), -20000, 20000) : 0;
	}();
	return ppm;
}

static uint32_t audio_out_clock_rate(uint32_t freq) {
	const auto ppm = audio_out_clock_skew_ppm();
	return ppm == 0 ? freq
	                : static_cast<uint32_t>(static_cast<int64_t>(freq) - static_cast<int64_t>(freq) * ppm / 1000000);
}

// KYTY_AUDIO_PAD_PRIME: a port that joins a batch another port already paces never gets an unpaced
// burst, so it plays with whatever cushion it started with (often under one block: the pad
// speaker's main-output fallback opens after the bed). Such a port, when found under one block, is
// topped up to its target with silence instead, once, and then counts as primed. On by default;
// KYTY_AUDIO_PAD_PRIME=0 turns it off.
static std::atomic<int> g_pad_prime {-1};
static bool             audio_out_pad_prime() {
	int enabled = g_pad_prime.load(std::memory_order_relaxed);
	if (enabled < 0) {
		const char* value = std::getenv("KYTY_AUDIO_PAD_PRIME");
		enabled           = value == nullptr || value[0] != '0' ? 1 : 0;
		g_pad_prime.store(enabled, std::memory_order_relaxed);
	}
	return enabled != 0;
}

// The SDL queue level a blocking port keeps (KYTY_AUDIO_LATENCY_MS, 10-500; default 40 ms).
static uint64_t audio_out_target_latency_us() {
	static const uint64_t latency = [] {
		const char* value = std::getenv("KYTY_AUDIO_LATENCY_MS");
		const long  ms    = value != nullptr ? std::strtol(value, nullptr, 10) : 0;
		return ms > 0 ? static_cast<uint64_t>(std::clamp(ms, 10L, 500L)) * 1000 : AUDIO_OUT_TARGET_LATENCY_US;
	}();
	return latency;
}

// KYTY_AUDIO_QUEUE_CATCHUP: a primed port whose SDL queue has fallen more than one block below its
// target skips the sample-clock wait for that block, so the device's consumption rate, not the
// process clock, sets the long-term pace. Without it a device that plays a few ppm faster than the
// process clock (calibrated TSC) drains the cushion at a steady rate until the queue runs dry; the
// port then re-primes with an unpaced burst, which a guest that cannot render ahead does not fill.
// On by default; KYTY_AUDIO_QUEUE_CATCHUP=0 turns it off.
static std::atomic<int> g_queue_catchup {-1}; // -1: not read yet (tests set it directly).
static bool             audio_out_queue_catchup() {
	int enabled = g_queue_catchup.load(std::memory_order_relaxed);
	if (enabled < 0) {
		const char* value = std::getenv("KYTY_AUDIO_QUEUE_CATCHUP");
		enabled           = value == nullptr || value[0] != '0' ? 1 : 0;
		g_queue_catchup.store(enabled, std::memory_order_relaxed);
	}
	return enabled != 0;
}

// KYTY_AUDIO_TRACE: per output port, what happened in the last Diag::TRACE_INTERVAL_US.
struct OutTrace {
	static constexpr int WAIT_BUCKETS = 8; // 0, <1, <2, <5, <10, <20, <50 ms, cleared (200 ms)

	uint64_t start        = 0;
	uint32_t writes       = 0;
	uint32_t blocking     = 0;
	uint64_t bytes        = 0;
	int64_t  queued_sum   = 0;
	int      queued_min   = INT_MAX;
	int      queued_max   = 0;
	int      queued_first = -1;
	int      queued_last  = 0;
	uint32_t low          = 0; // Writes that found less than one block queued.
	uint32_t clears       = 0;
	uint32_t reprimes     = 0;
	uint32_t catch_ups    = 0;
	uint32_t events       = 0; // Event lines printed in this window.
	uint64_t sleep_us     = 0; // Sample-clock waits.
	uint32_t sleeps       = 0;
	uint64_t gap_sum      = 0; // Guest time between two submissions (its mixing time).
	uint64_t gap_max      = 0;
	uint32_t gaps         = 0;
	uint32_t late_gaps    = 0; // Gaps longer than one block.
	uint64_t last_return  = 0;
	uint64_t total_clears = 0;
	uint64_t total_low    = 0;
	uint64_t total_short  = 0;
	uint32_t pads         = 0;
	// What the guest handed over (as sent to SDL): all-zero blocks, zero blocks right after sound
	// (dropouts), sample jumps at block boundaries far above the block's own sample-to-sample
	// steps (clicks), and the mean power.
	uint32_t zero_blocks  = 0;
	uint32_t dropouts     = 0;
	uint32_t jumps        = 0;
	double   power_sum    = 0.0;
	uint64_t power_n      = 0;
	bool     have_prev    = false;
	bool     prev_sound   = false;
	float    prev_last[8] {};

	std::array<uint32_t, WAIT_BUCKETS> wait {};

	static int WaitBucket(uint64_t us, bool cleared) {
		if (cleared) {
			return WAIT_BUCKETS - 1;
		}
		static constexpr uint64_t LIMITS[] = {1, 1000, 2000, 5000, 10000, 20000, 50000};
		for (int i = 0; i < WAIT_BUCKETS - 1; i++) {
			if (us < LIMITS[i]) {
				return i;
			}
		}
		return WAIT_BUCKETS - 2;
	}
};

// Filled by SDL's device thread through the stream's get callback: what the device asked the
// stream for and how much of it was missing (played as silence).
struct DeviceTrace {
	std::atomic<uint64_t> requests {0};
	std::atomic<uint64_t> bytes {0};
	std::atomic<uint64_t> short_requests {0};
	std::atomic<uint64_t> short_bytes {0};
	// Time between two requests: SDL's device thread running late (the host device then plays
	// out whatever its own buffer holds, or a gap).
	std::atomic<uint64_t> last_ns {0};
	std::atomic<uint64_t> max_interval_ns {0};
	std::atomic<uint32_t> late15 {0};
	std::atomic<uint32_t> late30 {0};

	void Reset() {
		requests        = 0;
		bytes           = 0;
		short_requests  = 0;
		short_bytes     = 0;
		last_ns         = 0;
		max_interval_ns = 0;
		late15          = 0;
		late30          = 0;
	}
};

static void SDLCALL trace_device_request(void* userdata, SDL_AudioStream* /*stream*/,
                                         int additional_amount, int total_amount) {
	auto*      device = static_cast<DeviceTrace*>(userdata);
	const auto now_ns = SDL_GetTicksNS();
	const auto last   = device->last_ns.exchange(now_ns, std::memory_order_relaxed);
	if (last != 0 && now_ns > last) {
		const auto interval = now_ns - last;
		if (interval > device->max_interval_ns.load(std::memory_order_relaxed)) {
			device->max_interval_ns.store(interval, std::memory_order_relaxed);
		}
		device->late15.fetch_add(interval > 15000000 ? 1 : 0, std::memory_order_relaxed);
		device->late30.fetch_add(interval > 30000000 ? 1 : 0, std::memory_order_relaxed);
	}
	device->requests.fetch_add(1, std::memory_order_relaxed);
	device->bytes.fetch_add(static_cast<uint64_t>(std::max(total_amount, 0)), std::memory_order_relaxed);
	if (additional_amount > 0) {
		device->short_requests.fetch_add(1, std::memory_order_relaxed);
		device->short_bytes.fetch_add(static_cast<uint64_t>(additional_amount), std::memory_order_relaxed);
	}
}

static bool audio_out_port_type_is_valid(int type) {
	return (type >= AUDIO_OUT_PORT_TYPE_MAIN && type <= AUDIO_OUT_PORT_TYPE_PADSPK) ||
	       type == AUDIO_OUT_PORT_TYPE_VIBRATION || type == AUDIO_OUT_PORT_TYPE_AUDIO3D ||
	       type == AUDIO_OUT_PORT_TYPE_AUX;
}

static const char* audio_out_port_type_name(int type) {
	switch (type) {
		case AUDIO_OUT_PORT_TYPE_MAIN: return "main";
		case AUDIO_OUT_PORT_TYPE_BGM: return "bgm";
		case AUDIO_OUT_PORT_TYPE_VOICE: return "voice";
		case AUDIO_OUT_PORT_TYPE_PERSONAL: return "personal";
		case AUDIO_OUT_PORT_TYPE_PADSPK: return "pad speaker";
		case AUDIO_OUT_PORT_TYPE_VIBRATION: return "vibration";
		case AUDIO_OUT_PORT_TYPE_AUDIO3D: return "audio3d";
		case AUDIO_OUT_PORT_TYPE_AUX: return "aux";
		default: return "unknown";
	}
}

// The host's default playback device as SDL sees it, once: a device whose format or period
// differs from the usual 48 kHz stereo changes how output is paced.
static void report_default_playback_device() {
	static std::atomic_bool reported {false};
	if (!Diag::Console().Enabled() || reported.exchange(true)) {
		return;
	}
	SDL_AudioSpec spec {};
	int           frames = 0;
	const char*   name   = SDL_GetAudioDeviceName(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK);
	if (SDL_GetAudioDeviceFormat(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, &frames)) {
		Diag::Print("default playback device '%s': %d Hz, %d ch, format 0x%04x, %d-frame period",
		            name != nullptr ? name : "?", spec.freq, spec.channels,
		            static_cast<unsigned>(spec.format), frames);
	} else {
		Diag::Print("default playback device '%s': format unknown (%s)", name != nullptr ? name : "?",
		            SDL_GetError());
	}
	int   count   = 0;
	auto* devices = SDL_GetAudioPlaybackDevices(&count);
	for (int i = 0; devices != nullptr && i < count && i < 16; i++) {
		SDL_AudioSpec device_spec {};
		int           device_frames = 0;
		const char*   device_name   = SDL_GetAudioDeviceName(devices[i]);
		const bool    known = SDL_GetAudioDeviceFormat(devices[i], &device_spec, &device_frames);
		Diag::Print("playback device %d: '%s' %d Hz, %d ch, %d-frame period", i,
		            device_name != nullptr ? device_name : "?", known ? device_spec.freq : 0,
		            known ? device_spec.channels : 0, known ? device_frames : 0);
	}
	SDL_free(devices);
}

} // namespace

class Audio {
public:
	using Format = AudioInternal::Format;

	class Id {
	public:
		explicit Id(int id): m_id(id - 1) {}
		[[nodiscard]] int  ToInt() const { return m_id + 1; }
		[[nodiscard]] bool IsValid() const { return m_id >= 0; }

		friend class Audio;

	private:
		Id() = default;
		static Id Invalid() { return {}; }
		static Id Create(int audio_id) {
			Id r;
			r.m_id = audio_id;
			return r;
		}
		[[nodiscard]] int GetId() const { return m_id; }

		int m_id = -1;
	};

	struct OutputParam {
		Id           handle;
		const void*  data  = nullptr;
		const float* gains = nullptr;
	};

	Audio() = default;
	virtual ~Audio();

	KYTY_CLASS_NO_COPY(Audio);

	Id       AudioOutOpen(int type, uint32_t samples_num, uint32_t freq, Format format);
	bool     AudioOutClose(Id handle);
	bool     AudioOutValid(Id handle);
	bool     AudioOutHasDevice(Id handle);
	bool     AudioOutSetVolume(Id handle, uint32_t bitflag, const int* volume);
	uint32_t AudioOutOutputs(OutputParam* params, uint32_t num, bool blocking = true);
	bool     AudioOutGetStatus(Id handle, int* type, int* channels_num);

	void SetMixSettings(const Mix::Settings& settings) { m_mix = settings; }
	[[nodiscard]] const Mix::Settings& GetMixSettings() const { return m_mix; }
	// Every LEVEL_LOG_INTERVAL_US, print each active output port's RMS and peak (KYTY_AUDIO_LEVELS).
	void SetLevelLogging(bool enabled) { m_level_logging = enabled; }
	static constexpr uint64_t LEVEL_LOG_INTERVAL_US = 5000000;

	Id       AudioInOpen(uint32_t samples_num, uint32_t freq, Format format, bool asynchronous);
	int      AudioInClose(Id handle);
	int      AudioInGetSilentState(Id handle);
	int      AudioInInput(Id handle, void* dest);

	static constexpr int OUT_PORTS_MAX = 32;
	static constexpr int IN_PORTS_MAX  = 8;

private:
	struct PortOut {
		bool        used        = false;
		int         type        = 0;
		uint32_t    samples_num = 0;
		uint32_t    freq        = 0;
		Format      format      = Format::Unknown;
		OutputClock clock;
		bool        queue_primed = false;
		int         channels_num = 0;
		int         volume[12]   = {};
		// A 7.1 (or 7.1.4) port on a stereo device: downmixed here instead of by SDL.
		bool        downmix_stereo = false;

		SDL_AudioStream*                      stream  = nullptr;
		Controller::DualSenseHaptics::Stream* haptics = nullptr;

		Mix::LevelMeter meter;
		uint64_t        meter_start   = 0;
		uint32_t        meter_on_main = 0; // Blocks that played on the main output.
		uint32_t        meter_blocks  = 0;
		float           meter_gain    = 1.0f;
		// What SDL is given (after volume, host gain and a stereo downmix), and samples beyond
		// full scale in it.
		bool            meter_device   = false;
		Mix::LevelMeter device_meter;
		uint32_t        device_clipped = 0;

		int      trace_id = 0; // 1-based port number in trace lines.
		bool     catch_up = false; // The last write found the queue below target (no clock wait).
		OutTrace trace;
	};

	struct PortIn {
		bool              used            = false;
		bool              busy            = false;
		bool              asynchronous    = false;
		uint32_t          samples_num     = 0;
		uint32_t          freq            = 0;
		uint32_t          bytes_per_frame = 0;
		uint64_t          last_input_time = 0;
		SDL_AudioStream*  stream          = nullptr;
		SDL_AudioDeviceID device          = 0;
	};

	PortIn* GetAudioInPort(Id handle); // Caller holds m_mutex.

	Common::Mutex m_mutex;
	PortOut       m_out_ports[OUT_PORTS_MAX];
	PortIn        m_in_ports[IN_PORTS_MAX];
	Mix::Settings m_mix;
	bool          m_level_logging = false;

	void MeterOutput(int port_id, PortOut* port, const void* data, bool on_main, float gain);

	static bool            FormatIsFloat(Format format);
	static bool            FormatIsStd(Format format);
	static uint32_t        BytesPerSample(Format format);
	static uint32_t        OutputChannels(const PortOut& port);
	static SDL_AudioFormat SdlFormat(Format format);
	static bool            OpenSdlDevice(PortOut* port);
	static void            CloseSdlDevice(PortOut* port);
	static void            OpenSdlDevice(PortIn* port, Format format);
	static void            CloseSdlDevice(PortIn* port);
	static const void*     PrepareOutputBuffer(const PortOut& port, const void* data,
	                                           std::vector<uint8_t>* buffer, float gain,
	                                           const float* gains);
	static bool QueueSdlAudio(PortOut* port, const void* data, bool blocking, float gain,
	                          const float* gains, bool batch_paced = false);
	static void TraceSummary(PortOut* port, uint64_t now);
	static void TraceContent(PortOut* port, const void* data);

	static DeviceTrace s_device_trace[OUT_PORTS_MAX];
};

DeviceTrace Audio::s_device_trace[Audio::OUT_PORTS_MAX];

static Audio* g_audio = nullptr;

namespace AudioInternal {

int AudioOutOpen(int type, uint32_t samples_num, uint32_t freq, Format format) {
	if (g_audio == nullptr) {
		return 0;
	}

	auto id = g_audio->AudioOutOpen(type, samples_num, freq, format);
	return id.IsValid() ? id.ToInt() : 0;
}

void AudioOutClose(int handle) {
	if (g_audio != nullptr && handle > 0) {
		(void)g_audio->AudioOutClose(Audio::Id(handle));
	}
}

bool AudioOutHasDevice(int handle) {
	return g_audio != nullptr && handle > 0 && g_audio->AudioOutHasDevice(Audio::Id(handle));
}

uint32_t AudioOutOutputs(const OutputParam* params, uint32_t num, bool blocking) {
	if (g_audio == nullptr || params == nullptr || num == 0) {
		return 0;
	}

	std::vector<Audio::OutputParam> output_params;
	output_params.reserve(num);
	for (uint32_t i = 0; i < num; i++) {
		if (params[i].handle > 0 && params[i].data != nullptr) {
			output_params.push_back(
			    Audio::OutputParam {Audio::Id(params[i].handle), params[i].data, params[i].gains});
		}
	}

	if (output_params.empty()) {
		return 0;
	}

	return g_audio->AudioOutOutputs(output_params.data(),
	                                static_cast<uint32_t>(output_params.size()), blocking);
}

float AudioOutObjectsGain() {
	return g_audio != nullptr ? Mix::ObjectsGain(g_audio->GetMixSettings()) : 1.0f;
}

bool AudioOutObjectsEnabled() {
	return g_audio == nullptr || g_audio->GetMixSettings().objects_enabled;
}

} // namespace AudioInternal

void Initialize() {
	EXIT_IF(g_audio != nullptr);

	// Audio diagnostics reach the printf file too: players send _kyty.txt, not the console.
	Diag::ConsoleLogWriter().store(+[](const std::string& text) { Log::WriteToConsoleAndLog(text); });

	g_audio = new Audio;
	if (Diag::TraceEnabled()) {
		Diag::Trace("on: output target latency %.0f ms, queue catch-up %s",
		            static_cast<double>(audio_out_target_latency_us()) / 1000.0,
		            audio_out_queue_catchup() ? "on" : "off");
	}

	Mix::Settings mix;
	mix.master      = Config::GetAudioMasterVolume();
	mix.main        = Config::GetAudioMainVolume();
	mix.music       = Config::GetAudioMusicVolume();
	mix.pad_on_main = Config::GetAudioPadSpeakerOnMainVolume();
	mix.objects     = Config::GetAudioObjectsVolume();
	mix.objects_enabled = Config::AudioObjectsEnabled();
	mix             = Mix::ApplyEnvironment(mix, [](const char* name) { return std::getenv(name); });
	g_audio->SetMixSettings(mix);
	const char* levels = std::getenv("KYTY_AUDIO_LEVELS");
	g_audio->SetLevelLogging(levels != nullptr && levels[0] != '\0' && levels[0] != '0');
	std::printf("Kyty audio mix: master %u%%, main %u%%, music (BGM ports) %u%%, pad speaker on main "
	            "output %u%%, 3D objects %s %u%%\n",
	            mix.master, mix.main, mix.music, mix.pad_on_main, mix.objects_enabled ? "on" : "off",
	            mix.objects);
}

void Shutdown() {
	delete g_audio;
	g_audio = nullptr;
}

Audio::~Audio() {
	for (auto& port: m_out_ports) {
		CloseSdlDevice(&port);
	}
	for (auto& port: m_in_ports) {
		CloseSdlDevice(&port);
	}
}

bool Audio::FormatIsFloat(Format format) {
	switch (format) {
		case Format::FloatMono:
		case Format::FloatStereo:
		case Format::Float8Ch:
		case Format::Float8ChStd:
		case Format::Float12Ch: return true;
		default: return false;
	}
}

bool Audio::FormatIsStd(Format format) {
	return (format == Format::Signed16bit8ChStd || format == Format::Float8ChStd);
}

uint32_t Audio::BytesPerSample(Format format) {
	return FormatIsFloat(format) ? sizeof(float) : sizeof(int16_t);
}

uint32_t Audio::OutputChannels(const PortOut& port) {
	// SDL only takes up to 8 channels. Keep the guest buffer's channel count separate.
	return port.downmix_stereo ? 2u : static_cast<uint32_t>(std::min(port.channels_num, 8));
}

// SDL's 7.1-to-stereo matrix keeps eight full-scale channels unclipped, so it plays the front pair
// at 0.21 (-13.5 dB): game audio on a 7.1 bed came out far too quiet on stereo devices. Such ports
// are downmixed here instead with the usual coefficients (front 1, centre/surrounds/heights 0.707,
// LFE dropped). KYTY_AUDIO_STEREO_DOWNMIX=0 leaves it to SDL.
static bool StereoDownmixEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_AUDIO_STEREO_DOWNMIX");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

SDL_AudioFormat Audio::SdlFormat(Format format) {
	return FormatIsFloat(format) ? SDL_AUDIO_F32 : SDL_AUDIO_S16;
}

bool Audio::OpenSdlDevice(PortOut* port) {
	EXIT_IF(port == nullptr);

	if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
		LOGF("AudioOut: SDL audio init failed: %s\n", SDL_GetError());
		Diag::Print("SDL audio init failed: %s", SDL_GetError());
		return false;
	}
	report_default_playback_device();

	port->downmix_stereo = false;
	if (port->channels_num >= 8 && StereoDownmixEnabled()) {
		SDL_AudioSpec device {};
		port->downmix_stereo =
		    SDL_GetAudioDeviceFormat(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &device, nullptr) &&
		    device.channels == 2;
		if (port->downmix_stereo) {
			Diag::Print("%d-channel output on a stereo device: downmixed by Kyty (front 1, centre/"
			            "surrounds 0.707, no LFE)",
			            port->channels_num);
		}
	}

	SDL_AudioSpec desired {};
	desired.freq     = static_cast<int>(port->freq);
	desired.format   = SdlFormat(port->format);
	desired.channels = static_cast<int>(OutputChannels(*port));

	port->stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &desired, nullptr,
	                                         nullptr);
	if (port->stream == nullptr) {
		LOGF("AudioOut: SDL_OpenAudioDeviceStream failed: %s\n", SDL_GetError());
		Diag::Print("cannot open the default playback device (%d Hz, %d ch): %s", desired.freq,
		            desired.channels, SDL_GetError());
		SDL_QuitSubSystem(SDL_INIT_AUDIO);
		return false;
	}
	if (!SDL_ResumeAudioStreamDevice(port->stream)) {
		LOGF("AudioOut: SDL_ResumeAudioStreamDevice failed: %s\n", SDL_GetError());
		Diag::Print("cannot start the default playback device: %s", SDL_GetError());
		CloseSdlDevice(port);
		return false;
	}

	LOGF("AudioOut: opened SDL stream (%d Hz, %d ch, format 0x%04x)\n", desired.freq,
	     desired.channels, static_cast<unsigned>(desired.format));
	return true;
}

void Audio::CloseSdlDevice(PortOut* port) {
	EXIT_IF(port == nullptr);
	Controller::DualSenseHaptics::Close(port->haptics);
	port->haptics = nullptr;

	if (port->stream != nullptr) {
		SDL_DestroyAudioStream(port->stream);
		SDL_QuitSubSystem(SDL_INIT_AUDIO);
	}
	port->stream = nullptr;
}

const void* Audio::PrepareOutputBuffer(const PortOut& port, const void* data,
                                       std::vector<uint8_t>* buffer, float gain,
                                       const float* gains) {
	EXIT_IF(data == nullptr);
	EXIT_IF(buffer == nullptr);

	const auto frames           = port.samples_num;
	const auto channels         = static_cast<uint32_t>(port.channels_num);
	const auto output_channels  = OutputChannels(port);
	const auto bytes_per_sample = BytesPerSample(port.format);
	const bool reorder          = channels >= 8 && !FormatIsStd(port.format);

	if (port.downmix_stereo) {
		// Both 8-channel orders have the left surrounds at 4 and 6 and the right ones at 5 and 7; the
		// 12-channel one adds its left heights at 8 and 10 and the right ones at 9 and 11.
		const bool is_float = FormatIsFloat(port.format);
		float      scale[12] {};
		for (uint32_t ch = 0; ch < channels && ch < 12; ch++) {
			scale[ch] = static_cast<float>(port.volume[ch]) / 32768.0f * gain *
			            (gains != nullptr ? gains[ch] : 1.0f);
		}
		const auto sample = [&](uint32_t frame, uint32_t ch) {
			const auto  index = static_cast<size_t>(frame) * channels + ch;
			const float value = is_float ? static_cast<const float*>(data)[index]
			                             : static_cast<float>(static_cast<const int16_t*>(data)[index]) / 32768.0f;
			return value * scale[ch];
		};
		constexpr float SIDE = 0.70710677f;
		buffer->resize(static_cast<size_t>(frames) * 2 * bytes_per_sample);
		for (uint32_t frame = 0; frame < frames; frame++) {
			const float centre = sample(frame, 2);
			float left  = sample(frame, 0) + SIDE * (centre + sample(frame, 4) + sample(frame, 6));
			float right = sample(frame, 1) + SIDE * (centre + sample(frame, 5) + sample(frame, 7));
			if (channels == 12) {
				left += SIDE * (sample(frame, 8) + sample(frame, 10));
				right += SIDE * (sample(frame, 9) + sample(frame, 11));
			}
			if (is_float) {
				reinterpret_cast<float*>(buffer->data())[frame * 2]     = left;
				reinterpret_cast<float*>(buffer->data())[frame * 2 + 1] = right;
			} else {
				reinterpret_cast<int16_t*>(buffer->data())[frame * 2] =
				    static_cast<int16_t>(std::clamp(left * 32768.0f, -32768.0f, 32767.0f));
				reinterpret_cast<int16_t*>(buffer->data())[frame * 2 + 1] =
				    static_cast<int16_t>(std::clamp(right * 32768.0f, -32768.0f, 32767.0f));
			}
		}
		return buffer->data();
	}

	std::array<double, 12> scales {};
	bool                   volume_changed = false;
	for (uint32_t ch = 0; ch < channels; ch++) {
		scales[ch] = (static_cast<double>(port.volume[ch]) / 32768.0) * gain *
		             (gains != nullptr ? gains[ch] : 1.0f);
		volume_changed |= scales[ch] != 1.0;
	}

	if (!volume_changed && !reorder) {
		return data;
	}

	buffer->resize(frames * output_channels * bytes_per_sample);

	// SDL wants back speakers before side speakers; non-STD PCM has them reversed.
	static constexpr uint32_t SDL_8CH_MAP[8] = {0, 1, 2, 3, 6, 7, 4, 5};

	if (FormatIsFloat(port.format)) {
		auto*       dst = reinterpret_cast<float*>(buffer->data());
		const auto* src = static_cast<const float*>(data);

		for (uint32_t frame = 0; frame < frames; frame++) {
			for (uint32_t ch = 0; ch < output_channels; ch++) {
				const auto src_ch = reorder ? SDL_8CH_MAP[ch] : ch;
				dst[frame * output_channels + ch] =
				    static_cast<float>(src[frame * channels + src_ch] * scales[src_ch]);
			}
			if (channels == 12) {
				// Add the four top channels to their front/back channels, turning 12 into 8.
				// SDL handles the rest if the output device has fewer channels.
				static constexpr uint32_t HEIGHT_DST[4] = {0, 1, 4, 5};
				for (uint32_t ch = 0; ch < 4; ch++) {
					dst[frame * output_channels + HEIGHT_DST[ch]] +=
					    static_cast<float>(src[frame * channels + 8 + ch] * scales[8 + ch]);
				}
			}
		}
	} else {
		auto*       dst = reinterpret_cast<int16_t*>(buffer->data());
		const auto* src = static_cast<const int16_t*>(data);

		for (uint32_t frame = 0; frame < frames; frame++) {
			for (uint32_t ch = 0; ch < output_channels; ch++) {
				const auto src_ch = reorder ? SDL_8CH_MAP[ch] : ch;
				const double sample = src[frame * channels + src_ch] * scales[src_ch];
				dst[frame * output_channels + ch] = static_cast<int16_t>(
				    std::clamp(sample, -32768.0, 32767.0));
			}
		}
	}

	return buffer->data();
}

bool Audio::QueueSdlAudio(PortOut* port, const void* data, bool blocking, float gain,
                          const float* gains, bool batch_paced) {
	EXIT_IF(port == nullptr);

	if (port->stream == nullptr || data == nullptr) {
		return false;
	}

	std::vector<uint8_t> prepared_buffer;
	const void*          prepared_data = PrepareOutputBuffer(*port, data, &prepared_buffer, gain, gains);
	const auto           output_channels = OutputChannels(*port);
	const auto           prepared_size =
	    BytesPerSample(port->format) * output_channels * port->samples_num;
	if (port->meter_device) {
		port->device_meter.Add(prepared_data, port->samples_num, output_channels, FormatIsFloat(port->format));
		if (FormatIsFloat(port->format)) {
			const auto* samples = static_cast<const float*>(prepared_data);
			for (uint32_t i = 0; i < port->samples_num * output_channels; i++) {
				port->device_clipped += std::abs(samples[i]) > 1.0f ? 1 : 0;
			}
		}
	}

	const bool trace       = Diag::TraceEnabled();
	const bool was_primed  = port->queue_primed;
	const auto frame_bytes = BytesPerSample(port->format) * output_channels;
	const auto to_ms       = [&](int bytes) {
		return frame_bytes != 0 && port->freq != 0
		           ? static_cast<double>(bytes) / frame_bytes * 1000.0 / port->freq
		           : 0.0;
	};
	uint32_t min_queued_size = 0;
	int      queued_before   = -1;
	port->catch_up           = false;
	if (blocking) {
		const auto buffer_us = port->freq != 0 ? (1000000ULL * port->samples_num) / port->freq : 0;
		const auto buffers =
		    buffer_us != 0
		        ? static_cast<uint32_t>((audio_out_target_latency_us() + buffer_us - 1) / buffer_us)
		        : 2u;
		min_queued_size           = prepared_size * std::clamp(buffers, 2u, 96u);
		const auto wait_start      = LibKernel::KernelGetProcessTime();
		auto       queued          = SDL_GetAudioStreamQueued(port->stream);
		const int  queued_on_entry = queued;
		bool       cleared         = false;
		while (queued > static_cast<int>(min_queued_size)) {
			if (LibKernel::KernelGetProcessTime() - wait_start > 200000) {
				SDL_ClearAudioStream(port->stream);
				port->queue_primed = false;
				cleared            = true;
				break;
			}
			Common::Thread::SleepMicro(1000);
			queued = SDL_GetAudioStreamQueued(port->stream);
		}
		if (queued < static_cast<int>(prepared_size)) {
			port->queue_primed = false;
		}
		// Below target by more than a block: the device is consuming faster than the sample clock
		// paces this port. Write the next block without the clock wait.
		port->catch_up = audio_out_queue_catchup() && port->queue_primed &&
		                 queued + static_cast<int>(prepared_size) < static_cast<int>(min_queued_size);
		queued_before = queued;
		if (trace) {
			const auto now    = LibKernel::KernelGetProcessTime();
			auto&      t      = port->trace;
			const auto waited = now - wait_start;
			t.blocking++;
			t.wait[OutTrace::WaitBucket(waited, cleared)]++;
			t.catch_ups += port->catch_up ? 1 : 0;
			if (cleared) {
				t.clears++;
				t.total_clears++;
				if (t.events++ < 8) {
					Diag::Trace("t=%.3f s port %d CLEAR: queue did not drain for %.1f ms, %d bytes "
					            "(%.1f ms) queued on entry, %d (%.1f ms) at the clear; target %.1f ms",
					            static_cast<double>(now) / 1e6, port->trace_id,
					            static_cast<double>(waited) / 1000.0, queued_on_entry,
					            to_ms(queued_on_entry), queued, to_ms(queued), to_ms(static_cast<int>(min_queued_size)));
				}
			}
			if (queued < static_cast<int>(prepared_size)) {
				t.low++;
				t.total_low++;
				if (t.events++ < 8) {
					Diag::Trace("t=%.3f s port %d LOW: %d bytes (%.2f ms) queued before the write, "
					            "under one block (%.2f ms)%s",
					            static_cast<double>(now) / 1e6, port->trace_id, queued, to_ms(queued),
					            to_ms(static_cast<int>(prepared_size)),
					            was_primed ? "; was primed: re-priming" : "");
				}
			}
		}
	} else if (trace) {
		queued_before = SDL_GetAudioStreamQueued(port->stream);
	}
	if (trace) {
		auto& t = port->trace;
		t.writes++;
		t.bytes += prepared_size;
		if (queued_before >= 0) {
			t.queued_sum += queued_before;
			t.queued_min = std::min(t.queued_min, queued_before);
			t.queued_max = std::max(t.queued_max, queued_before);
			if (t.queued_first < 0) {
				t.queued_first = queued_before;
			}
			t.queued_last = queued_before;
		}
		if (was_primed && !port->queue_primed) {
			t.reprimes++;
		}
	}

	if (blocking && batch_paced && !port->queue_primed && queued_before >= 0 &&
	    queued_before < static_cast<int>(prepared_size)) {
		// Up to the target before this block, as a burst-primed port stands.
		int pad = static_cast<int>(min_queued_size) - queued_before;
		pad -= pad % static_cast<int>(frame_bytes);
		if (pad > 0) {
			const std::vector<uint8_t> silence(static_cast<size_t>(pad), 0);
			if (SDL_PutAudioStreamData(port->stream, silence.data(), pad) && trace) {
				port->trace.pads++;
				if (port->trace.events++ < 8) {
					Diag::Trace("t=%.3f s port %d PAD: %.2f ms queued in a paced batch; %.2f ms of silence "
					            "added to reach the target",
					            static_cast<double>(LibKernel::KernelGetProcessTime()) / 1e6, port->trace_id,
					            to_ms(queued_before), to_ms(pad));
				}
			}
		}
	}
	if (trace) {
		TraceContent(port, prepared_data);
	}

	if (!SDL_PutAudioStreamData(port->stream, prepared_data, static_cast<int>(prepared_size))) {
		LOGF("AudioOut: SDL_PutAudioStreamData failed: %s\n", SDL_GetError());
		if (trace) {
			Diag::Trace("port %d: SDL_PutAudioStreamData failed: %s", port->trace_id, SDL_GetError());
		}
		port->queue_primed = false;
		return false;
	}
	// Unpaced priming must not advance the playback deadline into the future.
	if (!port->queue_primed) {
		port->clock = {};
	}
	if (blocking && !port->queue_primed &&
	    SDL_GetAudioStreamQueued(port->stream) >= static_cast<int>(min_queued_size)) {
		port->queue_primed = true;
	}
	if (trace) {
		TraceSummary(port, LibKernel::KernelGetProcessTime());
	}

	return true;
}

void Audio::TraceContent(PortOut* port, const void* data) {
	auto&      t        = port->trace;
	const auto channels = std::min<uint32_t>(OutputChannels(*port), 8);
	const auto frames   = port->samples_num;
	const bool is_float = FormatIsFloat(port->format);
	const auto stride   = OutputChannels(*port);
	const auto sample   = [&](uint32_t frame, uint32_t ch) {
		const auto index = static_cast<size_t>(frame) * stride + ch;
		return is_float ? static_cast<const float*>(data)[index]
		                : static_cast<float>(static_cast<const int16_t*>(data)[index]) / 32768.0f;
	};
	if (frames == 0 || channels == 0) {
		return;
	}
	bool   zero  = true;
	double power = 0.0;
	bool   jump  = false;
	for (uint32_t ch = 0; ch < channels; ch++) {
		double steps = 0.0;
		float  prev  = sample(0, ch);
		for (uint32_t frame = 0; frame < frames; frame++) {
			const float value = sample(frame, ch);
			zero &= value == 0.0f;
			power += static_cast<double>(value) * value;
			if (frame != 0) {
				steps += std::abs(value - prev);
			}
			prev = value;
		}
		const double mean_step = frames > 1 ? steps / (frames - 1) : 0.0;
		const double boundary  = std::abs(sample(0, ch) - t.prev_last[ch]);
		jump |= t.have_prev && boundary > 0.05 && boundary > 12.0 * mean_step + 0.02;
		t.prev_last[ch] = sample(frames - 1, ch);
	}
	const double mean_power = power / (static_cast<double>(frames) * channels);
	t.zero_blocks += zero ? 1 : 0;
	t.dropouts += zero && t.prev_sound ? 1 : 0;
	t.jumps += jump ? 1 : 0;
	t.power_sum += mean_power;
	t.power_n++;
	t.have_prev  = true;
	t.prev_sound = mean_power > 1e-6; // -60 dBFS
}

void Audio::TraceSummary(PortOut* port, uint64_t now) {
	auto& t = port->trace;
	if (t.start == 0) {
		t.start = now;
		return;
	}
	if (now - t.start < Diag::TRACE_INTERVAL_US) {
		return;
	}
	const double elapsed     = static_cast<double>(now - t.start) / 1e6;
	const auto   frame_bytes = static_cast<double>(BytesPerSample(port->format) * OutputChannels(*port));
	const auto   to_ms       = [&](double bytes) { return bytes / frame_bytes * 1000.0 / port->freq; };
	auto&        device      = s_device_trace[port->trace_id - 1];
	const auto   dev_bytes   = device.bytes.exchange(0, std::memory_order_relaxed);
	const auto   dev_req     = device.requests.exchange(0, std::memory_order_relaxed);
	const auto   dev_short   = device.short_requests.exchange(0, std::memory_order_relaxed);
	const auto   short_bytes = device.short_bytes.exchange(0, std::memory_order_relaxed);
	t.total_short += dev_short;
	const int queued_now = SDL_GetAudioStreamQueued(port->stream);
	// The device's consumption over the window: what was put minus the change in queue level.
	const double consumed = static_cast<double>(t.bytes) -
	                        (t.queued_first >= 0 ? static_cast<double>(queued_now - t.queued_first) : 0.0);
	Diag::Trace("t=%.1f s port %d %s %u Hz: %u writes (%u blocking) put %.1f Hz, device took %.1f Hz "
	            "(requests %.1f Hz, %llu requests); queue ms min/avg/max %.1f/%.1f/%.1f first %.1f last %.1f; "
	            "low %u, clears %u, re-primes %u, catch-ups %u; device short %llu (%.1f ms silence); wait "
	            "hist 0/<1/<2/<5/<10/<20/<50ms/clear %u/%u/%u/%u/%u/%u/%u/%u; clock sleep %.1f ms in %u; "
	            "guest gap avg %.2f max %.2f ms, %u over a block; totals clears %llu low %llu short %llu",
	            static_cast<double>(now) / 1e6, port->trace_id, audio_out_port_type_name(port->type), port->freq,
	            t.writes, t.blocking, static_cast<double>(t.bytes) / frame_bytes / elapsed,
	            consumed / frame_bytes / elapsed, static_cast<double>(dev_bytes) / frame_bytes / elapsed,
	            static_cast<unsigned long long>(dev_req), t.queued_min == INT_MAX ? 0.0 : to_ms(t.queued_min),
	            t.writes != 0 ? to_ms(static_cast<double>(t.queued_sum) / t.writes) : 0.0, to_ms(t.queued_max),
	            to_ms(std::max(t.queued_first, 0)), to_ms(t.queued_last), t.low, t.clears, t.reprimes, t.catch_ups,
	            static_cast<unsigned long long>(dev_short), to_ms(static_cast<double>(short_bytes)), t.wait[0],
	            t.wait[1], t.wait[2], t.wait[3], t.wait[4], t.wait[5], t.wait[6], t.wait[7],
	            static_cast<double>(t.sleep_us) / 1000.0, t.sleeps,
	            t.gaps != 0 ? static_cast<double>(t.gap_sum) / t.gaps / 1000.0 : 0.0,
	            static_cast<double>(t.gap_max) / 1000.0, t.late_gaps,
	            static_cast<unsigned long long>(t.total_clears), static_cast<unsigned long long>(t.total_low),
	            static_cast<unsigned long long>(t.total_short));
	const auto max_interval = device.max_interval_ns.exchange(0, std::memory_order_relaxed);
	const auto late15       = device.late15.exchange(0, std::memory_order_relaxed);
	const auto late30       = device.late30.exchange(0, std::memory_order_relaxed);
	Diag::Trace("t=%.1f s port %d content: rms %.1f dBFS, %u zero blocks, %u dropouts (zero block after "
	            "sound), %u boundary jumps, %u silence pads; device request gaps max %.1f ms, %u over 15 ms, "
	            "%u over 30 ms",
	            static_cast<double>(now) / 1e6, port->trace_id,
	            t.power_n != 0 && t.power_sum > 0.0 ? 10.0 * std::log10(t.power_sum / t.power_n) : -200.0,
	            t.zero_blocks, t.dropouts, t.jumps, t.pads, static_cast<double>(max_interval) / 1e6, late15,
	            late30);
	OutTrace next;
	next.start        = now;
	next.last_return  = t.last_return;
	next.total_clears = t.total_clears;
	next.total_low    = t.total_low;
	next.total_short  = t.total_short;
	next.have_prev    = t.have_prev;
	next.prev_sound   = t.prev_sound;
	std::copy(std::begin(t.prev_last), std::end(t.prev_last), std::begin(next.prev_last));
	t = next;
}

Audio::Id Audio::AudioOutOpen(int type, uint32_t samples_num, uint32_t freq, Format format) {
	if (freq == 0) {
		return Id::Invalid();
	}

	Common::LockGuard lock(m_mutex);

	for (int id = 0; id < OUT_PORTS_MAX; id++) {
		if (!m_out_ports[id].used) {
			auto& port = m_out_ports[id];

			port.used        = true;
			port.type        = type;
			port.samples_num = samples_num;
			port.freq        = freq;
			port.format      = format;

			switch (format) {
				case Format::Signed16bitMono:
				case Format::FloatMono: port.channels_num = 1; break;
				case Format::Signed16bitStereo:
				case Format::FloatStereo: port.channels_num = 2; break;
				case Format::Signed16bit8Ch:
				case Format::Float8Ch:
				case Format::Signed16bit8ChStd:
				case Format::Float8ChStd: port.channels_num = 8; break;
				case Format::Float12Ch: port.channels_num = 12; break;
				default: EXIT("unknown format");
			}

			for (int i = 0; i < port.channels_num; i++) {
				port.volume[i] = 32768;
			}

			// Pad speaker ports keep the main output too; it plays them whenever no DualSense does.
			if (type != AUDIO_OUT_PORT_TYPE_VIBRATION) {
				OpenSdlDevice(&port);
			}
			port.trace_id = id + 1;
			port.trace    = {};
			s_device_trace[id].Reset();
			if (Diag::TraceEnabled() && port.stream != nullptr) {
				SDL_SetAudioStreamGetCallback(port.stream, trace_device_request, &s_device_trace[id]);
				Diag::Trace("t=%.3f s port %d open: %s, %u frames (%.2f ms), %u Hz, %d ch",
				            static_cast<double>(LibKernel::KernelGetProcessTime()) / 1e6, id + 1,
				            audio_out_port_type_name(type), samples_num,
				            1000.0 * samples_num / freq, freq, port.channels_num);
			}
			if (type == AUDIO_OUT_PORT_TYPE_VIBRATION || type == AUDIO_OUT_PORT_TYPE_PADSPK) {
				port.haptics =
				    Controller::DualSenseHaptics::Open(freq, type == AUDIO_OUT_PORT_TYPE_PADSPK);
			}

			Diag::Print("output port %d open: %s, %u frames, %u Hz, %s (%d ch); main output %s, "
			            "DualSense stream %s",
			            id + 1, audio_out_port_type_name(type), samples_num, freq,
			            magic_enum::enum_name(format).data(), port.channels_num,
			            port.stream != nullptr ? "open" : "none",
			            port.haptics != nullptr ? "created" : "none");

			return Id::Create(id);
		}
	}

	Diag::Print("output port open failed: all %d ports in use (%s, %u frames, %u Hz)",
	            OUT_PORTS_MAX, audio_out_port_type_name(type), samples_num, freq);
	return Id::Invalid();
}

bool Audio::AudioOutClose(Id handle) {
	Common::LockGuard lock(m_mutex);

	if (AudioOutValid(handle)) {
		auto& port = m_out_ports[handle.GetId()];

		Diag::Print("output port %d close: %s", handle.ToInt(), audio_out_port_type_name(port.type));
		if (Diag::TraceEnabled()) {
			Diag::Trace("t=%.3f s port %d close: %s; totals clears %llu low %llu short %llu",
			            static_cast<double>(LibKernel::KernelGetProcessTime()) / 1e6, handle.ToInt(),
			            audio_out_port_type_name(port.type),
			            static_cast<unsigned long long>(port.trace.total_clears),
			            static_cast<unsigned long long>(port.trace.total_low),
			            static_cast<unsigned long long>(port.trace.total_short));
		}
		CloseSdlDevice(&port);
		port = {};

		return true;
	}

	return false;
}

bool Audio::AudioOutValid(Id handle) {
	Common::LockGuard lock(m_mutex);

	return (handle.GetId() >= 0 && handle.GetId() < OUT_PORTS_MAX &&
	        m_out_ports[handle.GetId()].used);
}

bool Audio::AudioOutHasDevice(Id handle) {
	Common::LockGuard lock(m_mutex);

	return (handle.GetId() >= 0 && handle.GetId() < OUT_PORTS_MAX &&
	        m_out_ports[handle.GetId()].used && m_out_ports[handle.GetId()].stream != nullptr);
}

void Audio::MeterOutput(int port_id, PortOut* port, const void* data, bool on_main, float gain) {
	const auto now = LibKernel::KernelGetProcessTime();
	if (port->meter_start == 0) {
		port->meter_start = now;
	}
	port->meter.Add(data, port->samples_num, static_cast<uint32_t>(port->channels_num),
	                FormatIsFloat(port->format));
	port->meter_blocks++;
	port->meter_on_main += on_main ? 1 : 0;
	port->meter_gain = gain;
	if (now - port->meter_start < LEVEL_LOG_INTERVAL_US) {
		return;
	}
	// Raw levels are the guest's samples; "out" adds the guest's port volume (channel 0) and
	// the host gain, i.e. what reaches the device.
	const double volume = static_cast<double>(port->volume[0]) / 32768.0;
	const double rms    = port->meter.Rms();
	const double peak   = port->meter.Peak();
	std::printf("AudioLevel: t=%.1f s port %d %s %dch %uHz: raw rms %.1f dBFS peak %.1f dBFS; guest vol "
	            "%.2f host gain %.2f -> out rms %.1f dBFS; %u/%u blocks on main output (%.1f s)\n",
	            static_cast<double>(now) / 1e6, port_id + 1, audio_out_port_type_name(port->type),
	            port->channels_num, port->freq, Mix::LevelMeter::ToDb(rms), Mix::LevelMeter::ToDb(peak), volume,
	            static_cast<double>(port->meter_gain),
	            Mix::LevelMeter::ToDb(rms * volume * port->meter_gain), port->meter_on_main,
	            port->meter_blocks, static_cast<double>(now - port->meter_start) / 1e6);
	if (port->device_meter.Samples() != 0) {
		std::printf("AudioLevel: t=%.1f s port %d to SDL: %uch rms %.1f dBFS peak %.1f dBFS, %u samples "
		            "clipped\n",
		            static_cast<double>(now) / 1e6, port_id + 1, OutputChannels(*port),
		            Mix::LevelMeter::ToDb(port->device_meter.Rms()),
		            Mix::LevelMeter::ToDb(port->device_meter.Peak()), port->device_clipped);
		port->device_meter.Reset();
		port->device_clipped = 0;
	}
	std::fflush(stdout);
	port->meter.Reset();
	port->meter_start   = now;
	port->meter_blocks  = 0;
	port->meter_on_main = 0;
}

bool Audio::AudioOutGetStatus(Id handle, int* type, int* channels_num) {
	Common::LockGuard lock(m_mutex);

	if (AudioOutValid(handle)) {
		auto& port = m_out_ports[handle.GetId()];

		*type         = port.type;
		*channels_num = port.channels_num;

		return true;
	}

	return false;
}

bool Audio::AudioOutSetVolume(Id handle, uint32_t bitflag, const int* volume) {
	Common::LockGuard lock(m_mutex);

	if (AudioOutValid(handle)) {
		auto& port = m_out_ports[handle.GetId()];

		for (int i = 0; i < port.channels_num; i++, bitflag >>= 1u) {
			auto bit = bitflag & 0x1u;

			if (bit == 1) {
				if (Diag::TraceEnabled() && port.volume[i] != volume[i]) {
					static std::atomic<int> lines {0};
					if (lines.fetch_add(1, std::memory_order_relaxed) < 64) {
						Diag::Trace("t=%.3f s port %d volume ch %d: %d -> %d",
						            static_cast<double>(LibKernel::KernelGetProcessTime()) / 1e6,
						            handle.ToInt(), i, port.volume[i], volume[i]);
					}
				}
				port.volume[i] = volume[i];

				LOGF("\t port.volume[%d] = volume[%d] (%d)\n", i, i, volume[i]);
			}
		}

		return true;
	}

	return false;
}

uint32_t Audio::AudioOutOutputs(OutputParam* params, uint32_t num, bool blocking) {
	EXIT_NOT_IMPLEMENTED(num == 0 || num > OUT_PORTS_MAX);
	EXIT_NOT_IMPLEMENTED(!AudioOutValid(params[0].handle));

	auto&                           first_port = m_out_ports[params[0].handle.GetId()];
	std::array<bool, OUT_PORTS_MAX> paced {};
	bool                            any_device = false;
	uint64_t                        deadline   = 0;
	const bool                      trace      = Diag::TraceEnabled();
	if (trace) {
		auto&      t     = first_port.trace;
		const auto start = LibKernel::KernelGetProcessTime();
		if (t.last_return != 0 && start > t.last_return) {
			const auto gap    = start - t.last_return;
			const auto period = first_port.freq != 0 ? 1000000ULL * first_port.samples_num / first_port.freq : 0;
			t.gap_sum += gap;
			t.gap_max = std::max(t.gap_max, gap);
			t.gaps++;
			t.late_gaps += gap > period ? 1 : 0;
		}
	}
	// Which ports already pace this batch (see KYTY_AUDIO_PAD_PRIME): a port is padded only when
	// another one does.
	std::array<bool, OUT_PORTS_MAX> primed_before {};
	uint32_t                        primed_count = 0;
	for (uint32_t i = 0; i < num && blocking && audio_out_pad_prime(); i++) {
		const auto& port = m_out_ports[params[i].handle.GetId()];
		primed_before[i] = port.stream != nullptr && port.queue_primed;
		primed_count += primed_before[i] ? 1 : 0;
	}

	for (uint32_t i = 0; i < num; i++) {
		auto& port = m_out_ports[params[i].handle.GetId()];
		// The DualSense speaker/vibration setting also scales a pad speaker port that falls back
		// to the main output, so the speaker hotkey keeps working there.
		const float gain =
		    port.type == AUDIO_OUT_PORT_TYPE_PADSPK
		        ? Controller::GetSettingScale(Controller::Setting::SpeakerVolume)
		    : port.type == AUDIO_OUT_PORT_TYPE_VIBRATION
		        ? Controller::GetSettingScale(Controller::Setting::VibrationIntensity)
		        : 1.0f;

		uint64_t controller_queued_us = 0;
		bool controller_uses_bluetooth = false;
		if (port.haptics != nullptr) {
			// Keep the stream alive against close and volume changes.
			Common::LockGuard lock(m_mutex);
			controller_queued_us = Controller::DualSenseHaptics::Queue(
			    port.haptics, Controller::GetActiveControllerId(), params[i].data, port.samples_num,
			    static_cast<uint32_t>(port.channels_num), FormatIsFloat(port.format), port.volume,
			    gain, params[i].gains);
			controller_uses_bluetooth =
			    Controller::DualSenseHaptics::UsesBluetooth(port.haptics);
		}
		if (controller_queued_us == 0) {
			// No DualSense took it (e.g. it was unplugged); a pad speaker port plays on the main
			// output instead, and a vibration port has none.
			const float main_gain = gain * Mix::MainOutputGain(m_mix, port.type, true);
			if (m_level_logging) {
				MeterOutput(params[i].handle.GetId(), &port, params[i].data,
				            port.stream != nullptr, main_gain);
			}
			port.meter_device = m_level_logging;
			const bool paced_by_other = primed_count > (primed_before[i] ? 1u : 0u);
			if (QueueSdlAudio(&port, params[i].data, blocking, main_gain, params[i].gains, paced_by_other)) {
				any_device = true;
				paced[i]   = port.queue_primed && !port.catch_up;
			}
		} else {
			if (m_level_logging) {
				MeterOutput(params[i].handle.GetId(), &port, params[i].data, false, gain);
			}
			port.queue_primed = false;
			port.catch_up     = false;
			// Bluetooth HID has its own sender. Let the sample clock pace a batch
			// without another audio device, instead of waiting on the HID queue.
			if (port.type == AUDIO_OUT_PORT_TYPE_PADSPK && !controller_uses_bluetooth) {
				any_device = true;
				if (blocking && controller_queued_us > AUDIO_OUT_TARGET_LATENCY_US) {
					// Limit a stalled USB endpoint to one audio block of waiting.
					const uint64_t block_us = 1000000ULL * port.samples_num / port.freq;
					deadline = std::max(deadline, LibKernel::KernelGetProcessTime() +
					                                  std::min(controller_queued_us - AUDIO_OUT_TARGET_LATENCY_US,
					                                           block_us));
				}
			}
		}
	}

	// Submit the whole batch before pacing it. Audible devices own its cadence while
	// priming; silent/vibration-only batches use the same sample clock as SDL output.
	const auto now = LibKernel::KernelGetProcessTime();
	// One port short of its target skips the clock for the whole batch: every port gets one block
	// per batch, so a clock kept by a fuller port would go on draining the short one. The fuller
	// ports' own queue waits keep them at their targets.
	bool batch_catch_up = false;
	for (uint32_t i = 0; i < num; i++) {
		const auto& port = m_out_ports[params[i].handle.GetId()];
		batch_catch_up |= port.catch_up && port.stream != nullptr;
	}
	for (uint32_t i = 0; i < num; i++) {
		auto& port = m_out_ports[params[i].handle.GetId()];
		if (batch_catch_up) {
			paced[i] = false;
		}
		if (blocking && (paced[i] || !any_device)) {
			deadline = std::max(deadline, port.clock.Advance(now, port.samples_num, audio_out_clock_rate(port.freq)));
		} else {
			port.clock = {};
		}
		if (!blocking) {
			port.queue_primed = false;
		}
	}
	if (deadline > now) {
		Common::Thread::SleepMicro(deadline - now);
		if (trace) {
			first_port.trace.sleep_us += deadline - now;
			first_port.trace.sleeps++;
		}
	}
	if (trace) {
		first_port.trace.last_return = LibKernel::KernelGetProcessTime();
	}

	return first_port.samples_num;
}

static bool RecordingDevicePresent(SDL_AudioDeviceID device) {
	int                count   = 0;
	SDL_AudioDeviceID* devices = SDL_GetAudioRecordingDevices(&count);
	bool               present = false;
	for (int i = 0; i < count; i++) {
		if (devices[i] == device) {
			present = true;
			break;
		}
	}
	SDL_free(devices);
	return present;
}

void Audio::OpenSdlDevice(PortIn* port, Format format) {
	const auto& name = Config::GetAudioInputDevice();
	if (name.empty()) {
		Diag::Print("input port: no microphone configured; the game records silence");
		return;
	}
	if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
		LOGF("AudioIn: SDL init failed: %s\n", SDL_GetError());
		Diag::Print("input port: SDL audio init failed: %s", SDL_GetError());
		return;
	}
	int                device_count = 0;
	SDL_AudioDeviceID* devices      = SDL_GetAudioRecordingDevices(&device_count);
	SDL_AudioDeviceID  device       = 0;
	for (int i = 0; i < device_count; i++) {
		const char* device_name = SDL_GetAudioDeviceName(devices[i]);
		if (device_name != nullptr && name == device_name) {
			device = devices[i];
			break;
		}
	}
	SDL_free(devices);
	if (device == 0) {
		LOGF("AudioIn: cannot find '%s'; using silence\n", name.c_str());
		Diag::Print("input port: microphone '%s' not found; the game records silence",
		            name.c_str());
		SDL_QuitSubSystem(SDL_INIT_AUDIO);
		return;
	}
	SDL_AudioSpec desired {};
	desired.freq     = static_cast<int>(port->freq);
	desired.format   = SdlFormat(format);
	desired.channels = static_cast<int>(port->bytes_per_frame / BytesPerSample(format));
	port->stream     = SDL_OpenAudioDeviceStream(device, &desired, nullptr, nullptr);
	if (port->stream == nullptr) {
		LOGF("AudioIn: cannot open '%s': %s; using silence\n", name.c_str(), SDL_GetError());
		Diag::Print("input port: cannot open microphone '%s': %s", name.c_str(), SDL_GetError());
		SDL_QuitSubSystem(SDL_INIT_AUDIO);
		return;
	}
	port->device = device;
	LOGF("AudioIn: opened '%s' (%u Hz, %d ch)\n", name.c_str(), port->freq, desired.channels);
	Diag::Print("input port: recording from '%s' (%u Hz, %d ch)", name.c_str(), port->freq,
	            desired.channels);
}

void Audio::CloseSdlDevice(PortIn* port) {
	if (port->stream != nullptr) {
		SDL_DestroyAudioStream(port->stream);
		SDL_QuitSubSystem(SDL_INIT_AUDIO);
		port->stream = nullptr;
		port->device = 0;
	}
}

Audio::Id Audio::AudioInOpen(uint32_t samples_num, uint32_t freq, Format format, bool asynchronous) {
	Common::LockGuard lock(m_mutex);

	for (int id = 0; id < IN_PORTS_MAX; id++) {
		if (!m_in_ports[id].used) {
			auto& port = m_in_ports[id];

			port                 = {};
			port.used            = true;
			port.asynchronous    = asynchronous;
			port.samples_num     = samples_num;
			port.freq            = freq;
			port.bytes_per_frame = BytesPerSample(format);
			if (format == Format::Signed16bitStereo || format == Format::FloatStereo) {
				port.bytes_per_frame *= 2;
			}
			OpenSdlDevice(&port, format);

			return Id::Create(id);
		}
	}

	return Id::Invalid();
}

Audio::PortIn* Audio::GetAudioInPort(Id handle) {
	if (handle.GetId() < 0 || handle.GetId() >= IN_PORTS_MAX ||
	    !m_in_ports[handle.GetId()].used) {
		return nullptr;
	}
	return &m_in_ports[handle.GetId()];
}

int Audio::AudioInClose(Id handle) {
	Common::LockGuard lock(m_mutex);
	auto* port = GetAudioInPort(handle);
	if (port == nullptr) {
		return AUDIO_IN_ERROR_INVALID_HANDLE;
	}
	if (port->busy) {
		return AUDIO_IN_ERROR_BUSY;
	}
	CloseSdlDevice(port);
	*port = {};
	return OK;
}

int Audio::AudioInGetSilentState(Id handle) {
	Common::LockGuard lock(m_mutex);
	auto*             port = GetAudioInPort(handle);
	if (port == nullptr) {
		return AUDIO_IN_ERROR_INVALID_HANDLE;
	}
	if (port->stream == nullptr || !RecordingDevicePresent(port->device)) {
		return AUDIO_IN_SILENT_STATE_DEVICE_NONE;
	}
	return 0;
}

int Audio::AudioInInput(Id handle, void* dest) {
	PortIn snapshot;
	{
		Common::LockGuard lock(m_mutex);
		auto*             port = GetAudioInPort(handle);
		if (port == nullptr) {
			return AUDIO_IN_ERROR_INVALID_HANDLE;
		}
		if (port->busy) {
			return AUDIO_IN_ERROR_BUSY;
		}
		port->busy = true;
		snapshot   = *port;
	}
	bool failed = snapshot.stream != nullptr && dest != nullptr &&
	              (!RecordingDevicePresent(snapshot.device) ||
	               !SDL_ResumeAudioStreamDevice(snapshot.stream));

	const uint64_t block_time = (1000000ULL * snapshot.samples_num) / snapshot.freq;
	uint64_t       wait_time  = 0;
	if (snapshot.asynchronous) {
		if (dest != nullptr) {
			wait_time = block_time;
		}
	} else if (snapshot.last_input_time != 0 && (snapshot.stream == nullptr || dest == nullptr)) {
		const auto now  = LibKernel::KernelGetProcessTime();
		const auto next = snapshot.last_input_time + block_time;
		if (next > now) {
			wait_time = next - now;
		}
	}
	Common::Thread::SleepMicro(wait_time);
	uint32_t frames = snapshot.samples_num;
	bool     timed_out = false;
	if (dest != nullptr) {
		if (snapshot.stream != nullptr && !failed) {
			const int requested = static_cast<int>(frames * snapshot.bytes_per_frame);
			const auto deadline = LibKernel::KernelGetProcessTime() +
			                      std::max<uint64_t>(20000, block_time * 4);
			const auto device = SDL_GetAudioStreamDevice(snapshot.stream);
			int        available = SDL_GetAudioStreamAvailable(snapshot.stream);
			while (!snapshot.asynchronous && available >= 0 &&
			       available < requested &&
			       device != 0 && !SDL_AudioDevicePaused(device) &&
			       LibKernel::KernelGetProcessTime() < deadline) {
				Common::Thread::SleepMicro(1000);
				available = SDL_GetAudioStreamAvailable(snapshot.stream);
			}
			failed = available < 0 || device == 0 || SDL_AudioDevicePaused(device);
			timed_out = !snapshot.asynchronous && available < requested;
			if (!failed && !timed_out) {
				if (snapshot.asynchronous) {
					frames = AUDIO_IN_GRAIN_MAX_ASYNC;
				}
				const int bytes = SDL_GetAudioStreamData(
				    snapshot.stream, dest, static_cast<int>(frames * snapshot.bytes_per_frame));
				failed = bytes < 0;
				if (!failed) {
					frames = static_cast<uint32_t>(bytes) / snapshot.bytes_per_frame;
				}
			}
		}
		if (snapshot.stream == nullptr || failed || timed_out) {
			std::memset(dest, 0, frames * snapshot.bytes_per_frame);
		}
	} else if (snapshot.stream != nullptr) {
		SDL_PauseAudioStreamDevice(snapshot.stream);
		SDL_ClearAudioStream(snapshot.stream);
	}
	{
		Common::LockGuard lock(m_mutex);
		auto&             port = m_in_ports[handle.GetId()];
		if (failed) {
			LOGF("AudioIn: capture stopped; using silence\n");
			CloseSdlDevice(&port);
		}
		port.last_input_time = dest != nullptr ? LibKernel::KernelGetProcessTime() : 0;
		port.busy            = false;
	}
	return dest != nullptr ? static_cast<int>(frames) : 0;
}

namespace AudioOut {

LIB_NAME("AudioOut", "AudioOut");

struct AudioOutOutputParam {
	int         handle;
	const void* ptr;
};

struct AudioOutPortState {
	uint16_t output;
	uint8_t  channel;
	uint8_t  reserved1[1];
	int16_t  volume;
	uint16_t reroute_counter;
	uint64_t flag;
	uint64_t reserved2[2];
};

int KYTY_SYSV_ABI AudioOutInit() {
	PRINT_NAME();

	return OK;
}

int KYTY_SYSV_ABI AudioOutOpen(int user_id, int type, int index, uint32_t len, uint32_t freq,
                               uint32_t param) {
	PRINT_NAME();

	LOGF("\t user_id = %d\n"
	     "\t type    = %d\n"
	     "\t index   = %d\n"
	     "\t len     = %u\n"
	     "\t freq    = %u\n",
	     user_id, type, index, len, freq);

	Diag::Print("sceAudioOutOpen(user %d, type %d (%s), index %d, %u frames, %u Hz, param 0x%x)",
	            user_id, type, audio_out_port_type_name(type), index, len, freq, param);

	if (!audio_out_port_type_is_valid(type)) {
		return Diag::Error("sceAudioOutOpen", AUDIO_OUT_ERROR_INVALID_PORT_TYPE, type);
	}
	if (freq == 0) {
		return Diag::Error("sceAudioOutOpen", AUDIO_OUT_ERROR_INVALID_SAMPLE_FREQ, freq);
	}
	EXIT_NOT_IMPLEMENTED(index != 0);

	Audio::Format format       = Audio::Format::Unknown;
	const auto    format_param = param & AUDIO_OUT_PARAM_FORMAT_MASK;

	switch (format_param) {
		case 0: format = Audio::Format::Signed16bitMono; break;
		case 1: format = Audio::Format::Signed16bitStereo; break;
		case 2: format = Audio::Format::Signed16bit8Ch; break;
		case 3: format = Audio::Format::FloatMono; break;
		case 4: format = Audio::Format::FloatStereo; break;
		case 5: format = Audio::Format::Float8Ch; break;
		case 6: format = Audio::Format::Signed16bit8ChStd; break;
		case 7: format = Audio::Format::Float8ChStd; break;
		default:;
	}

	LOGF("\t param   = %u (format=%u, %s)\n", param, format_param, magic_enum::enum_name(format));

	EXIT_NOT_IMPLEMENTED(format == Audio::Format::Unknown);

	EXIT_IF(g_audio == nullptr);

	auto id = g_audio->AudioOutOpen(type, len, freq, format);

	if (!id.IsValid()) {
		return Diag::Error("sceAudioOutOpen", AUDIO_OUT_ERROR_PORT_FULL, type);
	}

	Diag::Print("sceAudioOutOpen -> handle %d", id.ToInt());
	return id.ToInt();
}

int KYTY_SYSV_ABI AudioOutClose(int handle) {
	PRINT_NAME();

	if (!g_audio->AudioOutClose(Audio::Id(handle))) {
		return Diag::Error("sceAudioOutClose", AUDIO_OUT_ERROR_INVALID_PORT, handle);
	}

	return OK;
}

int KYTY_SYSV_ABI AudioOutGetPortState(int handle, AudioOutPortState* state) {
	PRINT_NAME();

	int type         = 0;
	int channels_num = 0;

	if (!g_audio->AudioOutGetStatus(Audio::Id(handle), &type, &channels_num)) {
		return Diag::Error("sceAudioOutGetPortState", AUDIO_OUT_ERROR_INVALID_PORT, handle);
	}

	EXIT_NOT_IMPLEMENTED(state == nullptr);

	state->reroute_counter = 0;
	state->volume          = 127;

	switch (type) {
		case AUDIO_OUT_PORT_TYPE_MAIN:
		case AUDIO_OUT_PORT_TYPE_BGM:
		case AUDIO_OUT_PORT_TYPE_AUDIO3D:
			state->output  = 1;
			state->channel = (channels_num > 2 ? 2 : channels_num);
			break;
		case AUDIO_OUT_PORT_TYPE_VOICE:
		case AUDIO_OUT_PORT_TYPE_PERSONAL:
			state->output  = 0x40;
			state->channel = 1;
			break;
		case AUDIO_OUT_PORT_TYPE_PADSPK:
		case AUDIO_OUT_PORT_TYPE_VIBRATION:
			state->output  = 4;
			state->channel = 1;
			break;
		case AUDIO_OUT_PORT_TYPE_AUX:
			state->output  = 0x80;
			state->channel = 0;
			break;
		default: EXIT("unknown port type: %d\n", type);
	}

	LOGF("\t output  = %" PRIu16 "\n"
	     "\t channel = %" PRIu8 "\n",
	     state->output, state->channel);

	return OK;
}

int KYTY_SYSV_ABI AudioOutSetVolume(int handle, uint32_t flag, int* vol) {
	PRINT_NAME();

	LOGF("\t handle = %d\n"
	     "\t flag   = %u\n",
	     handle, flag);

	EXIT_IF(g_audio == nullptr);
	EXIT_NOT_IMPLEMENTED(vol == nullptr);

	if (!g_audio->AudioOutSetVolume(Audio::Id(handle), flag, vol)) {
		return Diag::Error("sceAudioOutSetVolume", AUDIO_OUT_ERROR_INVALID_PORT, handle);
	}

	return OK;
}

int KYTY_SYSV_ABI AudioOutOutputs(AudioOutOutputParam* param, uint32_t num) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(param == nullptr);
	if (num == 0 || num > Audio::OUT_PORTS_MAX) {
		return Diag::Error("sceAudioOutOutputs", AUDIO_OUT_ERROR_INVALID_SIZE, num);
	}

	Audio::OutputParam params[Audio::OUT_PORTS_MAX];

	EXIT_IF(g_audio == nullptr);

	for (uint32_t i = 0; i < num; i++) {
		params[i].handle = Audio::Id(param[i].handle);
		params[i].data   = param[i].ptr;

		if (!g_audio->AudioOutValid(params[i].handle)) {
			return Diag::Error("sceAudioOutOutputs", AUDIO_OUT_ERROR_INVALID_PORT, param[i].handle);
		}
	}

	return static_cast<int>(g_audio->AudioOutOutputs(params, num));
}

int KYTY_SYSV_ABI AudioOutOutput(int handle, const void* ptr) {
	// EXIT_NOT_IMPLEMENTED(ptr == nullptr);

	Audio::OutputParam params[1];

	EXIT_IF(g_audio == nullptr);

	params[0].handle = Audio::Id(handle);
	params[0].data   = ptr;

	if (!g_audio->AudioOutValid(params[0].handle)) {
		return Diag::Error("sceAudioOutOutput", AUDIO_OUT_ERROR_INVALID_PORT, handle);
	}

	return static_cast<int>(g_audio->AudioOutOutputs(params, 1));
}

} // namespace AudioOut

namespace AudioIn {

LIB_NAME("AudioIn", "AudioIn");

static int OpenPort(int user_id, int type, int index, uint32_t len, uint32_t freq,
                    uint32_t param, bool asynchronous) {
	LOGF("\t user_id = %d\n"
	     "\t type    = %d\n"
	     "\t index   = %d\n"
	     "\t len     = %u\n"
	     "\t freq    = %u\n",
	     user_id, type, index, len, freq);

	const char* api = asynchronous ? "sceAudioInHqOpen" : "sceAudioInOpen";
	Diag::Print("%s(user %d, type %d, index %d, %u frames, %u Hz, param 0x%x)", api, user_id, type,
	            index, len, freq, param);

	if (type != 0 && type != 1) {
		return Diag::Error(api, AUDIO_IN_ERROR_INVALID_TYPE, type);
	}
	if (index != 0) {
		return Diag::Error(api, AUDIO_IN_ERROR_INVALID_PARAM, index);
	}
	if (len != 128 && (asynchronous || len != 256)) {
		return Diag::Error(api, AUDIO_IN_ERROR_INVALID_SIZE, len);
	}
	if (freq != 48000 && (asynchronous || freq != 16000)) {
		return Diag::Error(api, AUDIO_IN_ERROR_INVALID_FREQ, freq);
	}

	Audio::Format format = Audio::Format::Unknown;

	switch (param) {
		case 1: format = Audio::Format::Signed16bitMono; break;
		case 2: format = Audio::Format::Signed16bitStereo; break;
		case 0x10:
		case 0x11: format = Audio::Format::FloatMono; break;
		case 0x12: format = Audio::Format::FloatStereo; break;
		default: return Diag::Error(api, AUDIO_IN_ERROR_INVALID_PARAM, param);
	}

	LOGF("\t param   = %u (%s)\n", param, magic_enum::enum_name(format));

	EXIT_IF(g_audio == nullptr);

	auto id = g_audio->AudioInOpen(len, freq, format, asynchronous);

	if (!id.IsValid()) {
		return Diag::Error(api, AUDIO_IN_ERROR_PORT_FULL, len);
	}

	Diag::Print("%s -> handle %d", api, id.ToInt());
	return id.ToInt();
}

int KYTY_SYSV_ABI AudioInOpen(int user_id, int type, int index, uint32_t len,
                              uint32_t freq, uint32_t param) {
	PRINT_NAME();
	return OpenPort(user_id, type, index, len, freq, param, false);
}

int KYTY_SYSV_ABI AudioInHqOpen(int user_id, int type, int index, uint32_t len,
                                uint32_t freq, uint32_t param) {
	PRINT_NAME();
	return OpenPort(user_id, type, index, len, freq, param, true);
}

int KYTY_SYSV_ABI AudioInClose(int handle) {
	PRINT_NAME();
	EXIT_IF(g_audio == nullptr);
	if (handle <= 0) {
		return Diag::Error("sceAudioInClose", AUDIO_IN_ERROR_INVALID_HANDLE, handle);
	}
	const int result = g_audio->AudioInClose(Audio::Id(handle));
	return result < 0 ? Diag::Error("sceAudioInClose", result, handle) : result;
}

int KYTY_SYSV_ABI AudioInInput(int handle, void* dest) {
	PRINT_NAME();
	EXIT_IF(g_audio == nullptr);
	if (handle <= 0) {
		return Diag::Error("sceAudioInInput", AUDIO_IN_ERROR_INVALID_HANDLE, handle);
	}

	const int result = g_audio->AudioInInput(Audio::Id(handle), dest);
	return result < 0 ? Diag::Error("sceAudioInInput", result, handle) : result;
}

int KYTY_SYSV_ABI AudioInGetSilentState(int handle) {
	PRINT_NAME();

	EXIT_IF(g_audio == nullptr);

	if (handle <= 0) {
		return Diag::Error("sceAudioInGetSilentState", AUDIO_IN_ERROR_INVALID_HANDLE, handle);
	}

	const int result = g_audio->AudioInGetSilentState(Audio::Id(handle));
	return result < 0 ? Diag::Error("sceAudioInGetSilentState", result, handle) : result;
}

} // namespace AudioIn

namespace VoiceQoS {

LIB_NAME("VoiceQoS", "VoiceQoS");

int KYTY_SYSV_ABI VoiceQoSInit(void* mem_block, uint32_t mem_size, int32_t app_type) {
	PRINT_NAME();

	LOGF("\t mem_block = %016" PRIx64 "\n"
	     "\t mem_size = %" PRIu32 "\n"
	     "\t app_type = %" PRId32 "\n",
	     reinterpret_cast<uint64_t>(mem_block), mem_size, app_type);

	return OK;
}

} // namespace VoiceQoS

} // namespace Libs::Audio
