#include "libs/dualSenseHaptics.h"

#include "common/logging/log.h"
#include "common/threads.h"
#include "libs/audioDiag.h"
#include "libs/dualSenseBluetooth.h"

#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <map>
#include <vector>

namespace Libs::Controller::DualSenseHaptics {

struct Stream {
	uint32_t                    freq       = 0;
	bool                        speaker    = false;
	int                         controller = -1;
	SDL_AudioStream*            sdl        = nullptr;
	DualSenseBluetooth::Stream* bluetooth  = nullptr;
	SDL_AudioDeviceID           device     = 0;
	uint64_t                    next_check = 0;
	std::vector<float>          buffer;

	// Haptics converted for pads that are not a DualSense
	int                     rumble_pad    = -1;
	int                     rumble_mode   = 0; // HOST_HAPTICS_*
	uint32_t                rumble_frames = 0;
	std::array<double, 2>   rumble_sum {};
	std::array<uint32_t, 2> rumble_crossings {};
	std::array<int, 2>      rumble_sign {};
	std::array<float, 2>    rumble_freq {};
	std::array<uint16_t, 2> rumble_last {};
};

namespace {

constexpr int      FRAME_BYTES  = 4 * sizeof(float);
constexpr uint32_t MAX_QUEUE_MS = 80;
Common::Mutex      g_mutex;
int                g_controller      = -1;
int                g_haptics_streams = 0;
uint8_t            g_large_motor = 0, g_small_motor = 0;
uint64_t           g_rumble_until = 0, g_haptics_until = 0;
int                g_speaker_controller = -1;
std::map<int, int> g_speaker_streams;

// Where pad speaker and vibration audio goes, for the console diagnostics: printed when it
// changes, per port kind, so a 2-second device recheck does not repeat it.
enum class Route { Unknown, NoWiredPad, NoUsbDevice, Usb, UsbFailed, Bluetooth, BluetoothFailed };
std::atomic<Route> g_reported_route[2] {Route::Unknown, Route::Unknown};

void ReportRoute(const Stream* stream, Route route, const char* detail) {
	auto& reported = g_reported_route[stream->speaker ? 1 : 0];
	if (reported.exchange(route) == route) {
		return;
	}
	const char* text = "unknown";
	switch (route) {
		case Route::NoWiredPad:
			text = "no single wired DualSense and no Bluetooth one; not played";
			break;
		case Route::NoUsbDevice:
			text = "wired DualSense, but no 4-channel 'Wireless Controller' audio device";
			break;
		case Route::Usb: text = "USB audio device"; break;
		case Route::UsbFailed: text = "USB audio device failed to open"; break;
		case Route::Bluetooth: text = "Bluetooth HID audio"; break;
		case Route::BluetoothFailed: text = "Bluetooth HID audio failed to open"; break;
		case Route::Unknown: break;
	}
	Libs::Audio::Diag::Print("DualSense %s audio (controller %d, %u Hz): %s%s%s",
	                         stream->speaker ? "speaker" : "vibration", stream->controller,
	                         stream->freq, text, detail != nullptr ? ": " : "",
	                         detail != nullptr ? detail : "");
}

void ApplyRumble(bool force = false) { // Caller holds g_mutex; no calls into Audio or Controller.
	SDL_LockJoysticks();
	if (auto* pad = SDL_GetGamepadFromID(static_cast<SDL_JoystickID>(g_controller));
	    pad != nullptr) {
		const auto now   = SDL_GetTicks();
		const bool muted = g_haptics_until > now || g_rumble_until <= now;
		if (force && !muted) {
			// SDL caches motor strengths. Force a report after raw speaker routing changed the
			// controller's motor mode without changing that cache.
			(void)SDL_RumbleGamepad(pad, 0, 0, 0);
		}
		(void)SDL_RumbleGamepad(
		    pad, muted ? 0 : g_large_motor * 0x101U, muted ? 0 : g_small_motor * 0x101U,
		    g_rumble_until > now ? static_cast<uint32_t>(g_rumble_until - now) : 0);
	}
	SDL_UnlockJoysticks();
}

void SelectController(int controller) {
	if (g_controller != controller) {
		g_haptics_until = g_rumble_until = 0;
		ApplyRumble();
		g_controller  = controller;
		g_large_motor = g_small_motor = 0;
	}
}

bool RouteSpeaker(int controller, bool speaker) { // Caller holds g_mutex.
	// Route the right channel to the speaker and mute the headphones with the values Linux uses
	// without headphones (output report 0x02). Plugged-in headphones are not checked.
	std::array<uint8_t, 38> report {};
	report[0] = 0x80; // Audio control valid
	if (speaker) {
		report[0] |= 0x20; // Speaker volume valid
		report[1]  = 0x80; // Audio control 2 valid
		report[5]  = 0x64; // Speaker volume
		report[7]  = 0x30; // Output path: right channel to the speaker, headphones muted
		report[37] = 0x02; // Speaker preamp +6 dB
	}
	bool routed = false;
	SDL_LockJoysticks();
	if (auto* pad = SDL_GetGamepadFromID(static_cast<SDL_JoystickID>(controller)); pad != nullptr) {
		routed = SDL_SendGamepadEffect(pad, report.data(), static_cast<int>(report.size()));
	}
	SDL_UnlockJoysticks();
	if (routed && controller == g_controller) {
		ApplyRumble(true);
	}
	return routed;
}

bool SelectSpeaker(int controller) { // Caller holds g_mutex; -1 restores headphone routing.
	if (g_speaker_controller == controller) {
		return true;
	}
	if (g_speaker_controller != -1) {
		RouteSpeaker(g_speaker_controller, false);
	}
	g_speaker_controller = -1;
	if (controller != -1 && !RouteSpeaker(controller, true)) {
		return false;
	}
	g_speaker_controller = controller;
	return true;
}

void SDLCALL UpdateRumble(void*, SDL_AudioStream*, int, int) {
	Common::LockGuard lock(g_mutex);
	if (g_haptics_until != 0 && SDL_GetTicks() >= g_haptics_until) {
		g_haptics_until = 0;
		ApplyRumble();
	}
}

void UpdateBluetoothRumble() {
	UpdateRumble(nullptr, nullptr, 0, 0);
}

void CloseDevice(Stream* stream) {
	const bool had_device = stream->sdl != nullptr || stream->bluetooth != nullptr;
	if (stream->bluetooth != nullptr) {
		DualSenseBluetooth::Close(stream->bluetooth);
		stream->bluetooth = nullptr;
	}
	if (stream->sdl != nullptr) {
		// Destroy outside g_mutex: SDL waits for any running get callback.
		SDL_DestroyAudioStream(stream->sdl);
		stream->sdl = nullptr;
	}
	if (had_device) {
		Common::LockGuard lock(g_mutex);
		if (!stream->speaker && --g_haptics_streams == 0 && g_haptics_until != 0) {
			g_haptics_until = 0;
			ApplyRumble();
		}
		// Route back to the headphones once no speaker port plays on the controller.
		if (stream->speaker && --g_speaker_streams.at(stream->controller) == 0) {
			g_speaker_streams.erase(stream->controller);
			if (g_speaker_controller == stream->controller) {
				SelectSpeaker(-1);
			}
		}
	}
}

bool IsWireless(int controller) {
	SDL_LockJoysticks();
	auto*      pad = SDL_GetGamepadFromID(static_cast<SDL_JoystickID>(controller));
	const bool wireless =
	    pad != nullptr && SDL_GetGamepadConnectionState(pad) == SDL_JOYSTICK_CONNECTION_WIRELESS;
	SDL_UnlockJoysticks();
	return wireless;
}

bool CanUseDevice(int controller) {
	// SDL exposes no portable association between a gamepad and its USB audio endpoint.
	// Require one wired DualSense; guessing with multiple pads could play on the wrong one.
	SDL_LockJoysticks();
	int   count                 = 0;
	auto* pads                  = SDL_GetGamepads(&count);
	int   wired_dualsense_count = 0;
	bool  found                 = false;
	for (int i = 0; i < count; i++) {
		auto* candidate = SDL_GetGamepadFromID(pads[i]);
		if (candidate != nullptr && SDL_GetGamepadTypeForID(pads[i]) == SDL_GAMEPAD_TYPE_PS5 &&
		    SDL_GetGamepadConnectionState(candidate) == SDL_JOYSTICK_CONNECTION_WIRED) {
			wired_dualsense_count++;
			found |= pads[i] == static_cast<SDL_JoystickID>(controller);
		}
	}
	SDL_free(pads);
	SDL_UnlockJoysticks();
	return found && wired_dualsense_count == 1;
}

SDL_AudioDeviceID FindDevice() {
	int               count   = 0;
	auto*             devices = SDL_GetAudioPlaybackDevices(&count);
	SDL_AudioDeviceID device  = 0;
	for (int i = 0; i < count; i++) {
		const char*   name = SDL_GetAudioDeviceName(devices[i]);
		SDL_AudioSpec spec {};
		const bool is_dualsense_name =
		    name != nullptr &&
		    (SDL_strcasestr(name, "DualSense") != nullptr ||
		     SDL_strcasestr(name, "Wireless Controller") != nullptr);
		if (is_dualsense_name &&
		    SDL_GetAudioDeviceFormat(devices[i], &spec, nullptr) && spec.channels == 4) {
			if (device != 0) {
				device = 0; // Multiple matching endpoints cannot be associated reliably either.
				break;
			}
			device = devices[i];
		}
	}
	SDL_free(devices);
	return device;
}

void RefreshDevice(Stream* stream, int controller, bool wireless) {
	if (stream->controller != controller || (wireless && stream->sdl != nullptr) ||
	    (!wireless && stream->bluetooth != nullptr)) {
		CloseDevice(stream);
		stream->controller = controller;
		stream->next_check = 0;
	}
	const auto now = SDL_GetTicks();
	if (stream->bluetooth != nullptr || now < stream->next_check) {
		return;
	}
	stream->next_check = now + 2000;
	const auto device  = wireless ? 0 : FindDevice();
	if (stream->device == device && stream->sdl != nullptr) {
		return;
	}
	CloseDevice(stream);
	stream->device = device;
	if (!wireless && device == 0) {
		ReportRoute(stream, Route::NoUsbDevice, nullptr);
		return;
	}
	SDL_AudioStream*            sdl       = nullptr;
	DualSenseBluetooth::Stream* bluetooth = nullptr;
	bool                        ready;
	if (wireless) {
		bluetooth = DualSenseBluetooth::Open(stream->freq, stream->speaker, controller,
		                                     UpdateBluetoothRumble);
		ready     = bluetooth != nullptr;
	} else {
		const SDL_AudioSpec desired {SDL_AUDIO_F32, 4, static_cast<int>(stream->freq)};
		SDL_AudioSpec       actual {};
		sdl   = SDL_OpenAudioDeviceStream(device, &desired, UpdateRumble, nullptr);
		ready = sdl != nullptr &&
		        SDL_GetAudioDeviceFormat(SDL_GetAudioStreamDevice(sdl), &actual, nullptr) &&
		        actual.channels == 4 && SDL_ResumeAudioStreamDevice(sdl);
	}
	if (ready) {
		Common::LockGuard lock(g_mutex);
		if (stream->speaker) {
			ready = SelectSpeaker(controller);
			if (ready) {
				g_speaker_streams[controller]++;
			}
		} else {
			g_haptics_streams++;
		}
	}
	if (ready) {
		stream->sdl       = sdl;
		stream->bluetooth = bluetooth;
		if (!wireless) {
			LOGF("DualSenseHaptics: playing on '%s'\n", SDL_GetAudioDeviceName(device));
		}
		ReportRoute(stream, wireless ? Route::Bluetooth : Route::Usb,
		            wireless ? nullptr : SDL_GetAudioDeviceName(device));
	} else {
		DualSenseBluetooth::Close(bluetooth);
		SDL_DestroyAudioStream(sdl);
		if (!wireless) {
			LOGF("DualSenseHaptics: cannot open quad output: %s\n", SDL_GetError());
		}
		ReportRoute(stream, wireless ? Route::BluetoothFailed : Route::UsbFailed,
		            wireless ? nullptr : SDL_GetError());
	}
}

uint64_t QueueUsbAudio(Stream* stream, uint32_t frames) {
	auto      queued = std::max(0, SDL_GetAudioStreamQueued(stream->sdl));
	const int bytes  = static_cast<int>(frames * FRAME_BYTES);
	const int max_bytes =
	    static_cast<int>(static_cast<uint64_t>(stream->freq) * FRAME_BYTES * MAX_QUEUE_MS / 1000);
	if (queued + bytes > max_bytes) {
		// Keep the newest sound after a brief device stall. Clearing the entire
		// stream produces a gap that is much longer than one audio block.
		SDL_AudioSpec              target {};
		std::array<uint8_t, 16384> discarded {};
		if (SDL_GetAudioStreamFormat(stream->sdl, nullptr, &target)) {
			const int frame_bytes = SDL_AUDIO_FRAMESIZE(target);
			if (frame_bytes > 0) {
				const int chunk_frames =
				    std::min<int>(static_cast<int>(discarded.size()) / frame_bytes,
				                  static_cast<int>((static_cast<uint64_t>(frames) * target.freq +
				                                    stream->freq - 1) /
				                                   stream->freq));
				while (queued + bytes > max_bytes && chunk_frames > 0 &&
				       SDL_GetAudioStreamData(stream->sdl, discarded.data(),
				                              chunk_frames * frame_bytes) > 0) {
					queued = std::max(0, SDL_GetAudioStreamQueued(stream->sdl));
				}
			}
		}
		if (queued + bytes > max_bytes) {
			SDL_ClearAudioStream(stream->sdl);
			queued = 0;
		}
	}
	if (!SDL_PutAudioStreamData(stream->sdl, stream->buffer.data(), bytes)) {
		return 0;
	}
	return static_cast<uint64_t>(queued + bytes) * 1000000 / (stream->freq * FRAME_BYTES);
}

// DualSense haptics are an audio stream for the two actuators. Other pads get, per 10 ms window,
// the loudness and the main frequency (from zero crossings) of each channel: the original Steam
// Controller plays both on its trackpad actuators, other pads rumble with the loudness only.
constexpr uint32_t HOST_HAPTICS_WINDOW_MS  = 10;
constexpr uint32_t HOST_HAPTICS_HOLD_MS    = 100; // stops by itself if the stream stalls
constexpr float    HOST_HAPTICS_HYSTERESIS = 1.0f / 128;
constexpr float    HOST_HAPTICS_MIN_FREQ   = 30.0f;
constexpr float    HOST_HAPTICS_MAX_FREQ   = 1000.0f;
constexpr int      HOST_HAPTICS_RUMBLE     = 1;
constexpr int      HOST_HAPTICS_STEAM      = 2;

// Must match SteamHapticEffect in the patched SDL (src/joystick/hidapi/SDL_hidapi_steam.c).
constexpr uint32_t STEAM_HAPTIC_EFFECT_MAGIC = 0x50484353;
struct SteamHapticEffect {
	uint32_t magic;
	uint16_t amplitude[2];
	uint16_t frequency[2];
	uint16_t duration_ms;
};

bool IsHostRumblePad(int controller) {
	SDL_LockJoysticks();
	const bool other_pad =
	    controller >= 0 && SDL_GetGamepadFromID(static_cast<SDL_JoystickID>(controller)) != nullptr &&
	    SDL_GetGamepadTypeForID(static_cast<SDL_JoystickID>(controller)) != SDL_GAMEPAD_TYPE_PS5;
	SDL_UnlockJoysticks();
	return other_pad;
}

int GetHostHapticsMode(int controller) {
	const auto id      = static_cast<SDL_JoystickID>(controller);
	const auto vendor  = SDL_GetGamepadVendorForID(id);
	const auto product = SDL_GetGamepadProductForID(id);
	// Original Steam Controller: wired, wireless dongle, Bluetooth
	const bool steam = vendor == 0x28de && (product == 0x1102 || product == 0x1142 || product == 0x1106);
	return steam ? HOST_HAPTICS_STEAM : HOST_HAPTICS_RUMBLE;
}

void SendHostHaptics(Stream* stream, std::array<uint16_t, 2> amplitude, std::array<uint16_t, 2> frequency) {
	const bool on = amplitude[0] != 0 || amplitude[1] != 0;
	SDL_LockJoysticks();
	if (auto* pad = SDL_GetGamepadFromID(static_cast<SDL_JoystickID>(stream->rumble_pad)); pad != nullptr) {
		if (stream->rumble_mode == HOST_HAPTICS_STEAM) {
			const SteamHapticEffect effect {STEAM_HAPTIC_EFFECT_MAGIC,
			                                {amplitude[0], amplitude[1]},
			                                {frequency[0], frequency[1]},
			                                static_cast<uint16_t>(on ? HOST_HAPTICS_HOLD_MS : 0)};
			if (!SDL_SendGamepadEffect(pad, &effect, sizeof(effect))) {
				// An SDL without the Steam Controller haptics patch: fall back to rumble.
				LOGF("Haptics: Steam Controller effect failed (%s), using rumble\n", SDL_GetError());
				stream->rumble_mode = HOST_HAPTICS_RUMBLE;
			}
		}
		if (stream->rumble_mode == HOST_HAPTICS_RUMBLE) {
			// Rumble motors need more push for quiet textures to be felt
			const auto motor = [](uint16_t a) {
				return static_cast<uint16_t>(std::lround(std::sqrt(a / 65535.0) * 65535.0));
			};
			(void)SDL_RumbleGamepad(pad, motor(amplitude[0]), motor(amplitude[1]),
			                        on ? HOST_HAPTICS_HOLD_MS : 0);
		}
	}
	SDL_UnlockJoysticks();
	stream->rumble_last = amplitude;
}

void ResetHostHaptics(Stream* stream) {
	stream->rumble_frames    = 0;
	stream->rumble_sum       = {};
	stream->rumble_crossings = {};
}

void StopHostRumble(Stream* stream) {
	if (stream->rumble_pad >= 0 && (stream->rumble_last[0] != 0 || stream->rumble_last[1] != 0)) {
		SendHostHaptics(stream, {0, 0}, {0, 0});
	}
	stream->rumble_pad  = -1;
	stream->rumble_mode = 0;
	stream->rumble_sign = {};
	stream->rumble_freq = {};
	ResetHostHaptics(stream);
}

void FlushHostHaptics(Stream* stream) {
	const float             seconds = static_cast<float>(stream->rumble_frames) / static_cast<float>(stream->freq);
	std::array<uint16_t, 2> amplitude {};
	std::array<uint16_t, 2> frequency {};
	for (uint32_t ch = 0; ch < 2; ch++) {
		const double rms = std::sqrt(stream->rumble_sum[ch] / stream->rumble_frames);
		// Peak level of a sine with that RMS
		const double level = std::clamp(rms * std::sqrt(2.0), 0.0, 1.0);
		amplitude[ch]      = level < 1.0 / 256 ? 0 : static_cast<uint16_t>(std::lround(level * 65535.0));
		if (amplitude[ch] == 0) {
			stream->rumble_freq[ch] = 0.0f;
			continue;
		}
		// A 10 ms window sees few crossings of a low tone; average over a few windows.
		const float measured = std::clamp(static_cast<float>(stream->rumble_crossings[ch]) / (2.0f * seconds),
		                                  HOST_HAPTICS_MIN_FREQ, HOST_HAPTICS_MAX_FREQ);
		stream->rumble_freq[ch] =
		    stream->rumble_freq[ch] == 0.0f ? measured : stream->rumble_freq[ch] * 0.5f + measured * 0.5f;
		frequency[ch] = static_cast<uint16_t>(std::lround(stream->rumble_freq[ch]));
	}
	ResetHostHaptics(stream);
	if (amplitude[0] == 0 && amplitude[1] == 0 && stream->rumble_last[0] == 0 && stream->rumble_last[1] == 0) {
		return;
	}
	SendHostHaptics(stream, amplitude, frequency);
}

void QueueHostRumble(Stream* stream, int controller, const void* data, uint32_t first_frame,
                     uint32_t frames, uint32_t channels, bool is_float, const int* volume, float gain) {
	if (stream->rumble_pad != controller) {
		StopHostRumble(stream);
		stream->rumble_pad  = controller;
		stream->rumble_mode = GetHostHapticsMode(controller);
	}
	const uint32_t window = std::max<uint32_t>(1, stream->freq * HOST_HAPTICS_WINDOW_MS / 1000);
	for (uint32_t frame = 0; frame < frames; frame++) {
		for (uint32_t ch = 0; ch < 2; ch++) {
			const auto src_ch = channels == 1 ? 0 : ch;
			const auto index  = (static_cast<size_t>(first_frame) + frame) * channels + src_ch;
			float      value  = is_float ? static_cast<const float*>(data)[index]
			                             : static_cast<const int16_t*>(data)[index] / 32768.0f;
			value *= volume[src_ch] / 32768.0f * gain;
			if (!std::isfinite(value)) {
				continue;
			}
			stream->rumble_sum[ch] += static_cast<double>(value) * value;
			const int sign = value > HOST_HAPTICS_HYSTERESIS ? 1 : (value < -HOST_HAPTICS_HYSTERESIS ? -1 : 0);
			if (sign != 0) {
				if (stream->rumble_sign[ch] != 0 && sign != stream->rumble_sign[ch]) {
					stream->rumble_crossings[ch]++;
				}
				stream->rumble_sign[ch] = sign;
			}
		}
		if (++stream->rumble_frames >= window) {
			FlushHostHaptics(stream);
		}
	}
}

} // namespace

Stream* Open(uint32_t freq, bool speaker) {
	if (freq == 0 || !SDL_InitSubSystem(SDL_INIT_AUDIO)) {
		return nullptr;
	}
	return new Stream {.freq = freq, .speaker = speaker};
}

void Close(Stream* stream) {
	if (stream != nullptr) {
		StopHostRumble(stream);
		CloseDevice(stream);
		delete stream;
		SDL_QuitSubSystem(SDL_INIT_AUDIO);
	}
}

bool UsesBluetooth(const Stream* stream) {
	return stream != nullptr && stream->bluetooth != nullptr;
}

uint64_t Queue(Stream* stream, int controller, const void* data, uint32_t frames, uint32_t channels,
               bool is_float, const int* volume, float gain) {
	if (stream == nullptr || data == nullptr || frames == 0 || channels == 0 || volume == nullptr) {
		return 0;
	}
	const auto max_frames =
	    static_cast<uint32_t>(static_cast<uint64_t>(stream->freq) * MAX_QUEUE_MS / 1000);
	const auto first_frame = frames > max_frames ? frames - max_frames : 0;
	frames -= first_frame;
	if (frames == 0) {
		return 0;
	}
	if (!stream->speaker && IsHostRumblePad(controller)) {
		// No DualSense to play on: the port keeps pacing itself as it does without a device.
		QueueHostRumble(stream, controller, data, first_frame, frames, channels, is_float, volume, gain);
		return 0;
	}
	if (stream->rumble_pad >= 0) {
		StopHostRumble(stream);
	}
	const bool wireless = IsWireless(controller);
	if (!wireless && !CanUseDevice(controller)) {
		CloseDevice(stream);
		stream->next_check = 0; // Look for the device as soon as a DualSense is active again.
		ReportRoute(stream, Route::NoWiredPad, nullptr);
		return 0;
	}
	RefreshDevice(stream, controller, wireless);
	if (stream->sdl == nullptr && stream->bluetooth == nullptr) {
		return 0;
	}
	if (stream->speaker) {
		bool routed;
		{
			Common::LockGuard lock(g_mutex);
			routed = SelectSpeaker(controller);
		}
		if (!routed) {
			CloseDevice(stream);
			return 0;
		}
	}
	const uint32_t output_channels = wireless ? 2 : 4;
	const uint32_t channel_offset  = wireless || stream->speaker ? 0 : 2;
	stream->buffer.assign(static_cast<size_t>(frames) * output_channels, 0.0f);
	bool audible = false;
	for (uint32_t frame = 0; frame < frames; frame++) {
		for (uint32_t ch = 0; ch < 2; ch++) {
			const auto src_ch = channels == 1 ? 0 : ch;
			const auto index  = (static_cast<size_t>(first_frame) + frame) * channels + src_ch;
			float      value  = is_float ? static_cast<const float*>(data)[index]
			                             : static_cast<const int16_t*>(data)[index] / 32768.0f;
			value *= volume[src_ch] / 32768.0f * gain;
			// USB places the speaker in front and the actuators in back; Bluetooth takes stereo.
			stream->buffer[static_cast<size_t>(frame) * output_channels + ch + channel_offset] =
			    value;
			audible |= std::isfinite(value) && std::fabs(value) > 1.0f / 1024;
		}
	}
	const auto queued_us =
	    wireless ? DualSenseBluetooth::Queue(stream->bluetooth, stream->buffer.data(), frames)
	             : QueueUsbAudio(stream, frames);
	if (queued_us == 0) {
		CloseDevice(stream);
		stream->next_check = SDL_GetTicks() + 2000;
		return 0;
	}
	if (audible && !stream->speaker) {
		Common::LockGuard lock(g_mutex);
		SelectController(controller);
		const auto now      = SDL_GetTicks();
		const bool starting = g_haptics_until <= now;
		g_haptics_until =
		    std::max(g_haptics_until, now + std::max<uint64_t>(queued_us / 1000, 250));
		if (starting) {
			ApplyRumble();
		}
	}
	return queued_us;
}

bool SetVibration(int controller, uint8_t large_motor, uint8_t small_motor, uint32_t duration_ms) {
	if (SDL_GetGamepadTypeForID(static_cast<SDL_JoystickID>(controller)) != SDL_GAMEPAD_TYPE_PS5) {
		return false;
	}
	Common::LockGuard lock(g_mutex);
	SelectController(controller);
	g_large_motor  = large_motor;
	g_small_motor  = small_motor;
	g_rumble_until = SDL_GetTicks() + duration_ms;
	ApplyRumble();
	return true;
}

void Shutdown() {
	DualSenseBluetooth::Shutdown();
	Common::LockGuard lock(g_mutex);
	SelectController(-1);
	SelectSpeaker(-1);
}

} // namespace Libs::Controller::DualSenseHaptics
