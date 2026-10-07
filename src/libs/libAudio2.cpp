#include "common/assert.h"
#include "common/common.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "kernel/pthread.h"
#include "libs/audio.h"
#include "libs/audioDiag.h"
#include "libs/audioMix.h"
#include "libs/audioObjects.h"
#include "libs/audio_internal.h"
#include "libs/errno.h"
#include "libs/libs.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace Libs::Audio {

namespace {

constexpr int AUDIO_OUT_PORT_TYPE_MAIN      = 0;
constexpr int AUDIO_OUT_PORT_TYPE_BGM       = 1;
constexpr int AUDIO_OUT_PORT_TYPE_VOICE     = 2;
constexpr int AUDIO_OUT_PORT_TYPE_PERSONAL  = 3;
constexpr int AUDIO_OUT_PORT_TYPE_PADSPK    = 4;
constexpr int AUDIO_OUT_PORT_TYPE_VIBRATION = 10;
constexpr int AUDIO_OUT_PORT_TYPE_AUX       = 127;

} // namespace

namespace AudioOut2 {

LIB_NAME("AudioOut2", "AudioOut");

struct AudioOut2ContextParam {
	uint32_t max_ports;
	uint32_t max_object_ports;
	uint32_t guarantee_object_ports;
	uint32_t queue_depth;
	uint32_t num_grains;
	uint32_t flags;
	uint32_t reserved[10];
};

struct AudioOut2PortParam {
	uint16_t            port_type;
	uint16_t            pad;
	uint32_t            data_format;
	uint32_t            sampling_freq;
	uint32_t            flags;
	AudioOut2UserHandle user_handle;
	uint32_t            reserved[10];
};

struct AudioOut2Attribute {
	uint32_t    attribute_id;
	int32_t     reserved;
	const void* value;
	size_t      value_size;
};

struct AudioOut2Pcm {
	const void* data;
};

struct AudioOut2Position {
	float x;
	float y;
	float z;
};

struct AudioOut2PortState {
	uint16_t output;
	uint8_t  num_channels;
	uint8_t  pad1;
	int16_t  volume;
	uint16_t reroute_counter;
	uint32_t flags;
	uint32_t pad2;
	uint64_t reserved[6];
};

struct AudioOut2SystemState {
	float    loudness;
	uint32_t pad;
	uint64_t reserved[7];
};

struct AudioOut2SpeakerAngle {
	int16_t azimuth;
	int16_t elevation;
};

struct AudioOut2SpeakerInfo {
	uint8_t               type;
	uint8_t               pad1;
	int16_t               pad2;
	uint32_t              available_bits;
	uint32_t              flags;
	uint32_t              pad3;
	AudioOut2SpeakerAngle speaker_angle[16];
};

struct AudioOut2SystemDebugStateParam {
	uint32_t debug_state_id;
	int32_t  reserved;
	void*    param;
	size_t   param_size;
};

struct AudioOut2MasteringParamsHeader {
	uint32_t params_id;
};

struct AudioOut2MasteringStatesHeader {
	uint32_t states_id;
};

struct AudioOut2MasteringStatesDescriptor {
	uint32_t id;
	uint32_t size;
};

static constexpr uint32_t AUDIO_OUT2_MASTERING_MAX_SYSTEM_CHANNELS = 8;
static constexpr uint32_t AUDIO_OUT2_MASTERING_COMPRESSOR_BANDS    = 3;
static constexpr uint32_t AUDIO_OUT2_MASTERING_COMPRESSOR_BANDS_V2 = 4;

struct AudioOut2MasteringCompressorStates {
	AudioOut2MasteringStatesDescriptor descriptor;
	uint32_t                           reserved[2];
	float                              input_rms[AUDIO_OUT2_MASTERING_COMPRESSOR_BANDS]
	                                            [AUDIO_OUT2_MASTERING_MAX_SYSTEM_CHANNELS];
	float                              compression_coeff[AUDIO_OUT2_MASTERING_COMPRESSOR_BANDS]
	                                                    [AUDIO_OUT2_MASTERING_MAX_SYSTEM_CHANNELS];
};

struct AudioOut2MasteringCompressorStatesV2 {
	AudioOut2MasteringStatesDescriptor descriptor;
	uint32_t                           reserved[2];
	float                              input_rms[AUDIO_OUT2_MASTERING_COMPRESSOR_BANDS_V2]
	                                            [AUDIO_OUT2_MASTERING_MAX_SYSTEM_CHANNELS];
	float                              compression_coeff[AUDIO_OUT2_MASTERING_COMPRESSOR_BANDS_V2]
	                                                    [AUDIO_OUT2_MASTERING_MAX_SYSTEM_CHANNELS];
};

struct AudioOut2MasteringLimiterStates {
	AudioOut2MasteringStatesDescriptor descriptor;
	uint32_t                           reserved[2];
	float                              input_peak[AUDIO_OUT2_MASTERING_MAX_SYSTEM_CHANNELS];
	float                              output_peak[AUDIO_OUT2_MASTERING_MAX_SYSTEM_CHANNELS];
	float                              gain_peak[AUDIO_OUT2_MASTERING_MAX_SYSTEM_CHANNELS];
};

struct AudioOut2MasteringStates {
	AudioOut2MasteringStatesHeader     states_header;
	uint32_t                           reserved[3];
	AudioOut2MasteringCompressorStates compressor_states;
	AudioOut2MasteringLimiterStates    limiter_states;
};

struct AudioOut2MasteringStatesV2 {
	AudioOut2MasteringStatesHeader       states_header;
	uint32_t                             reserved[3];
	AudioOut2MasteringCompressorStatesV2 compressor_states;
	AudioOut2MasteringLimiterStates      limiter_states;
};

static std::atomic_uint64_t g_audioout2_next_context {1};
static std::atomic_uint64_t g_audioout2_next_port {1};
static AudioOut2UserHandle g_audioout2_next_user = 1;

struct AudioOut2ContextState {
	bool                   used        = false;
	AudioOut2ContextHandle handle      = 0;
	uint32_t               queue_depth = 4;
	uint32_t               queued      = 0;
	uint32_t               num_grains  = 512;
	uint64_t               last_update = 0;
};

struct AudioOut2PortStateEntry {
	bool                   used          = false;
	AudioOut2PortHandle    handle        = 0;
	AudioOut2ContextHandle context       = 0;
	AudioOut2UserHandle    user          = 0;
	uint16_t               port_type     = 0;
	uint32_t               data_format   = 0;
	uint32_t               sampling_freq = 48000;
	uint32_t               samples_num   = 512;
	AudioInternal::Format  audio_format  = AudioInternal::Format::Unknown;
	int                    audio_handle  = 0;
	std::vector<uint8_t>   pcm_data;
	Mix::LevelMeter        meter;     // KYTY_AUDIO_LEVELS: PCM the guest set since the last report.
	uint32_t               pcm_sets = 0;
	// Object ports: placement attributes, and whether PCM was set since the last push (an object
	// that is not fed again is silent, not a repeated block).
	Objects::Params        object;
	bool                   pcm_fresh = false;
};

struct AudioOut2SpeakerArrayState {
	bool                        used          = false;
	uint32_t                    num_speakers  = 0;
	uint8_t                     is_3d         = 0;
	uint8_t                     is_ambisonics = 0;
	AudioOut2SpeakerArrayHandle handle        = nullptr;
};

struct AudioOut2LatencyState {
	bool     used       = false;
	uint32_t user_id    = 0;
	uint32_t output     = 0;
	uint32_t latency_us = 0;
};

static Common::Mutex                              g_audioout2_context_mutex;
static std::array<AudioOut2ContextState, 16>      g_audioout2_contexts;
static Common::Mutex                              g_audioout2_port_mutex;
static std::array<AudioOut2PortStateEntry, 256>   g_audioout2_ports;
static std::vector<AudioOut2UserHandle>          g_audioout2_users;
static Common::Mutex                              g_audioout2_speaker_array_mutex;
static std::array<AudioOut2SpeakerArrayState, 32> g_audioout2_speaker_arrays;
static Common::Mutex                              g_audioout2_latency_mutex;
static std::array<AudioOut2LatencyState, 16>      g_audioout2_latencies;

static constexpr int AUDIO_OUT2_ERROR_NOT_READY                   = -2144960504; /* 0x80268008 */
static constexpr int AUDIO_OUT2_ERROR_PORT_FULL                   = -2144960494; /* 0x80268012 */
static constexpr int AUDIO_OUT2_ERROR_INVALID_PARAM               = -2144960511; /* 0x80268001 */
static constexpr int AUDIO_OUT2_ERROR_BUSY                        = -2144960505; /* 0x80268007 */
static constexpr int AUDIO_OUT2_ERROR_MASTERING_INVALID_API_PARAM = -2144959999; /* 0x80268201 */
static constexpr int AUDIO_OUT2_ERROR_MASTERING_INVALID_STATES_ID = -2144959996; /* 0x80268204 */
static constexpr uint32_t AUDIO_OUT2_PORT_ATTRIBUTE_ID_PCM        = 0;
static constexpr uint32_t AUDIO_OUT2_MASTERING_OUTPUT_RECORDING   = 2;
static constexpr uint32_t AUDIO_OUT2_MASTERING_STATES_ID_DEFAULT  = 1;
static constexpr uint32_t AUDIO_OUT2_MASTERING_STATES_ID_V2       = 2;
static constexpr uint32_t AUDIO_OUT2_MASTERING_STATES_STRUCT_ID_COMPRESSOR_DEFAULT = 0x01010001;
static constexpr uint32_t AUDIO_OUT2_MASTERING_STATES_STRUCT_ID_COMPRESSOR_V2      = 0x01020001;
static constexpr uint32_t AUDIO_OUT2_MASTERING_STATES_STRUCT_ID_LIMITER_DEFAULT    = 0x01010003;

static AudioOut2ContextState* audioout2_find_context_locked(AudioOut2ContextHandle ctx) {
	for (auto& state: g_audioout2_contexts) {
		if (state.used && state.handle == ctx) {
			return &state;
		}
	}

	return nullptr;
}

static uint32_t audioout2_grain_micros(uint32_t grains) {
	const auto sample_count = (grains == 0 ? 512u : grains);
	return std::max<uint32_t>((sample_count * 1000000u) / 48000u, 1000u);
}

static uint8_t audioout2_data_format_channels(uint32_t data_format) {
	const auto channels = (data_format >> 8u) & 0xffu;
	return static_cast<uint8_t>(channels == 0 ? 2u : std::min(channels, 16u));
}

static AudioInternal::Format audioout2_data_format_to_audio_format(uint32_t data_format) {
	const auto channels  = audioout2_data_format_channels(data_format);
	const auto data_type = data_format & 0x7fu;
	const auto is_std    = (data_format & 0x80u) != 0;

	switch (data_type) {
		case 0:
			switch (channels) {
				case 1: return AudioInternal::Format::FloatMono;
				case 2: return AudioInternal::Format::FloatStereo;
				case 8:
					return is_std ? AudioInternal::Format::Float8ChStd
					              : AudioInternal::Format::Float8Ch;
				case 12:
					if (!is_std) {
						return AudioInternal::Format::Float12Ch;
					}
					break;
				default: break;
			}
			break;
		case 1:
			switch (channels) {
				case 1: return AudioInternal::Format::Signed16bitMono;
				case 2: return AudioInternal::Format::Signed16bitStereo;
				case 8:
					return is_std ? AudioInternal::Format::Signed16bit8ChStd
					              : AudioInternal::Format::Signed16bit8Ch;
				default: break;
			}
			break;
		default: break;
	}

	return AudioInternal::Format::Unknown;
}

static int audioout2_port_type_to_audio_out_type(uint16_t port_type) {
	switch (port_type & 0xffu) {
		case 0: return AUDIO_OUT_PORT_TYPE_MAIN;
		case 1: return AUDIO_OUT_PORT_TYPE_BGM;
		case 2: return AUDIO_OUT_PORT_TYPE_VOICE;
		case 3: return AUDIO_OUT_PORT_TYPE_PADSPK;
		case 4: return AUDIO_OUT_PORT_TYPE_PERSONAL;
		case 5: return AUDIO_OUT_PORT_TYPE_AUX;
		case 6: return AUDIO_OUT_PORT_TYPE_VIBRATION;
		default: return AUDIO_OUT_PORT_TYPE_MAIN;
	}
}

static bool audioout2_port_type_is_object(uint16_t port_type) {
	return (port_type & 0xff00u) == 0x0100u;
}

static void audioout2_update_context_locked(AudioOut2ContextState* state) {
	if (state == nullptr) {
		return;
	}

	const auto now = LibKernel::KernelGetProcessTime();
	if (state->last_update == 0 || state->queued == 0) {
		state->last_update = now;
		return;
	}

	const auto grain_micros = static_cast<uint64_t>(audioout2_grain_micros(state->num_grains));
	if (now <= state->last_update || grain_micros == 0) {
		return;
	}

	const auto elapsed = now - state->last_update;
	const auto drained = std::min<uint64_t>(state->queued, elapsed / grain_micros);
	if (drained == 0) {
		return;
	}

	state->queued -= static_cast<uint32_t>(drained);
	state->last_update += drained * grain_micros;
	if (state->queued == 0) {
		state->last_update = now;
	}
}

static AudioOut2PortStateEntry* audioout2_find_port_locked(AudioOut2PortHandle port) {
	for (auto& state: g_audioout2_ports) {
		if (state.used && state.handle == port) {
			return &state;
		}
	}

	return nullptr;
}

static size_t audioout2_pcm_size(const AudioOut2PortStateEntry& state) {
	const auto bytes_per_sample = (state.data_format & 0x7fu) == 1 ? sizeof(int16_t) : sizeof(float);
	return static_cast<size_t>(state.samples_num) *
	       audioout2_data_format_channels(state.data_format) * bytes_per_sample;
}

static bool audioout2_context_has_queueable_device(AudioOut2ContextHandle ctx) {
	Common::LockGuard lock(g_audioout2_port_mutex);
	for (const auto& state: g_audioout2_ports) {
		if (state.used && state.context == ctx && state.audio_handle > 0 &&
		    !state.pcm_data.empty() && AudioInternal::AudioOutHasDevice(state.audio_handle)) {
			return true;
		}
	}
	return false;
}

static bool audioout2_level_logging();

// KYTY_AUDIO_LEVELS: what the object mix added to the bed's front pair, and the gains of the plain
// (positioned) objects that were mixed, since the last report. Guarded by g_audioout2_port_mutex.
struct AudioOut2ObjectMixStats {
	Mix::LevelMeter added;
	uint32_t        plain_objects  = 0;
	double          plain_gain     = 0.0;
	float           plain_gain_max = 0.0f;
};
static AudioOut2ObjectMixStats g_audioout2_object_mix;

static bool audioout2_is_object_bed(const AudioOut2PortStateEntry& state) {
	if ((state.port_type & 0xffu) != 0 || audioout2_port_type_is_object(state.port_type) ||
	    state.audio_handle <= 0 || state.pcm_data.empty()) {
		return false;
	}
	switch (state.audio_format) {
		case AudioInternal::Format::FloatStereo:
		case AudioInternal::Format::Float8Ch:
		case AudioInternal::Format::Float8ChStd:
		case AudioInternal::Format::Float12Ch: return true;
		default: return false;
	}
}

// Object ports have no output of their own: their fresh PCM is panned into the context's first
// float main port (the bed), at the objects mix level, before the bed is played. Returns the mixed
// bed, or the bed's own PCM when no object added anything. Caller holds g_audioout2_port_mutex.
static const void* audioout2_mix_objects_locked(AudioOut2ContextHandle         ctx,
                                                const AudioOut2PortStateEntry& bed,
                                                std::vector<float>*            mix) {
	const float objects_gain = AudioInternal::AudioOutObjectsGain();
	const auto  channels     = audioout2_data_format_channels(bed.data_format);
	const auto  frames       = bed.samples_num;
	bool        mixed        = false;
	for (auto& object: g_audioout2_ports) {
		if (!object.used || object.context != ctx || !audioout2_port_type_is_object(object.port_type)) {
			continue;
		}
		const bool fresh = object.pcm_fresh;
		object.pcm_fresh = false;
		if (!fresh || objects_gain <= 0.0f || object.pcm_data.empty() ||
		    object.audio_format != AudioInternal::Format::FloatMono || object.samples_num != frames) {
			continue;
		}
		float gains[Objects::MAX_CHANNELS] {};
		if (!Objects::SpeakerGains(object.object, channels, gains)) {
			continue;
		}
		if (!mixed) {
			const auto* pcm = reinterpret_cast<const float*>(bed.pcm_data.data());
			mix->assign(pcm, pcm + static_cast<size_t>(frames) * channels);
			mixed = true;
		}
		Objects::MixInto(mix->data(), channels, frames,
		                 reinterpret_cast<const float*>(object.pcm_data.data()), gains, objects_gain);
		if (audioout2_level_logging() && !Objects::IsAmbisonicsChannel(object.object.ambisonics) &&
		    object.object.passthrough == Objects::PASSTHROUGH_NONE) {
			auto& stats = g_audioout2_object_mix;
			stats.plain_objects++;
			stats.plain_gain += object.object.gain;
			stats.plain_gain_max = std::max(stats.plain_gain_max, object.object.gain);
		}
	}
	if (mixed && audioout2_level_logging()) {
		const auto* pcm = reinterpret_cast<const float*>(bed.pcm_data.data());
		float       added[2] {};
		for (uint32_t frame = 0; frame < frames; frame++) {
			const auto index = static_cast<size_t>(frame) * channels;
			added[0]         = (*mix)[index] - pcm[index];
			added[1]         = (*mix)[index + 1] - pcm[index + 1];
			g_audioout2_object_mix.added.Add(added, 1, 2, true);
		}
	}
	return mixed ? static_cast<const void*>(mix->data()) : bed.pcm_data.data();
}

static void audioout2_queue_context_audio(AudioOut2ContextHandle ctx, bool blocking) {
	// The backend consumes the buffers synchronously, but can block while pacing SDL's queue.
	// Keep both PCM storage and port handles alive until it returns.
	std::vector<AudioInternal::OutputParam> params;
	params.reserve(AudioInternal::OUT_PORTS_MAX);

	thread_local std::vector<float> object_mix;

	Common::LockGuard lock(g_audioout2_port_mutex);
	bool objects_mixed = false;
	for (const auto& state: g_audioout2_ports) {
		if (state.used && state.context == ctx && state.audio_handle > 0 &&
		    !state.pcm_data.empty() && params.size() < AudioInternal::OUT_PORTS_MAX) {
			const void* data = state.pcm_data.data();
			if (!objects_mixed && audioout2_is_object_bed(state)) {
				data          = audioout2_mix_objects_locked(ctx, state, &object_mix);
				objects_mixed = true;
			}
			params.push_back(AudioInternal::OutputParam {state.audio_handle, data});
		}
	}
	if (!objects_mixed) {
		// No bed to carry them: drop this push's object PCM rather than replay it later.
		for (auto& state: g_audioout2_ports) {
			if (state.used && state.context == ctx) {
				state.pcm_fresh = false;
			}
		}
	}

	if (!params.empty()) {
		(void)AudioInternal::AudioOutOutputs(params.data(), static_cast<uint32_t>(params.size()),
		                                     blocking);
	}
}

static void audioout2_close_audio_handle(int audio_handle) {
	if (audio_handle > 0) {
		AudioInternal::AudioOutClose(audio_handle);
	}
}

int KYTY_SYSV_ABI AudioOut2Initialize() {
	PRINT_NAME();

	return OK;
}

int KYTY_SYSV_ABI AudioOut2ContextResetParam(AudioOut2ContextParam* params) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(params == nullptr);

	std::memset(params, 0, sizeof(AudioOut2ContextParam));
	params->max_ports              = 256;
	params->max_object_ports       = 256;
	params->guarantee_object_ports = 0;
	params->queue_depth            = 4;
	params->num_grains             = 512;
	params->flags                  = 1;

	return OK;
}

int KYTY_SYSV_ABI AudioOut2ContextQueryMemory(const AudioOut2ContextParam* params,
                                              size_t*                      memory_size) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(params == nullptr);
	EXIT_NOT_IMPLEMENTED(memory_size == nullptr);

	const auto queue_depth = (params->queue_depth == 0 ? 4u : params->queue_depth);
	*memory_size           = 0x10000u + static_cast<size_t>(queue_depth) * 0x590u;

	LOGF("\t memory_size = 0x%016" PRIx64 "\n", static_cast<uint64_t>(*memory_size));

	return OK;
}

int KYTY_SYSV_ABI AudioOut2ContextCreate(const AudioOut2ContextParam* params, void* buffer,
                                         size_t buffer_size, AudioOut2ContextHandle* ctx) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(params == nullptr);
	EXIT_NOT_IMPLEMENTED(ctx == nullptr);

	*ctx = g_audioout2_next_context.fetch_add(1, std::memory_order_relaxed);

	g_audioout2_context_mutex.Lock();
	auto* state = audioout2_find_context_locked(0);
	if (state == nullptr) {
		for (auto& candidate: g_audioout2_contexts) {
			if (!candidate.used) {
				state = &candidate;
				break;
			}
		}
	}
	EXIT_NOT_IMPLEMENTED(state == nullptr);
	*state             = AudioOut2ContextState {};
	state->used        = true;
	state->handle      = *ctx;
	state->queue_depth = (params->queue_depth == 0 ? 4u : params->queue_depth);
	state->queued      = 0;
	state->num_grains  = (params->num_grains == 0 ? 512u : params->num_grains);
	state->last_update = LibKernel::KernelGetProcessTime();
	g_audioout2_context_mutex.Unlock();

	LOGF("\t buffer      = 0x%016" PRIx64 "\n"
	     "\t buffer_size = 0x%016" PRIx64 "\n"
	     "\t ctx         = 0x%016" PRIx64 "\n"
	     "\t queue_depth = %" PRIu32 ", num_grains = %" PRIu32 "\n",
	     reinterpret_cast<uint64_t>(buffer), static_cast<uint64_t>(buffer_size), *ctx,
	     state->queue_depth, state->num_grains);
	Diag::Print("sceAudioOut2ContextCreate -> context %" PRIu64 " (max ports %" PRIu32
	            ", queue depth %" PRIu32 ", %" PRIu32 " grains, flags 0x%x, buffer 0x%" PRIx64
	            " size 0x%" PRIx64 ")",
	            *ctx, params->max_ports, params->queue_depth, params->num_grains, params->flags,
	            reinterpret_cast<uint64_t>(buffer), static_cast<uint64_t>(buffer_size));

	return OK;
}

int KYTY_SYSV_ABI AudioOut2ContextDestroy(AudioOut2ContextHandle ctx) {
	PRINT_NAME();
	LOGF("\t ctx = 0x%016" PRIx64 "\n", ctx);
	Diag::Print("sceAudioOut2ContextDestroy(context %" PRIu64 ")", ctx);

	g_audioout2_context_mutex.Lock();
	if (auto* state = audioout2_find_context_locked(ctx); state != nullptr) {
		*state = AudioOut2ContextState {};
	}
	g_audioout2_context_mutex.Unlock();

	std::array<int, 256> audio_handles {};
	size_t               audio_handles_num = 0;

	g_audioout2_port_mutex.Lock();
	for (auto& port_state: g_audioout2_ports) {
		if (port_state.used && port_state.context == ctx) {
			if (port_state.audio_handle > 0 && audio_handles_num < audio_handles.size()) {
				audio_handles[audio_handles_num++] = port_state.audio_handle;
			}
			port_state = AudioOut2PortStateEntry {};
		}
	}
	g_audioout2_port_mutex.Unlock();

	for (size_t i = 0; i < audio_handles_num; i++) {
		audioout2_close_audio_handle(audio_handles[i]);
	}

	return OK;
}

int KYTY_SYSV_ABI AudioOut2ContextSetAttributes(AudioOut2ContextHandle    ctx,
                                                const AudioOut2Attribute* attributes,
                                                uint32_t                  num) {
	PRINT_NAME();
	LOGF("\t ctx = 0x%016" PRIx64 ", num = %" PRIu32 "\n", ctx, num);
	EXIT_NOT_IMPLEMENTED(num != 0 && attributes == nullptr);
	return OK;
}

int KYTY_SYSV_ABI AudioOut2ContextAdvance(AudioOut2ContextHandle ctx) {
	g_audioout2_context_mutex.Lock();
	if (auto* state = audioout2_find_context_locked(ctx); state != nullptr) {
		audioout2_update_context_locked(state);
	}
	g_audioout2_context_mutex.Unlock();

	return OK;
}

int KYTY_SYSV_ABI AudioOut2ContextPush(AudioOut2ContextHandle ctx, uint32_t blocking) {
	uint32_t sleep_micros = audioout2_grain_micros(512);

	for (;;) {
		// Only a synchronous submission carrying PCM to a real device can rely on the SDL queue for
		// pacing. Async pushes must retain queue-depth backpressure, and a handle without PCM (or a
		// vibration/failed-open handle) has no downstream operation that can block this call.
		const bool use_device_clock =
		    blocking != 0 && audioout2_context_has_queueable_device(ctx);

		g_audioout2_context_mutex.Lock();
		if (auto* state = audioout2_find_context_locked(ctx); state != nullptr) {
			audioout2_update_context_locked(state);
			sleep_micros = audioout2_grain_micros(state->num_grains);
			if (state->queued < state->queue_depth || use_device_clock) {
				if (state->queued == 0) {
					state->last_update = LibKernel::KernelGetProcessTime();
				}
				if (state->queued < state->queue_depth) {
					state->queued++;
				}
				g_audioout2_context_mutex.Unlock();
				audioout2_queue_context_audio(ctx, blocking != 0);
				return OK;
			}
		}
		g_audioout2_context_mutex.Unlock();

		if (blocking == 0) {
			return AUDIO_OUT2_ERROR_NOT_READY;
		}

		Common::Thread::SleepMicro(sleep_micros);
	}
}

int KYTY_SYSV_ABI AudioOut2ContextGetQueueLevel(AudioOut2ContextHandle ctx, uint32_t* queue_level,
                                                uint32_t* available_queues) {
	if (queue_level != nullptr) {
		*queue_level = 0;
	}
	if (available_queues != nullptr) {
		*available_queues = 4;
	}

	g_audioout2_context_mutex.Lock();
	if (auto* state = audioout2_find_context_locked(ctx); state != nullptr) {
		audioout2_update_context_locked(state);
		if (queue_level != nullptr) {
			*queue_level = state->queued;
		}
		if (available_queues != nullptr) {
			*available_queues =
			    (state->queued < state->queue_depth ? state->queue_depth - state->queued : 0);
		}
	}
	g_audioout2_context_mutex.Unlock();

	return OK;
}

int KYTY_SYSV_ABI AudioOut2PortCreate(AudioOut2ContextHandle ctx, const AudioOut2PortParam* params,
                                      AudioOut2PortHandle* port) {
	EXIT_NOT_IMPLEMENTED(params == nullptr);
	EXIT_NOT_IMPLEMENTED(port == nullptr);

	const auto next_port    = g_audioout2_next_port.fetch_add(1, std::memory_order_relaxed);
	const auto audio_format = audioout2_data_format_to_audio_format(params->data_format);
	const auto audio_type   = audioout2_port_type_to_audio_out_type(params->port_type);

	g_audioout2_context_mutex.Lock();
	const auto* context_state = audioout2_find_context_locked(ctx);
	if (context_state == nullptr) {
		g_audioout2_context_mutex.Unlock();
		Diag::Print("sceAudioOut2PortCreate: unknown context %" PRIu64 " (port type 0x%x)", ctx,
		            params->port_type);
		return Diag::Error("sceAudioOut2PortCreate", AUDIO_OUT2_ERROR_INVALID_PARAM, 1);
	}
	const auto samples_num = context_state->num_grains == 0 ? 512u : context_state->num_grains;

	g_audioout2_port_mutex.Lock();
	if (std::find(g_audioout2_users.begin(), g_audioout2_users.end(), params->user_handle) ==
	    g_audioout2_users.end()) {
		g_audioout2_port_mutex.Unlock();
		g_audioout2_context_mutex.Unlock();
		Diag::Print("sceAudioOut2PortCreate: unknown user handle %" PRIu64 " (port type 0x%x)",
		            static_cast<uint64_t>(params->user_handle), params->port_type);
		return Diag::Error("sceAudioOut2PortCreate", AUDIO_OUT2_ERROR_INVALID_PARAM, 2);
	}
	AudioOut2PortStateEntry* port_state = nullptr;
	for (auto& candidate: g_audioout2_ports) {
		if (!candidate.used) {
			port_state = &candidate;
			break;
		}
	}
	if (port_state != nullptr) {
		*port_state               = AudioOut2PortStateEntry {};
		port_state->used          = true;
		port_state->handle        = next_port;
		port_state->context       = ctx;
		port_state->user          = params->user_handle;
		port_state->port_type     = params->port_type;
		port_state->data_format   = params->data_format;
		port_state->sampling_freq = params->sampling_freq;
		port_state->samples_num   = samples_num;
		port_state->audio_format  = audio_format;
	}
	g_audioout2_port_mutex.Unlock();
	g_audioout2_context_mutex.Unlock();

	if (port_state == nullptr) {
		return Diag::Error("sceAudioOut2PortCreate", AUDIO_OUT2_ERROR_PORT_FULL, params->port_type);
	}

	int audio_handle = 0;

	if (audio_format != AudioInternal::Format::Unknown &&
	    !audioout2_port_type_is_object(params->port_type)) {
		audio_handle = AudioInternal::AudioOutOpen(audio_type, samples_num, params->sampling_freq,
		                                           audio_format);
	}

	g_audioout2_port_mutex.Lock();
	const bool reserved = port_state->used && port_state->handle == next_port;
	if (reserved) {
		port_state->audio_handle = audio_handle;
	}
	g_audioout2_port_mutex.Unlock();
	if (!reserved) {
		audioout2_close_audio_handle(audio_handle);
		return Diag::Error("sceAudioOut2PortCreate", AUDIO_OUT2_ERROR_INVALID_PARAM, 3);
	}

	*port = next_port;

	// Object ports can be created per sound; the first ones and every non-object port suffice.
	if (!audioout2_port_type_is_object(params->port_type) || next_port <= 16) {
		Diag::Print("sceAudioOut2PortCreate(context %" PRIu64 ", type 0x%x, format 0x%x, %" PRIu32
		            " Hz, flags 0x%x, user %" PRIu64 ") -> port %" PRIu64 ", %u frames, %u ch, "
		            "output handle %d%s",
		            ctx, params->port_type, params->data_format, params->sampling_freq,
		            params->flags, static_cast<uint64_t>(params->user_handle), next_port,
		            samples_num, audioout2_data_format_channels(params->data_format), audio_handle,
		            audio_format == AudioInternal::Format::Unknown ? " (format not played)" : "");
	}

	if (next_port <= 16 || (next_port % 600) == 0) {
		PRINT_NAME();
		LOGF("\t ctx           = 0x%016" PRIx64 "\n"
		     "\t port          = 0x%016" PRIx64 "\n"
		     "\t port_type     = %" PRIu16 "\n"
		     "\t data_format   = 0x%08" PRIx32 "\n"
		     "\t sampling_freq = %" PRIu32 "\n",
		     ctx, *port, params->port_type, params->data_format, params->sampling_freq);
	}

	return OK;
}

int KYTY_SYSV_ABI AudioOut2PortDestroy(AudioOut2PortHandle port) {
	PRINT_NAME();
	LOGF("\t port = 0x%016" PRIx64 "\n", port);

	int      audio_handle = 0;
	bool     found        = false;
	uint16_t port_type    = 0;
	g_audioout2_port_mutex.Lock();
	if (auto* state = audioout2_find_port_locked(port); state != nullptr) {
		audio_handle = state->audio_handle;
		port_type    = state->port_type;
		found        = true;
		*state       = AudioOut2PortStateEntry {};
	}
	g_audioout2_port_mutex.Unlock();

	if (!found) {
		Diag::Print("sceAudioOut2PortDestroy(port %" PRIu64 "): no such port", port);
	} else if (!audioout2_port_type_is_object(port_type) || port <= 16) {
		Diag::Print("sceAudioOut2PortDestroy(port %" PRIu64 ", type 0x%x, output handle %d)", port,
		            port_type, audio_handle);
	}

	audioout2_close_audio_handle(audio_handle);

	return OK;
}

// KYTY_AUDIO_LEVELS: every 5 s, the PCM the guest hands each AudioOut2 port class (bed ports by
// type, and all object ports together, which are mixed into the main bed), and the attribute ids
// seen per class. The output ports themselves are also metered in audio.cpp.
static bool audioout2_level_logging() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_AUDIO_LEVELS");
		return value != nullptr && value[0] != '\0' && value[0] != '0';
	}();
	return enabled;
}

static void audioout2_meter_locked(AudioOut2PortStateEntry* state, const void* pcm,
                                   uint32_t attribute_mask) {
	static uint64_t report_start = 0;
	static uint32_t seen_masks[3] {};
	const bool      object = audioout2_port_type_is_object(state->port_type);
	const int       klass  = object ? 2 : ((state->port_type & 0xffu) == 1 ? 1 : 0);
	if ((seen_masks[klass] | attribute_mask) != seen_masks[klass]) {
		seen_masks[klass] |= attribute_mask;
		std::printf("AudioLevel: AudioOut2 %s port type 0x%x attribute ids mask 0x%x\n",
		            object ? "object" : "bed", state->port_type, seen_masks[klass]);
	}
	if (pcm != nullptr) {
		const bool is_float = (state->data_format & 0x7fu) != 1;
		state->meter.Add(pcm, state->samples_num, audioout2_data_format_channels(state->data_format),
		                 is_float);
		state->pcm_sets++;
	}
	const auto now = LibKernel::KernelGetProcessTime();
	if (report_start == 0) {
		report_start = now;
	}
	if (now - report_start < 5000000) {
		return;
	}
	report_start = now;
	// Objects: the power sum over all object ports (as if mixed incoherently), their peak and
	// how many carried PCM.
	double   object_power = 0.0;
	double   object_peak  = 0.0;
	uint32_t object_ports = 0;
	uint64_t object_sets  = 0;
	for (auto& entry: g_audioout2_ports) {
		if (!entry.used || entry.pcm_sets == 0) {
			continue;
		}
		if (audioout2_port_type_is_object(entry.port_type)) {
			const double rms = entry.meter.Rms();
			object_power += rms * rms * (static_cast<double>(entry.pcm_sets));
			object_peak = std::max(object_peak, entry.meter.Peak());
			object_ports++;
			object_sets += entry.pcm_sets;
		} else {
			std::printf("AudioLevel: t=%.1f s AudioOut2 port %llu type 0x%x: %u PCM blocks, rms %.1f "
			            "dBFS peak %.1f dBFS%s\n",
			            static_cast<double>(now) / 1e6, static_cast<unsigned long long>(entry.handle),
			            entry.port_type, entry.pcm_sets, Mix::LevelMeter::ToDb(entry.meter.Rms()),
			            Mix::LevelMeter::ToDb(entry.meter.Peak()),
			            entry.audio_handle > 0 ? "" : " (not played)");
		}
		entry.meter.Reset();
		entry.pcm_sets = 0;
	}
	if (object_ports != 0) {
		// Normalise the power by the number of blocks a single port pushes in 5 s (~469).
		const double blocks = 5.0 * 48000.0 / 512.0;
		std::printf("AudioLevel: t=%.1f s AudioOut2 objects (guest PCM, mixed into the bed): %u ports, %llu "
		            "blocks, summed rms %.1f dBFS, max peak %.1f dBFS\n",
		            static_cast<double>(now) / 1e6, object_ports,
		            static_cast<unsigned long long>(object_sets),
		            Mix::LevelMeter::ToDb(std::sqrt(object_power / blocks)),
		            Mix::LevelMeter::ToDb(object_peak));
		auto& stats = g_audioout2_object_mix;
		std::printf("AudioLevel: t=%.1f s AudioOut2 objects as mixed into the bed front pair (objects "
		            "volume applied): rms %.1f dBFS peak %.1f dBFS; %u plain object blocks, mean gain %.3f, "
		            "max gain %.3f\n",
		            static_cast<double>(now) / 1e6, Mix::LevelMeter::ToDb(stats.added.Rms()),
		            Mix::LevelMeter::ToDb(stats.added.Peak()), stats.plain_objects,
		            stats.plain_objects != 0 ? stats.plain_gain / stats.plain_objects : 0.0,
		            static_cast<double>(stats.plain_gain_max));
		stats = {};
	}
	std::fflush(stdout);
}

// KYTY_AUDIO_LEVELS: the raw placement attributes of the first object ports (id, size, value),
// then every 5 s the decoded placement of a few active objects.
static void audioout2_log_object_attributes_locked(const AudioOut2PortStateEntry& state,
                                                   const AudioOut2Attribute* attributes, uint32_t num) {
	static uint32_t raw_lines[34] {}; // per attribute id (0-16), objects then other ports
	static uint64_t last_report = 0;
	for (uint32_t i = 0; i < num; i++) {
		const auto& a    = attributes[i];
		auto&       seen = raw_lines[std::min<uint32_t>(a.attribute_id, 16) +
		                         (audioout2_port_type_is_object(state.port_type) ? 0 : 17)];
		if (a.attribute_id == AUDIO_OUT2_PORT_ATTRIBUTE_ID_PCM || a.value == nullptr || seen >= 24) {
			continue;
		}
		uint32_t words[4] {};
		std::memcpy(words, a.value, std::min<size_t>(sizeof(words), static_cast<size_t>(a.value_size)));
		float floats[4] {};
		std::memcpy(floats, words, sizeof(floats));
		std::printf("AudioObject: port %" PRIu64 " type 0x%x attr %u size %" PRIu64
		            " raw %08x %08x %08x %08x float %g %g %g %g\n",
		            static_cast<uint64_t>(state.handle), state.port_type, a.attribute_id,
		            static_cast<uint64_t>(a.value_size), words[0], words[1], words[2], words[3], floats[0],
		            floats[1], floats[2], floats[3]);
		seen++;
	}
	const auto now = LibKernel::KernelGetProcessTime();
	if (now - last_report < 5000000) {
		return;
	}
	last_report  = now;
	int reported = 0;
	for (const auto& entry: g_audioout2_ports) {
		const auto& o = entry.object;
		if (!entry.used || !audioout2_port_type_is_object(entry.port_type) || entry.pcm_data.empty() ||
		    Objects::IsAmbisonicsChannel(o.ambisonics) || o.passthrough != Objects::PASSTHROUGH_NONE) {
			continue;
		}
		std::printf("AudioObject: t=%.1f s port %" PRIu64 " gain %.3f pos%s (%.2f, %.2f, %.2f) spread %.2f "
		            "passthrough %u ambisonics 0x%x\n",
		            static_cast<double>(now) / 1e6, static_cast<uint64_t>(entry.handle), o.gain,
		            o.has_position ? "" : " (none)", o.x, o.y, o.z, o.spread, o.passthrough, o.ambisonics);
		if (++reported == 12) {
			break;
		}
	}
	std::fflush(stdout);
}

int KYTY_SYSV_ABI AudioOut2PortSetAttributes(AudioOut2PortHandle       port,
                                             const AudioOut2Attribute* attributes, uint32_t num) {
	EXIT_NOT_IMPLEMENTED(num != 0 && attributes == nullptr);

	const void* pcm_data = nullptr;
	bool        has_pcm  = false;
	uint32_t    ids      = 0;
	for (uint32_t i = 0; i < num; i++) {
		ids |= attributes[i].attribute_id < 32 ? (1u << attributes[i].attribute_id) : 0x80000000u;
		if (attributes[i].attribute_id == AUDIO_OUT2_PORT_ATTRIBUTE_ID_PCM &&
		    attributes[i].value != nullptr && attributes[i].value_size >= sizeof(AudioOut2Pcm)) {
			AudioOut2Pcm pcm {};
			std::memcpy(&pcm, attributes[i].value, sizeof(AudioOut2Pcm));
			pcm_data = pcm.data;
			has_pcm  = true;
		}
	}

	if (audioout2_level_logging()) {
		g_audioout2_port_mutex.Lock();
		if (auto* state = audioout2_find_port_locked(port); state != nullptr) {
			audioout2_meter_locked(state, pcm_data, ids);
		}
		g_audioout2_port_mutex.Unlock();
	}

	if (has_pcm || (ids & ~(1u << AUDIO_OUT2_PORT_ATTRIBUTE_ID_PCM)) != 0) {
		g_audioout2_port_mutex.Lock();
		if (auto* state = audioout2_find_port_locked(port); state != nullptr) {
			if (audioout2_port_type_is_object(state->port_type)) {
				for (uint32_t i = 0; i < num; i++) {
					Objects::ApplyAttribute(&state->object, attributes[i].attribute_id,
					                        attributes[i].value, attributes[i].value_size);
				}
			}
			if (audioout2_level_logging()) {
				audioout2_log_object_attributes_locked(*state, attributes, num);
			}
			if (has_pcm) {
				if (pcm_data != nullptr && state->audio_format != AudioInternal::Format::Unknown) {
					const auto* bytes = static_cast<const uint8_t*>(pcm_data);
					state->pcm_data.assign(bytes, bytes + audioout2_pcm_size(*state));
					state->pcm_fresh = true;
				} else {
					state->pcm_data.clear();
					state->pcm_fresh = false;
				}
			}
		}
		g_audioout2_port_mutex.Unlock();
	}

	return OK;
}

int KYTY_SYSV_ABI AudioOut2PortGetState(AudioOut2PortHandle port, AudioOut2PortState* state) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(state == nullptr);

	std::memset(state, 0, sizeof(AudioOut2PortState));
	state->output          = 1;
	state->num_channels    = 2;
	state->volume          = 127;
	state->reroute_counter = 0;
	state->flags           = 0;

	g_audioout2_port_mutex.Lock();
	if (auto* port_state = audioout2_find_port_locked(port); port_state != nullptr) {
		state->num_channels = audioout2_data_format_channels(port_state->data_format);
	}
	g_audioout2_port_mutex.Unlock();

	// LOGF("\t port = 0x%016" PRIx64 "\n", port);
	// LOGF("\t num_channels = %" PRIu8 "\n", state->num_channels);

	return OK;
}

int KYTY_SYSV_ABI AudioOut2GetSystemState(AudioOut2SystemState* state) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(state == nullptr);
	std::memset(state, 0, sizeof(AudioOut2SystemState));
	return OK;
}

int KYTY_SYSV_ABI AudioOut2UserCreate(uint32_t user_id, AudioOut2UserHandle* handle) {
	PRINT_NAME();
	if (handle == nullptr) {
		return AUDIO_OUT2_ERROR_INVALID_PARAM;
	}
	Common::LockGuard lock(g_audioout2_port_mutex);
	*handle = g_audioout2_next_user++;
	g_audioout2_users.push_back(*handle);
	LOGF("\t user_id = %" PRIu32 ", handle = 0x%016" PRIx64 "\n", user_id,
	     static_cast<uint64_t>(*handle));
	Diag::Print("sceAudioOut2UserCreate(user id %" PRIu32 ") -> user handle %" PRIu64, user_id,
	            static_cast<uint64_t>(*handle));
	return OK;
}

int KYTY_SYSV_ABI AudioOut2UserDestroy(AudioOut2UserHandle handle) {
	PRINT_NAME();
	LOGF("\t handle = 0x%016" PRIx64 "\n", static_cast<uint64_t>(handle));
	Diag::Print("sceAudioOut2UserDestroy(user handle %" PRIu64 ")", static_cast<uint64_t>(handle));
	Common::LockGuard lock(g_audioout2_port_mutex);
	const auto it = std::find(g_audioout2_users.begin(), g_audioout2_users.end(), handle);
	if (it == g_audioout2_users.end()) {
		return Diag::Error("sceAudioOut2UserDestroy", AUDIO_OUT2_ERROR_INVALID_PARAM,
		                   static_cast<int64_t>(handle));
	}
	if (std::any_of(g_audioout2_ports.begin(), g_audioout2_ports.end(),
	                [handle](const auto& port) { return port.used && port.user == handle; })) {
		return Diag::Error("sceAudioOut2UserDestroy", AUDIO_OUT2_ERROR_BUSY,
		                   static_cast<int64_t>(handle));
	}
	g_audioout2_users.erase(it);
	return OK;
}

int KYTY_SYSV_ABI AudioOut2UserGetSupportedAttributes(AudioOut2UserHandle handle,
                                                       uint32_t* context_attributes,
                                                       uint32_t* port_attributes) {
	if (context_attributes == nullptr || port_attributes == nullptr) {
		return Diag::Error("sceAudioOut2UserGetSupportedAttributes", AUDIO_OUT2_ERROR_INVALID_PARAM,
		                   0);
	}
	Common::LockGuard lock(g_audioout2_port_mutex);
	if (std::find(g_audioout2_users.begin(), g_audioout2_users.end(), handle) ==
	    g_audioout2_users.end()) {
		return Diag::Error("sceAudioOut2UserGetSupportedAttributes", AUDIO_OUT2_ERROR_INVALID_PARAM,
		                   static_cast<int64_t>(handle));
	}
	*context_attributes = 0;
	*port_attributes    = 1u << AUDIO_OUT2_PORT_ATTRIBUTE_ID_PCM;
	return OK;
}

size_t KYTY_SYSV_ABI AudioOut2GetSpeakerArrayMemorySize(uint32_t num_speakers, uint8_t is_3d,
                                                        uint8_t is_ambisonics) {
	PRINT_NAME();

	const auto speakers = std::clamp<uint32_t>(num_speakers, 1, 32);
	const auto size = static_cast<size_t>(0x400 + speakers * (is_ambisonics != 0 ? 0x100 : 0x40) +
	                                      (is_3d != 0 ? 0x200 : 0));

	LOGF("\t num_speakers  = %" PRIu32 "\n"
	     "\t is_3d         = %" PRIu8 "\n"
	     "\t is_ambisonics = %" PRIu8 "\n"
	     "\t memory_size   = 0x%016" PRIx64 "\n",
	     num_speakers, is_3d, is_ambisonics, static_cast<uint64_t>(size));

	return size;
}

int KYTY_SYSV_ABI AudioOut2SpeakerArrayCreate(AudioOut2SpeakerArrayHandle* handle,
                                              const void* vbap_params, const void* ambi_params) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(handle == nullptr);

	*handle = nullptr;

	g_audioout2_speaker_array_mutex.Lock();
	for (auto& state: g_audioout2_speaker_arrays) {
		if (!state.used) {
			state              = AudioOut2SpeakerArrayState {};
			state.used         = true;
			state.handle       = &state;
			state.num_speakers = 2;
			*handle            = state.handle;
			break;
		}
	}
	g_audioout2_speaker_array_mutex.Unlock();

	LOGF("\t handle      = 0x%016" PRIx64 "\n"
	     "\t vbap_params = 0x%016" PRIx64 "\n"
	     "\t ambi_params = 0x%016" PRIx64 "\n",
	     reinterpret_cast<uint64_t>(*handle), reinterpret_cast<uint64_t>(vbap_params),
	     reinterpret_cast<uint64_t>(ambi_params));

	return (*handle != nullptr
	            ? OK
	            : Diag::Error("sceAudioOut2SpeakerArrayCreate", AUDIO_OUT2_ERROR_PORT_FULL, 0));
}

int KYTY_SYSV_ABI AudioOut2SpeakerArrayDestroy(AudioOut2SpeakerArrayHandle handle) {
	PRINT_NAME();
	LOGF("\t handle = 0x%016" PRIx64 "\n", reinterpret_cast<uint64_t>(handle));

	g_audioout2_speaker_array_mutex.Lock();
	for (auto& state: g_audioout2_speaker_arrays) {
		if (state.used && state.handle == handle) {
			state = AudioOut2SpeakerArrayState {};
			break;
		}
	}
	g_audioout2_speaker_array_mutex.Unlock();

	return OK;
}

int KYTY_SYSV_ABI AudioOut2GetSpeakerArrayCoefficients(
    AudioOut2SpeakerArrayHandle handle, AudioOut2Position pos, float spread, float* coefficients,
    uint32_t num_coefficients, uint8_t height_aware, float downmix_spread_radius) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(coefficients == nullptr && num_coefficients != 0);

	if (coefficients != nullptr) {
		std::fill(coefficients, coefficients + num_coefficients, 0.0f);
		if (num_coefficients > 0) {
			coefficients[0] = 1.0f;
		}
		if (num_coefficients > 1) {
			coefficients[1] = 1.0f;
		}
	}

	LOGF("\t handle = 0x%016" PRIx64 ", coeffs = %" PRIu32
	     ", pos = (%f, %f, %f), spread = %f, height = %" PRIu8 ", downmix = %f\n",
	     reinterpret_cast<uint64_t>(handle), num_coefficients, static_cast<double>(pos.x),
	     static_cast<double>(pos.y), static_cast<double>(pos.z), static_cast<double>(spread),
	     height_aware, static_cast<double>(downmix_spread_radius));

	return OK;
}

int KYTY_SYSV_ABI AudioOut2GetSpeakerArrayAmbisonicsCoefficients(AudioOut2SpeakerArrayHandle handle,
                                                                 uint32_t ambisonics_channel,
                                                                 float*   coefficients,
                                                                 uint32_t num_coefficients) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(coefficients == nullptr && num_coefficients != 0);

	if (coefficients != nullptr) {
		std::fill(coefficients, coefficients + num_coefficients, 0.0f);
		if (num_coefficients > 0) {
			coefficients[0] =
			    (ambisonics_channel == 0 || ambisonics_channel == 64 ? 0.70710677f : 1.0f);
		}
	}

	LOGF("\t handle = 0x%016" PRIx64 ", channel = %" PRIu32 ", coeffs = %" PRIu32 "\n",
	     reinterpret_cast<uint64_t>(handle), ambisonics_channel, num_coefficients);

	return OK;
}

int KYTY_SYSV_ABI AudioOut2GetSpeakerInfo(AudioOut2SpeakerInfo* info, uint32_t flags) {
	EXIT_NOT_IMPLEMENTED(info == nullptr);
	std::memset(info, 0, sizeof(AudioOut2SpeakerInfo));
	info->type             = 0;
	info->available_bits   = 0x03;
	info->flags            = 0;
	info->speaker_angle[0] = {-30, 0};
	info->speaker_angle[1] = {30, 0};

	return OK;
}

int KYTY_SYSV_ABI AudioOut2SetSystemDebugState(const AudioOut2SystemDebugStateParam* param) {
	PRINT_NAME();
	LOGF("\t param = 0x%016" PRIx64 "\n", reinterpret_cast<uint64_t>(param));
	return OK;
}

int KYTY_SYSV_ABI AudioOut2Set3DLatency(uint32_t user_id, uint32_t output, uint32_t latency_us) {
	// Not sure

	PRINT_NAME();
	LOGF("\t user_id = %" PRIu32 ", output = %" PRIu32 ", latency_us = %" PRIu32 "\n", user_id,
	     output, latency_us);

	if (output > AUDIO_OUT2_MASTERING_OUTPUT_RECORDING) {
		return Diag::Error("sceAudioOut2Set3DLatency", AUDIO_OUT2_ERROR_INVALID_PARAM, output);
	}

	g_audioout2_latency_mutex.Lock();
	AudioOut2LatencyState* free_state = nullptr;
	for (auto& state: g_audioout2_latencies) {
		if (state.used && state.user_id == user_id && state.output == output) {
			state.latency_us = latency_us;
			g_audioout2_latency_mutex.Unlock();
			return OK;
		}
		if (!state.used && free_state == nullptr) {
			free_state = &state;
		}
	}

	if (free_state != nullptr) {
		*free_state = AudioOut2LatencyState {true, user_id, output, latency_us};
	}
	g_audioout2_latency_mutex.Unlock();

	return (free_state != nullptr
	            ? OK
	            : Diag::Error("sceAudioOut2Set3DLatency", AUDIO_OUT2_ERROR_PORT_FULL, user_id));
}

int KYTY_SYSV_ABI AudioOut2MasteringInit(uint32_t flags) {
	PRINT_NAME();
	LOGF("\t flags = 0x%08" PRIx32 "\n", flags);
	return OK;
}

int KYTY_SYSV_ABI AudioOut2MasteringSetParam(const AudioOut2MasteringParamsHeader* param,
                                             uint32_t output, uint32_t flags) {
	PRINT_NAME();
	LOGF("\t param = 0x%016" PRIx64 ", output = %" PRIu32 ", flags = 0x%08" PRIx32 "\n",
	     reinterpret_cast<uint64_t>(param), output, flags);
	return OK;
}

int KYTY_SYSV_ABI AudioOut2MasteringGetState(AudioOut2MasteringStatesHeader* state, uint32_t output,
                                             AudioOut2UserHandle user) {
	PRINT_NAME();
	LOGF("\t state = 0x%016" PRIx64 ", output = %" PRIu32 ", user = 0x%016" PRIx64 "\n",
	     reinterpret_cast<uint64_t>(state), output, static_cast<uint64_t>(user));

	if (state == nullptr) {
		return Diag::Error("sceAudioOut2MasteringGetState",
		                   AUDIO_OUT2_ERROR_MASTERING_INVALID_API_PARAM, output);
	}

	const auto states_id = state->states_id;
	switch (states_id) {
		case AUDIO_OUT2_MASTERING_STATES_ID_DEFAULT: {
			auto* full_state = reinterpret_cast<AudioOut2MasteringStates*>(state);
			std::memset(full_state, 0, sizeof(AudioOut2MasteringStates));
			full_state->states_header.states_id = AUDIO_OUT2_MASTERING_STATES_ID_DEFAULT;
			full_state->compressor_states.descriptor.id =
			    AUDIO_OUT2_MASTERING_STATES_STRUCT_ID_COMPRESSOR_DEFAULT;
			full_state->compressor_states.descriptor.size =
			    sizeof(AudioOut2MasteringCompressorStates);
			full_state->limiter_states.descriptor.id =
			    AUDIO_OUT2_MASTERING_STATES_STRUCT_ID_LIMITER_DEFAULT;
			full_state->limiter_states.descriptor.size = sizeof(AudioOut2MasteringLimiterStates);
			return OK;
		}
		case AUDIO_OUT2_MASTERING_STATES_ID_V2: {
			auto* full_state = reinterpret_cast<AudioOut2MasteringStatesV2*>(state);
			std::memset(full_state, 0, sizeof(AudioOut2MasteringStatesV2));
			full_state->states_header.states_id = AUDIO_OUT2_MASTERING_STATES_ID_V2;
			full_state->compressor_states.descriptor.id =
			    AUDIO_OUT2_MASTERING_STATES_STRUCT_ID_COMPRESSOR_V2;
			full_state->compressor_states.descriptor.size =
			    sizeof(AudioOut2MasteringCompressorStatesV2);
			full_state->limiter_states.descriptor.id =
			    AUDIO_OUT2_MASTERING_STATES_STRUCT_ID_LIMITER_DEFAULT;
			full_state->limiter_states.descriptor.size = sizeof(AudioOut2MasteringLimiterStates);
			return OK;
		}
		default:
			return Diag::Error("sceAudioOut2MasteringGetState",
			                   AUDIO_OUT2_ERROR_MASTERING_INVALID_STATES_ID, states_id);
	}
}

int KYTY_SYSV_ABI AudioOut2MasteringTerm() {
	PRINT_NAME();
	return OK;
}

} // namespace AudioOut2
} // namespace Libs::Audio
