// AJM batch jobs run when their batch starts (sceAjmBatchStart), not when they are added
// (KYTY_AJM_DEFER, default on; todojuegos67 d74ca6333). The library source is compiled into this
// test so the batch structures are its own.
#include "libs/ajm.cpp" // NOLINT(bugprone-suspicious-include)

#include <array>
#include <cstdint>
#include <cstdio>
#include <opus_multistream.h>
#include <cstdlib>
#include <cstring>

namespace {
namespace Ajm = Libs::Audio::Ajm;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "AjmDeferTests: %s\n", message);
		std::abort();
	}
}

// A clear-context job on an instance that does not exist writes its (error) result into the result
// buffer: not when it is added, but when the batch starts.
void TestJobRunsAtBatchStart() {
	std::array<uint8_t, 4096> batch_buffer {};
	Ajm::AjmBatchInfo         info {};
	Check(Ajm::AjmBatchInitialize(batch_buffer.data(), batch_buffer.size(), &info) == OK, "initialize");
	std::array<uint8_t, 64> result {};
	result.fill(0xcd);
	Check(Ajm::AjmBatchJobClearContext(&info, 0x7fffffffu, result.data()) == OK, "add clear-context");
	Check(result[0] == 0xcd && result[1] == 0xcd && result[2] == 0xcd && result[3] == 0xcd,
	      "the job did not run when it was added");
	Check(info.offset != 0, "the job took its space in the batch when it was added");
	uint32_t batch = 0;
	Check(Ajm::AjmBatchStart(0, &info, 0, nullptr, &batch) == OK, "start");
	Check(!(result[0] == 0xcd && result[1] == 0xcd && result[2] == 0xcd && result[3] == 0xcd),
	      "the job ran when the batch started");
	// A started batch has no jobs left: starting it again runs nothing.
	result.fill(0xcd);
	Check(Ajm::AjmBatchStart(0, &info, 0, nullptr, &batch) == OK, "start again");
	Check(result[0] == 0xcd, "a started batch's jobs run once");
}

// Re-initializing a batch buffer drops the jobs added to it and never started.
void TestInitializeDropsUnstartedJobs() {
	std::array<uint8_t, 4096> batch_buffer {};
	Ajm::AjmBatchInfo         info {};
	Check(Ajm::AjmBatchInitialize(batch_buffer.data(), batch_buffer.size(), &info) == OK, "initialize");
	std::array<uint8_t, 64> result {};
	result.fill(0xcd);
	Check(Ajm::AjmBatchJobClearContext(&info, 0x7fffffffu, result.data()) == OK, "add clear-context");
	Check(Ajm::AjmBatchInitialize(batch_buffer.data(), batch_buffer.size(), &info) == OK, "initialize again");
	uint32_t batch = 0;
	Check(Ajm::AjmBatchStart(0, &info, 0, nullptr, &batch) == OK, "start");
	Check(result[0] == 0xcd, "a job of the overwritten batch does not run");
}

// Opus channel mapping family 1 (KYTY_OPUS_MULTISTREAM): refused by default; with the switch, a
// 4-channel multistream packet from libopus decodes to 960 4-channel samples.
void TestOpusMultistream() {
	const Ajm::AjmDecOpusInitializeParameters family1 {4, 48000, 1};
	{
		Ajm::AjmOpusDecoder off(8, Ajm::AjmSampleEncoding::S16, false);
		Check((off.Initialize(&family1, sizeof(family1)).result & Ajm::AJM_RESULT_FATAL) != 0,
		      "family 1 is refused with the switch off");
	}
	Ajm::AjmOpusDecoder decoder(8, Ajm::AjmSampleEncoding::S16, true);
	Check(decoder.Initialize(&family1, sizeof(family1)).result == OK, "family 1 initializes");

	constexpr int     frame_samples = 960;
	const uint8_t     mapping[4] {0, 1, 2, 3};
	int               error   = 0;
	OpusMSEncoder*    encoder = opus_multistream_encoder_create(48000, 4, 2, 2, mapping,
	                                                            OPUS_APPLICATION_AUDIO, &error);
	Check(encoder != nullptr && error == OPUS_OK, "create the multistream encoder");
	std::array<int16_t, frame_samples * 4> pcm {};
	for (int i = 0; i < frame_samples; i++) {
		for (int c = 0; c < 4; c++) {
			pcm[i * 4 + c] = static_cast<int16_t>(((i * (c + 1) * 37) % 2000) - 1000);
		}
	}
	std::array<uint8_t, 4002> input {};
	const int size = opus_multistream_encode(encoder, pcm.data(), frame_samples, input.data() + 2,
	                                         static_cast<int>(input.size() - 2));
	opus_multistream_encoder_destroy(encoder);
	Check(size > 0, "encode a multistream packet");
	input[0] = static_cast<uint8_t>(size & 0xff);
	input[1] = static_cast<uint8_t>(size >> 8);

	std::array<int16_t, frame_samples * 4> output {};
	const auto result = decoder.Decode(input.data(), static_cast<size_t>(size) + 2, output.data(),
	                                   sizeof(output), false, nullptr);
	Check(result.result == OK && result.frames == 1 &&
	          result.input_consumed == static_cast<size_t>(size) + 2 &&
	          result.output_written == sizeof(output),
	      "a 4-channel family 1 packet decodes");
}

} // namespace

// This line keeps the switch default off (int21 behaviour); the test turns it on unless set.
static void DefaultEnv(const char* name, const char* value) {
	if (std::getenv(name) != nullptr) {
		return;
	}
#ifdef _WIN32
	_putenv_s(name, value);
#else
	setenv(name, value, 0);
#endif
}

int main() {
	TestOpusMultistream();
	DefaultEnv("KYTY_AJM_DEFER", "1");
	if (const char* defer = std::getenv("KYTY_AJM_DEFER"); defer != nullptr && defer[0] == '0') {
		std::printf("AjmDeferTests: skipped (KYTY_AJM_DEFER=0)\n");
		return 0;
	}
	TestJobRunsAtBatchStart();
	TestInitializeDropsUnstartedJobs();
	std::printf("AjmDeferTests: all passed\n");
	return 0;
}
