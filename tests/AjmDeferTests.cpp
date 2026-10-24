// AJM batch jobs run when their batch starts (sceAjmBatchStart), not when they are added
// (KYTY_AJM_DEFER, default on; todojuegos67 d74ca6333). The library source is compiled into this
// test so the batch structures are its own.
#include "libs/ajm.cpp" // NOLINT(bugprone-suspicious-include)

#include <array>
#include <cstdio>
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

} // namespace

int main() {
	if (const char* defer = std::getenv("KYTY_AJM_DEFER"); defer != nullptr && defer[0] == '0') {
		std::printf("AjmDeferTests: skipped (KYTY_AJM_DEFER=0)\n");
		return 0;
	}
	TestJobRunsAtBatchStart();
	TestInitializeDropsUnstartedJobs();
	std::printf("AjmDeferTests: all passed\n");
	return 0;
}
