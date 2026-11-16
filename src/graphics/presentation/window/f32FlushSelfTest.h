#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_F32FLUSHSELFTEST_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_F32FLUSHSELFTEST_H_

#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/F32FlushSelfTest.h"

#include <string>

namespace Libs::Graphics {

struct GraphicContext;

// One run of the KYTY_F32_FLUSH_HOST self-test job (recompiler/F32FlushSelfTest.h) on a device:
// emits the module, dispatches it once on `queue` and classifies the read-back.
struct F32FlushSelfTestRun {
	bool                                            ran = false; // the job completed on the GPU
	std::string                                     error;       // why it did not
	ShaderRecompiler::FlushSelfTest::Verdict        verdict;
	ShaderRecompiler::FlushSelfTest::Verdict        reference;   // the shader's expected vs the model
	double                                          milliseconds = 0.0;
	// Phases of milliseconds: module emission, buffer + pipeline creation, submit to fence, the
	// read-back copy and classification.
	double emit_ms = 0.0, pipeline_ms = 0.0, gpu_ms = 0.0, check_ms = 0.0;
	[[nodiscard]] bool Passed() const { return ran && verdict.Passed(); }
};

// force_mismatch: KYTY_F32_FLUSH_SELFTEST=fail (ForceMismatch on the read-back).
[[nodiscard]] F32FlushSelfTestRun RunF32FlushSelfTest(vk::Device device,
                                                      const vk::PhysicalDeviceMemoryProperties& memory,
                                                      vk::Queue queue, uint32_t queue_family,
                                                      bool force_mismatch);

// The device layer, once the device and its queue exist and before anything is translated: with
// KYTY_F32_FLUSH_HOST=nvidia on an NVIDIA device (ShaderRecompiler::F32FlushHostAwaitsSelfTest)
// runs the job and keeps the switch only when it passes, printing one "Kyty GPU capabilities:"
// line. KYTY_F32_FLUSH_SELFTEST=off keeps the switch without the test, =fail forces a mismatch.
void SettleF32FlushHostSelfTest(GraphicContext& graphics);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_F32FLUSHSELFTEST_H_
