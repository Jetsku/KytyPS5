#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_THREADSAMPLER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_THREADSAMPLER_H_

namespace Libs::Graphics {

// Diagnostics (Windows; after BryanKAdams/KytyPS5's thread sampler):
// KYTY_CP_SAMPLER=<period in microseconds> (unset or 0: off; values below 100 select 500) makes a
// helper thread sample the calling thread's call stack. The sampled thread is stopped only while
// its registers and the top of its stack (KYTY_CP_SAMPLER_BYTES, default 32768) are copied; the
// copy is unwound afterwards. Every 10 s the window's stacks are written to
// <KYTY_CP_SAMPLER_DIR or the working directory>/cp-sample-<name>-<window>.txt as
// "count module+offset;module+offset;..." lines (innermost frame first) for llvm-symbolizer.
// Needs no administrator rights (unlike ETW sampling), so unattended timing runs can profile the
// command processor. It costs the sampled thread a few microseconds per sample.
void StartThreadSampler(const char* name);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_THREADSAMPLER_H_
