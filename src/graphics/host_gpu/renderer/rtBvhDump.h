#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RTBVHDUMP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RTBVHDUMP_H_

#include <array>
#include <cstdint>
#include <span>

namespace Libs::Graphics {

// KYTY_RT_DUMP_BVH=<n>[,<n>...] (or / between the numbers; off by default; diagnostic only). At the n-th compute dispatch
// of a program with BVH instructions (1-based over the run, whether KYTY_RT_STUB translated the
// program or it was skipped), dump the Psr TLAS the dispatch's SRT names (s[0:1]: +0 in the
// lighting shaders, +40 in the GI shaders, other offsets as a fallback) and every BLAS its
// instances reference. Files go to KYTY_RT_DUMP_BVH_DIR (default <shader log folder>/bvh): a
// .kbvh snapshot (graphics/host_gpu/rt/bvhDump.h) and a .txt summary. The dump reads the BVHs back
// coherently (GPU-written bytes are synchronized first), so it stalls the GPU thread while it
// runs. A dispatch whose SRT names no plausible TLAS is retried at the next BVH dispatch (at most
// 64 times per requested ordinal).
[[nodiscard]] bool RtBvhDumpEnabled() noexcept;

struct RtBvhDispatch {
	uint64_t                  shader_hash    = 0; // 0 when the program was skipped
	uint64_t                  shader_address = 0;
	uint64_t                  args_address   = 0; // indirect dispatches
	std::span<const uint32_t> user_sgpr;
	std::array<uint32_t, 3>   groups {};
	bool                      indirect = false;
	bool                      skipped  = false;
	int                       frame    = 0;
};

// GPU thread, before the dispatch prepares its bindings: the readbacks may flush the recording.
void RtBvhDumpOnDispatch(const RtBvhDispatch& dispatch);

// KYTY_WATCH_READ=<hash>:<sgpr>:<bytes>[+<sgpr>:<bytes>...][,<hash>:...] (diagnostic): at each
// dispatch of a listed compute shader, the buffers whose base address sits in user SGPRs
// <sgpr>,<sgpr+1> (V# word 0 and the low 16 bits of word 1) are read coherently and saved; at the
// same time the regions last seen for every listed shader are read again, so a later listed
// dispatch shows what an earlier one wrote. Files go to KYTY_WATCH_READ_DIR (default
// _Shaders/watch), at most KYTY_WATCH_READ_MAX (default 48) dispatches.
[[nodiscard]] bool WatchReadEnabled() noexcept;
void               WatchReadOnDispatch(uint64_t shader_hash, std::span<const uint32_t> user_sgpr, int frame);

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RTBVHDUMP_H_ */
