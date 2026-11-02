#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_NATIVERAYTRACING_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_NATIVERAYTRACING_H_

#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <functional>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

namespace ShaderRecompiler::IR {
struct ShaderInfo;
} // namespace ShaderRecompiler::IR

class BufferCache;
class CommandScheduler;
struct GraphicContext;

// Host acceleration structures for guest ray tracing (KYTY_RT_HW; VK_KHR_acceleration_structure and
// VK_KHR_ray_query). Adapted from TheCruZ's native-hardware-raytracing branch of KytyPS5-GTA:
// the GTA V HLE stages (native TLAS builds, HLE refits) and the host table of sce::Psr builds the
// command processor saw are left out; the guest BVH layout is Miles Morales' sce::Psr 2.0 one.
//
// Games using sce::Psr build AMD BVHs and traverse them in software with IMAGE_BVH_INTERSECT_RAY,
// one node at a time. Before a dispatch whose program the recompiler patched
// (RayTracingPatch.h), this class mirrors the guest BVH on the GPU, without the CPU reading what
// the GPU wrote:
// - every leaf of the guest top-level BVH (an instance record) becomes a host TLAS instance with
//   the inverse of the record's world-to-object matrix;
// - every guest bottom-level BVH becomes a host BLAS of the triangles of its leaf lists (opaque
//   ones as geometry 0, alpha tested ones as geometry 1), with a table from the primitive to the
//   guest list entry (its leaf pointer, info word, triangle node).
// The patched shader takes every triangle the hardware reports as a candidate and runs the guest
// leaf code for it. BLASes the host has not seen are reported by the top-level walk, counted on
// the GPU and read back asynchronously; the host allocates them and builds them the next time.
// Slots whose guest BLAS changed (the verification walk compares a signature of its boxes) are
// rebuilt before every use while they keep changing; a changed slot whose leaf count stayed
// (a refit) is rebuilt by gathering its vertices through its entry table (no walk).
//
// The host memory is bounded: a slot no TLAS referenced for EvictFrames frames is evicted, the
// structures have a budget (a share of the VRAM, KYTY_RT_ARENA_MB overrides it), and whatever the
// GPU may still read is destroyed only after the work recorded so far has completed.
//
// Guest TLAS layout (Psr 2.0): the header ("PSR_BVHL") has the instance array offset at +40 and
// the node count (T# size) at +80; a 128-byte instance record holds the BLAS address at +0, a
// column-major 3x4 world-to-object matrix at +8, the BLAS root at +56 and the instance flags (low
// 4 bits) and mask (bits 8-31) at +60. The header address is read from the V# at the user data
// pointer + vsharp_offset (its base) + header_offset.
class NativeRayTracing {
public:
	NativeRayTracing(GraphicContext& graphics, CommandScheduler& scheduler, BufferCache& buffer_cache);
	~NativeRayTracing();
	KYTY_CLASS_NO_COPY(NativeRayTracing);

	// Whether the device supports host ray tracing and KYTY_RT_HW is on.
	[[nodiscard]] bool Enabled() const noexcept { return m_enabled; }

	struct TlasSource {
		uint32_t user_sgpr     = 0;
		uint32_t vsharp_offset = 0;
		uint32_t header_offset = 0;
		uint32_t root          = 0;
	};
	// The source of a program that traces through a guest TLAS (its info), or false.
	[[nodiscard]] static bool FindTlasSource(const ShaderRecompiler::IR::ShaderInfo& info, TlasSource& source) noexcept;

	// Records, before the dispatch, the builds of pending BLASes, the walk of the guest TLAS of
	// `source` from the user data pointer `user_pointer`, and the TLAS build. Outside a render
	// pass, into the native command buffer `command`. KYTY_RT_HW_NATIVE=0 (live) invalidates the
	// context instead: the patched shaders run their guest loops.
	// header_hint: the TLAS header address the host read from guest memory (0: the walk reads it).
	void Prepare(vk::CommandBuffer command, uint64_t user_pointer, const TlasSource& source, uint64_t frame,
	             uint64_t header_hint = 0);

	// The device address of the context the shaders look their TLAS up in (ContextEntry in
	// rt_tlas_walk.comp), 0 when disabled.
	[[nodiscard]] uint64_t ContextAddress() const noexcept { return m_context_address; }

	// Verify counters (KYTY_RT_HW_VERIFY, IR::RayQueryCounter) at this offset of the context buffer.
	static constexpr uint32_t CountersOffset = 1024;
	static constexpr uint32_t CounterCount   = 10;

private:
	struct Allocation {
		Buffer*           buffer  = nullptr;
		uint64_t          offset  = 0;
		vk::DeviceAddress address = 0;
		uint64_t          size    = 0;
	};

	// Sub-allocates device memory for acceleration structures and their inputs (best fit in the
	// fullest chunk; empty chunks are released; a large allocation gets a chunk of its own).
	class Arena {
	public:
		Arena(GraphicContext& graphics, CommandScheduler& scheduler): m_graphics(graphics), m_scheduler(scheduler) {}
		Allocation Allocate(uint64_t size, uint64_t alignment);
		void       Free(const Allocation& allocation);
		[[nodiscard]] uint64_t Used() const noexcept { return m_used; }
		[[nodiscard]] uint64_t Reserved() const noexcept { return m_reserved; }

	private:
		static constexpr uint64_t ChunkSize = uint64_t {64} << 20u;
		struct Chunk {
			std::unique_ptr<Buffer>      buffer;
			std::map<uint64_t, uint64_t> free_ranges;
			uint64_t                     free_bytes = 0;
			bool                         dedicated  = false;
		};
		GraphicContext&                     m_graphics;
		CommandScheduler&                   m_scheduler;
		std::vector<std::unique_ptr<Chunk>> m_chunks;
		uint64_t                            m_used     = 0;
		uint64_t                            m_reserved = 0;
	};

	struct SlotSizes {
		uint32_t capacity0 = 0;
		uint32_t capacity1 = 0;
		uint64_t storage   = 0;
		uint64_t vertices0 = 0;
		uint64_t vertices1 = 0;
		uint64_t entries   = 0;
		uint64_t scratch   = 0;
		[[nodiscard]] uint64_t Total() const noexcept { return storage + vertices0 + vertices1 + entries; }
	};

	struct Slot {
		uint64_t                     blas_va       = 0;
		uint32_t                     root          = 0;
		uint32_t                     table_index   = 0;
		uint32_t                     capacity0     = 0;
		uint32_t                     capacity1     = 0;
		uint32_t                     leaf_count    = 0;
		Allocation                   vertices0;
		Allocation                   vertices1;
		Allocation                   entries;
		Allocation                   storage;
		vk::AccelerationStructureKHR handle        = nullptr;
		vk::DeviceAddress            as_address    = 0;
		uint64_t                     build_scratch = 0;
		bool                         built         = false;
		bool                         needs_build   = false;
		// Until this preparation the slot is rebuilt every time (the guest changed it lately).
		uint32_t                     dynamic_until = 0;
		uint64_t                     built_frame   = UINT64_MAX;
		// Changed since the last build with the same leaf count (a refit): gathered, not walked.
		bool                         refit         = false;
		// The last change the verification walk reported was a refit (same leaf count): the
		// rebuilds of a dynamic slot gather.
		bool                         refit_change  = false;
		bool                         fast_built    = false;
		uint32_t                     gathers       = 0;
		uint64_t                     last_use_frame = 0;
		bool                         evicted        = false;
	};

	struct Retired {
		uint64_t                     tick       = 0;
		uint64_t                     prepare_id = 0;
		vk::AccelerationStructureKHR handle     = nullptr;
		std::array<Allocation, 4>    allocations {};
		uint32_t                     slot       = UINT32_MAX;
	};

	struct PendingSlot {
		uint64_t blas_va     = 0;
		uint32_t root        = 0;
		uint32_t table_index = 0;
		uint32_t leaf_count  = 0;
		uint32_t opaque      = 0;
		uint32_t alpha       = 0;
	};

	struct Readback {
		std::unique_ptr<Buffer> buffer;
		uint64_t                tick        = 0;
		uint32_t                build_count = 0;
		std::vector<uint32_t>   build_slots;
		uint32_t                source_key  = 0;
		uint64_t                frame       = 0;
	};

	void CreatePipelines();
	void CreateBuffers();
	void CreateTlas(uint32_t capacity);
	bool CreateSlot(const PendingSlot& pending);
	bool ComputeSlotSizes(uint32_t opaque, uint32_t alpha, SlotSizes& sizes, bool exact = false);
	void AllocateSlot(Slot& slot, const SlotSizes& sizes);
	[[nodiscard]] static uint64_t SlotBytes(const Slot& slot) noexcept;
	void ReportAgain(uint64_t blas_va, uint32_t root, uint32_t table_index);
	void RetireResources(Slot& slot, uint32_t slot_index);
	void DestroyRetired();
	void EvictUnused(std::vector<uint32_t>& evicted);
	void EvictOverBudget(std::vector<uint32_t>& evicted);
	void EvictSlot(uint32_t index);
	void RecordBuilds(vk::CommandBuffer command);
	void ProcessReadback(Readback& readback);
	void Report();

	GraphicContext&   m_graphics;
	CommandScheduler& m_scheduler;
	BufferCache&      m_buffer_cache;
	bool              m_enabled = false;
	bool              m_cleared = false;
	bool              m_context_valid = false;

	vk::DescriptorSetLayout m_descriptor_layout = nullptr;
	vk::PipelineLayout      m_tlas_layout       = nullptr;
	vk::PipelineLayout      m_blas_layout       = nullptr;
	vk::Pipeline            m_tlas_pipeline     = nullptr;
	vk::Pipeline            m_blas_pipeline     = nullptr;

	std::unique_ptr<Buffer> m_state;
	std::unique_ptr<Buffer> m_queue;
	std::unique_ptr<Buffer> m_instance_nodes;
	std::unique_ptr<Buffer> m_instances;
	std::unique_ptr<Buffer> m_table;
	std::unique_ptr<Buffer> m_slots;
	std::unique_ptr<Buffer> m_missing;
	std::unique_ptr<Buffer> m_context;
	std::unique_ptr<Buffer> m_pool;
	std::unique_ptr<Buffer> m_verify_list;
	std::unique_ptr<Buffer> m_changed;
	std::unique_ptr<Buffer> m_jobs;
	std::unique_ptr<Buffer> m_build_scratch;
	std::unique_ptr<Buffer> m_tlas_storage;
	vk::AccelerationStructureKHR m_tlas              = nullptr;
	vk::DeviceAddress            m_tlas_address      = 0;
	uint64_t                     m_tlas_scratch_size = 0;
	uint32_t                     m_tlas_capacity     = 0;
	uint64_t                     m_context_address   = 0;
	Arena                        m_arena;

	std::array<Readback, 32> m_readbacks;
	uint32_t                 m_readback_index = 0;

	Common::Mutex                        m_mutex;
	std::vector<PendingSlot>             m_new_slots;
	std::vector<std::unique_ptr<Slot>>   m_slot_list;
	std::unordered_map<uint64_t, std::vector<uint32_t>> m_slots_by_blas;
	std::vector<uint32_t>                m_build_queue;
	std::vector<uint32_t>                m_dynamic;
	std::vector<std::array<uint32_t, 3>> m_grow;
	std::vector<uint32_t>                m_retry;
	struct RetryState {
		uint32_t count       = 0;
		uint32_t table_index = 0;
	};
	std::unordered_map<uint64_t, RetryState>   m_retries;
	std::vector<std::pair<uint64_t, uint32_t>> m_late_retries;
	std::vector<uint32_t>                      m_free_slots;
	std::deque<Retired>                        m_retired;
	uint64_t                                   m_retired_bytes     = 0;
	uint64_t                                   m_budget            = 0;
	uint64_t                                   m_last_scan_prepare = 0;
	std::unordered_map<uint32_t, uint32_t>     m_last_instance_counts;
	uint32_t                                   m_prepare_id = 0;
	uint64_t                                   m_frame      = 0;

	// Statistics (KYTY_RT_HW_STATS=<prepares>, default 600): sums since the last report.
	struct Stats {
		uint64_t prepares = 0, instances = 0, inactive = 0, unknown = 0, builds = 0, gathers = 0,
		         changed = 0, missing = 0, overflow = 0, empty = 0, hinted = 0, reused = 0;
		std::array<uint64_t, CounterCount> counters {};
		uint64_t last_header = 0;
		uint32_t headers     = 0;
	} m_stats;
	uint64_t m_report_every = 600;
	// KYTY_RT_HW_CHECK_EVERY=<n> (default 8): each built slot's guest BLAS is compared with its
	// signature every n preparations (one n-th of the slots per preparation).
	uint32_t m_verify_every = 8;
	// The header and frame of the last preparation (KYTY_RT_HW_ONCE_PER_FRAME).
	uint64_t m_prepared_header = 0;
	uint64_t m_prepared_frame  = UINT64_MAX;
};

// KYTY_RT_HW on Windows: creates and destroys a ray tracing device before the guest address space is
// reserved, so the driver's runtime owns the low virtual addresses it needs (adapted from
// ace31d73d's PrepareHardwareRuntime). Returns whether a device was created.
bool PrepareNativeRayTracingRuntime(int gpu_index);

// Before a dispatch of a patched program (ShaderInfo::ray_query_sites): Prepare() for its TLAS, the
// user data pointer read from user_data.
void PrepareNativeRayTracingDispatch(NativeRayTracing& rt, vk::CommandBuffer command,
                                     const ShaderRecompiler::IR::ShaderInfo& info,
                                     std::span<const uint32_t> user_data, uint64_t frame,
                                     const std::function<bool(uint64_t, uint64_t)>& readable);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_NATIVERAYTRACING_H_
