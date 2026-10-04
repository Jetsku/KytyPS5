#ifndef KYTY_RENDERER_OCCLUSION_H_
#define KYTY_RENDERER_OCCLUSION_H_

#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/host_gpu/renderer/occlusionReset.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace Libs::Graphics {
class Buffer;
class RenderContext;

// GPU-resident cumulative counter. A guest dump is reduced by the GPU into a private
// host-visible slot and published into guest memory by a completion callback once that GPU
// work has finished, so results (including the ready bit) appear no earlier than on hardware.
// Publishing from the host keeps the dump pages CPU-owned: the game keeps EOP labels on the
// same pages, and GPU-owned dump pages made every label write fault and drain the GPU.
class OcclusionCounter {
public:
	explicit OcclusionCounter(RenderContext& context);
	~OcclusionCounter();
	static bool Enabled();
	void Prepare(uint32_t control); // outside rendering
	void Begin();                  // after beginning guest rendering
	void End();                    // before ending guest rendering
	void Accumulate();             // after ending rendering; flush only when pool is full
	// KYTY_OCCLUSION_BATCH=1 (default off, needs KYTY_GPU_OCCLUSION=1). Without it every dump
	// reduces the pending queries and writes its publish slot at once: a query copy, up to two
	// dispatches and four full barriers per dump, recorded through Handle() (a recorder drain each
	// with KYTY_CP_RECORDER). Creamy Canyon's start view makes ~9,400 dumps per frame (one
	// depth-only proxy query per pair), which cost ~80 of its ~98 GPU ms per frame. With the batch
	// a dump only records how many of the pending queries precede it (its prefix, in a host-visible
	// per-slot table); one dispatch (gpu_dcc_occlusion_batch.comp) then reduces the queries and
	// writes the publish slots of every dump of the command buffer, right before the scheduler
	// submits it (CommandScheduler pre-submit hook), or earlier when the query pool fills. Each
	// dump still ends rendering, keeps its own slot and completion tick (the current tick, as
	// before) and its own publication registered at the dump, in the same FIFO order with deferred
	// labels, so the guest reads the same values at the same points.
	[[nodiscard]] static bool BatchEnabled();
	// KYTY_OCCLUSION_BATCH=verify: the batch, plus the per-dump reduction of every dump into a
	// second set of slots (on a copy of the query results, with its own counter); each publication
	// compares the 16 values it publishes with that reduction's and reports mismatches. Slow (the
	// per-dump GPU work comes back); for checking only.
	[[nodiscard]] static bool BatchVerifyEnabled();
	// KYTY_OCCLUSION_SLOTS=<n> (default 1024, 16..262144): publish slots (256 host-visible bytes
	// each). A dump reuses a slot only after the GPU work and the publication of its previous dump
	// completed (the CP waits otherwise), so more slots let the CP run further ahead of the GPU.
	[[nodiscard]] static uint32_t SlotCount();
	// Records the batched reductions (the pre-submit hook; also when the query pool is full).
	void FlushBatch();
	// Publications whose slot did not carry their dump's tag (must stay 0).
	[[nodiscard]] uint64_t BatchMismatches() const noexcept {
		return m_batch_mismatches.load(std::memory_order_relaxed);
	}
	// The value the batched reduction writes for each dump, given the counter before the batch,
	// the batch's query results and each dump's prefix (test reference for the shader; equal to
	// the per-dump reductions' values).
	static void ReferenceBatch(uint64_t counter, const std::vector<uint64_t>& results,
	                           const std::vector<uint32_t>& prefixes, std::vector<uint64_t>& values,
	                           uint64_t& counter_after);
	// Visibility-proxy handling (an end dump closing a depth-only scope, whose result Astro Bot
	// reads right after the next end-of-pipe label). KYTY_OCCLUSION_PROXY_MODE:
	//   defer-label (default): the CP continues; its next label write is deferred until this
	//                          dump's tick has completed and its result has been published.
	//   sync:                  the CP waits for the publication (BufferWait), as before.
	// KYTY_OCCLUSION_SYNC_PROXY=0 disables proxy handling (plain asynchronous publication).
	enum class ProxyMode { Off, Sync, DeferLabel };
	[[nodiscard]] static ProxyMode GetProxyMode();
	// Returns true when this dump is a proxy end dump the caller must order (see ProxyMode).
	[[nodiscard]] bool Dump(uint64_t address);
	// Proxy detection enabled (any mode but Off).
	[[nodiscard]] static bool SyncProxyDumps();
	// Publications run on the completion (priority) runner instead of the GPU thread's pending
	// operations: required whenever labels may be deferred, so a deferred label, registered later
	// on the same FIFO runner, is written only after the result it announces.
	[[nodiscard]] static bool PriorityPublication();
	// True while a dump has been recorded but not yet published to guest memory.
	[[nodiscard]] bool HasUnpublishedDumps() const noexcept {
		return m_published.load(std::memory_order_acquire) != m_issued;
	}
	[[nodiscard]] bool Active() const noexcept { return m_active; }
	// KYTY_OCCLUSION_GATE (default on, needs KYTY_GPU_OCCLUSION=1). The guest reads only
	// end - begin differences of the cumulative counter, taken by interleaved dump pairs (begin at
	// A, A % 16 == 0; end at A + 8). Samples of rendering instances begun while no pair is open
	// cannot reach any such difference, so they are not counted. Every dump ends rendering, so
	// an instance never straddles a pair boundary. Unexpected dump patterns disable the gate for
	// the rest of the process (always-on counting, as without the gate).
	[[nodiscard]] static bool GateEnabled();
	// True when a rendering instance begun now with DB_COUNT_CONTROL `control` would be counted.
	[[nodiscard]] bool WouldCount(uint32_t control) const noexcept;
	// Hang-trace diagnostics: the depth target of the latest counted scope, reported with the
	// next dump.
	void NoteScope(uint64_t depth_address, uint32_t width, uint32_t height, uint32_t colors,
	               bool has_depth, uint32_t depth_format) {
		m_last_scope = {depth_address, width, height, colors, has_depth, depth_format};
	}
private:
	static constexpr uint32_t DefaultPublishSlots = 1024;
	static constexpr uint64_t PublishSlotSize     = 256;
	static constexpr uint64_t TableEntrySize      = 8; // batch table: prefix and tag per slot
	void Initialize();
	void InitializeBatch();
	void FlushPending();
	void Dispatch(uint32_t mode, vk::Buffer output, uint64_t offset, uint64_t range);
	static void PreSubmit(void* context);
	[[nodiscard]] static bool ControlCounts(uint32_t control) noexcept {
		return (control & 1u) == 0 && (control & 0xf00u) != 0;
	}
	[[nodiscard]] bool GateOpen() const noexcept {
		return !GateEnabled() || m_gate_broken || !m_open_pairs.empty();
	}
	void UpdateOpenPairs(uint64_t address);
	void BreakGate(const char* reason, uint64_t address);
	static constexpr size_t MaxOpenPairs = 64;
	std::vector<uint64_t> m_open_pairs; // begin addresses of dump pairs awaiting their end
	bool m_gate_broken = false;
	static constexpr uint32_t QueryCapacity = 1024;
	RenderContext& m_context;
	vk::QueryPool m_pool;
	vk::DescriptorSetLayout m_descriptors;
	vk::PipelineLayout m_layout;
	vk::Pipeline m_pipeline;
	std::unique_ptr<Buffer> m_counter;
	std::unique_ptr<Buffer> m_result;
	std::unique_ptr<Buffer> m_publish;
	uint32_t m_slot_count = DefaultPublishSlots;
	OcclusionResetWindow m_reset_window;
	std::vector<uint64_t> m_slot_ticks;
	// KYTY_OCCLUSION_BATCH: the batch reduction pipeline, the per-slot prefix table and the dumps
	// recorded since the last batch (consecutive slots from m_batch_first_slot).
	vk::DescriptorSetLayout m_batch_descriptors;
	vk::PipelineLayout m_batch_layout;
	vk::Pipeline m_batch_pipeline;
	std::unique_ptr<Buffer> m_prefix;
	uint32_t m_batch_first_slot = 0;
	uint32_t m_batch_count = 0;
	std::atomic<uint64_t> m_batch_mismatches {0}; // publications whose slot tag did not match
	// KYTY_OCCLUSION_BATCH=verify: the per-dump reduction's buffers, the queries it has reduced
	// so far in this batch, and the publications checked / found different.
	std::unique_ptr<Buffer> m_verify_counter;
	std::unique_ptr<Buffer> m_verify_result;
	std::unique_ptr<Buffer> m_verify_publish;
	uint32_t m_verified = 0;
	std::atomic<uint64_t> m_verify_checks {0};
	std::atomic<uint64_t> m_verify_mismatches {0};
	void VerifyReduce();
	void VerifyDispatch(uint32_t mode, vk::Buffer output, uint64_t offset, uint64_t range, uint32_t count);
	uint64_t m_issued = 0;
	std::atomic<uint64_t> m_published {0};
	bool m_prepared = false;
	bool m_active = false;
	uint32_t m_pending = 0;
	uint32_t m_scopes_since_dump = 0; // hang-trace diagnostics only
	struct ScopeInfo {
		uint64_t depth_address = 0;
		uint32_t width         = 0;
		uint32_t height        = 0;
		uint32_t colors        = 0;
		bool     has_depth     = false;
		uint32_t depth_format  = 0;
	};
	ScopeInfo m_last_scope {}; // hang-trace diagnostics only
};
}
#endif
