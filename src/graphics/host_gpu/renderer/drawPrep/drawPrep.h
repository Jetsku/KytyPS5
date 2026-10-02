#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_DRAWPREP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_DRAWPREP_H_

#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/drawPrep/packetClass.h"
#include "graphics/host_gpu/renderer/drawPrep/readSet.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/render.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

// Draw-prep S5/S6: prepare-then-commit for direct draws.
//
// KYTY_DRAW_PREP=off (default): nothing here runs; the command processor executes draws exactly
// as before.
// KYTY_DRAW_PREP=inline (S5): each direct draw snapshots the registers, is prepared on the
// command processor (GPU) thread from that snapshot with every guest read recorded, and is then
// committed: the command buffer reads the snapshot, the certificate is checked where the serial
// path would call GetGraphicsPrograms, and the prepared programs are used when it holds (the
// serial preparation runs otherwise). This is the whole protocol without threads.
// KYTY_DRAW_PREP=parallel (S6): the command processor keeps parsing while DrawPrep workers
// prepare the published draws of a window; packets that could change what a later draw reads
// (fences) first commit the whole window in guest order.
//
// KYTY_DRAW_PREP_VERIFY=1|exit: after every committed preparation the serial preparation runs
// on copies and the outputs are compared (logged and counted; "exit" stops on a difference).
// KYTY_DRAW_PREP_CERT=log (default)|value: see Validate() in drawPrep.cpp.
// KYTY_DRAW_PREP_CERT_RANGES=worker (default)|commit: where the coherence-log check's merged
// certificate ranges are built (ReadSet::BuildCertificate). "worker": by the preparing thread,
// once, after the read set is finished; "commit": by the command processor in Validate, as
// before (0.41 ms/flip of CP time at the Sky Garden start, U55). The list is a pure function of
// the finished read set either way. KYTY_DRAW_PREP_CERT_RANGES_VERIFY=1|exit rebuilds it at
// commit with the commit-time function and compares (DrawPrepCertRangesVerify* events).
// KYTY_DRAW_PREP_WORKERS (default 6, 1..32), KYTY_DRAW_PREP_WINDOW (default 32 slots, rounded
// up to a power of two), KYTY_DRAW_PREP_SPIN_US (default 200: how long an idle hot worker spins
// before parking): parallel mode only.
// KYTY_DRAW_PREP_HOT (default 2): workers that spin like that; the others park as soon as nothing
// is claimable and are woken, one at a time, when KYTY_DRAW_PREP_WAKE_BACKLOG (default 2; 8 until
// U54) published slots wait unclaimed; a woken one spins KYTY_DRAW_PREP_COLD_SPIN_US (default 50)
// before parking again (workerGate.h). KYTY_DRAW_PREP_HOT >= KYTY_DRAW_PREP_WORKERS keeps every
// worker hot, the behaviour before the gate.
// KYTY_DRAW_PREP_STEAL (default 0 = off): N >= 1 lets the command processor, while a worker holds
// the head slot it must commit, prepare later unclaimed slots itself, exactly as a worker does,
// as long as at least N slots wait unclaimed; KYTY_DRAW_PREP_STEAL_AFTER_US (default 0) first
// spins that long on the held head (Engine::CommitHead, workerGate.h AwaitHead).
// KYTY_DRAW_PREP_HISTOGRAM=1: the S0 draws-per-fence histogram also in off mode (the packet
// classification runs, nothing else changes).
namespace Libs::Graphics {

class RenderContext;
struct ShaderVertexInputInfo;
struct ShaderPixelInputInfo;

namespace DrawPrep {

enum class Mode : uint8_t { Off, Inline, Parallel };
enum class CertMode : uint8_t { Value, Log };

[[nodiscard]] Mode     GetMode();
[[nodiscard]] int      VerifyMode(); // 0 off, 1 count/log, 2 exit on difference
[[nodiscard]] CertMode GetCertMode();
// KYTY_DRAW_PREP_CERT_RANGES: the preparing thread builds the log check's certificate ranges.
[[nodiscard]] bool CertRangesOnWorker();
// KYTY_DRAW_PREP_CERT_RANGES_VERIFY: 0 off, 1 count/log, 2 exit on difference.
[[nodiscard]] int CertRangesVerifyMode();
// The command processor's per-packet hook (window fences and the S0 histogram) is needed.
[[nodiscard]] bool PacketHookEnabled();
// KYTY_DRAW_PREP_REG_INDIRECT_WINDOW (default on; =0 keeps them fences): a SET_*_REG_INDIRECT
// packet whose register pairs are clean for a backing read leaves the preparation window open.
// Its handler then only reads those clean bytes (no synchronization, nothing recorded) and
// writes command-processor registers, which pending draws do not read (they use their register
// snapshots). The bytes read are those the serial path reads unless a pending draw (earlier in
// the stream, not yet recorded) writes the pairs with its shaders: a readback publication only
// writes bytes that are GPU-dirty or being published now, and the pairs are neither. Without a
// guest wait between that draw and the load (a wait, acquire or end-of-pipe packet, which is a
// fence and commits the window first) the hardware command processor races the draw in the same
// way: it fetches the pairs when it reaches the packet, not after earlier draws completed. Guest
// CPU writes that race the load race it on the hardware as well.
[[nodiscard]] bool RegisterIndirectWindowEnabled();
// PacketHookEnabled(), or the passive S0/fence-kind histogram in off mode: aggregate diagnostics
// with a connected profiler and KYTY_DRAW_PREP_FENCE_HISTOGRAM=1 (opt-in). The hook changes
// nothing in off mode; it only classifies packets and counts.
[[nodiscard]] bool PacketHookActive();

// The register state a draw reads, copied when the draw packet is parsed.
struct RegisterSnapshot {
	HW::Context    context;
	HW::UserConfig user_config;
	HW::Shader     shaders;
};

enum class Failure : uint8_t {
	None,
	Ineligible,
	Unclean,
	Backing,
	Overflow,
	Inconsistent,
	Uncertified,
	NotPublished,
	ShaderMap,
	CertUnclean,
	CertChanged,
	CoherenceLog,
	Mismatch,
};

// Process-wide totals of the engine's decisions, always counted (relaxed; written on the GPU
// thread). The DrawPrep* frame events only count with a connected profiler; tests read these.
struct Totals {
	std::atomic<uint64_t> committed {0};              // Validate accepted a preparation
	std::atomic<uint64_t> fallbacks {0};              // Validate rejected one (serial preparation)
	std::atomic<uint64_t> drains {0};                 // Drain calls that found pending draws
	std::atomic<uint64_t> register_indirect_kept {0}; // SET_*_REG_INDIRECT packets kept in a window
	std::atomic<Failure>  last_failure {Failure::None};
	// KYTY_DRAW_PREP_CERT_RANGES: log checks that used the preparing thread's certificate ranges,
	// and KYTY_DRAW_PREP_CERT_RANGES_VERIFY comparisons and differences.
	std::atomic<uint64_t> cert_ranges_prebuilt {0};
	std::atomic<uint64_t> cert_ranges_verify_checks {0};
	std::atomic<uint64_t> cert_ranges_verify_mismatches {0};
	// KYTY_CP_SEQ_PREFETCH (P3c): slots published by a speculative parse, adopted by the real
	// parse, and retired unused (SkipSlots).
	std::atomic<uint64_t> prefetch_published {0};
	std::atomic<uint64_t> prefetch_adopted {0};
	std::atomic<uint64_t> prefetch_skipped {0};
};
[[nodiscard]] Totals& GetTotals();

// One draw's speculative preparation and its certificate. Reused across draws (vectors keep
// their capacity).
struct PreparedDraw {
	bool                                                ok      = false;
	Failure                                             failure = Failure::None;
	bool                                                pixel_active = false;
	std::array<Prospero::ColorComponentMapping, 8>      target_export_mapping {};
	PipelineCache::GraphicsPrograms                     programs;
	ShaderVertexInputInfo                               vertex_info;
	ShaderPixelInputInfo                                pixel_info;
	PipelineCache::StagePrep                            vertex_prep;
	PipelineCache::StagePrep                            pixel_prep;
	ReadSet                                             reads;
	uint64_t                                            coherence_generation  = 0;
	uint64_t                                            shader_map_generation = 0;
	// KYTY_DRAW_PREP_CERT_RANGES=worker: reads.BuildCertificate() of a successful preparation,
	// built by the preparing thread (certificate_built); Validate uses it for the log check.
	std::vector<Coherence::Range> certificate;
	bool                          certificate_built = false;
};

// Whether the draw path would reach its program preparation for a draw with these registers and
// counts (renderDraw.cpp: the early returns before PrepareDrawRenderState, with target operations
// as a superset). Only such draws are prepared, so a preparation never evaluates registers the
// serial path would not have evaluated for that draw.
[[nodiscard]] bool DrawReachesPrograms(const HW::Context& context, const HW::UserConfig& user_config,
                                       const HW::Shader& shaders, uint32_t count,
                                       uint32_t instance_count);

// The pure preparation of one draw from a register snapshot. `exact`: the caller is the GPU
// thread (exact clean predicate for reads); otherwise a DrawPrep worker. Never touches the
// texture/buffer caches, the scheduler or Vulkan, and never dereferences guest memory (every
// read goes through the recorder). `eligible`: DrawReachesPrograms for the draw; otherwise the
// preparation is skipped (Ineligible).
void Prepare(PipelineCache& pipeline_cache, const RegisterSnapshot& registers, bool eligible,
             bool exact, PreparedDraw& prepared);

// GPU thread, at the point where the serial path would prepare the programs: whether the
// prepared outputs equal what the serial preparation would produce now. Counts the committed
// draw or the fallback reason.
[[nodiscard]] bool Validate(PreparedDraw& prepared, bool pixel_active,
                            std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping);

// KYTY_DRAW_PREP_VERIFY: compares committed outputs with a serial preparation's (the pixel stage
// only when it is active). Returns true when equal; otherwise counts, logs and (exit mode) stops.
bool VerifyCommitted(bool pixel_active, const PipelineCache::GraphicsPrograms& programs,
                     const ShaderVertexInputInfo& vertex_info, const ShaderPixelInputInfo& pixel_info,
                     const PipelineCache::GraphicsStagePreps& preps,
                     const PipelineCache::GraphicsPrograms& serial_programs,
                     const ShaderVertexInputInfo&            serial_vertex_info,
                     const ShaderPixelInputInfo&             serial_pixel_info,
                     const PipelineCache::GraphicsStagePreps& serial_preps);

enum class DrawKind : uint8_t { Index, Auto };

// Owned by the graphics command processor; its methods run only on the GPU thread.
//
// Parallel mode (S6): Submit publishes the draw (register snapshot + resolved arguments) into
// the window, where KYTY_DRAW_PREP_WORKERS DrawPrep#k threads claim and prepare it. The command
// processor keeps parsing. Drain commits every pending draw in guest order; the command processor
// drains before every fence packet, before servicing commands from other threads, when a
// draw's instance count may be GPU data, and at the end of every command-stream slice. A head
// slot no worker has claimed yet is prepared by the command processor itself. For a claimed one,
// the command processor can first prepare later unclaimed slots as a worker would
// (KYTY_DRAW_PREP_STEAL, off by default). Then it waits, spinning; only after 2 ms, as a deadlock
// guard, does the wait service commands from other threads. Draws are committed in order either way. Workers
// never touch the caches, the scheduler or Vulkan, and never dereference guest memory; a failed
// preparation or certificate runs the serial preparation at commit.
class Engine {
public:
	// service_commands runs the GPU thread's pending cross-thread commands (used while waiting).
	// after_commit runs after every committed draw has been recorded, with the command buffer
	// bound to the live registers again (the command processor's per-draw hooks, e.g. the
	// idle-GPU early submit). Commits happen at packet boundaries (before a fence packet or a
	// service command, at the end of a slice) or at the start of a draw packet, before the
	// packet records anything, so a flush there splits no packet.
	Engine(RenderContext& renderer, std::function<void()> service_commands,
	       std::function<void()> after_commit);
	~Engine();
	Engine(const Engine&)            = delete;
	Engine& operator=(const Engine&) = delete;

	// Takes a direct draw whose arguments the command processor has fully resolved. Returns
	// false when the caller must execute the draw serially now (off mode).
	[[nodiscard]] bool Submit(uint64_t submit_id, const DrawIndexArgs* index_args,
	                          const DrawAutoArgs* auto_args, const HW::Context& context,
	                          const HW::UserConfig& user_config, const HW::Shader& shaders);
	// Commits every pending draw in submission order.
	void Drain();
	[[nodiscard]] bool Pending() const noexcept;

	// P3b (KYTY_CP_SEQ=1, cpSequencer.h): the sequencer thread publishes each direct draw (the
	// window's producer) and the resolver commits it when it executes the draw's op, so the
	// window spans every packet but the ordering points. Parallel mode only.
	[[nodiscard]] bool Parallel() const noexcept { return m_workers != nullptr; }
	// Sequencer: publishes a draw from the front's registers and returns its window position.
	// While the window is full it calls `wait_for_space`, which returns false when the caller
	// stops (then nothing is published and the result is UINT64_MAX). The submission id and an
	// inherited instance count are only known at commit (CommitPublished).
	[[nodiscard]] uint64_t Publish(const DrawIndexArgs* index_args, const DrawAutoArgs* auto_args,
	                               const HW::Context& context, const HW::UserConfig& user_config,
	                               const HW::Shader& shaders,
	                               const std::function<bool()>& wait_for_space);
	[[nodiscard]] bool WindowHasSpace() const noexcept;
	// Sequencer, diagnostics: it passed a barrier (kind 1, a lockstep op) or another stop (kind 2:
	// a submission start, the frame fence, a handoff, a pending CP write); the next published draw
	// is marked as the first after it (FrameEvent DrawPrepCommitWaitsBarrier/Start,
	// CpSeqBarrierDrawBursts). A barrier outranks another stop.
	void NoteStop(uint8_t kind) noexcept {
		if (m_stop_pending != 1) {
			m_stop_pending = kind;
		}
	}
	// Resolver: commits the draw at window position `position` (the head), recorded with
	// `submit_id` and, unless UINT32_MAX, `instance_count` (resolved in order).
	void CommitPublished(uint64_t position, uint64_t submit_id, uint32_t instance_count);

	// P3c (KYTY_CP_SEQ_PREFETCH, cpOps.h): the sequencer's speculative parse past a wait publishes
	// draws without ops. Each such slot carries the number of packets and the hash of every byte
	// the speculative parse consumed from the wait up to and including the draw packet
	// (`packets`, `inputs`). The front's state is a deterministic function of its state at the wait
	// and those bytes, so the real parse's draw with the same packets and hash has the same
	// registers and arguments, and the slot's preparation is the one it would publish.
	// Sequencer: publishes a speculative slot unless the window is full (false, nothing done).
	[[nodiscard]] bool PublishSpeculative(const DrawIndexArgs* index_args,
	                                      const DrawAutoArgs* auto_args, const HW::Context& context,
	                                      const HW::UserConfig& user_config,
	                                      const HW::Shader& shaders, uint64_t packets,
	                                      uint64_t inputs);
	// Sequencer: speculative slots not adopted or dropped yet.
	[[nodiscard]] uint64_t SpeculativeSlots() const noexcept;
	enum class Adoption : uint8_t { None, Adopted, Dropped };
	// Sequencer, before publishing a draw: None (no speculative slot), Adopted (the next one has the
	// draw's packets and inputs: `position` is its window position, publish nothing), or Dropped
	// (it differs: every speculative slot is given up, `dropped` of them, which the caller hands
	// to the resolver as a SkipSlots op before it publishes the draw).
	[[nodiscard]] Adoption TryAdopt(uint64_t packets, uint64_t inputs, uint64_t& position,
	                                uint64_t& dropped);
	// Sequencer: gives up every speculative slot (a barrier, the stream end); returns how many.
	[[nodiscard]] uint64_t DropSpeculative();
	// Resolver: retires `count` head slots without drawing (their SkipSlots op).
	void SkipPublished(uint64_t count);

	// Per-packet hook of the command processor (before the packet's handler runs). fence_kind
	// only matters for a fence (counted as FrameEvent DrawPrepFence<kind>).
	void OnPacket(PacketClass packet_class, FenceKind fence_kind = FenceKind::Other);

	struct Slot;

private:
	struct Workers;

	void Commit(Slot& slot);
	// `patch` (P3b): applied to the head once no other thread works on it, before its commit.
	void CommitHead(const std::function<void(Slot&)>* patch = nullptr);
	void NoteFence();
	void FillSlot(Slot& slot, uint64_t submit_id, const DrawIndexArgs* index_args,
	              const DrawAutoArgs* auto_args, const HW::Context& context,
	              const HW::UserConfig& user_config, const HW::Shader& shaders);

	RenderContext&           m_renderer;
	Mode                     m_mode;
	std::function<void()>    m_service_commands;
	std::function<void()>    m_after_commit;
	std::unique_ptr<Slot>    m_inline_slot;
	std::unique_ptr<Workers> m_workers; // parallel mode: the window and its threads
	uint64_t                 m_draws_since_fence = 0;
	uint8_t                  m_stop_pending      = 0; // sequencer only (NoteStop)
	// P3c, sequencer only: the window position of the real parse's next draw; the slots from here
	// to the window's tail are speculative.
	uint64_t                 m_real_tail = 0;
};

} // namespace DrawPrep
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_DRAWPREP_H_
