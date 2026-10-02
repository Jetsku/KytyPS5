#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H

#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/command_processor/cpOps.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <span>
#include <vector>

namespace Libs::Graphics {

namespace DrawPrep {
class Engine;
} // namespace DrawPrep

namespace CpSeq {
class Verifier;
class Sequencer;
struct Intake;
struct RegisterState;
} // namespace CpSeq

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func);

// CP read of guest memory the GPU may have written: the clean backing when no GPU-owned byte
// overlaps, else after synchronizing those bytes (the drain a page fault would cause), else the
// mapped read with its page-fault readback path. GPU thread, at packet boundaries only.
void ReadGuestForCp(uint64_t vaddr, uint64_t size, void* dst);

template <typename T>
[[nodiscard]] T ReadGuestForCp(uint64_t vaddr) {
	T value {};
	ReadGuestForCp(vaddr, sizeof(T), &value);
	return value;
}

// A CPU write of guest memory by the command processor (labels, WRITE_DATA, DUMP_CONST_RAM, ...),
// after the bytes are written. With KYTY_CP_SEQ=1 draw preparations span such writes, so they are
// logged for the draw-prep certificate (Coherence::Source::CpWrite); otherwise a no-op.
void NoteCpWrite(uint64_t address, uint64_t size);

enum class Pm4ProcessResult { Complete, Blocked };

enum class ContextStateOperation : uint32_t {
	Clear     = 0,
	Push      = 1,
	Pop       = 2,
	PushClear = 3,
};

class Pm4Execution {
public:
	[[nodiscard]] bool MadeProgress() const noexcept { return m_made_progress; }
	// The last Process call stopped after a completed packet to let other queues run.
	[[nodiscard]] bool Yielded() const noexcept { return m_yielded; }

private:
	friend class CommandProcessor;
	friend class CpSeq::Verifier;

	struct BufferCursor {
		std::span<const uint32_t> commands;
		uint32_t                  offset_dw = 0;
		// KYTY_CP_SEQ=1, the sequencer's parse: the barrier epoch at which the rest of this buffer
		// was proven readable (0: not yet), and, when it had to be read in lockstep, the copy that
		// `commands` now points into and the guest address of the copy's first dword.
		uint64_t                               checked_epoch = 0;
		std::shared_ptr<std::vector<uint32_t>> copy;
		uint64_t                               guest_address = 0;
	};

	std::vector<BufferCursor> m_buffer_stack;
	std::span<const uint32_t> m_next_buffer;
	bool                      m_chain         = false;
	bool                      m_suspended     = false;
	bool                      m_made_progress = false;
	bool                      m_yield         = false; // stop after the current packet
	bool                      m_yielded       = false;
	// KYTY_CP_SEQ (cpOps.h): the stream this execution parses (assigned when it starts; stable
	// while the owning submission moves between queues), the packets completed in it, and with
	// KYTY_CP_SEQ_VERIFY the running hash of their dwords. m_verify = false: the reference front
	// does not follow this stream (constant-engine submissions).
	uint64_t m_stream_id    = 0;
	uint64_t m_packets      = 0;
	uint64_t m_packets_hash = 0;
	bool     m_verify       = true;

public:
	// KYTY_CP_SEQ_VERIFY: the reference front does not follow this stream.
	void DisableSequencerVerify() noexcept { m_verify = false; }
};

// P3 (cpOps.h): the packet handlers call the "front" methods below. Front state (the register
// files, context push state, user-data marker, index and indirect-argument state, the known
// instance count, DE/CE counters, constant RAM, flip info, predication state) belongs to the
// parse. Every effect beyond it is an op (CpSeq::OpKind) that the front hands to Submit(), and
// that the "back" executes with ExecuteOp()/Exec*(), which own the back state (submission id,
// flush batching, label deferral, idle flushes, slice draws, instance counts read back from guest
// memory, draw prep). KYTY_CP_SEQ=0 executes ops directly (FrontMode::Direct); =inline passes
// them through the op ring and executes them at once (FrontMode::Inline). A reference front
// (FrontMode::Reference, KYTY_CP_SEQ_VERIFY) only captures its ops for comparison.
class CommandProcessor {
public:
	struct FlipInfo {
		int     handle    = 0;
		int     index     = 0;
		int     flip_mode = 0;
		int64_t flip_arg  = 0;
	};

	// Thread: the graphics queue's front on the sequencer thread (KYTY_CP_SEQ=1); ops go to the
	// resolver through the op ring, lockstep ops wait for its answer.
	// Prefetch: P3c (KYTY_CP_SEQ_PREFETCH), a copy of the sequencer's front parsing past a wait
	// while the resolver catches up: it emits no op, publishes its draws to the draw-prep window
	// as speculative slots, and stops at anything it cannot parse without the resolver.
	enum class FrontMode : uint8_t { Direct, Inline, Reference, Thread, Prefetch };

	CommandProcessor(RenderContext& renderer, int interrupt_event_id);
	~CommandProcessor();

	KYTY_CLASS_NO_COPY(CommandProcessor);

	void Reset();
	void ApplyContextStateOperation(ContextStateOperation operation);

	void            BufferInit();
	void            BufferFlush();
	void            BufferFlushAndWait();
	static uint32_t EopFlushPacketLimit();
	void            BufferWait();
	// Early submit when the GPU ran out of submitted work (KYTY_IDLE_FLUSH_DRAWS). Call after
	// recording a draw or dispatch, at a packet boundary.
	void            MaybeFlushIdleGpu();
	// Graphics queue: after KYTY_GFX_SLICE_DRAWS draws in a slice, end the slice after the
	// current packet when another queue has runnable work. Call after recording a draw.
	void            MaybeYieldSlice();
	HW::Context&    GetCtx() { return m_ctx; }
	HW::UserConfig& GetUcfg() { return m_ucfg; }
	HW::Shader&     GetShCtx() { return m_sh_ctx; }

	void SetIndexType(uint32_t index_type_and_size);
	void SetIndexBaseAddress(uint64_t index_base_addr);
	void SetIndexBufferSize(uint32_t index_buffer_size);
	void SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr);
	void SetDispatchIndirectArgsBaseAddress(uint64_t dispatch_indirect_args_base_addr);
	[[nodiscard]] uint64_t GetDispatchIndirectArgsBaseAddress() const {
		return m_dispatch_indirect_args_base_addr;
	}
	void SetNumInstances(uint32_t num_instances);
	void DrawIndex(DrawIndexArgs args);
	void DrawIndexOffset(uint32_t index_offset, uint32_t index_count);
	void DrawIndexAuto(DrawAutoArgs args);
	// GET_LOD_STATS: the packet's 4 body dwords (the report, or its KYTY_LOD_STATS_MODE fallback).
	void GetLodStats(const uint32_t* body);
	void DrawIndirect(uint32_t data_offset, uint32_t draw_initiator, bool indexed);
	void DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
	                       const volatile uint32_t* count_addr, uint32_t stride_in_bytes,
	                       uint32_t draw_initiator, bool indexed);
	void WriteAtEndOfPipe32(uint32_t cache_policy, uint32_t event_write_dest,
	                        uint32_t eop_event_type, uint32_t cache_action, uint32_t event_index,
	                        uint32_t event_write_source, void* dst_gpu_addr, uint32_t value,
	                        uint32_t interrupt_selector, uint32_t interrupt_context_id = 0);
	void WriteAtEndOfPipe64(uint32_t cache_policy, uint32_t event_write_dest,
	                        uint32_t eop_event_type, uint32_t cache_action, uint32_t event_index,
	                        uint32_t event_write_source, void* dst_gpu_addr, uint64_t value,
	                        uint32_t interrupt_selector, uint32_t interrupt_context_id = 0);
	void Flip();
	void Flip(void* dst_gpu_addr, uint32_t value);
	void FlipWithInterrupt(uint32_t eop_event_type, uint32_t cache_action, void* dst_gpu_addr,
	                       uint32_t value);
	void PrepareCpuFlip(uint64_t request_id);
	void SynchronizeGpu();
	// RELEASE_MEM: the packet's 7 body dwords (barrier, dropped data, interrupt, end-of-pipe
	// write and flush batching, as the handler decoded them).
	void ReleaseMem(const uint32_t* body);
	void DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y, uint32_t thread_group_z,
	                    uint32_t mode);
	void DispatchIndirect(uint64_t args_addr, uint32_t mode);
	void WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index);
	void TriggerEvent(uint32_t event_type, uint32_t event_index, uint64_t event_address = 0);
	// COND_EXEC: whether the condition dword at `address` is non-zero (read in order).
	[[nodiscard]] bool CondExec(uint64_t address);
	// Conditional INDIRECT_BUFFER: whether the then-buffer is taken (the 64-bit value at
	// `compare_addr` compared as WAIT_REG_MEM does, read in order).
	[[nodiscard]] bool Branch(uint64_t compare_addr, uint64_t mask, uint64_t reference,
	                          uint32_t function);
	// SET_*_REG_INDIRECT: the register pairs (offset, value), read once, synchronized, before any
	// register is applied. Valid until the next call on this processor.
	[[nodiscard]] const uint32_t* ReadRegisterPairs(uint64_t address, uint32_t num_regs);

	void SetUserDataMarker(HW::UserSgprType type) { m_user_data_marker = type; }
	[[nodiscard]] HW::UserSgprType GetUserDataMarker() const { return m_user_data_marker; }

	void ResetDeCe();
	void SetCeComplete(bool complete);
	void WaitCe();
	void WaitDeDiff(uint32_t diff);
	void WaitForRewind(bool valid);
	void IncrementDe();
	void IncrementCe();

	void WriteConstRam(uint32_t offset, const uint32_t* src, uint32_t dw_num);
	void DumpConstRam(uint32_t* dst, uint32_t offset, uint32_t dw_num);

	template <typename T>
	void WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll, uint32_t wait_op);
	void WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num, uint32_t write_control);
	void WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes);
	void DmaData(uint8_t engine, uint8_t dst_sel, uint8_t dst_cache_policy,
	             uint64_t dst_address_or_offset, uint8_t src_sel, uint8_t src_cache_policy,
	             uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes,
	             uint8_t wait_for_previous, uint8_t write_confirm, uint8_t block_engine);
	void SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
	                    const volatile void* address, uint32_t count_in_dwords);
	[[nodiscard]] bool ShouldSkipPredicatedPackets() const { return m_predicate_skip; }
	// A predicated packet under a GPU predicate that conditional rendering cannot gate: takes the
	// GPU's decision on the CPU (m_predicate_skip) and ends the GPU predicate. False: suspended.
	[[nodiscard]] bool ResolveGpuPredicate();

	Pm4ProcessResult Process(Pm4Execution& execution, std::span<const uint32_t> commands);
	void             ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain);

	void SetFlip(const FlipInfo& flip) { m_flip = flip; }

	[[nodiscard]] uint64_t GetSubmitId() const { return m_submit_id; }
	void                   SetSubmitId(uint64_t submit_id) { m_submit_id = submit_id; }
	[[nodiscard]] bool     IsAsyncComputeQueue() const { return m_interrupt_event_id >= 0x20; }

	[[nodiscard]] FrontMode GetFrontMode() const noexcept { return m_front_mode; }
	// A reference front (KYTY_CP_SEQ_VERIFY) and a speculative one (KYTY_CP_SEQ_PREFETCH) only
	// parse: packet handlers skip their counters and diagnostics for them.
	[[nodiscard]] bool IsReferenceFront() const noexcept {
		return m_front_mode == FrontMode::Reference || m_front_mode == FrontMode::Prefetch;
	}

	// ---- P3b, KYTY_CP_SEQ=1 (cpSequencer.h) ----
	// GuestGpu, before any submission: the graphics processor's front moves to the sequencer's
	// thread. Creates the draw-prep engine (its producer is then the sequencer).
	void AttachSequencer(CpSeq::Sequencer* sequencer);
	[[nodiscard]] bool Sequenced() const noexcept { return m_sequencer != nullptr; }
	// Resolver (GPU thread): executes the ops of graphics submission `submission` (admission
	// sequence) until its end (Complete), a suspended op, a slice yield, or an empty op ring
	// (Blocked; `execution` tells yielded from suspended). `handoff`: the submission is to be
	// parsed and executed by this thread itself (constant engine): run it with Process() in
	// Direct mode, then EndHandoff().
	Pm4ProcessResult ResolveSubmission(Pm4Execution& execution, uint64_t submission,
	                                   std::span<const uint32_t> commands, bool& handoff);
	void             BeginHandoff();
	void             EndHandoff(uint64_t submission);
	// Tests: back to Direct once the sequencer is stopped and its ops are executed.
	void             DetachSequencer();

private:
	friend class CpSeq::Verifier;
	friend class CpSeq::Sequencer;

	// ---- P3: ops ----
	// The front hands an op to the back: executed directly (Direct), through the op ring
	// (Inline), or captured (Reference). Returns the op's result (lockstep ops).
	CpSeq::Result Submit(CpSeq::OpKind kind, const void* payload, uint32_t payload_size,
	                     const void* data = nullptr, uint32_t data_size = 0);
	template <typename T>
	CpSeq::Result Submit(const T& payload, const void* data = nullptr, uint32_t data_size = 0) {
		return Submit(CpSeq::KindOf<T>::Kind, &payload, static_cast<uint32_t>(sizeof(T)), data,
		              data_size);
	}
	CpSeq::Result SubmitInline(CpSeq::OpKind kind, const void* payload, uint32_t payload_size,
	                           const void* data, uint32_t data_size);
	// Thread mode: encodes the op for the resolver; a lockstep op waits for its result.
	CpSeq::Result SubmitThread(CpSeq::OpKind kind, const void* payload, uint32_t payload_size,
	                           const void* data, uint32_t data_size);
	// Sequencer thread: parses one admitted submission and emits its ops. False when stopping.
	[[nodiscard]] bool SequenceSubmission(const CpSeq::Intake& intake);
	// Sequencer thread: before a packet of `cursor`: the rest of its buffer may be parsed (no
	// pending CP write over it, and never GPU-touched, else it is read in lockstep).
	[[nodiscard]] bool CheckCommandBytes(Pm4Execution::BufferCursor& cursor);
	// Sequencer thread: waits until no pending CP write overlaps [begin, end). False: stopping.
	[[nodiscard]] bool AwaitPendingWrites(uint64_t begin, uint64_t end);
	// Sequencer thread: reads guest bytes the resolver may have to read in order (lockstep).
	[[nodiscard]] bool ReadGuestForFront(uint64_t address, uint64_t size, void* dst);
	// Sequencer thread: a snapshot of the front registers for an op; its index.
	[[nodiscard]] uint32_t TakeSnapshot();
	// Sequencer thread: waits for a free draw-prep window slot. False: stopping.
	[[nodiscard]] bool WaitForWindowSpace();
	// Sequencer thread: the oldest of `ops` still pending, as a wake threshold (op + 1).
	[[nodiscard]] uint64_t OldestPendingOp(std::deque<uint64_t>& ops) const;
	// The guest range a CP-write op writes (empty when none), for the pending-write set.
	[[nodiscard]] static bool CpWriteRange(CpSeq::OpKind kind, const void* payload,
	                                       uint64_t& begin, uint64_t& end);
	// Resolver: register state for an op with a snapshot (bound around its execution).
	[[nodiscard]] CpSeq::RegisterState* OpSnapshot(CpSeq::OpKind kind, const void* payload);

	// ---- P3c, KYTY_CP_SEQ_PREFETCH (cpOps.h) ----
	// Sequencer: the last end-of-pipe label's value, when it is plain data (NoteLastLabel), and
	// whether a WAIT_REG_MEM waits for exactly that label (the "wait for idle" idiom): no guest
	// CPU write can be ordered before such a wait, so draws after it may be prepared before it.
	void               NoteLastLabel(CpSeq::OpKind kind, const void* payload);
	[[nodiscard]] bool IsSelfLabelWait(const CpSeq::WaitRegMemOp& op) const;
	// Sequencer, at such a wait (lockstep op `wait_op` emitted, its answer pending): runs the
	// speculative front from the packet after the wait until the answer comes or it must stop.
	void RunPrefetch(uint64_t wait_op);
	// Sequencer: hands every speculative slot not adopted yet to the resolver to retire.
	void DropSpeculativeSlots();
	void EmitSkipSlots(uint64_t count);
	// Sequencer: a draw of the real parse takes the next speculative slot when its adoption key
	// matches (sets DrawFlagPublished and the window position); false: publish as usual.
	template <typename Op>
	[[nodiscard]] bool AdoptSpeculativeDraw(Op& op);
	// Both fronts after a wait: the packets and bytes consumed so far (the adoption key).
	void TrackAdoptPacket(const uint32_t* packet, uint32_t packet_dw, uint64_t guest_address);
	void TrackAdoptInputs(const void* data, uint64_t size, uint64_t guest_address);
	// The adoption key of the draw packet being parsed: the inputs hash with its bytes added.
	[[nodiscard]] uint64_t CurrentPacketKey() const;
	// Speculative front: the op's CP write is noted (bytes the parse must not read before it
	// executes); a lockstep op stops the parse (suspended).
	CpSeq::Result SubmitPrefetch(CpSeq::OpKind kind, const void* payload);
	template <typename Op>
	void               PrefetchDraw(const Op& op);
	[[nodiscard]] bool CheckCommandBytesPrefetch(Pm4Execution::BufferCursor& cursor);
	[[nodiscard]] bool ReadGuestForPrefetch(uint64_t address, uint64_t size, void* dst);
	[[nodiscard]] bool PrefetchOverlaps(uint64_t begin, uint64_t end) const;
	void               StopPrefetch(Profiler::FrameEvent reason);
	// The speculative front takes the sequencer's front state (not the constant RAM, which the
	// thread-mode stream never changes and draws never read).
	void CopyFrontStateForPrefetch(const CommandProcessor& from);
	CpSeq::Result ExecLockstepRead(const CpSeq::LockstepReadOp& op);
	// The back: executes one op with today's code (the direct path's bodies).
	CpSeq::Result ExecuteOp(CpSeq::OpKind kind, const void* payload, const void* data);
	void          ExecDrawIndex(const CpSeq::DrawIndexOp& op);
	void          ExecDrawAuto(const CpSeq::DrawAutoOp& op);
	void          ExecDrawIndirect(const CpSeq::DrawIndirectOp& op);
	void          ExecDrawIndirectMulti(const CpSeq::DrawIndirectOp& op);
	void          ExecDispatchDirect(const CpSeq::DispatchDirectOp& op);
	void          ExecDispatchIndirect(const CpSeq::DispatchIndirectOp& op);
	void          ExecEndOfPipe(const CpSeq::EndOfPipeOp& op);
	void          ExecReleaseMem(const CpSeq::ReleaseMemOp& op);
	void          ExecEventWrite(const CpSeq::EventWriteOp& op);
	void          ExecWriteData(const CpSeq::WriteDataOp& op, const uint32_t* src);
	void          ExecReferenceClock(const CpSeq::ReferenceClockOp& op);
	void          ExecDmaData(const CpSeq::DmaDataOp& op);
	void          ExecLodStats(const CpSeq::LodStatsOp& op);
	void          ExecFlip(const CpSeq::FlipOp& op);
	CpSeq::Result ExecWaitRegMem(const CpSeq::WaitRegMemOp& op);
	template <typename T>
	CpSeq::Result ExecWaitRegMemSized(const CpSeq::WaitRegMemOp& op);
	CpSeq::Result ExecWaitFlipDone(const CpSeq::WaitFlipDoneOp& op);
	void          ExecDumpConstRam(const CpSeq::DumpConstRamOp& op, const uint32_t* src);
	CpSeq::Result ExecPredication(const CpSeq::PredicationOp& op);
	CpSeq::Result ExecCondExec(const CpSeq::CondExecOp& op);
	CpSeq::Result ExecBranch(const CpSeq::BranchOp& op);
	// The front's state of an indirect draw packet.
	[[nodiscard]] CpSeq::DrawIndirectOp IndirectDrawOp(uint32_t data_offset,
	                                                   uint32_t draw_initiator, bool indexed);
	// Reference front: one step of the parse of `execution` (KYTY_CP_SEQ_VERIFY). Returns whether
	// the step left it suspended.
	[[nodiscard]] bool ReferenceStep(Pm4Execution& execution);
	// Reference front: takes the primary's front state at the start of a stream.
	void CopyFrontState(const CommandProcessor& from);
	// A back-only helper: GET_LOD_STATS with KYTY_LOD_STATS_MODE=gpu.
	void ReportLodStats(uint64_t destination, uint32_t size, uint32_t control);
	// Back-only helpers of RELEASE_MEM and EVENT_WRITE.
	void EmitGlobalBarrier();
	void TriggerEopEventAtEndOfPipe(uint32_t interrupt_context_id);
	// Flush requested by an end-of-pipe interrupt. Interrupts fire when their tick completes, so
	// only every KYTY_EOP_FLUSH_BATCH-th request flushes (default 8; 1 = flush every time); a
	// slice always ends with a flush, so a pending interrupt is submitted before the CP blocks.
	void BufferFlushForEop();

	template <typename T>
	void WriteAtEndOfPipe(uint32_t cache_policy, uint32_t event_write_dest, uint32_t eop_event_type,
	                      uint32_t cache_action, uint32_t event_index, uint32_t event_write_source,
	                      void* dst_gpu_addr, T value, uint32_t interrupt_selector,
	                      uint32_t interrupt_context_id);
	void ProcessPm4(Pm4Execution& execution);
	// One step of the parse: pops a finished buffer, or processes one packet. Returns false when
	// the parse stops there (the packet suspended, or the slice yields after it).
	[[nodiscard]] bool ProcessPacket(Pm4Execution& execution);
	void               SuspendPm4();
	// Defers an end-of-pipe label (and its interrupt) to the completion of the current tick when
	// a visibility-proxy dump armed it (KYTY_OCCLUSION_PROXY_MODE=defer-label), every label is
	// deferred (KYTY_LABEL_MODE=completion), or an older deferred label to the same address is
	// still pending (write order). Returns false when the label must be written now.
	// timestamp_slot: the label is a clock value with a KYTY_EOP_TIMESTAMPS=gpu query
	// (RecordEopTimestamp); a deferred write then writes the query's GPU time instead.
	[[nodiscard]] bool TryDeferLabel(void* dst, uint64_t value, uint32_t size, bool interrupt,
	                                 uint32_t interrupt_context_id,
	                                 uint32_t timestamp_slot = UINT32_MAX);
	// KYTY_EOP_TIMESTAMPS=gpu, at an end-of-pipe clock write (a fence: the draw-prep window is
	// empty): publishes the completed timestamps, then records a query at this point. Returns its
	// slot, or UINT32_MAX in record mode or without a free slot.
	[[nodiscard]] uint32_t RecordEopTimestamp();
	// The clock value `value` was just written to `dst` at record time: rewrite it with the
	// query's GPU time once the current tick completes (no-op without a slot).
	void QueueEopTimestamp(uint32_t slot, const void* dst, uint64_t value);
	// KYTY_GDS_EOP_MODE=defer: snapshot a GDS range at this packet and write it to `dst` at the
	// tick's completion instead of draining the GPU. Returns false for the synchronous path.
	[[nodiscard]] bool TryDeferGdsRead(uint32_t* dst, uint32_t dw_offset, uint32_t dw_size,
	                                   bool interrupt, uint32_t interrupt_context_id);
	// Writes (or defers) the data of an end-of-pipe event Kyty used to drop (counted under
	// `counter`). Returns true when a deferred write took over raising the interrupt.
	// timestamp: `value` is the reference clock (KYTY_EOP_TIMESTAMPS).
	[[nodiscard]] bool WriteDroppedLabel(void* dst, uint64_t value, uint32_t size, bool interrupt,
	                                     uint32_t interrupt_context_id,
	                                     Profiler::FrameEvent counter, bool timestamp = false);
	// RELEASE_MEM with INT_SEL=4 and DATA_SEL != 0 (previously no data was written): writes the
	// data (1: 32-bit, 2: 64-bit, 3: reference clock). Returns true when a deferred write took
	// over raising the interrupt (the caller then only flushes).
	[[nodiscard]] bool WriteReleaseMemDroppedData(void* dst, uint64_t value, uint32_t data_sel,
	                                              bool interrupt, uint32_t interrupt_context_id);

	// The source's index state (index_base_addr, index_buffer_size, index_type_and_size) is the
	// packet's (front state carried by the op).
	[[nodiscard]] bool  TryDrawIndirectNative(DrawIndirectSource source);
	void                ValidateIndirectSource(const DrawIndirectSource& source);
	[[nodiscard]] uint32_t NumInstances();
	// Draw-prep (drawPrep.h): the engine of the graphics processor, created on first use when
	// KYTY_DRAW_PREP (or the S0 histogram) is enabled; null otherwise.
	[[nodiscard]] DrawPrep::Engine* DrawPrepEngine();
	[[nodiscard]] bool TrySubmitPreparedDraw(const DrawIndexArgs* index_args,
	                                         const DrawAutoArgs*  auto_args);
	void               DrainPreparedDraws();
	void               NoteRepeatTraceDrawPacket();
	CommandScheduler&   GetScheduler() const { return m_renderer.GetCommandScheduler(); }
	CommandBuffer&      CurrentBuffer() { return GetScheduler().Current(); }

	RenderContext&   m_renderer;
	HW::Context      m_ctx;
	HW::Context      m_saved_ctx;
	bool             m_context_state_pushed = false;
	HW::UserConfig   m_ucfg;
	HW::Shader       m_sh_ctx;
	HW::UserSgprType m_user_data_marker                 = HW::UserSgprType::Unknown;
	uint32_t         m_index_type_and_size              = 0;
	uint32_t         m_index_buffer_size                = 0;
	uint64_t         m_index_base_addr                  = 0;
	uint64_t         m_draw_indirect_args_base_addr     = 0;
	uint64_t         m_dispatch_indirect_args_base_addr = 0;
	// Front view of the persistent instance count (KYTY_CP_SEQ inline/reference): the last
	// SET_NUM_INSTANCES value while no indirect draw followed it (known), which draws without
	// their own count use. After an indirect draw the count is back state (m_num_instances,
	// possibly GPU data): such draws inherit it (DrawFlagInheritInstances), and the next indirect
	// draw applies a newer SET_NUM_INSTANCES first (IndirectFlagSetInstances). Direct mode keeps
	// the back state current at every packet instead, as before.
	uint32_t m_front_num_instances   = 1;
	bool     m_front_instances_known = true;
	// SET_*_REG_INDIRECT pairs of the current packet (ReadRegisterPairs).
	std::vector<uint32_t> m_register_pairs;
	// Persistent draw state: indirect draws update it for subsequent draws.
	uint32_t m_num_instances = 1;
	// After native indirect draws the instance count is still GPU data: instance_count of the
	// last record drawn, which a draw without its own count reads back (NumInstances). With a
	// GPU count that may be zero, older sources remain candidates, newest last. Empty whenever
	// m_num_instances is current; SetNumInstances and CPU-read indirect draws clear it.
	struct PendingNumInstances {
		uint64_t args_addr  = 0;
		uint32_t stride     = 0;
		uint32_t max_count  = 0;
		uint64_t count_addr = 0;
		bool     has_expected = false; // KYTY_INDIRECT_VALIDATE: value read at the draw
		uint32_t expected     = 0;
	};
	std::vector<PendingNumInstances> m_pending_num_instances;

	uint32_t m_de_count    = 0;
	uint32_t m_ce_count    = 0;
	bool     m_ce_complete = false;

	uint32_t m_const_ram[0x3000] = {0};

	FlipInfo  m_flip;
	const int m_interrupt_event_id;
	uint64_t  m_submit_id                   = 0;
	uint64_t  m_synthetic_occlusion_counter = 0;
	bool      m_predicate_skip              = false;
	// KYTY_PREDICATION_MODE=gpu (gpuPredication.h): the active GPU predicate id (front state; 0:
	// none), and the id the packet being handled passes to its draw (0: not gated on the GPU).
	uint32_t  m_predicate_gpu               = 0;
	uint32_t  m_packet_predicate            = 0;
	uint32_t  m_deferred_eop_flushes        = 0;
	uint32_t  m_packets_since_eop_request   = 0;
	// A visibility-proxy end dump was recorded: defer the next end-of-pipe label (defer-label).
	bool      m_defer_next_label            = false;
	// The current WAIT_FLIP_DONE packet already flushed and suspended (retries skip the flush).
	bool      m_flip_wait_suspended         = false;
	// MaybeFlushIdleGpu: recording tick being counted and its draws/dispatches so far.
	uint64_t  m_idle_flush_tick             = 0;
	uint32_t  m_idle_flush_draws            = 0;
	// Draws recorded in the current graphics slice (MaybeYieldSlice).
	uint32_t  m_slice_draws                 = 0;

	struct DrawPrepDeleter {
		void operator()(DrawPrep::Engine* engine) const noexcept;
	};
	std::unique_ptr<DrawPrep::Engine, DrawPrepDeleter> m_draw_prep;
	// Packets processed, for the CP's placement samples (common/cpuPlacement.h).
	uint32_t m_placement_packets = 0;

	// ---- P3 (cpOps.h) ----
	FrontMode m_front_mode = FrontMode::Direct;
	// Inline: an op is being executed (ops never emit ops).
	bool      m_executing  = false;
	// Streams started on this processor (Pm4Execution::m_stream_id).
	uint64_t  m_next_stream_id = 0;
	struct OpStreamDeleter {
		void operator()(CpSeq::OpStream* stream) const noexcept;
	};
	struct VerifierDeleter {
		void operator()(CpSeq::Verifier* verifier) const noexcept;
	};
	// Inline: the op ring (created on the first op).
	std::unique_ptr<CpSeq::OpStream, OpStreamDeleter> m_ops;
	// KYTY_CP_SEQ_VERIFY: the reference front following this processor's streams.
	std::unique_ptr<CpSeq::Verifier, VerifierDeleter> m_verifier;
	// Reference front: the verifier its ops are captured by.
	CpSeq::Verifier* m_capture = nullptr;

	// ---- P3b, thread mode ----
	CpSeq::Sequencer* m_sequencer = nullptr;
	// Front (sequencer thread): a sync-epoch fence packet was parsed since the last op; the
	// barrier epoch (bumped after every lockstep wait, CheckCommandBytes); CP writes emitted but
	// maybe not executed yet (range, op sequence).
	bool     m_epoch_pending = false;
	uint64_t m_barrier_epoch = 1;
	struct PendingWrite {
		uint64_t begin = 0;
		uint64_t end   = 0;
		uint64_t op    = 0;
	};
	std::vector<PendingWrite> m_pending_writes;
	// Sequencer: the destination of the last end-of-pipe label it emitted (EndOfPipe/ReleaseMem),
	// which classifies the next WAIT_REG_MEM (FrameEvent CpSeqBarrierWaitSelfLabel), and, for P3c,
	// its data (known: plain data, and no other CP write was emitted since).
	uint64_t m_last_label_begin = 0;
	uint64_t m_last_label_end   = 0;
	uint64_t m_last_label_value = 0;
	uint32_t m_last_label_size  = 0;
	bool     m_last_label_known = false;
	// ---- P3c ----
	// Sequencer: the speculative front (created on first use). Speculative front: its owner, the
	// wait it runs for, the CP writes of the ops it did not emit, whether it must stop, and the
	// draws it published.
	struct PrefetchDeleter {
		void operator()(CommandProcessor* processor) const noexcept;
	};
	std::unique_ptr<CommandProcessor, PrefetchDeleter> m_prefetch;
	CommandProcessor*         m_prefetch_owner = nullptr;
	uint64_t                  m_prefetch_wait  = 0;
	std::vector<PendingWrite> m_prefetch_writes;
	bool                      m_prefetch_stop  = false;
	uint32_t                  m_prefetch_draws = 0;
	// Both fronts after a wait (P3c): the adoption key so far. skip_one: the wait packet itself
	// completes first and is not counted.
	bool     m_adopt_tracking = false;
	bool     m_adopt_skip_one = false;
	uint64_t m_adopt_packets  = 0;
	uint64_t m_adopt_inputs   = 0;
	// The draw ops of published window slots and the ops of taken snapshots, oldest first: the
	// op whose execution frees the next slot or snapshot (the sequencer's wake threshold).
	std::deque<uint64_t> m_published_ops;
	std::deque<uint64_t> m_snapshot_ops;
	// Resolver (GPU thread): the register files bound outside ops with a snapshot; the packet
	// count of the last op (EOP flush batching); a suspended op that is retried (sequence + 1;
	// the verifier compared it at its first execution).
	HW::Context    m_back_ctx;
	HW::UserConfig m_back_ucfg;
	HW::Shader     m_back_sh;
	uint64_t       m_resolver_packets = 0;
	uint64_t       m_retry_op         = 0;
	// Process(): the draw-prep packet hook runs for this slice's packets (decided per slice).
	bool m_packet_hook = false;
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H
