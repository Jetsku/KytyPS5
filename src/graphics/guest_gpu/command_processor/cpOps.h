#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPOPS_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPOPS_H_

// P3: the command processor split into a front (the PM4 parse) and a resolver (the back), design
// and exactness argument in Profiling/analysis/P3-SEQUENCER.md.
//
// The front parses PM4 with the packet handlers and owns the command processor's register state
// (CommandProcessor "front state"). Every effect of a packet outside that state (a draw, a
// dispatch, a label, a wait, an event, a data write, a flip, ...) is an op: a kind and a payload
// that carries every front value the effect needs. The resolver executes the ops in stream order
// with the command processor's own code (CommandProcessor::Exec*), which owns the "back state"
// (submission ids, flush batching, label deferral, instance counts read from guest memory, ...).
//
// KYTY_CP_SEQ=0 (default) | inline | 1
//   0:      the front executes each op's body directly (CommandProcessor::ExecuteOp) where its
//           packet handler asks for it: the path before P3, same code.
//   inline: every op is encoded into the op ring (OpStream) and the resolver decodes and executes
//           it at once, on the same thread (P3a: the transport without a second thread). Effects
//           keep the position of the direct path; only their arguments travel through the ring.
//   1:      (P3b, also "thread") the graphics queue's front runs on its own thread, CpSequencer
//           (cpSequencer.h), and the resolver is the GPU thread (Thread_Gpu), which executes the
//           ops of each graphics submission in order. The sequencer stops at every ordering point
//           (waits, predication, conditions, register reads it cannot prove clean) until the
//           resolver has executed it. Compute queues keep the direct path.
// KYTY_CP_SEQ_VERIFY=1|exit (with inline): a reference front, a second CommandProcessor in
//   reference mode, parses the same PM4 streams in lockstep with the resolver (CpSeq::Verifier).
//   Before the resolver executes an op it runs the reference to its next op and compares: kind,
//   packet position, payload and inline data, and the hash of every packet dword the front parsed
//   before it. Guest reads of the front itself (register-indirect pairs) travel as ReadCheck ops
//   with a hash of the bytes, which the reference repeats with the serial read (ReadGuestForCp).
//   Ops whose result decides the parse (lockstep: waits, predication, COND_EXEC, branches) are
//   executed once, by the resolver, and the result is handed to the reference. Differences count
//   FrameEvent CpSeqVerifyMismatches and are logged; "exit" stops at the first. When the two
//   fronts read different guest bytes (a ReadCheck, or command bytes: another packet count or
//   packets hash before an op), only a guest race can explain it (class E2): that counts
//   CpSeqVerifyReadDivergences, is logged, and does not stop "exit". The front's register state
//   is a deterministic function of the parsed bytes and the reads, so equal inputs and ops mean
//   equal register state.
//
// Op records (OpStream): an OpHeader, the fixed payload of its kind, then inline data. Records
// are 8-byte aligned and never straddle the ring's end (CommandStream::Ring pads with a Wrap
// record). Payloads are trivially copyable, have no padding bytes (compared and hashed as bytes)
// and hold values only, never a pointer into front memory (guest addresses are values).

#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/commandStream.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace Libs::Graphics::CpSeq {

enum class Mode : uint8_t { Off, Inline, Thread };

// KYTY_CP_SEQ (evaluated once).
[[nodiscard]] Mode ConfiguredMode();
// KYTY_CP_SEQ_VERIFY: 0 off, 1 count/log, 2 exit at the first difference.
[[nodiscard]] int VerifyMode();
// The front hashes the packets it parses (verify mode with an op stream).
[[nodiscard]] bool PacketHashing();

enum class OpKind : uint16_t {
	DrawIndex,
	DrawAuto,
	DrawIndirect,
	DrawIndirectMulti,
	DispatchDirect,
	DispatchIndirect,
	EndOfPipe,
	ReleaseMem,
	EventWrite,
	WriteData,
	ReferenceClock,
	DmaData,
	LodStats,
	Flip,
	WaitRegMem,
	WaitFlipDone,
	DumpConstRam,
	// Lockstep ops: their result decides the front's parse (Result::value, Result::suspended).
	Predication,
	CondExec,
	Branch,
	// KYTY_CP_SEQ_VERIFY: a guest read of the front itself (address, size, hash of the bytes).
	// Executing it does nothing.
	ReadCheck,
	// Thread mode (transport, not compared by the verifier): a graphics submission's first and
	// last op; a submission the resolver parses and executes itself (constant engine); a guest
	// read the sequencer cannot prove clean, done by the resolver in order (lockstep).
	StreamBegin,
	StreamEnd,
	Handoff,
	LockstepRead,
	Count,
};

[[nodiscard]] const char* OpKindName(OpKind kind) noexcept;

// Ops whose result the front needs before it continues: the resolver executes them and hands the
// result back (and, in verify mode, to the reference front). A suspended result makes the front
// suspend its packet, which is parsed again (and the op emitted again) on the next slice.
[[nodiscard]] constexpr bool IsLockstep(OpKind kind) noexcept {
	switch (kind) {
		case OpKind::WaitRegMem:
		case OpKind::WaitFlipDone:
		case OpKind::Predication:
		case OpKind::CondExec:
		case OpKind::Branch:
		case OpKind::LockstepRead: return true;
		default: return false;
	}
}

// What executing an op returns to the front.
struct Result {
	bool     suspended = false; // the packet is retried later (a failed wait, a pending query)
	uint64_t value     = 0;     // predication: skip; COND_EXEC: condition != 0; branch: taken
};

// Thread-mode transport ops: the verifier's reference front has no counterpart for them.
[[nodiscard]] constexpr bool IsTransport(OpKind kind) noexcept {
	switch (kind) {
		case OpKind::StreamBegin:
		case OpKind::StreamEnd:
		case OpKind::Handoff:
		case OpKind::LockstepRead: return true;
		default: return false;
	}
}

inline constexpr uint16_t FlagVerify       = 1u << 0u; // the header's verify_hash is filled
inline constexpr uint16_t FlagAdvanceEpoch = 1u << 1u; // thread mode: a sync-epoch fence packet
                                                       // was parsed since the previous op

// Every record starts with the ring's header; ring.op is always CommandStream::Op::Count ("not a
// recorder packet": the op ring reuses the CP recorder's ring and its Wrap records).
struct OpHeader {
	CommandStream::Header ring;
	OpKind                kind  = OpKind::Count;
	uint16_t              flags = 0;
	uint32_t              data_size = 0; // inline data bytes after the payload
	uint64_t              sequence  = 0; // op number in its stream (the ordering token)
	uint64_t              packet    = 0; // packets the front completed in this PM4 stream before
	                                     // the emitting packet
	uint64_t              packets_hash = 0; // verify: running hash of those packets' dwords
	uint64_t              verify_hash  = 0; // verify: payload and inline data at encode
};
static_assert(sizeof(OpHeader) == 48);

// ---- Payloads ----------------------------------------------------------------------------------

// DrawIndexOp / DrawAutoOp flags.
inline constexpr uint32_t DrawFlagInheritInstances = 1u << 0u; // instance count from the resolver's
                                                             // instance state (NumInstances)
// Thread mode: the draw was published to the draw-prep window at position `window`, or carries
// the register snapshot `snapshot` (cpSequencer.h). Transport fields: the verifier ignores them.
inline constexpr uint32_t DrawFlagPublished = 1u << 1u;
inline constexpr uint32_t DrawFlagSnapshot  = 1u << 2u;

// A draw packet (DRAW_INDEX_2, DRAW_INDEX_OFFSET_2, DRAW_INDEX_AUTO): the DrawIndexArgs /
// DrawAutoArgs the direct path would pass on, with the index type of the packet's front state.
struct DrawIndexOp {
	uint64_t index_addr                 = 0;
	uint32_t index_count                = 0;
	uint32_t instance_count             = 0;
	uint32_t index_type_and_size        = 0;
	int32_t  base_vertex                = 0;
	uint32_t first_instance             = 0;
	uint32_t render_target_slice_offset = 0;
	uint32_t offset_source              = 0; // DrawOffsetSource
	uint32_t flags                      = 0;
	uint64_t window                     = 0; // DrawFlagPublished
	uint32_t snapshot                   = 0; // DrawFlagSnapshot
	uint32_t predicate                  = 0; // GPU predicate id (gpuPredication.h), 0: none
};

struct DrawAutoOp {
	uint32_t vertex_count               = 0;
	uint32_t instance_count             = 0;
	uint32_t first_vertex               = 0;
	uint32_t first_instance             = 0;
	uint32_t offset_source              = 0; // DrawOffsetSource
	uint32_t render_target_slice_offset = 0;
	uint32_t flags                      = 0;
	uint32_t snapshot                   = 0; // DrawFlagSnapshot
	uint64_t window                     = 0; // DrawFlagPublished
	uint32_t predicate                  = 0; // GPU predicate id (gpuPredication.h), 0: none
	uint32_t reserved                   = 0;
};

// DrawIndirectOp flags.
inline constexpr uint32_t IndirectFlagIndexed      = 1u << 0u;
inline constexpr uint32_t IndirectFlagSetInstances = 1u << 1u; // apply num_instances first (the
                                                               // front's SET_NUM_INSTANCES value)
inline constexpr uint32_t IndirectFlagSnapshot     = 1u << 2u; // thread mode: `snapshot` is bound

// DRAW_INDIRECT / DRAW_INDEX_INDIRECT (multi: DRAW_INDIRECT_MULTI / DRAW_INDEX_INDIRECT_MULTI).
// The front's draw state at the packet travels with the op: the resolver reads the arguments.
struct DrawIndirectOp {
	uint64_t args_base           = 0; // SET_BASE draw-indirect base
	uint64_t index_base_addr     = 0;
	uint64_t count_addr          = 0; // multi: GPU count (0: none)
	uint32_t data_offset         = 0;
	uint32_t draw_initiator      = 0;
	uint32_t index_buffer_size   = 0;
	uint32_t index_type_and_size = 0;
	uint32_t max_count_or_count  = 0; // multi
	uint32_t stride              = 0; // multi
	uint32_t flags               = 0;
	uint32_t num_instances       = 0; // IndirectFlagSetInstances
	uint32_t snapshot            = 0; // IndirectFlagSnapshot
	uint32_t reserved            = 0;
};

// Dispatch op flags.
inline constexpr uint32_t DispatchFlagSnapshot = 1u << 0u; // thread mode: `snapshot` is bound

struct DispatchDirectOp {
	uint32_t x        = 0;
	uint32_t y        = 0;
	uint32_t z        = 0;
	uint32_t mode     = 0;
	uint32_t flags    = 0;
	uint32_t snapshot = 0;
};

// DISPATCH_INDIRECT: the resolver reads thread-dimension arguments itself.
struct DispatchIndirectOp {
	uint64_t args_addr = 0;
	uint32_t mode      = 0;
	uint32_t flags     = 0;
	uint32_t snapshot  = 0;
	uint32_t reserved  = 0;
};

// EVENT_WRITE_EOP / EVENT_WRITE_EOS: the WriteAtEndOfPipe arguments.
struct EndOfPipeOp {
	uint64_t dst                  = 0;
	uint64_t value                = 0;
	uint32_t cache_policy         = 0;
	uint32_t event_write_dest     = 0;
	uint32_t eop_event_type       = 0;
	uint32_t cache_action         = 0;
	uint32_t event_index          = 0;
	uint32_t event_write_source   = 0;
	uint32_t interrupt_selector   = 0;
	uint32_t interrupt_context_id = 0;
	uint32_t size                 = 0; // 4 or 8
	uint32_t reserved             = 0;
};

// RELEASE_MEM: the packet body; the resolver decodes and executes it as the handler did
// (barrier, dropped data, interrupt, end-of-pipe write, flush batching).
struct ReleaseMemOp {
	uint32_t body[8] {}; // 7 dwords used
};

struct EventWriteOp {
	uint64_t address     = 0;
	uint32_t event_type  = 0;
	uint32_t event_index = 0;
};

// WRITE_DATA: dw_num dwords follow as inline data.
struct WriteDataOp {
	uint64_t dst           = 0;
	uint32_t dw_num        = 0;
	uint32_t write_control = 0;
};

// COPY_DATA from the reference clock.
struct ReferenceClockOp {
	uint64_t dst       = 0;
	uint32_t num_bytes = 0;
	uint32_t reserved  = 0;
};

// DMA_DATA and the other COPY_DATA forms.
struct DmaDataOp {
	uint64_t dst               = 0;
	uint64_t src               = 0;
	uint32_t num_bytes         = 0;
	uint8_t  engine            = 0;
	uint8_t  dst_sel           = 0;
	uint8_t  dst_cache_policy  = 0;
	uint8_t  src_sel           = 0;
	uint8_t  src_cache_policy  = 0;
	uint8_t  wait_for_previous = 0;
	uint8_t  write_confirm     = 0;
	uint8_t  block_engine      = 0;
	uint32_t reserved          = 0;
};

// GET_LOD_STATS: the packet body; the resolver runs the report or its KYTY_LOD_STATS_MODE
// fallback write.
struct LodStatsOp {
	uint32_t body[4] {};
};

enum class FlipVariant : uint32_t { Plain = 0, Label = 1, LabelInterrupt = 2 };

// The flip markers: the front's flip info (SetFlip) and the label.
struct FlipOp {
	int64_t  flip_arg       = 0;
	uint64_t dst            = 0;
	int32_t  handle         = 0;
	int32_t  index          = 0;
	int32_t  flip_mode      = 0;
	uint32_t variant        = 0; // FlipVariant
	uint32_t value          = 0;
	uint32_t eop_event_type = 0;
	uint32_t cache_action   = 0;
	uint32_t reserved       = 0;
};

struct WaitRegMemOp {
	uint64_t addr    = 0;
	uint64_t ref     = 0;
	uint64_t mask    = 0;
	uint32_t func    = 0;
	uint32_t poll    = 0;
	uint32_t wait_op = 0;
	uint32_t size    = 0; // 4 or 8
};

struct WaitFlipDoneOp {
	uint32_t video_out_handle     = 0;
	uint32_t display_buffer_index = 0;
};

// DUMP_CONST_RAM: dw_num dwords of the front's constant RAM follow as inline data.
struct DumpConstRamOp {
	uint64_t dst    = 0;
	uint32_t offset = 0;
	uint32_t dw_num = 0;
};

// SET_PREDICATION with a memory source (op 1: Z-pass counters, op 3: boolean). Result: suspended,
// or value = 0 (run the predicated packets), 1 (skip them), or (id << 1) for a GPU predicate
// (gpuPredication.h). resolve != 0: no packet; the CPU decision (0/1) of GPU predicate `resolve`.
struct PredicationOp {
	uint64_t address         = 0;
	uint32_t condition       = 0;
	uint32_t op              = 0;
	uint32_t wait_op         = 0;
	uint32_t count_in_dwords = 0;
	uint32_t resolve         = 0;
	uint32_t reserved        = 0;
};

// COND_EXEC. Result: value = the condition dword is non-zero.
struct CondExecOp {
	uint64_t address = 0;
};

// Conditional INDIRECT_BUFFER. Result: value = the then-buffer is taken.
struct BranchOp {
	uint64_t compare_addr = 0;
	uint64_t mask         = 0;
	uint64_t reference    = 0;
	uint32_t function     = 0;
	uint32_t reserved     = 0;
};

struct ReadCheckOp {
	uint64_t address = 0;
	uint64_t size    = 0;
	uint64_t hash    = 0;
};

// Thread mode. `state` (verify only): a CommandProcessor holding the sequencer's front state at
// the start of the stream, owned by the op (the resolver's verifier attaches with it, then
// deletes it). `verify`: the reference front follows this stream.
struct StreamBeginOp {
	uint64_t submission = 0; // GuestGpu admission sequence
	uint64_t state      = 0;
	uint32_t verify     = 0;
	uint32_t reserved   = 0;
};

struct StreamEndOp {
	uint64_t submission = 0;
	uint64_t packets    = 0; // packets the sequencer parsed in the stream
};

struct HandoffOp {
	uint64_t submission = 0;
};

// The sequencer is blocked in the lockstep wait for this op, so the resolver may write the bytes
// to `destination` (sequencer memory; the one exception to "values only").
struct LockstepReadOp {
	uint64_t address     = 0;
	uint64_t size        = 0;
	uint64_t destination = 0;
};

template <typename T>
inline constexpr bool IsPayload = std::is_trivially_copyable_v<T> && alignof(T) <= 8 &&
                                  (sizeof(T) % 8) == 0 &&
                                  std::has_unique_object_representations_v<T>;

static_assert(IsPayload<DrawIndexOp> && IsPayload<DrawAutoOp> && IsPayload<DrawIndirectOp> &&
              IsPayload<DispatchDirectOp> && IsPayload<DispatchIndirectOp> &&
              IsPayload<EndOfPipeOp> && IsPayload<ReleaseMemOp> && IsPayload<EventWriteOp> &&
              IsPayload<WriteDataOp> && IsPayload<ReferenceClockOp> && IsPayload<DmaDataOp> &&
              IsPayload<LodStatsOp> && IsPayload<FlipOp> && IsPayload<WaitRegMemOp> &&
              IsPayload<WaitFlipDoneOp> && IsPayload<DumpConstRamOp> &&
              IsPayload<PredicationOp> && IsPayload<CondExecOp> && IsPayload<BranchOp> &&
              IsPayload<ReadCheckOp> && IsPayload<StreamBeginOp> && IsPayload<StreamEndOp> &&
              IsPayload<HandoffOp> && IsPayload<LockstepReadOp>);

// The kind of each payload type (compile-time dispatch for Emit/Submit).
template <typename T>
struct KindOf;
#define KYTY_CPSEQ_PAYLOAD(kind_name, type)                                                        \
	template <>                                                                                  \
	struct KindOf<type> {                                                                        \
		static constexpr OpKind Kind = OpKind::kind_name;                                        \
	};
KYTY_CPSEQ_PAYLOAD(DrawIndex, DrawIndexOp)
KYTY_CPSEQ_PAYLOAD(DrawAuto, DrawAutoOp)
KYTY_CPSEQ_PAYLOAD(DispatchDirect, DispatchDirectOp)
KYTY_CPSEQ_PAYLOAD(DispatchIndirect, DispatchIndirectOp)
KYTY_CPSEQ_PAYLOAD(EndOfPipe, EndOfPipeOp)
KYTY_CPSEQ_PAYLOAD(ReleaseMem, ReleaseMemOp)
KYTY_CPSEQ_PAYLOAD(EventWrite, EventWriteOp)
KYTY_CPSEQ_PAYLOAD(WriteData, WriteDataOp)
KYTY_CPSEQ_PAYLOAD(ReferenceClock, ReferenceClockOp)
KYTY_CPSEQ_PAYLOAD(DmaData, DmaDataOp)
KYTY_CPSEQ_PAYLOAD(LodStats, LodStatsOp)
KYTY_CPSEQ_PAYLOAD(Flip, FlipOp)
KYTY_CPSEQ_PAYLOAD(WaitRegMem, WaitRegMemOp)
KYTY_CPSEQ_PAYLOAD(WaitFlipDone, WaitFlipDoneOp)
KYTY_CPSEQ_PAYLOAD(DumpConstRam, DumpConstRamOp)
KYTY_CPSEQ_PAYLOAD(Predication, PredicationOp)
KYTY_CPSEQ_PAYLOAD(CondExec, CondExecOp)
KYTY_CPSEQ_PAYLOAD(Branch, BranchOp)
KYTY_CPSEQ_PAYLOAD(ReadCheck, ReadCheckOp)
KYTY_CPSEQ_PAYLOAD(StreamBegin, StreamBeginOp)
KYTY_CPSEQ_PAYLOAD(StreamEnd, StreamEndOp)
KYTY_CPSEQ_PAYLOAD(Handoff, HandoffOp)
KYTY_CPSEQ_PAYLOAD(LockstepRead, LockstepReadOp)
#undef KYTY_CPSEQ_PAYLOAD
// DrawIndirectOp serves two kinds (DrawIndirect, DrawIndirectMulti): no KindOf.

[[nodiscard]] uint32_t PayloadSize(OpKind kind) noexcept;

// The whole record size of an op with `data_size` inline bytes.
[[nodiscard]] constexpr uint32_t RecordSize(uint32_t payload_size, uint32_t data_size) noexcept {
	return static_cast<uint32_t>(sizeof(OpHeader)) + payload_size +
	       CommandStream::Align8(data_size);
}

// The payload without its thread-mode transport fields (window slot, snapshot), which only one
// side of a verify comparison has.
void NormalizeForCompare(OpKind kind, void* payload) noexcept;

// Hash of an op's payload and inline data (the encode and decode sides compute the same).
[[nodiscard]] uint64_t HashOp(OpKind kind, const void* payload, uint32_t payload_size,
                              const void* data, uint32_t data_size) noexcept;

// A decoded record (valid until the stream advances past it).
struct OpView {
	const OpHeader* header  = nullptr;
	const uint8_t*  payload = nullptr;
	const uint8_t*  data    = nullptr;

	template <typename T>
	[[nodiscard]] const T& As() const noexcept {
		return *reinterpret_cast<const T*>(payload);
	}
	[[nodiscard]] OpKind   Kind() const noexcept { return header->kind; }
	[[nodiscard]] uint32_t DataSize() const noexcept { return header->data_size; }
};

// The op ring between the front and the resolver: one producer, one consumer. In inline mode both
// are the same thread and every record is consumed right after it is committed.
class OpStream {
public:
	// capacity: a power of two, at least 64 KiB; the largest record is capacity / 4.
	explicit OpStream(uint64_t capacity);
	~OpStream();
	KYTY_CLASS_NO_COPY(OpStream);

	// ---- Producer ----
	// Encodes an op: header, payload, `data_size` inline bytes. Returns its sequence number.
	uint64_t Emit(OpKind kind, const void* payload, uint32_t payload_size, const void* data,
	              uint32_t data_size, uint64_t packet, uint64_t packets_hash, bool verify,
	              uint16_t flags = 0);

	// ---- Consumer ----
	// The next published op (Wrap records skipped), or false when none is published.
	[[nodiscard]] bool Peek(OpView& view);
	// Releases the op returned by the last Peek.
	void Pop();

	[[nodiscard]] uint64_t Emitted() const noexcept { return m_next_sequence; }
	[[nodiscard]] uint64_t Bytes() const noexcept { return m_bytes; }
	[[nodiscard]] uint32_t MaxRecord() const noexcept { return m_ring.MaxPacket(); }
	// Producer: waits for ring space so far (thread mode: the resolver consumes).
	[[nodiscard]] const CommandStream::WaitStats& ProducerStats() const noexcept {
		return m_producer_stats;
	}
	// Producer: waits in Emit are spin, then block (0: block at once).
	void SetSpinNs(uint64_t spin_ns) noexcept { m_policy.spin_ns = spin_ns; }

private:
	CommandStream::Ring       m_ring;
	CommandStream::WaitPolicy m_policy;
	CommandStream::WaitStats  m_producer_stats;
	CommandStream::WaitStats  m_consumer_stats;
	uint64_t                  m_next_sequence = 0;
	uint64_t                  m_bytes         = 0;
	uint32_t                  m_peeked_size   = 0;
};

// KYTY_CP_SEQ_VERIFY totals, process-wide (tests read them; relaxed).
struct VerifyTotals {
	std::atomic<uint64_t> streams {0};          // PM4 streams the reference front followed
	std::atomic<uint64_t> checks {0};           // ops compared
	std::atomic<uint64_t> mismatches {0};       // ops or stream ends that differed
	std::atomic<uint64_t> read_divergences {0}; // ReadCheck ops whose bytes differed
	std::atomic<uint64_t> lockstep_answers {0}; // lockstep results handed to the reference
};
[[nodiscard]] VerifyTotals& GetVerifyTotals();

} // namespace Libs::Graphics::CpSeq

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPOPS_H_
