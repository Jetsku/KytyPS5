#include "graphics/guest_gpu/command_processor/cpOps.h"

#include "common/liveSwitch.h"

#include <cstdlib>
#include <cstring>
#include <xxhash.h>

namespace Libs::Graphics::CpSeq {

namespace {

const char* EnvValue(const char* name) {
	const auto* value = std::getenv(name);
	return value != nullptr && *value != '\0' ? value : nullptr;
}

} // namespace

Mode ConfiguredMode() {
	static const Mode mode = [] {
		const auto* value = EnvValue("KYTY_CP_SEQ");
		if (value == nullptr || std::strcmp(value, "0") == 0 || std::strcmp(value, "off") == 0) {
			return Mode::Off;
		}
		if (std::strcmp(value, "inline") == 0) {
			return Mode::Inline;
		}
		if (std::strcmp(value, "1") == 0 || std::strcmp(value, "thread") == 0) {
			return Mode::Thread;
		}
		EXIT("KYTY_CP_SEQ must be 0, inline or 1 (got '%s')\n", value);
		return Mode::Off;
	}();
	return mode;
}

int VerifyMode() {
	static const int mode = [] {
		const auto* value = EnvValue("KYTY_CP_SEQ_VERIFY");
		if (value == nullptr || std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	}();
	return mode;
}

bool PacketHashing() {
	static const bool enabled = ConfiguredMode() != Mode::Off && VerifyMode() != 0;
	return enabled;
}

namespace {

// KYTY_CP_SEQ_PREFETCH: a live switch (common/liveSwitch.h), read at every self-label wait and
// adoption. -1: a value that is not 0, 1 or mismatch (an exit at startup, off when set live).
int64_t ParsePrefetchMode(const char* value) {
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0 ||
	    std::strcmp(value, "off") == 0) {
		return 0;
	}
	if (std::strcmp(value, "1") == 0 || std::strcmp(value, "on") == 0) {
		return 1;
	}
	if (std::strcmp(value, "mismatch") == 0) {
		return 2;
	}
	return -1;
}

Live::Switch g_prefetch_mode("KYTY_CP_SEQ_PREFETCH", ParsePrefetchMode);

} // namespace

int PrefetchMode() {
	static const bool checked = [] {
		if (g_prefetch_mode.Get() < 0) {
			EXIT("KYTY_CP_SEQ_PREFETCH must be 0, 1 or mismatch (got '%s')\n",
			     EnvValue("KYTY_CP_SEQ_PREFETCH"));
		}
		return true;
	}();
	(void)checked;
	const auto mode = g_prefetch_mode.Get();
	return mode > 0 ? static_cast<int>(mode) : 0;
}

uint32_t PrefetchDraws() {
	static const uint32_t draws = [] {
		const auto* value = EnvValue("KYTY_CP_SEQ_PREFETCH_DRAWS");
		const auto  parsed = value != nullptr ? std::strtoul(value, nullptr, 10) : 16ul;
		return static_cast<uint32_t>(parsed < 1ul ? 1ul : (parsed > 1024ul ? 1024ul : parsed));
	}();
	return draws;
}

VerifyTotals& GetVerifyTotals() {
	static VerifyTotals totals;
	return totals;
}

const char* OpKindName(OpKind kind) noexcept {
	switch (kind) {
		case OpKind::DrawIndex: return "DrawIndex";
		case OpKind::DrawAuto: return "DrawAuto";
		case OpKind::DrawIndirect: return "DrawIndirect";
		case OpKind::DrawIndirectMulti: return "DrawIndirectMulti";
		case OpKind::DispatchDirect: return "DispatchDirect";
		case OpKind::DispatchIndirect: return "DispatchIndirect";
		case OpKind::EndOfPipe: return "EndOfPipe";
		case OpKind::ReleaseMem: return "ReleaseMem";
		case OpKind::EventWrite: return "EventWrite";
		case OpKind::WriteData: return "WriteData";
		case OpKind::ReferenceClock: return "ReferenceClock";
		case OpKind::DmaData: return "DmaData";
		case OpKind::LodStats: return "LodStats";
		case OpKind::Flip: return "Flip";
		case OpKind::WaitRegMem: return "WaitRegMem";
		case OpKind::WaitFlipDone: return "WaitFlipDone";
		case OpKind::DumpConstRam: return "DumpConstRam";
		case OpKind::Predication: return "Predication";
		case OpKind::CondExec: return "CondExec";
		case OpKind::Branch: return "Branch";
		case OpKind::ReadCheck: return "ReadCheck";
		case OpKind::StreamBegin: return "StreamBegin";
		case OpKind::StreamEnd: return "StreamEnd";
		case OpKind::Handoff: return "Handoff";
		case OpKind::LockstepRead: return "LockstepRead";
		case OpKind::SkipSlots: return "SkipSlots";
		case OpKind::Count: break;
	}
	return "?";
}

uint32_t PayloadSize(OpKind kind) noexcept {
	switch (kind) {
		case OpKind::DrawIndex: return sizeof(DrawIndexOp);
		case OpKind::DrawAuto: return sizeof(DrawAutoOp);
		case OpKind::DrawIndirect:
		case OpKind::DrawIndirectMulti: return sizeof(DrawIndirectOp);
		case OpKind::DispatchDirect: return sizeof(DispatchDirectOp);
		case OpKind::DispatchIndirect: return sizeof(DispatchIndirectOp);
		case OpKind::EndOfPipe: return sizeof(EndOfPipeOp);
		case OpKind::ReleaseMem: return sizeof(ReleaseMemOp);
		case OpKind::EventWrite: return sizeof(EventWriteOp);
		case OpKind::WriteData: return sizeof(WriteDataOp);
		case OpKind::ReferenceClock: return sizeof(ReferenceClockOp);
		case OpKind::DmaData: return sizeof(DmaDataOp);
		case OpKind::LodStats: return sizeof(LodStatsOp);
		case OpKind::Flip: return sizeof(FlipOp);
		case OpKind::WaitRegMem: return sizeof(WaitRegMemOp);
		case OpKind::WaitFlipDone: return sizeof(WaitFlipDoneOp);
		case OpKind::DumpConstRam: return sizeof(DumpConstRamOp);
		case OpKind::Predication: return sizeof(PredicationOp);
		case OpKind::CondExec: return sizeof(CondExecOp);
		case OpKind::Branch: return sizeof(BranchOp);
		case OpKind::ReadCheck: return sizeof(ReadCheckOp);
		case OpKind::StreamBegin: return sizeof(StreamBeginOp);
		case OpKind::StreamEnd: return sizeof(StreamEndOp);
		case OpKind::Handoff: return sizeof(HandoffOp);
		case OpKind::LockstepRead: return sizeof(LockstepReadOp);
		case OpKind::SkipSlots: return sizeof(SkipSlotsOp);
		case OpKind::Count: break;
	}
	return 0;
}

void NormalizeForCompare(OpKind kind, void* payload) noexcept {
	switch (kind) {
		case OpKind::DrawIndex: {
			auto* op = static_cast<DrawIndexOp*>(payload);
			op->flags &= ~(DrawFlagPublished | DrawFlagSnapshot);
			op->window   = 0;
			op->snapshot = 0;
			break;
		}
		case OpKind::DrawAuto: {
			auto* op = static_cast<DrawAutoOp*>(payload);
			op->flags &= ~(DrawFlagPublished | DrawFlagSnapshot);
			op->window   = 0;
			op->snapshot = 0;
			break;
		}
		case OpKind::DrawIndirect:
		case OpKind::DrawIndirectMulti: {
			auto* op = static_cast<DrawIndirectOp*>(payload);
			op->flags &= ~IndirectFlagSnapshot;
			op->snapshot = 0;
			break;
		}
		case OpKind::DispatchDirect: {
			auto* op = static_cast<DispatchDirectOp*>(payload);
			op->flags &= ~DispatchFlagSnapshot;
			op->snapshot = 0;
			break;
		}
		case OpKind::DispatchIndirect: {
			auto* op = static_cast<DispatchIndirectOp*>(payload);
			op->flags &= ~DispatchFlagSnapshot;
			op->snapshot = 0;
			break;
		}
		default: break;
	}
}

uint64_t HashOp(OpKind kind, const void* payload, uint32_t payload_size, const void* data,
                uint32_t data_size) noexcept {
	// Chained one-shot hashes (the streaming state is opaque in this build of xxHash).
	const auto hash =
	    XXH3_64bits_withSeed(payload, payload_size, static_cast<uint64_t>(kind) + 1u);
	return data_size != 0 ? XXH3_64bits_withSeed(data, data_size, hash) : hash;
}

OpStream::OpStream(uint64_t capacity): m_ring(capacity) {
	// Inline consumption follows every commit at once: publish every record.
	m_ring.SetPublishBatch(0);
	m_policy.spin_ns = 30000;
}

OpStream::~OpStream() = default;

uint64_t OpStream::Emit(OpKind kind, const void* payload, uint32_t payload_size, const void* data,
                        uint32_t data_size, uint64_t packet, uint64_t packets_hash, bool verify,
                        uint16_t flags) {
	EXIT_IF(payload_size != PayloadSize(kind));
	const auto bytes = RecordSize(payload_size, data_size);
	if (bytes > m_ring.MaxPacket()) {
		EXIT("CpSeq: %s op of %u bytes exceeds the op ring's record limit %u\n", OpKindName(kind),
		     bytes, m_ring.MaxPacket());
	}
	auto* base   = m_ring.Reserve(bytes, m_policy, m_producer_stats);
	auto* header = reinterpret_cast<OpHeader*>(base);
	std::memset(header, 0, sizeof(OpHeader));
	header->ring.op      = CommandStream::Op::Count;
	header->ring.flags   = 0;
	header->ring.size    = bytes;
	header->kind         = kind;
	header->flags        = static_cast<uint16_t>(flags | (verify ? FlagVerify : 0u));
	header->data_size    = data_size;
	header->sequence     = m_next_sequence;
	header->packet       = packet;
	header->packets_hash = packets_hash;
	auto* out            = base + sizeof(OpHeader);
	std::memcpy(out, payload, payload_size);
	if (data_size != 0) {
		std::memcpy(out + payload_size, data, data_size);
		const auto padded = CommandStream::Align8(data_size);
		if (padded != data_size) {
			std::memset(out + payload_size + data_size, 0, padded - data_size);
		}
	}
	if (verify) {
		header->verify_hash = HashOp(kind, payload, payload_size, data, data_size);
	}
	m_ring.Commit(bytes);
	m_ring.Publish();
	m_bytes += bytes;
	return m_next_sequence++;
}

bool OpStream::Peek(OpView& view) {
	for (;;) {
		const auto* header = m_ring.Peek();
		if (header == nullptr) {
			return false;
		}
		if (header->op == CommandStream::Op::Wrap) {
			m_ring.Advance(header->size);
			continue;
		}
		EXIT_IF(header->op != CommandStream::Op::Count);
		const auto* op = reinterpret_cast<const OpHeader*>(header);
		view.header    = op;
		view.payload   = reinterpret_cast<const uint8_t*>(op) + sizeof(OpHeader);
		view.data      = view.payload + PayloadSize(op->kind);
		m_peeked_size  = header->size;
		return true;
	}
}

void OpStream::Pop() {
	EXIT_IF(m_peeked_size == 0);
	m_ring.Advance(m_peeked_size);
	m_peeked_size = 0;
	m_ring.Release(m_consumer_stats);
}

} // namespace Libs::Graphics::CpSeq
