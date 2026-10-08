#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_BUFFERLOOKUPSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_BUFFERLOOKUPSTATS_H_

#include <cstdint>

// KYTY_BUFFER_LOOKUP_STATS=1 (diagnostic, default off): where the command processor's V# binding
// work goes (RenderExecutor::FindBuffers / RebindBuffers and the read-binding memo of
// BufferCache::ObtainReadBinding), measured with the TSC: the cost of each binding by its outcome
// and of the other parts of the two phases, and how often the tracker and the sync epoch moved.
// Nothing it measures feeds back into emulation; unset, every hook is one predictable branch.
// GPU thread only. One console line every 10 s (flushed), "BufferLookup 10s: ...".
namespace Libs::Graphics::BufferLookupStats {

// What one V# binding (NativeStorageBuffer) did.
enum class Outcome : uint8_t {
	None,          // not classified (no memo: KYTY_BINDING_EPOCH_MEMO=0, another thread)
	MemoHit,       // a same-epoch cache-buffer memo hit after its signature check
	MemoCross,     // a cross-epoch cache-buffer memo hit (KYTY_BINDING_MEMO_CROSS_EPOCH)
	MemoStream,    // a stream-copy memo hit
	MissCached,    // memo miss: the normal path bound a cache buffer
	MissStream,    // memo miss: the normal path made a stream copy
	MissLean,      // memo miss taken by KYTY_BINDING_LEAN_MISS
	Texel,         // a formatted (texel) read
	Written,       // a writable binding
	WrittenNarrow, // a writable binding with a proven write range (ObtainWrittenBuffer)
	Null,          // no range: the null buffer
	Count
};

// Other work of the FindBuffers and RebindBuffers phases.
enum class Part : uint8_t {
	FindBuffer,  // FindBuffers: V# decode/plan and BufferCache::FindBuffer, per binding
	PrepareBda,  // FindBuffers: RenderContext::PrepareBda
	Prefetch,    // RebindBuffers: the memo-slot prefetch loop
	WriteRanges, // RebindBuffers: ResolveWrittenRanges
	Invalidate,  // NativeStorageBuffer: TextureCache::InvalidateMemoryFromGPU of written bindings
	MipStats,    // RebindBuffers: WriteMipStatsFields
	Tables,      // RebindBuffers: flattened SRT and shader-data uploads
	Descriptor,  // NativeStorageBuffer after the lookup: offsets, range, descriptor info
	ReadNow,     // ObtainReadBinding miss: ObtainBufferNow
	ReadRecord,  // ObtainReadBinding miss: RecordBinding
	ReadTouch,   // ObtainReadBinding cache-buffer hit: the slot record and TouchBuffer
	FindCall,    // FindBuffers: the BufferCache::FindBuffer calls (per binding)
	BdaSkip,     // SynchronizeBdaBuffers: skipped (same sync epoch and buffer structure)
	BdaNone,     // a pass that found nothing logged (epochs unchanged, no hot runs)
	BdaHot,      // a pass over the recorded hot runs only
	BdaLog,      // a dirty-log pass (KYTY_BDA_DIRTY_LOG)
	BdaFull,     // a full scan of every mapped buffer
	BdaFullNew,  // of which: the buffer structure changed since the last scan
	CreateBuffer, // BufferCache::CreateBuffer (with its joins)
	Count
};

// Why a read binding missed its memo (ObtainReadBinding).
enum class Miss : uint8_t { NoRegion, Empty, OtherKey, Signature, Guard, Epoch, Count };

namespace Detail {
[[nodiscard]] bool ReadEnabled();
} // namespace Detail

[[nodiscard]] inline bool Enabled() {
	static const bool enabled = Detail::ReadEnabled();
	return enabled;
}
// KYTY_BUFFER_LOOKUP_STATS=keys: also the distinct read-binding keys per frame (NoteKey; a hash-set
// insert per lookup, which the binding times then include).
[[nodiscard]] bool KeysEnabled();

namespace Detail {
[[nodiscard]] uint64_t Tsc();
void                   SetOutcome(Outcome outcome);
[[nodiscard]] Outcome  TakeOutcome();
void                   AddBinding(Outcome outcome, uint64_t cycles);
void                   AddPart(Part part, uint64_t cycles);
void                   AddMiss(Miss miss);
void                   NoteKey(uint64_t vaddr, uint64_t size, uint32_t frame);
void                   NoteStage(uint64_t epoch);
} // namespace Detail

// Every hook below is inline and does nothing (one branch) while the diagnostic is off.

// TSC (0 when disabled).
[[nodiscard]] inline uint64_t Now() {
	return Enabled() ? Detail::Tsc() : 0;
}

// The outcome of the binding being obtained, set by the buffer cache; taken (and reset) by the
// binding's caller.
inline void SetOutcome(Outcome outcome) {
	if (Enabled()) {
		Detail::SetOutcome(outcome);
	}
}
[[nodiscard]] inline Outcome TakeOutcome() {
	return Enabled() ? Detail::TakeOutcome() : Outcome::None;
}

inline void AddBinding(Outcome outcome, uint64_t cycles) {
	if (Enabled()) {
		Detail::AddBinding(outcome, cycles);
	}
}
inline void AddPart(Part part, uint64_t cycles) {
	if (Enabled()) {
		Detail::AddPart(part, cycles);
	}
}
inline void AddMiss(Miss miss) {
	if (Enabled()) {
		Detail::AddMiss(miss);
	}
}
// A read binding's key (ObtainReadBinding) in guest frame `frame`: distinct keys per frame and
// how many of them the previous frame had.
inline void NoteKey(uint64_t vaddr, uint64_t size, uint32_t frame) {
	if (Enabled()) {
		Detail::NoteKey(vaddr, size, frame);
	}
}
// The sync epoch at the end of a RebindBuffers (its advances between calls are summed), and the
// stage count.
inline void NoteStage(uint64_t epoch) {
	if (Enabled()) {
		Detail::NoteStage(epoch);
	}
}

// Measures [construction, Stop()) into a part when enabled.
class PartScope {
public:
	explicit PartScope(Part part): m_part(part), m_start(Enabled() ? Now() : 0) {}
	~PartScope() { Stop(); }
	void Stop() {
		if (m_start != 0) {
			AddPart(m_part, Now() - m_start);
			m_start = 0;
		}
	}
	PartScope(const PartScope&)            = delete;
	PartScope& operator=(const PartScope&) = delete;

private:
	Part     m_part;
	uint64_t m_start;
};

} // namespace Libs::Graphics::BufferLookupStats

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_BUFFERLOOKUPSTATS_H_
