#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_INPLACESTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_INPLACESTATS_H_

#include <cstdint>

// KYTY_INPLACE_STATS=1 (diagnostic, default off): what in-place GPU reads of guest memory would
// replace. Per 4 KiB guest page it remembers the last frame the page took a guest write fault (and
// for how many consecutive frames), was uploaded by BufferCache::SynchronizeBuffer, or was copied
// into the stream ring by a small read binding. From that it classifies
//   - each write fault: consecutive-frame streak (1, 2, 3+) and whether the page was uploaded or
//     streamed in this or the previous frame;
//   - each command-processor write of guest memory (WRITE_DATA, DMA_DATA fill / copy, DUMP_CONST_RAM,
//     labels and other CP writes): whether it lands on a page that faulted, was uploaded or was
//     streamed in this or the previous frame (the in-place candidate ranges).
// One console line every 10 s at a guest flip, "InplaceStats 10s: ...", per-frame averages.
// KYTY_INPLACE_STATS=lite keeps only the byte and fault counts (no page map, no mutex-held map
// work per page): cheap enough for timing runs.
// Unset, every hook is one predictable branch. Nothing here feeds back into emulation.
namespace Libs::Graphics::InplaceStats {

enum class CpKind : uint8_t { WriteData, DmaFill, DmaCopy, ConstRam, LabelOther, Count };

[[nodiscard]] bool Enabled() noexcept;

void NoteFault(uint64_t vaddr) noexcept;
void NoteUpload(uint64_t vaddr, uint64_t size) noexcept;
// Full mode: one upload copy by path, hot (a changed hot page's copy) or not. Each 4 KiB page's
// contents are hashed and compared with its previous upload: bytes of unchanged pages are counted,
// and so are bytes of pages that took no write fault of their own this or the previous frame
// (opened by fault-ahead, or hot).
enum class UploadPath : uint8_t { BdaSync, Written, Texel, Read, BdaReach, Count };
void NoteUploadDetail(UploadPath path, bool hot, uint64_t vaddr, uint64_t size) noexcept;
void NoteUploadBytes(uint64_t bytes, uint64_t copies) noexcept;
void NoteStream(uint64_t vaddr, uint64_t size) noexcept;
void NoteCpWrite(CpKind kind, uint64_t vaddr, uint64_t size) noexcept;
// The kind NoteCpWriteDefault records on this thread (ExecWriteData / ExecDumpConstRam scopes).
void NoteCpWriteDefault(uint64_t vaddr, uint64_t size) noexcept;
void AdvanceFrame() noexcept;

class ScopedCpKind final {
public:
	explicit ScopedCpKind(CpKind kind) noexcept;
	~ScopedCpKind();
	ScopedCpKind(const ScopedCpKind&)            = delete;
	ScopedCpKind& operator=(const ScopedCpKind&) = delete;

private:
	CpKind m_previous;
};

} // namespace Libs::Graphics::InplaceStats

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_INPLACESTATS_H_ */
