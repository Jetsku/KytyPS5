#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RT_BVHDUMP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RT_BVHDUMP_H_

// Psr BVH dumps (KYTY_RT_DUMP_BVH): the guest memory of one TLAS and of every BLAS its instances
// reference, captured by the emulator at a dispatch that uses BVH instructions and loaded again by
// the offline tools (rt_hardware_tests --bvh-dump <file>).
//
// A dump holds whole 4 KiB pages: every page the psrBvh.h walkers read, plus each BVH's declared
// extent (header +88 node count, +96 leaf records), so offline analysis can also examine layouts
// the walkers do not assume (for example leaf children of type-6 nodes).
//
// File (.kbvh, little-endian): "KYTYBVH1", u32 meta bytes, meta text (key=value lines), u32 run
// count, then per run: u64 address, u32 page count, u32 flags, page count * 4096 bytes. Runs are
// sorted and disjoint; a run's pages share its flags.

#include "graphics/host_gpu/rt/psrBvh.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace Libs::Graphics::Rt {

// Sparse guest memory in 4 KiB pages. A read of a missing page fetches it through the fetcher when
// the page lies inside an allowed window and the byte budget allows; a loaded dump has no fetcher.
class GuestSnapshot {
public:
	static constexpr uint64_t PageBytes = 4096u;
	enum PageFlags : uint32_t {
		// GPU-dirty when captured: written by GPU work the CPU had not read back.
		GpuWritten = 1u,
	};
	// Fills `dst` with the bytes of whole pages at `address` and sets one PageFlags value per page
	// in `page_flags`. Returns false when those bytes are not readable.
	using Fetch = std::function<bool(uint64_t address, std::span<uint8_t> dst,
	                                 std::span<uint32_t> page_flags)>;
	struct Page {
		std::array<uint8_t, PageBytes> bytes {};
		uint32_t                       flags = 0;
	};
	using Meta = std::vector<std::pair<std::string, std::string>>;

	GuestSnapshot() = default;
	explicit GuestSnapshot(Fetch fetch, uint64_t byte_budget = 2048ull << 20u)
	    : m_fetch(std::move(fetch)), m_budget(byte_budget) {}

	bool                     Read(uint64_t address, void* dst, size_t size);
	[[nodiscard]] ReadMemory Reader();

	// Fetches only happen inside windows [begin, end).
	void AllowWindow(uint64_t begin, uint64_t end);
	// Fetches the missing pages of [begin, end) that lie inside a window; returns the pages added.
	uint64_t Prefetch(uint64_t begin, uint64_t end);
	void     AddPage(uint64_t address, std::span<const uint8_t> bytes, uint32_t flags);

	[[nodiscard]] const std::map<uint64_t, Page>& Pages() const { return m_pages; }
	[[nodiscard]] uint64_t Bytes() const { return m_pages.size() * PageBytes; }
	[[nodiscard]] uint64_t BytesWithFlag(uint32_t flag) const;
	// Page fetches refused (outside every window or over the budget) and pages that failed (page
	// numbers; fetch-ahead past the end of a mapping lands here too).
	[[nodiscard]] uint64_t                  Refused() const { return m_refused; }
	[[nodiscard]] uint64_t                  Unreadable() const { return m_missing.size(); }
	[[nodiscard]] const std::set<uint64_t>& UnreadablePages() const { return m_missing; }

	bool Save(const std::filesystem::path& path) const;
	bool Load(const std::filesystem::path& path);
	// The value of a meta key, or `fallback`.
	[[nodiscard]] std::string MetaValue(const std::string& key,
	                                    const std::string& fallback = {}) const;

	Meta meta;

private:
	bool InWindow(uint64_t page) const;
	bool FetchRun(uint64_t first_page, uint64_t count);

	Fetch                        m_fetch;
	uint64_t                     m_budget = 0;
	std::map<uint64_t, Page>     m_pages; // by page number
	std::set<uint64_t>           m_missing;
	std::map<uint64_t, uint64_t> m_windows; // disjoint [begin, end), by begin
	uint64_t                     m_refused = 0;
};

// The qbvh_header fields the traversal reads, and the fake instance's world-to-object matrix.
struct PsrHeader {
	uint32_t flags           = 0; // +24; bit 18 = S_TRAP ("trinity"/PS5 Pro formats)
	uint64_t instance_offset = 0; // +48
	uint64_t size_field      = 0; // +88, the T# size field
	uint32_t leaf_records    = 0; // +96, V# num_records of the leaf buffer (stride 8)
};
bool ReadPsrHeader(const ReadMemory& read, uint64_t base, PsrHeader& out);
// Bytes the header declares: max(64 * (size_field + 1), 8 * leaf_records, 384), at most `limit`.
uint64_t DeclaredExtent(const PsrHeader& header, uint64_t limit);
// A plausible Psr BVH at `base`: 256-byte aligned, a readable header with sane sizes and a root
// box32 (pointer 37, byte 256) whose children are valid pointers with ordered, finite bounds.
bool LooksLikePsrBvh(const ReadMemory& read, uint64_t base);

struct BvhCaptureLimits {
	uint64_t   window_bytes     = 256ull << 20u; // lazy fetches allowed this far past a BVH base
	uint64_t   max_extent_bytes = 256ull << 20u; // cap of a prefetched declared extent
	uint32_t   max_blas         = 1u << 16u;
	WalkLimits walk {};
};

struct BvhCaptureBlas {
	uint64_t   base = 0;
	uint32_t   root = 0;
	PsrHeader  header {};
	bool       header_read = false;
	uint64_t   extent      = 0;
	uint32_t   instances   = 0; // TLAS leaves that reference this (base, root)
	WalkStatus walk        = WalkStatus::Ok;
	WalkStatus expand      = WalkStatus::Ok;
	uint32_t   leaves = 0, box_nodes = 0, shared_exp_nodes = 0, max_depth = 0, triangles = 0;
	// Walker reads relative to the base, and how many fell outside [base, base + extent).
	uint64_t touched_begin = 0, touched_end = 0;
	uint64_t outside_extent = 0;
};

struct BvhCapture {
	uint64_t                    tlas_base = 0;
	PsrHeader                   header {};
	bool                        header_read    = false;
	uint64_t                    extent         = 0;
	WalkStatus                  tlas_walk      = WalkStatus::Ok;
	uint32_t                    tlas_box_nodes = 0;
	uint32_t                    instances = 0, fake_instances = 0, mask_zero = 0;
	uint32_t                    cull_disabled = 0, facing_flag = 0, max_instance_index = 0;
	std::vector<BvhCaptureBlas> blas;
	uint32_t                    blas_skipped = 0; // distinct BLASes over max_blas
	double                      ms           = 0;
};

// Walks the TLAS at `tlas_base` and every BLAS it reaches through `snapshot`, prefetching each
// BVH's declared extent. Afterwards the snapshot's pages are the dump.
BvhCapture CaptureBvh(GuestSnapshot& snapshot, uint64_t tlas_base, const BvhCaptureLimits& limits);
// Appends the capture's results as meta keys ("capture.*", "blas.<n>").
void AppendCaptureMeta(const BvhCapture& capture, const GuestSnapshot& snapshot,
                       GuestSnapshot::Meta& meta);
// A readable summary of a capture.
std::string DescribeCapture(const BvhCapture& capture, const GuestSnapshot& snapshot);

} // namespace Libs::Graphics::Rt

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RT_BVHDUMP_H_ */
