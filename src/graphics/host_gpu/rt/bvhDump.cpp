#include "graphics/host_gpu/rt/bvhDump.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <tuple>

namespace Libs::Graphics::Rt {

namespace {

constexpr char     Magic[8]        = {'K', 'Y', 'T', 'Y', 'B', 'V', 'H', '1'};
constexpr uint64_t LazyFetchPages  = 16u;  // a walker miss fetches up to 64 KiB
constexpr uint64_t PrefetchPages   = 256u; // declared extents are fetched 1 MiB at a time
constexpr uint32_t MaxMetaBytes    = 64u << 20u;
constexpr uint32_t MaxRunPages     = 1u << 20u;
constexpr uint64_t MaxHeaderSize   = 1ull << 32u;
constexpr uint64_t MaxHeaderOffset = 1ull << 36u;

uint64_t PageOf(uint64_t address) {
	return address / GuestSnapshot::PageBytes;
}

std::string Hex(uint64_t value) {
	char text[24];
	std::snprintf(text, sizeof(text), "0x%" PRIx64, value);
	return text;
}

std::string Dec(uint64_t value) {
	return std::to_string(value);
}

void Put32(std::ofstream& file, uint32_t value) {
	file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void Put64(std::ofstream& file, uint64_t value) {
	file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <typename T>
bool Get(std::ifstream& file, T& value) {
	return static_cast<bool>(file.read(reinterpret_cast<char*>(&value), sizeof(value)));
}

} // namespace

// ---------------------------------------------------------------------------------------------
// GuestSnapshot

bool GuestSnapshot::InWindow(uint64_t page) const {
	const uint64_t begin = page * PageBytes;
	const uint64_t end   = begin + PageBytes;
	auto           it    = m_windows.upper_bound(begin);
	if (it != m_windows.begin() && std::prev(it)->second > begin) {
		return true;
	}
	return it != m_windows.end() && it->first < end;
}

void GuestSnapshot::AllowWindow(uint64_t begin, uint64_t end) {
	if (end <= begin) {
		return;
	}
	auto it = m_windows.upper_bound(begin);
	if (it != m_windows.begin()) {
		const auto previous = std::prev(it);
		if (previous->second >= begin) {
			begin = previous->first;
			end   = std::max(end, previous->second);
			it    = m_windows.erase(previous);
		}
	}
	while (it != m_windows.end() && it->first <= end) {
		end = std::max(end, it->second);
		it  = m_windows.erase(it);
	}
	m_windows.emplace(begin, end);
}

bool GuestSnapshot::FetchRun(uint64_t first_page, uint64_t count) {
	if (!m_fetch || count == 0) {
		return false;
	}
	if (Bytes() + count * PageBytes > m_budget) {
		m_refused += count;
		return false;
	}
	std::vector<uint8_t>  bytes(count * PageBytes);
	std::vector<uint32_t> flags(count);
	if (m_fetch(first_page * PageBytes, bytes, flags)) {
		for (uint64_t i = 0; i < count; i++) {
			auto& page = m_pages[first_page + i];
			std::memcpy(page.bytes.data(), bytes.data() + i * PageBytes, PageBytes);
			page.flags = flags[i];
		}
		return true;
	}
	if (count == 1) {
		m_missing.insert(first_page);
		return false;
	}
	// Part of the run may still be readable (the end of a mapping, say).
	bool any = false;
	for (uint64_t i = 0; i < count; i++) {
		any = FetchRun(first_page + i, 1) || any;
	}
	return any;
}

bool GuestSnapshot::Read(uint64_t address, void* dst, size_t size) {
	if (size == 0) {
		return true;
	}
	if (address + size < address) {
		return false;
	}
	auto*    out  = static_cast<uint8_t*>(dst);
	uint64_t at   = address;
	size_t   left = size;
	while (left > 0) {
		const uint64_t page = PageOf(at);
		auto           it   = m_pages.find(page);
		if (it == m_pages.end()) {
			if (!m_fetch || m_missing.contains(page)) {
				return false;
			}
			if (!InWindow(page)) {
				m_refused++;
				return false;
			}
			uint64_t last = page;
			while ((last + 1) % LazyFetchPages != 0 && !m_pages.contains(last + 1) &&
			       !m_missing.contains(last + 1) && InWindow(last + 1)) {
				last++;
			}
			FetchRun(page, last - page + 1);
			it = m_pages.find(page);
			if (it == m_pages.end()) {
				return false;
			}
		}
		const uint64_t offset = at - page * PageBytes;
		const size_t   chunk  = static_cast<size_t>(std::min<uint64_t>(left, PageBytes - offset));
		std::memcpy(out, it->second.bytes.data() + offset, chunk);
		out += chunk;
		at += chunk;
		left -= chunk;
	}
	return true;
}

ReadMemory GuestSnapshot::Reader() {
	return [this](uint64_t address, void* dst, size_t size) { return Read(address, dst, size); };
}

uint64_t GuestSnapshot::Prefetch(uint64_t begin, uint64_t end) {
	if (!m_fetch || end <= begin) {
		return 0;
	}
	const size_t   before = m_pages.size();
	const uint64_t last   = PageOf(end - 1);
	for (uint64_t page = PageOf(begin); page <= last;) {
		if (m_pages.contains(page) || m_missing.contains(page) || !InWindow(page)) {
			page++;
			continue;
		}
		uint64_t run_last = page;
		while (run_last < last && run_last + 1 - page < PrefetchPages &&
		       !m_pages.contains(run_last + 1) && !m_missing.contains(run_last + 1) &&
		       InWindow(run_last + 1)) {
			run_last++;
		}
		if (Bytes() + (run_last - page + 1) * PageBytes > m_budget) {
			m_refused += last - page + 1;
			break;
		}
		FetchRun(page, run_last - page + 1);
		page = run_last + 1;
	}
	return m_pages.size() - before;
}

void GuestSnapshot::AddPage(uint64_t address, std::span<const uint8_t> bytes, uint32_t flags) {
	auto& page = m_pages[PageOf(address)];
	std::memcpy(page.bytes.data(), bytes.data(), std::min<size_t>(bytes.size(), PageBytes));
	page.flags = flags;
}

uint64_t GuestSnapshot::BytesWithFlag(uint32_t flag) const {
	uint64_t bytes = 0;
	for (const auto& [number, page]: m_pages) {
		bytes += (page.flags & flag) != 0u ? PageBytes : 0u;
	}
	return bytes;
}

std::string GuestSnapshot::MetaValue(const std::string& key, const std::string& fallback) const {
	for (const auto& [k, v]: meta) {
		if (k == key) {
			return v;
		}
	}
	return fallback;
}

bool GuestSnapshot::Save(const std::filesystem::path& path) const {
	std::ofstream file(path, std::ios::binary | std::ios::trunc);
	if (!file) {
		return false;
	}
	std::string text;
	for (const auto& [key, value]: meta) {
		text += key;
		text += '=';
		for (const char c: value) {
			text += (c == '\n' || c == '\r') ? ' ' : c;
		}
		text += '\n';
	}
	file.write(Magic, sizeof(Magic));
	Put32(file, static_cast<uint32_t>(text.size()));
	file.write(text.data(), static_cast<std::streamsize>(text.size()));

	// Runs of consecutive pages with equal flags.
	std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> runs;
	for (const auto& [number, page]: m_pages) {
		if (!runs.empty()) {
			auto& [first, count, flags] = runs.back();
			if (first + count == number && flags == page.flags && count < MaxRunPages) {
				count++;
				continue;
			}
		}
		runs.emplace_back(number, 1u, page.flags);
	}
	Put32(file, static_cast<uint32_t>(runs.size()));
	auto page = m_pages.begin();
	for (const auto& [first, count, flags]: runs) {
		Put64(file, first * PageBytes);
		Put32(file, count);
		Put32(file, flags);
		for (uint32_t i = 0; i < count; i++, ++page) {
			file.write(reinterpret_cast<const char*>(page->second.bytes.data()), PageBytes);
		}
	}
	return static_cast<bool>(file);
}

bool GuestSnapshot::Load(const std::filesystem::path& path) {
	m_pages.clear();
	m_missing.clear();
	meta.clear();
	std::ifstream file(path, std::ios::binary);
	char          magic[sizeof(Magic)] {};
	uint32_t      text_size = 0;
	if (!file.read(magic, sizeof(magic)) || std::memcmp(magic, Magic, sizeof(Magic)) != 0 ||
	    !Get(file, text_size) || text_size > MaxMetaBytes) {
		return false;
	}
	std::string text(text_size, '\0');
	if (!file.read(text.data(), text_size)) {
		return false;
	}
	for (size_t at = 0; at < text.size();) {
		const size_t end   = std::min(text.find('\n', at), text.size());
		const auto   line  = text.substr(at, end - at);
		const size_t equal = line.find('=');
		if (equal != std::string::npos) {
			meta.emplace_back(line.substr(0, equal), line.substr(equal + 1));
		}
		at = end + 1;
	}
	uint32_t runs = 0;
	if (!Get(file, runs)) {
		return false;
	}
	for (uint32_t r = 0; r < runs; r++) {
		uint64_t address = 0;
		uint32_t count   = 0;
		uint32_t flags   = 0;
		if (!Get(file, address) || !Get(file, count) || !Get(file, flags) ||
		    address % PageBytes != 0 || count > MaxRunPages) {
			return false;
		}
		for (uint32_t i = 0; i < count; i++) {
			auto& page = m_pages[PageOf(address) + i];
			page.flags = flags;
			if (!file.read(reinterpret_cast<char*>(page.bytes.data()), PageBytes)) {
				return false;
			}
		}
	}
	return true;
}

// ---------------------------------------------------------------------------------------------
// Headers and capture

bool ReadPsrHeader(const ReadMemory& read, uint64_t base, PsrHeader& out) {
	out = {};
	return read(base + Psr::HeaderFlagsOffset, &out.flags, sizeof(out.flags)) &&
	       read(base + Psr::HeaderInstancesAt, &out.instance_offset, sizeof(out.instance_offset)) &&
	       read(base + Psr::HeaderNodeSizeAt, &out.size_field, sizeof(out.size_field)) &&
	       read(base + Psr::HeaderLeafRecordsAt, &out.leaf_records, sizeof(out.leaf_records));
}

uint64_t DeclaredExtent(const PsrHeader& header, uint64_t limit) {
	const uint64_t nodes = header.size_field < limit / 64u ? 64u * (header.size_field + 1u) : limit;
	const uint64_t leaves = Psr::LeafEntryBytes * header.leaf_records;
	return std::min(limit, std::max({nodes, leaves, Psr::RootNodeOffset + Psr::Box32Bytes}));
}

bool LooksLikePsrBvh(const ReadMemory& read, uint64_t base) {
	if (base == 0 || (base & 255u) != 0u) {
		return false;
	}
	PsrHeader header {};
	if (!ReadPsrHeader(read, base, header) || header.size_field == 0 ||
	    header.size_field >= MaxHeaderSize || header.instance_offset >= MaxHeaderOffset ||
	    (header.instance_offset & 7u) != 0u) {
		return false;
	}
	std::array<uint8_t, Psr::Box32Bytes> bytes {};
	if (!read(base + Psr::RootNodeOffset, bytes.data(), bytes.size())) {
		return false;
	}
	BoxNode root {};
	DecodeBox(Psr::NodeType::Box32, bytes, root);
	for (uint32_t c = 0; c < 4; c++) {
		if (root.children[c] == Psr::InvalidNode) {
			continue;
		}
		const auto& b = root.bounds[c];
		for (int a = 0; a < 3; a++) {
			if (!std::isfinite(b.min[a]) || !std::isfinite(b.max[a]) || b.min[a] > b.max[a]) {
				return false;
			}
		}
	}
	return true;
}

BvhCapture CaptureBvh(GuestSnapshot& snapshot, uint64_t tlas_base, const BvhCaptureLimits& limits) {
	const auto start  = std::chrono::steady_clock::now();
	const auto reader = snapshot.Reader();
	BvhCapture capture {};
	capture.tlas_base = tlas_base;
	snapshot.AllowWindow(tlas_base, tlas_base + limits.window_bytes);
	capture.header_read = ReadPsrHeader(reader, tlas_base, capture.header);
	if (capture.header_read) {
		capture.extent = DeclaredExtent(capture.header, limits.max_extent_bytes);
		snapshot.Prefetch(tlas_base, tlas_base + capture.extent);
		const uint64_t instances = tlas_base + capture.header.instance_offset;
		snapshot.AllowWindow(instances, instances + uint64_t {limits.walk.max_instances} *
		                                                Psr::InstanceStride);
	}
	TlasWalk tlas;
	capture.tlas_walk      = WalkTlas(reader, tlas_base, tlas, limits.walk);
	capture.tlas_box_nodes = tlas.box_nodes;

	std::map<std::pair<uint64_t, uint32_t>, uint32_t> users;
	for (const auto& inst: tlas.instances) {
		capture.instances++;
		capture.fake_instances += inst.fake ? 1u : 0u;
		capture.mask_zero += inst.Mask() == 0u ? 1u : 0u;
		capture.cull_disabled += (inst.flags & Psr::CullDisableFlag) != 0u ? 1u : 0u;
		capture.facing_flag += (inst.flags & Psr::FacingFlag) != 0u ? 1u : 0u;
		if (!inst.fake) {
			capture.max_instance_index = std::max(capture.max_instance_index, inst.index);
		}
		users[{inst.blas_address, inst.root}]++;
	}

	for (const auto& [key, count]: users) {
		if (capture.blas.size() >= limits.max_blas) {
			capture.blas_skipped++;
			continue;
		}
		BvhCaptureBlas blas {};
		blas.base      = key.first;
		blas.root      = key.second;
		blas.instances = count;
		snapshot.AllowWindow(blas.base, blas.base + limits.window_bytes);
		blas.header_read = ReadPsrHeader(reader, blas.base, blas.header);
		if (blas.header_read) {
			blas.extent = DeclaredExtent(blas.header, limits.max_extent_bytes);
			snapshot.Prefetch(blas.base, blas.base + blas.extent);
		}
		uint64_t         lo      = UINT64_MAX;
		uint64_t         hi      = 0;
		const ReadMemory tracked = [&](uint64_t address, void* dst, size_t size) {
			lo = std::min(lo, address);
			hi = std::max(hi, address + size);
			if (address < blas.base || address + size > blas.base + blas.extent) {
				blas.outside_extent++;
			}
			return snapshot.Read(address, dst, size);
		};
		BlasWalk walk;
		blas.walk             = WalkBlas(tracked, blas.base, blas.root, walk, limits.walk);
		blas.leaves           = static_cast<uint32_t>(walk.leaves.size());
		blas.box_nodes        = walk.box_nodes;
		blas.shared_exp_nodes = walk.shared_exp_nodes;
		blas.max_depth        = walk.max_depth;
		// Leaves of a partial walk still bring their pages along.
		BlasTriangles triangles;
		blas.expand    = ExpandTriangles(tracked, blas.base, walk, triangles, limits.walk);
		blas.triangles = static_cast<uint32_t>(triangles.triangles.size());
		if (lo != UINT64_MAX) {
			blas.touched_begin = lo;
			blas.touched_end   = hi;
		}
		capture.blas.push_back(blas);
	}
	capture.ms =
	    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
	return capture;
}

void AppendCaptureMeta(const BvhCapture& capture, const GuestSnapshot& snapshot,
                       GuestSnapshot::Meta& meta) {
	const auto& h = capture.header;
	meta.emplace_back("capture.tlas", Hex(capture.tlas_base));
	meta.emplace_back("capture.tlas.header",
	                  capture.header_read
	                      ? "flags=" + Hex(h.flags) + " instance_offset=" + Hex(h.instance_offset) +
	                            " size_field=" + Dec(h.size_field) +
	                            " leaf_records=" + Dec(h.leaf_records)
	                      : "unreadable");
	meta.emplace_back("capture.tlas.extent", Dec(capture.extent));
	meta.emplace_back("capture.tlas.walk", ToString(capture.tlas_walk));
	meta.emplace_back("capture.tlas.box_nodes", Dec(capture.tlas_box_nodes));
	meta.emplace_back("capture.instances", Dec(capture.instances));
	meta.emplace_back("capture.instances.fake", Dec(capture.fake_instances));
	meta.emplace_back("capture.instances.mask_zero", Dec(capture.mask_zero));
	meta.emplace_back("capture.instances.cull_disabled", Dec(capture.cull_disabled));
	meta.emplace_back("capture.instances.facing_flag", Dec(capture.facing_flag));
	meta.emplace_back("capture.instances.max_index", Dec(capture.max_instance_index));
	meta.emplace_back("capture.blas", Dec(capture.blas.size()));
	meta.emplace_back("capture.blas.skipped", Dec(capture.blas_skipped));
	meta.emplace_back("capture.bytes", Dec(snapshot.Bytes()));
	meta.emplace_back("capture.bytes.gpu_written",
	                  Dec(snapshot.BytesWithFlag(GuestSnapshot::GpuWritten)));
	meta.emplace_back("capture.fetches.refused", Dec(snapshot.Refused()));
	meta.emplace_back("capture.pages.unreadable", Dec(snapshot.Unreadable()));
	char ms[32];
	std::snprintf(ms, sizeof(ms), "%.1f", capture.ms);
	meta.emplace_back("capture.ms", ms);
	for (size_t i = 0; i < capture.blas.size(); i++) {
		const auto& b = capture.blas[i];
		std::string v =
		    "base=" + Hex(b.base) + " root=" + Hex(b.root) + " instances=" + Dec(b.instances);
		if (b.header_read) {
			v += " flags=" + Hex(b.header.flags) + " size_field=" + Dec(b.header.size_field) +
			     " leaf_records=" + Dec(b.header.leaf_records) +
			     " instance_offset=" + Hex(b.header.instance_offset);
		} else {
			v += " header=unreadable";
		}
		v += " extent=" + Dec(b.extent) + " walk=" + ToString(b.walk) +
		     " expand=" + ToString(b.expand) + " leaves=" + Dec(b.leaves) +
		     " box_nodes=" + Dec(b.box_nodes) + " type6=" + Dec(b.shared_exp_nodes) +
		     " depth=" + Dec(b.max_depth) + " triangles=" + Dec(b.triangles) +
		     " touched=" + Hex(b.touched_begin) + "-" + Hex(b.touched_end) +
		     " outside_extent=" + Dec(b.outside_extent);
		meta.emplace_back("blas." + Dec(i), v);
	}
}

std::string DescribeCapture(const BvhCapture& capture, const GuestSnapshot& snapshot) {
	std::map<std::string, uint32_t> walks;
	uint64_t                        leaves = 0, boxes = 0, type6 = 0, triangles = 0, outside = 0;
	uint32_t                        depth = 0;
	for (const auto& b: capture.blas) {
		walks[std::string(ToString(b.walk)) + "/" + ToString(b.expand)]++;
		leaves += b.leaves;
		boxes += b.box_nodes;
		type6 += b.shared_exp_nodes;
		triangles += b.triangles;
		outside += b.outside_extent != 0u ? 1u : 0u;
		depth = std::max(depth, b.max_depth);
	}
	char text[1024];
	std::snprintf(text, sizeof(text),
	              "TLAS 0x%" PRIx64 ": walk %s, %u box nodes, %u instances (%u fake, %u mask 0, "
	              "%u cull off, %u facing flag), extent %" PRIu64 " bytes\n"
	              "BLAS: %zu distinct (%u skipped), %" PRIu64 " leaves, %" PRIu64
	              " box nodes (%" PRIu64 " type 6), %" PRIu64 " triangles, depth %u, %" PRIu64
	              " read outside their declared extent\n"
	              "Snapshot: %" PRIu64 " bytes (%" PRIu64 " GPU-written), %" PRIu64
	              " refused fetches, %" PRIu64 " unreadable pages, %.1f ms\n",
	              capture.tlas_base, ToString(capture.tlas_walk), capture.tlas_box_nodes,
	              capture.instances, capture.fake_instances, capture.mask_zero,
	              capture.cull_disabled, capture.facing_flag, capture.extent, capture.blas.size(),
	              capture.blas_skipped, leaves, boxes, type6, triangles, depth, outside,
	              snapshot.Bytes(), snapshot.BytesWithFlag(GuestSnapshot::GpuWritten),
	              snapshot.Refused(), snapshot.Unreadable(), capture.ms);
	std::string out = text;
	out += "BLAS walk/expand statuses:";
	for (const auto& [status, count]: walks) {
		out += " " + status + "=" + std::to_string(count);
	}
	out += "\n";
	return out;
}

} // namespace Libs::Graphics::Rt
