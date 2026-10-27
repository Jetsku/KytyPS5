#include "graphics/host_gpu/renderer/rtBvhDump.h"

#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/renderer/drawPrep/readSet.h"
#include "graphics/host_gpu/rt/bvhDump.h"
#include "kernel/memory.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace Libs::Graphics {

namespace {

// Requested dispatch ordinals, sorted and unique; empty = off.
const std::vector<uint64_t>& DumpOrdinals() {
	static const std::vector<uint64_t> ordinals = [] {
		std::vector<uint64_t> list;
		const char*           value = std::getenv("KYTY_RT_DUMP_BVH");
		if (value == nullptr) {
			return list;
		}
		const std::string text(value);
		for (size_t at = 0; at < text.size();) {
			const size_t end   = std::min(text.find_first_of(",;/ ", at), text.size());
			const auto   token = text.substr(at, end - at);
			char*        rest  = nullptr;
			const auto   n     = std::strtoull(token.c_str(), &rest, 10);
			if (!token.empty() && rest != nullptr && *rest == '\0' && n > 0) {
				list.push_back(n);
			}
			at = end + 1;
		}
		std::sort(list.begin(), list.end());
		list.erase(std::unique(list.begin(), list.end()), list.end());
		return list;
	}();
	return ordinals;
}

std::filesystem::path DumpFolder() {
	const char* dir = std::getenv("KYTY_RT_DUMP_BVH_DIR");
	return dir != nullptr && *dir != '\0' ? std::filesystem::path(dir)
	                                      : Config::GetShaderLogFolder() / "bvh";
}

// A coherent read of guest pages on the GPU thread: GPU-written bytes are synchronized to the
// backing first (the CP's ReadGuestForCp path, without its unchecked fallback).
bool FetchCoherent(uint64_t address, std::span<uint8_t> dst, std::span<uint32_t> page_flags) {
	for (size_t i = 0; i < page_flags.size(); i++) {
		const uint64_t page = address + i * Rt::GuestSnapshot::PageBytes;
		page_flags[i] = LibKernel::Memory::IsGpuCleanForRead(page, Rt::GuestSnapshot::PageBytes)
		                    ? 0u
		                    : Rt::GuestSnapshot::GpuWritten;
	}
	if (LibKernel::Memory::TryReadGpuCleanBacking(address, dst.data(), dst.size())) {
		return true;
	}
	return LibKernel::Memory::SynchronizeGpuBackingForRead(address, dst.size()) &&
	       LibKernel::Memory::TryReadGpuCleanBacking(address, dst.data(), dst.size());
}

std::string Hex(uint64_t value) {
	char text[24];
	std::snprintf(text, sizeof(text), "0x%" PRIx64, value);
	return text;
}

// Candidate SRT offsets of the TLAS pointer: the lighting shaders load it from +0, the GI shaders
// from TraceRayDescriptor::m_bvh at +40; the rest of the first 256 bytes are a fallback.
constexpr uint32_t SrtBytes = 256;

// KYTY_RT_PROBE_TLAS=<user sgpr>:<V# offset>:<qword offset> (diagnostic, with KYTY_RT_DUMP_BVH's
// ordinals): the TLAS header address is read along a path the guest kernel uses (Miles Morales'
// RT kernels: s[12:13] is the SRT, the V# at SRT+512 names a buffer whose qword at +80 is the TLAS
// header), and its header dwords and root box (box32 at +256) are logged with whether those pages
// were GPU-written. It tells whether the BVH the kernel traverses holds anything.
bool ProbeNow(const RtBvhDispatch& d, uint64_t ordinal, const char* spec) {
	// Several paths may be given, separated by '+' (the kernels differ: 0x6d9e... reads the V# at
	// SRT+544, 0x208e... at SRT+512); the first whose TLAS header is readable is reported.
	Rt::GuestSnapshot probe(FetchCoherent, 64ull << 20u);
	probe.AllowWindow(0, ~0ull);
	uint64_t   srt = 0, buffer = 0, tlas = 0;
	uint32_t   header[32] = {};
	uint8_t    root[Rt::Psr::Box32Bytes] = {};
	bool       header_ok = false, root_ok = false;
	std::string paths;
	for (const char* at = spec; at != nullptr && *at != '\0';) {
		unsigned sgpr = 0, vsharp = 0, qword = 0;
		const char* next = std::strchr(at, '+');
		if (std::sscanf(at, "%u:%u:%u", &sgpr, &vsharp, &qword) == 3 && sgpr + 1u < d.user_sgpr.size()) {
			srt = d.user_sgpr[sgpr] | static_cast<uint64_t>(d.user_sgpr[sgpr + 1u]) << 32u;
			uint32_t v[4] = {};
			buffer = 0;
			tlas   = 0;
			if (probe.Read(srt + vsharp, v, sizeof(v))) {
				buffer = v[0] | (static_cast<uint64_t>(v[1] & 0xffffu) << 32u);
				if (probe.Read(buffer + qword, &tlas, sizeof(tlas))) {
					tlas &= 0x0000ffffffffffffull;
				}
			}
			paths += fmt::format(" s{}:+{}:+{} -> 0x{:x}", sgpr, vsharp, qword, tlas);
			header_ok = tlas != 0 && probe.Read(tlas, header, sizeof(header));
			root_ok   = tlas != 0 && probe.Read(tlas + Rt::Psr::RootNodeOffset, root, sizeof(root));
			if (header_ok) break;
		}
		at = next != nullptr ? next + 1 : nullptr;
	}
	if (paths.empty()) {
		LOGF("RtBvhDump: KYTY_RT_PROBE_TLAS=%s does not fit this dispatch (%zu user SGPRs)\n", spec,
		     d.user_sgpr.size());
		return false;
	}
	LOGF("RtBvhDump: probe paths:%s\n", paths.c_str());
	std::string text;
	for (uint32_t i = 0; header_ok && i < 32u; i++) {
		text += (i % 8u == 0u ? "\n  " : " ") + Hex(header[i]);
	}
	std::string node;
	if (root_ok) {
		Rt::BoxNode box {};
		Rt::DecodeBox(Rt::Psr::NodeType::Box32, std::span<const uint8_t>(root, sizeof(root)), box);
		for (uint32_t c = 0; c < 4; c++) {
			const auto& b = box.bounds[c];
			char        line[200];
			std::snprintf(line, sizeof(line), "\n  child %u: 0x%08x [%g %g %g]..[%g %g %g]", c, box.children[c],
			              b.min[0], b.min[1], b.min[2], b.max[0], b.max[1], b.max[2]);
			node += line;
		}
	}
	// The node region (up to the instance array, at most 4 MiB): how many 64-byte blocks hold
	// anything, the first few non-empty offsets, and whether any of it was GPU-written.
	std::string scan;
	if (header_ok) {
		const uint64_t instances = static_cast<uint64_t>(header[10]) | static_cast<uint64_t>(header[11]) << 32u;
		const uint64_t limit     = std::min<uint64_t>(instances != 0 ? instances : (4u << 20u), 4u << 20u);
		std::vector<uint32_t> block(16);
		uint64_t              filled = 0, total = 0;
		std::string           first;
		for (uint64_t at = Rt::Psr::RootNodeOffset; at + 64u <= limit; at += 64u) {
			if (!probe.Read(tlas + at, block.data(), 64u)) break;
			total++;
			bool empty = true;
			for (uint32_t w = 0; w < 16u; w++) {
				// An empty box (pointers 0xffffffff, bounds +inf/-inf) or zeros counts as nothing.
				empty &= block[w] == 0u || block[w] == 0xffffffffu || block[w] == 0x7f800000u || block[w] == 0xff800000u;
			}
			if (!empty) {
				if (filled < 4u) first += fmt::format(" +0x{:x}:{:08x}/{:08x}", at, block[0], block[4]);
				filled++;
			}
		}
		uint32_t inst_nonzero = 0;
		for (uint32_t i = 0; instances != 0 && i < 64u; i++) {
			uint32_t rec[32] = {};
			if (!probe.Read(tlas + instances + i * 128u, rec, sizeof(rec))) break;
			bool any = false;
			for (const auto w: rec) any |= w != 0u;
			inst_nonzero += any ? 1u : 0u;
		}
		scan = fmt::format("\n  node region: {} of {} 64-byte blocks hold data{}; first 64 instance records: {} non-zero",
		                   filled, total, first, inst_nonzero);
	}
	uint32_t gpu_pages = 0;
	for (const auto& [number, page]: probe.Pages()) {
		gpu_pages += (page.flags & Rt::GuestSnapshot::GpuWritten) != 0u ? 1u : 0u;
	}
	LOGF("RtBvhDump: probe at dispatch %" PRIu64 " frame %d shader 0x%016" PRIx64 ": SRT 0x%" PRIx64
	     ", V# base 0x%" PRIx64 ", TLAS header 0x%" PRIx64 " (%s), %zu pages read, %u GPU-written;"
	     " header dwords:%s\n  root box32 at +256 (%s):%s%s\n",
	     ordinal, d.frame, d.shader_hash, srt, buffer, tlas, header_ok ? "readable" : "NOT readable",
	     probe.Pages().size(), gpu_pages, text.c_str(), root_ok ? "readable" : "NOT readable", node.c_str(),
	     scan.c_str());
	return true;
}

bool DumpNow(const RtBvhDispatch& d, uint64_t ordinal, uint32_t sequence) {
	if (d.user_sgpr.size() < 2) {
		LOGF("RtBvhDump: dispatch %" PRIu64 " has %zu user SGPRs, no SRT pointer\n", ordinal,
		     d.user_sgpr.size());
		return false;
	}
	const uint64_t srt = d.user_sgpr[0] | static_cast<uint64_t>(d.user_sgpr[1]) << 32u;

	// SRT and candidate headers are read through a throwaway snapshot.
	Rt::GuestSnapshot probe(FetchCoherent, 64ull << 20u);
	probe.AllowWindow(0, ~0ull);
	const auto            probe_reader = probe.Reader();
	std::vector<uint64_t> words(SrtBytes / 8u);
	uint32_t              srt_bytes = SrtBytes;
	if (srt == 0 || !probe.Read(srt, words.data(), SrtBytes)) {
		srt_bytes = 64;
		if (srt == 0 || !probe.Read(srt, words.data(), srt_bytes)) {
			LOGF("RtBvhDump: dispatch %" PRIu64 ": SRT 0x%" PRIx64 " is not readable\n", ordinal,
			     srt);
			return false;
		}
	}
	std::vector<uint32_t> offsets {0u, 40u};
	for (uint32_t offset = 8; offset < srt_bytes; offset += 8) {
		if (offset != 40u) {
			offsets.push_back(offset);
		}
	}
	// Every 256-aligned qword that looks like a Psr BVH (+0 and +40 first, at most four) is walked
	// as a TLAS; the first one whose walk succeeds with the most instances wins.
	uint64_t    tlas        = 0;
	uint32_t    tlas_offset = 0;
	size_t      best        = 0;
	bool        best_ok     = false;
	std::string tried;
	std::string candidates;
	uint32_t    plausible = 0;
	for (const uint32_t offset: offsets) {
		const uint64_t value = words[offset / 8u];
		if (offset >= srt_bytes || value == 0 || (value & 255u) != 0u || plausible >= 4u) {
			continue;
		}
		tried += " +" + std::to_string(offset) + "=" + Hex(value);
		if (!Rt::LooksLikePsrBvh(probe_reader, value)) {
			continue;
		}
		plausible++;
		Rt::TlasWalk walk;
		const auto   status = Rt::WalkTlas(probe_reader, value, walk);
		const bool   ok     = status == Rt::WalkStatus::Ok;
		candidates += " +" + std::to_string(offset) + "=" + Hex(value) + ":" +
		              Rt::ToString(status) + "/" + std::to_string(walk.instances.size());
		if (tlas == 0 || (ok && (!best_ok || walk.instances.size() > best))) {
			tlas        = value;
			tlas_offset = offset;
			best        = walk.instances.size();
			best_ok     = ok;
		}
	}
	if (tlas == 0) {
		LOGF("RtBvhDump: dispatch %" PRIu64 " (shader 0x%016" PRIx64
		     "): no Psr TLAS in the SRT at 0x%" PRIx64 " (256-aligned candidates:%s)\n",
		     ordinal, d.shader_hash, srt, tried.empty() ? " none" : tried.c_str());
		return false;
	}

	// At most 1 GiB of pages per dump (they are held in memory until written).
	Rt::GuestSnapshot snapshot(FetchCoherent, 1024ull << 20u);
	const auto        capture = Rt::CaptureBvh(snapshot, tlas, Rt::BvhCaptureLimits {});
	// The SRT bytes too, for offline reference.
	snapshot.AllowWindow(srt, srt + srt_bytes);
	snapshot.Prefetch(srt, srt + srt_bytes);

	auto& meta = snapshot.meta;
	meta.emplace_back("dump", "KYTY_RT_DUMP_BVH");
	meta.emplace_back("dispatch.ordinal", std::to_string(ordinal));
	meta.emplace_back("dispatch.frame", std::to_string(d.frame));
	meta.emplace_back("dispatch.shader_hash", Hex(d.shader_hash));
	meta.emplace_back("dispatch.shader_address", Hex(d.shader_address));
	meta.emplace_back("dispatch.program", d.skipped ? "skipped" : "translated");
	meta.emplace_back("dispatch.indirect", d.indirect ? "1" : "0");
	if (d.indirect) {
		meta.emplace_back("dispatch.args_address", Hex(d.args_address));
	} else {
		meta.emplace_back("dispatch.groups", std::to_string(d.groups[0]) + " " +
		                                         std::to_string(d.groups[1]) + " " +
		                                         std::to_string(d.groups[2]));
	}
	std::string user;
	for (const uint32_t value: d.user_sgpr) {
		user += (user.empty() ? "" : " ") + Hex(value);
	}
	meta.emplace_back("dispatch.user_sgpr", user);
	meta.emplace_back("srt.address", Hex(srt));
	meta.emplace_back("srt.tlas_offset", std::to_string(tlas_offset));
	meta.emplace_back("srt.candidates", candidates);
	Rt::AppendCaptureMeta(capture, snapshot, meta);

	std::error_code ec;
	const auto      folder = DumpFolder();
	std::filesystem::create_directories(folder, ec);
	char name[96];
	std::snprintf(name, sizeof(name), "bvh_%02u_d%" PRIu64 "_f%d_%" PRIx64, sequence, ordinal,
	              d.frame, tlas);
	const auto kbvh    = folder / (std::string(name) + ".kbvh");
	const auto summary = Rt::DescribeCapture(capture, snapshot);
	const bool saved   = snapshot.Save(kbvh);
	{
		std::ofstream text(folder / (std::string(name) + ".txt"), std::ios::trunc);
		text << summary << "\n";
		for (const auto& [key, value]: meta) {
			text << key << "=" << value << "\n";
		}
	}
	const auto path = kbvh.string();
	LOGF("RtBvhDump: dispatch %" PRIu64 " frame %d shader 0x%016" PRIx64 " (%s): TLAS 0x%" PRIx64
	     " at SRT+%u, %zu BLAS, %" PRIu64 " bytes -> %s%s\n%s",
	     ordinal, d.frame, d.shader_hash, d.skipped ? "skipped" : "translated", tlas, tlas_offset,
	     capture.blas.size(), snapshot.Bytes(), path.c_str(), saved ? "" : " (WRITE FAILED)",
	     summary.c_str());
	return true;
}

} // namespace

bool RtBvhDumpEnabled() noexcept {
	return !DumpOrdinals().empty();
}

void RtBvhDumpOnDispatch(const RtBvhDispatch& dispatch) {
	// GPU thread only: plain statics.
	static uint64_t count    = 0;
	static size_t   next     = 0;
	static uint32_t attempts = 0;
	static uint32_t written  = 0;
	const auto&     ordinals = DumpOrdinals();
	const uint64_t  ordinal  = ++count;
	if (next >= ordinals.size() || ordinal < ordinals[next]) {
		return;
	}
	if (DrawPrep::ActiveRecorder() != nullptr) {
		return; // never inside a draw preparation; the next BVH dispatch retries
	}
	const char* probe = std::getenv("KYTY_RT_PROBE_TLAS");
	const bool  done  = probe != nullptr && *probe != '\0' ? ProbeNow(dispatch, ordinal, probe)
	                                                       : DumpNow(dispatch, ordinal, written);
	written += done ? 1u : 0u;
	if (done || ++attempts >= 64u) {
		if (!done) {
			LOGF("RtBvhDump: giving up on requested dispatch %" PRIu64 "\n", ordinals[next]);
		}
		attempts = 0;
		while (next < ordinals.size() && ordinals[next] <= ordinal) {
			next++;
		}
	}
}

} // namespace Libs::Graphics
