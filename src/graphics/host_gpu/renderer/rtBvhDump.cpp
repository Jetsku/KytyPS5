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
			const size_t end   = std::min(text.find_first_of(",; ", at), text.size());
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
	const bool done = DumpNow(dispatch, ordinal, written);
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
