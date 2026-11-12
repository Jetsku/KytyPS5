// KYTY_BDA_REACH (default off, live): footprint-limited BDA synchronization. Based on
// TheCruZ/KytyPS5-GTA (RenderContext::SynchronizeDmaFootprint / PrepareBda), adapted to the
// interval tape of our write-range analysis. See bufferCacheReach.cpp for the safety argument.

#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/alignment.h"
#include "common/liveSwitch.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/shader/recompiler/ir/passes/WriteRangeAnalysis.h"
#include "graphics/shader/shader.h"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

namespace {

// 0 off (the full pass for every operation), 1 on, 2 verify (bufferCacheReach.cpp).
int64_t ParseBdaReach(const char* value) {
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
		return 0;
	}
	return std::strcmp(value, "verify") == 0 ? 2 : 1;
}
Live::Switch g_bda_reach("KYTY_BDA_REACH", ParseBdaReach);

using ShaderRecompiler::IR::DmaReachFailure;
using ShaderRecompiler::IR::DmaUnbounded;

constexpr size_t UnboundedCount = static_cast<size_t>(DmaUnbounded::Count);
constexpr size_t FailureCount   = static_cast<size_t>(DmaReachFailure::Count) + 1u; // + gpu fill

const char* UnboundedName(size_t index) {
	static constexpr const char* names[] = {"none",     "address-writes", "ray-tracing",
	                                        "full-address", "address-handle", "indirect-handle",
	                                        "other-access", "too-many",       "tape-full"};
	return index < std::size(names) ? names[index] : "?";
}

const char* FailureName(size_t index) {
	static constexpr const char* names[] = {"none", "not-bounded", "base",  "offset",  "table-entries",
	                                        "table-read", "swizzle", "size", "gpu-fill"};
	return index < std::size(names) ? names[index] : "?";
}

} // namespace

struct RenderContext::BdaReach {
	ShaderRecompiler::IR::WriteRangeEvaluator      evaluator;
	std::vector<uint32_t>                          flat;
	std::vector<std::pair<uint64_t, uint64_t>>     spans;
	std::vector<std::pair<uint64_t, uint64_t>>     mapped;
	std::unordered_set<uint64_t>                   logged;
	struct Totals {
		uint64_t limited = 0, full = 0, span_bytes = 0, upload_bytes = 0, spans = 0, ns = 0;
		std::array<uint64_t, UnboundedCount> unbounded {};
		std::array<uint64_t, FailureCount>   failures {};
	};
	Totals                                         totals;
	Totals                                         reported;
	BufferCache::BdaReachVerifyTotals              verify_reported;
	std::chrono::steady_clock::time_point          report_time {};
	int                                            report_frame = 0;
	uint32_t                                       report_calls = 0;
	int64_t                                        mode = 0;
};

void RenderContext::PrepareBda(std::span<const BdaReachStage> stages) {
	const auto mode = g_bda_reach.Get();
	if (mode == 0 && m_bda_reach == nullptr) {
		PrepareBda();
		return;
	}
	if (m_bda_reach == nullptr) {
		m_bda_reach = std::make_shared<BdaReach>();
	}
	auto&      reach = *m_bda_reach;
	const auto start = std::chrono::steady_clock::now();
	bool       limited = mode != 0;
	reach.spans.clear();
	for (const auto& stage: stages) {
		if (!limited) {
			break;
		}
		if (stage.runtime == nullptr || stage.runtime->program == nullptr || stage.runtime->resources == nullptr) {
			limited = false;
			reach.totals.failures[static_cast<size_t>(DmaReachFailure::NotBounded)]++;
			break;
		}
		const auto& program = *stage.runtime->program;
		if (!program.info.uses_dma) {
			continue;
		}
		const auto& ranges    = program.write_ranges;
		const auto& resources = *stage.runtime->resources;
		if (!ranges.dma.bounded) {
			limited = false;
			reach.totals.unbounded[std::min<size_t>(ranges.dma.reason, UnboundedCount - 1u)]++;
			if (reach.logged.size() < 256 && reach.logged.insert(program.shader_hash).second) {
				std::printf("BdaReach: program 0x%016" PRIx64 " stage %u reads through device addresses with an "
				     "unproven reach (%s): full pass\n",
				     program.shader_hash, static_cast<uint32_t>(program.stage), UnboundedName(ranges.dma.reason));
			}
			break;
		}
		// A flat word the GPU fills holds a placeholder on the host.
		if (!resources.gpu_fills.empty()) {
			limited = false;
			reach.totals.failures[FailureCount - 1u]++;
			break;
		}
		std::span<const uint32_t> flat = resources.flattened_srt;
		if (!stage.bindless_patches.empty()) {
			reach.flat.assign(flat.begin(), flat.end());
			for (const auto& [offset, region, entries]: stage.bindless_patches) {
				if (static_cast<size_t>(offset) + 1u < reach.flat.size()) {
					reach.flat[offset]      = region;
					reach.flat[offset + 1u] = entries;
				}
			}
			flat = reach.flat;
		}
		reach.evaluator.Evaluate(ranges, {.user_data     = resources.user_data,
		                                  .flattened_srt = flat,
		                                  .groups        = stage.groups,
		                                  .has_groups    = stage.has_groups});
		ShaderRecompiler::IR::DmaReachInputs inputs;
		inputs.user = &m_buffer_cache;
		if (stage.mesh_index_known) {
			inputs.mesh_index_known = true;
			inputs.mesh_index_begin = stage.mesh_index_begin;
			inputs.mesh_index_end   = stage.mesh_index_end;
		}
		inputs.read = [](void* user, uint64_t address, std::span<uint32_t> words) {
			return static_cast<BufferCache*>(user)->ReadGuestForBdaReach(address, words);
		};
		DmaReachFailure why = DmaReachFailure::None;
		if (!reach.evaluator.DmaSpans(ranges, inputs, reach.spans, why)) {
			limited = false;
			reach.totals.failures[std::min<size_t>(static_cast<size_t>(why), FailureCount - 1u)]++;
			if (reach.logged.size() < 256 && reach.logged.insert(program.shader_hash ^ 0x5bd1e995ull).second) {
				std::printf("BdaReach: program 0x%016" PRIx64 " stage %u reach not known for a draw (%s): %s\n",
				            program.shader_hash, static_cast<uint32_t>(program.stage),
				            FailureName(static_cast<size_t>(why)), reach.evaluator.DescribeDmaFailure(ranges).c_str());
			}
			break;
		}
	}
	if (!limited) {
		PrepareBda();
		// Verification traps go away once a full pass ran (or the switch is off).
		m_buffer_cache.RestoreBdaReachTraps(mode == 2);
		reach.totals.full++;
	} else {
		// Whole caching pages (the page table's granularity), merged, inside the GPU mappings.
		ShaderRecompiler::IR::NormalizeDmaSpans(reach.spans, BufferCache::CACHING_PAGESIZE);
		reach.mapped.clear();
		uint64_t bytes = 0;
		{
			std::shared_lock lock(m_mapped_ranges_mutex);
			for (const auto& [begin, end]: reach.spans) {
				m_mapped_ranges.ForEachInRange(begin, end - begin, [&](uint64_t first, uint64_t last) {
					// The mapped ranges are page aligned; keep caching-page alignment anyway.
					first = Common::AlignDown(first, BufferCache::CACHING_PAGESIZE);
					last  = Common::AlignUp(last, BufferCache::CACHING_PAGESIZE);
					if (!reach.mapped.empty() && first <= reach.mapped.back().second) {
						reach.mapped.back().second = std::max(reach.mapped.back().second, last);
					} else {
						reach.mapped.emplace_back(first, last);
					}
				});
			}
		}
		for (const auto& [begin, end]: reach.mapped) {
			bytes += end - begin;
		}
		reach.totals.upload_bytes += m_buffer_cache.SynchronizeBdaReach(reach.mapped, mode == 2);
		reach.totals.limited++;
		reach.totals.spans += reach.mapped.size();
		reach.totals.span_bytes += bytes;
		m_fault_process_pending = true;
	}
	reach.totals.ns += static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
	reach.mode = mode;
	// Every 10 s while the switch is on: one console line.
	if ((++reach.report_calls & 255u) != 0 || mode == 0) {
		return;
	}
	const auto now   = std::chrono::steady_clock::now();
	const int  frame = m_gpu != nullptr ? m_gpu->GetFrameNum() : 0;
	if (reach.report_time == std::chrono::steady_clock::time_point {}) {
		reach.report_time  = now;
		reach.report_frame = frame;
		reach.reported     = reach.totals;
		reach.verify_reported = m_buffer_cache.GetBdaReachVerifyTotals();
		return;
	}
	const auto elapsed = std::chrono::duration<double>(now - reach.report_time).count();
	if (elapsed < 10.0) {
		return;
	}
	const double frames = std::max(1, frame - reach.report_frame);
	const auto&  t      = reach.totals;
	const auto&  r      = reach.reported;
	std::string  why;
	for (size_t index = 1; index < UnboundedCount; index++) {
		if (t.unbounded[index] != r.unbounded[index]) {
			why += fmt::format(" {} {:.1f}", UnboundedName(index), (t.unbounded[index] - r.unbounded[index]) / frames);
		}
	}
	for (size_t index = 1; index < FailureCount; index++) {
		if (t.failures[index] != r.failures[index]) {
			why += fmt::format(" {} {:.1f}", FailureName(index), (t.failures[index] - r.failures[index]) / frames);
		}
	}
	std::string verify;
	if (mode == 2) {
		const auto v = m_buffer_cache.GetBdaReachVerifyTotals();
		const auto& p = reach.verify_reported;
		verify = fmt::format(
		    " | verify: checks {} inside-dirty pages {} (total {}), trapped {} restored {}, trap faults {} "
		    "(total {}), log overflows {}",
		    v.checks - p.checks, v.inside_dirty - p.inside_dirty, v.inside_dirty, v.trapped - p.trapped,
		    v.restored - p.restored, v.trap_faults - p.trap_faults, v.trap_faults, v.log_overflows - p.log_overflows);
		reach.verify_reported = v;
	}
	std::printf("BdaReach %.0fs (%s): %.0f frames | per frame: limited %.1f, full %.1f (%s), spans %.1f covering "
	            "%.2f MB, limited uploads %.3f MB, %.3f ms in PrepareBda%s\n",
	            elapsed, mode == 2 ? "verify" : "on", frames, (t.limited - r.limited) / frames,
	            (t.full - r.full) / frames, why.empty() ? "-" : why.c_str() + 1, (t.spans - r.spans) / frames,
	            (t.span_bytes - r.span_bytes) / frames / 1e6, (t.upload_bytes - r.upload_bytes) / frames / 1e6,
	            (t.ns - r.ns) / frames / 1e6, verify.c_str());
	std::fflush(stdout);
	reach.report_time  = now;
	reach.report_frame = frame;
	reach.reported     = t;
}

} // namespace Libs::Graphics
