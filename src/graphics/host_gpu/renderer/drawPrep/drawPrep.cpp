#include "graphics/host_gpu/renderer/drawPrep/drawPrep.h"

#include "graphics/host_gpu/renderer/drawPrep/ownedCounters.h"

#include "common/assert.h"
#include "common/prepPhases.h"
#include "common/cpuPlacement.h"
#include "common/hangWatchdog.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/command_processor/cpOps.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/coherenceLog.h"
#include "graphics/host_gpu/renderer/cache/texelImageLookup.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/cpCommit.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/drawPrep/bindingPlan.h"
#include "graphics/host_gpu/renderer/drawPrep/commitStats.h"
#include "graphics/host_gpu/renderer/drawPrep/cpGaps.h"
#include "graphics/host_gpu/renderer/drawPrep/drawRun.h"
#include "graphics/host_gpu/renderer/drawPrep/repeatTrace.h"
#include "graphics/host_gpu/renderer/drawPrep/window.h"
#include "graphics/host_gpu/renderer/drawPrep/workerGate.h"
#include "graphics/host_gpu/renderer/drawPrep/xframeReuse.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <thread>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Libs::Graphics::DrawPrep {

namespace {

const char* EnvValue(const char* name) {
	const auto* value = std::getenv(name);
	return value != nullptr && *value != '\0' ? value : nullptr;
}

Profiler::FrameEvent FallbackEvent(Failure failure) {
	using E = Profiler::FrameEvent;
	switch (failure) {
		case Failure::Unclean: return E::DrawPrepFallbackUnclean;
		case Failure::Backing: return E::DrawPrepFallbackBacking;
		case Failure::Overflow: return E::DrawPrepFallbackOverflow;
		case Failure::Inconsistent: return E::DrawPrepFallbackInconsistent;
		case Failure::Uncertified: return E::DrawPrepFallbackUncertified;
		case Failure::NotPublished: return E::DrawPrepFallbackNotPublished;
		case Failure::ShaderMap: return E::DrawPrepFallbackShaderMap;
		case Failure::CertUnclean: return E::DrawPrepFallbackCertUnclean;
		case Failure::CertChanged: return E::DrawPrepFallbackCertChanged;
		case Failure::CoherenceLog: return E::DrawPrepFallbackCoherenceLog;
		case Failure::Mismatch: return E::DrawPrepFallbackMismatch;
		case Failure::None:
		case Failure::Ineligible: break;
	}
	return E::DrawPrepFallbackIneligible;
}

Failure FromReadFailure(ReadFailure failure) {
	switch (failure) {
		case ReadFailure::Unclean: return Failure::Unclean;
		case ReadFailure::Backing: return Failure::Backing;
		case ReadFailure::Overflow: return Failure::Overflow;
		case ReadFailure::Inconsistent: return Failure::Inconsistent;
		case ReadFailure::Uncertified: return Failure::Uncertified;
		case ReadFailure::None: break;
	}
	return Failure::Uncertified;
}

// The same decisions RenderExecutor makes from the command buffer's registers
// (DrawHasActivePixelShader, RefreshShaders); the commit compares both.
bool HasActivePixelShader(const RegisterSnapshot& registers) {
	const auto& ctx     = registers.context;
	const auto& sh_regs = ctx.GetShaderRegisters();
	const auto& db      = sh_regs.db_shader_control;
	const bool  has_color_output = SkipInactivePixelShadersEnabled()
	                                   ? DrawColorOutputFilter(ctx) != 0
	                                   : (ctx.GetRenderTargetMask() & sh_regs.m_cbShaderMask) != 0;
	const bool  side_effects = db.shader_kill_enable || db.shader_z_export_enable ||
	                          db.shader_mask_export_enable || db.shader_dual_export_enable ||
	                          db.shader_execute_on_noop;
	return registers.shaders.GetPs().ps_regs.data_addr != 0 && (has_color_output || side_effects);
}

void TargetExportMapping(const HW::Context&                              ctx,
                         std::array<Prospero::ColorComponentMapping, 8>& mapping) {
	mapping = {};
	const auto output_filter = DrawColorOutputFilter(ctx);
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		const auto& rt = ctx.GetRenderTarget(slot);
		if (rt.base.addr != 0 && render_target_mask_slot(ctx.GetRenderTargetMask(), slot) != 0 &&
		    (output_filter & (1u << slot)) != 0) {
			mapping[slot] = TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
			                                             rt.info.channel_order)
			                    .export_mapping;
		}
	}
}

bool SameMapping(std::span<const Prospero::ColorComponentMapping, 8> a,
                 std::span<const Prospero::ColorComponentMapping, 8> b) {
	for (size_t i = 0; i < a.size(); i++) {
		if (a[i].packed != b[i].packed) {
			return false;
		}
	}
	return true;
}

// Every range a certificate covers (bytes and digests), sorted and merged, as the coherence log
// check requires.
std::span<const Coherence::Range> CertificateRanges(const ReadSet&                  reads,
                                                    std::vector<Coherence::Range>& scratch) {
	const auto digests = reads.DigestRanges();
	if (digests.empty()) {
		return reads.Ranges();
	}
	scratch.assign(reads.Ranges().begin(), reads.Ranges().end());
	scratch.insert(scratch.end(), digests.begin(), digests.end());
	std::sort(scratch.begin(), scratch.end(),
	          [](const Coherence::Range& a, const Coherence::Range& b) { return a.begin < b.begin; });
	size_t merged = 0;
	for (const auto& range: scratch) {
		if (merged != 0 && range.begin <= scratch[merged - 1].end) {
			scratch[merged - 1].end = std::max(scratch[merged - 1].end, range.end);
		} else {
			scratch[merged++] = range;
		}
	}
	scratch.resize(merged);
	return scratch;
}

Totals g_totals;

} // namespace

Totals& GetTotals() {
	return g_totals;
}

Mode GetMode() {
	static const Mode mode = [] {
		// Parallel by default: checked against the serial path (KYTY_DRAW_PREP_VERIFY) with no
		// mismatch in Astro Bot's Sky Garden, pool, hub and desert (U48-U50). =off reverts.
		const auto* value = EnvValue("KYTY_DRAW_PREP");
		if (value == nullptr) {
			return Mode::Parallel;
		}
		if (std::strcmp(value, "off") == 0 || std::strcmp(value, "0") == 0) {
			return Mode::Off;
		}
		if (std::strcmp(value, "inline") == 0) {
			return Mode::Inline;
		}
		if (std::strcmp(value, "parallel") == 0) {
			return Mode::Parallel;
		}
		EXIT("KYTY_DRAW_PREP must be off, inline or parallel (got '%s')\n", value);
	}();
	return mode;
}

int VerifyMode() {
	static const int mode = [] {
		const auto* value = EnvValue("KYTY_DRAW_PREP_VERIFY");
		if (value == nullptr || std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	}();
	return mode;
}

CertMode GetCertMode() {
	static const CertMode mode = [] {
		// Log by default (Run A audit, CP-ARCHITECTURE-20260927.md: DrawPrepLogMissed 0 in every
		// scene); KYTY_DRAW_PREP_CERT=value restores the byte and digest comparisons.
		const auto* value = EnvValue("KYTY_DRAW_PREP_CERT");
		return value != nullptr && std::strcmp(value, "value") == 0 ? CertMode::Value : CertMode::Log;
	}();
	return mode;
}

// KYTY_DRAW_PREP_LOG_AUDIT=1: with value certificates, also evaluate the log check and count
// where it would have decided differently (DrawPrepLogWouldReject, DrawPrepLogMissed).
static bool LogAuditEnabled() {
	static const bool enabled = [] {
		const auto* value = EnvValue("KYTY_DRAW_PREP_LOG_AUDIT");
		return value != nullptr && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

bool CertRangesOnWorker() {
	static const bool enabled = [] {
		// The preparing thread builds the list by default: it is a pure function of the finished
		// read set (ReadSet::BuildCertificate). =commit builds it in Validate, as before.
		const auto* value = EnvValue("KYTY_DRAW_PREP_CERT_RANGES");
		if (value == nullptr || std::strcmp(value, "worker") == 0) {
			return true;
		}
		if (std::strcmp(value, "commit") == 0) {
			return false;
		}
		EXIT("KYTY_DRAW_PREP_CERT_RANGES must be worker or commit (got '%s')\n", value);
	}();
	return enabled;
}

int CertRangesVerifyMode() {
	static const int mode = [] {
		const auto* value = EnvValue("KYTY_DRAW_PREP_CERT_RANGES_VERIFY");
		if (value == nullptr || std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	}();
	return mode;
}

// Whether a successful preparation's certificate ranges are needed at commit: the log check
// (log mode), or its audit on top of value certificates.
static bool CertificateRangesUsed() {
	return GetCertMode() == CertMode::Log || LogAuditEnabled();
}

// The certificate ranges of the log check at commit. A preparation that built them
// (KYTY_DRAW_PREP_CERT_RANGES=worker) hands over its list; otherwise they are built here into
// `scratch`, the path before this switch. KYTY_DRAW_PREP_CERT_RANGES_VERIFY rebuilds a handed-over
// list with that commit-time function and compares the two.
static std::span<const Coherence::Range> CommitCertificateRanges(
    const PreparedDraw& prepared, std::vector<Coherence::Range>& scratch) {
	if (!prepared.certificate_built) {
		return CertificateRanges(prepared.reads, scratch);
	}
	OwnedCounters::Add(g_totals.cert_ranges_prebuilt, 1);
	if (CertRangesVerifyMode() != 0) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepCertRangesVerifyChecks);
		g_totals.cert_ranges_verify_checks.fetch_add(1, std::memory_order_relaxed);
		const auto reference = CertificateRanges(prepared.reads, scratch);
		if (!std::equal(reference.begin(), reference.end(), prepared.certificate.begin(),
		                prepared.certificate.end())) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepCertRangesVerifyMismatches);
			g_totals.cert_ranges_verify_mismatches.fetch_add(1, std::memory_order_relaxed);
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
				LOGF("DrawPrepCertRangesVerify: prepared certificate has %zu ranges, the commit-time "
				     "list %zu (%zu byte ranges, %zu digest ranges)\n",
				     prepared.certificate.size(), reference.size(), prepared.reads.Ranges().size(),
				     prepared.reads.DigestRanges().size());
			}
			if (CertRangesVerifyMode() == 2) {
				EXIT("DrawPrepCertRangesVerify: prepared certificate ranges differ from the "
				     "commit-time list\n");
			}
			return reference; // the commit-time list decides, as before the switch
		}
	}
	return prepared.certificate;
}

bool PacketHookEnabled() {
	static const bool enabled = [] {
		const auto* value = EnvValue("KYTY_DRAW_PREP_HISTOGRAM");
		return GetMode() != Mode::Off || (value != nullptr && std::strcmp(value, "0") != 0);
	}();
	return enabled;
}

bool RegisterIndirectWindowEnabled() {
	static const bool enabled = [] {
		const auto* value = EnvValue("KYTY_DRAW_PREP_REG_INDIRECT_WINDOW");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

bool PacketHookActive() {
	if (PacketHookEnabled()) {
		return true;
	}
	// Opt-in: classifying every packet costs the command processor about 20-50 ns per packet,
	// which would skew aggregate comparisons. Inline and parallel modes count fence kinds anyway.
	static const bool passive = [] {
		const auto* value = EnvValue("KYTY_DRAW_PREP_FENCE_HISTOGRAM");
		return Profiler::AggregateEnabled() && value != nullptr && std::strcmp(value, "0") != 0;
	}();
	return passive && tracy::ProfilerAvailable() && TracyIsConnected;
}

void Prepare(PipelineCache& pipeline_cache, const RegisterSnapshot& registers, bool eligible,
             bool exact, PreparedDraw& prepared) {
	prepared.program_compile_ns = 0;
	prepared.ok         = false;
	prepared.failure    = Failure::None;
	prepared.programs   = {};
	prepared.pixel_info = {};
	prepared.reads.Reset();
	prepared.reads.SetCoalesce(DirectReadsEnabled());
	prepared.certificate.clear();
	prepared.certificate_built = false;
	if (!eligible) {
		prepared.failure = Failure::Ineligible;
		return;
	}
	Profiler::ScopedFrameWait wait(Profiler::FrameWait::DrawPrepPrepare);
	// Loaded before any read: every coherence transition after this point is newer.
	prepared.coherence_generation  = Coherence::Generation();
	prepared.shader_map_generation = ShaderMapGeneration();

	const auto& ctx    = registers.context;
	const auto& sh_ctx = registers.shaders;
	if (sh_ctx.GetVs().es_regs.data_addr == 0) {
		prepared.failure = Failure::Ineligible;
		return;
	}
	prepared.pixel_active = HasActivePixelShader(registers);
	TargetExportMapping(ctx, prepared.target_export_mapping);
	Common::PrepPhases::Mark(Common::PrepPhases::Phase::Reset);

	Recorder    recorder {&prepared.reads, exact};
	RecordScope scope(recorder);
	const auto  result = pipeline_cache.PrepareGraphicsProgramsSpeculative(
	    sh_ctx.GetVs(), sh_ctx.GetPs(), ctx.GetShaderRegisters(), ctx, registers.user_config,
	    prepared.target_export_mapping, prepared.pixel_active, prepared.vertex_info,
	    prepared.pixel_info, prepared.vertex_prep, prepared.pixel_prep, prepared.programs,
	    &prepared.program_compile_ns);
	switch (result) {
		case PipelineCache::SpeculativeResult::Ok: break;
		case PipelineCache::SpeculativeResult::Ineligible:
			prepared.failure = Failure::Ineligible;
			return;
		case PipelineCache::SpeculativeResult::NotPublished:
			prepared.failure = prepared.reads.Failed() ? FromReadFailure(prepared.reads.Failure())
			                                           : Failure::NotPublished;
			return;
		case PipelineCache::SpeculativeResult::ReadFailed:
			prepared.failure = FromReadFailure(prepared.reads.Failure());
			return;
	}
	if (!prepared.reads.Finish()) {
		prepared.failure = FromReadFailure(prepared.reads.Failure());
		return;
	}
	Common::PrepPhases::Mark(Common::PrepPhases::Phase::Finish);
	if (CertRangesOnWorker() && CertificateRangesUsed()) {
		// The read set is final: its certificate ranges are too. Validate on the command
		// processor uses them instead of sorting and merging there (KYTY_DRAW_PREP_CERT_RANGES).
		prepared.reads.BuildCertificate(prepared.certificate);
		prepared.certificate_built = true;
	}
	Common::PrepPhases::Mark(Common::PrepPhases::Phase::Certificate);
	prepared.ok = true;
}

// KYTY_DRAW_PREP_DIRECT_READS (readSet.h, kernel/memory.cpp TryReadForDrawPrep): live, default 1;
// 0 off, verify / exit compare each direct copy with a backing read.
static Live::Switch g_direct_reads("KYTY_DRAW_PREP_DIRECT_READS", [](const char* value) -> int64_t {
	if (value == nullptr || value[0] == '\0') {
		return 1;
	}
	if (std::strcmp(value, "verify") == 0) {
		return 2;
	}
	if (std::strcmp(value, "exit") == 0) {
		return 3;
	}
	return std::strcmp(value, "0") == 0 ? 0 : 1;
});

int DirectReadsMode() {
	return static_cast<int>(g_direct_reads.Get());
}

bool DirectReadsEnabled() {
	return g_direct_reads.Get() != 0;
}

// Live (common/liveSwitch.h): the sequencer reads it per dispatch it parses; the op carries the
// decision to the resolver, so a change applies from the next parsed dispatch.
static Live::Switch g_dispatch_prep("KYTY_DRAW_PREP_DISPATCH", Live::ParseDefaultOff);

bool DispatchPrepEnabled() {
	return g_dispatch_prep.On();
}

void MakeComputeInputInfo(const HW::ComputeShaderInfo& cs, const DispatchArgs& args,
                          ShaderComputeInputInfo& info) {
	constexpr uint32_t DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS = 1u << 5u;
	info                                  = {};
	if (args.indirect) {
		return; // RenderExecutor::DispatchIndirect's lookup: no counts, no thread dimensions
	}
	const bool use_thread_dimensions      = (args.mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	info.dispatch_thread_dimensions       = use_thread_dimensions;
	info.workgroup_counts[0]              = args.x;
	info.workgroup_counts[1]              = args.y;
	info.workgroup_counts[2]              = args.z;
	if (use_thread_dimensions) {
		const uint32_t group_sizes[] = {cs.cs_regs.num_thread_x, cs.cs_regs.num_thread_y,
		                                cs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			const auto size    = std::max(group_sizes[axis], 1u);
			const auto threads = info.workgroup_counts[axis];
			info.workgroup_counts[axis] = threads / size + (threads % size != 0u);
		}
	}
}

void PrepareDispatch(PipelineCache& pipeline_cache, const RegisterSnapshot& registers,
                     const DispatchArgs& args, bool eligible, bool exact, PreparedDraw& prepared) {
	prepared.program_compile_ns = 0;
	prepared.ok                 = false;
	prepared.failure            = Failure::None;
	prepared.compute_program    = {};
	prepared.reads.Reset();
	prepared.reads.SetCoalesce(DirectReadsEnabled());
	prepared.certificate.clear();
	prepared.certificate_built = false;
	if (!eligible) {
		prepared.failure = Failure::Ineligible;
		return;
	}
	Profiler::ScopedFrameWait wait(Profiler::FrameWait::DrawPrepPrepare);
	// Loaded before any read: every coherence transition after this point is newer.
	prepared.coherence_generation  = Coherence::Generation();
	prepared.shader_map_generation = ShaderMapGeneration();
	const auto& cs = registers.shaders.GetCs();
	MakeComputeInputInfo(cs, args, prepared.compute_info);
	Recorder    recorder {&prepared.reads, exact};
	RecordScope scope(recorder);
	const auto  result = pipeline_cache.PrepareComputeProgramSpeculative(
        cs, registers.context.GetShaderRegisters(), prepared.compute_info, prepared.compute_prep,
        prepared.compute_program, &prepared.program_compile_ns);
	switch (result) {
		case PipelineCache::SpeculativeResult::Ok: break;
		case PipelineCache::SpeculativeResult::Ineligible:
			prepared.failure = Failure::Ineligible;
			return;
		case PipelineCache::SpeculativeResult::NotPublished:
			prepared.failure = prepared.reads.Failed() ? FromReadFailure(prepared.reads.Failure())
			                                           : Failure::NotPublished;
			return;
		case PipelineCache::SpeculativeResult::ReadFailed:
			prepared.failure = FromReadFailure(prepared.reads.Failure());
			return;
	}
	if (!prepared.compute_program || !prepared.reads.Finish()) {
		prepared.failure = prepared.reads.Failed() ? FromReadFailure(prepared.reads.Failure())
		                                           : Failure::NotPublished;
		return;
	}
	if (CertRangesOnWorker() && CertificateRangesUsed()) {
		prepared.reads.BuildCertificate(prepared.certificate);
		prepared.certificate_built = true;
	}
	prepared.ok = true;
}

// KYTY_BACKING_INPLACE_VERIFY=1|exit: every in-place validation is repeated with copies and the two
// verdicts compared (the copied one is used). A guest write landing between the two runs can make
// them differ legitimately: a difference is counted as a mismatch only if a second pair of runs
// differs too, otherwise as a race.
static int InPlaceVerifyMode() {
	static const int mode = [] {
		const auto* value = EnvValue("KYTY_BACKING_INPLACE_VERIFY");
		if (value == nullptr || std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	}();
	return mode;
}

// The value certificate check (readSet.h), in place (KYTY_BACKING_INPLACE) or on copies.
static ValidateResult ValidateValues(const ReadSet& reads) {
	static thread_local std::vector<uint8_t> scratch;
	const auto copied = [&] {
		return reads.Validate(LibKernel::Memory::TryReadGpuCleanBacking, scratch);
	};
	if (!LibKernel::Memory::BackingInPlaceEnabled()) {
		return copied();
	}
	LibKernel::Memory::InPlaceStats stats;
	const auto in_place = [&] {
		return reads.ValidateInPlace(
		    [&](uint64_t address, const uint8_t* expected, uint64_t size) {
			    switch (LibKernel::Memory::CompareGpuCleanBacking(address, expected, size, &stats)) {
				    case LibKernel::Memory::BackingCompare::Equal: return ValidateResult::Ok;
				    case LibKernel::Memory::BackingCompare::Different: return ValidateResult::Changed;
				    case LibKernel::Memory::BackingCompare::Unavailable: break;
			    }
			    return ValidateResult::Unclean;
		    },
		    [&](uint64_t address, uint64_t size, uint64_t& digest) {
			    return LibKernel::Memory::HashGpuCleanBacking(address, size, digest, &stats);
		    });
	};
	auto result = in_place();
	if (InPlaceVerifyMode() != 0) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepValidateVerifyChecks);
		auto reference = copied();
		if (reference != result) {
			result    = in_place();
			reference = copied();
			if (reference == result) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepValidateVerifyRaces);
			} else {
				Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepValidateVerifyMismatches);
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
					LOGF("DrawPrepValidateVerify: in-place verdict %d, copied verdict %d (%zu ranges, "
					     "%zu digests)\n",
					     static_cast<int>(result), static_cast<int>(reference), reads.Ranges().size(),
					     reads.DigestRanges().size());
				}
				if (InPlaceVerifyMode() == 2) {
					EXIT("DrawPrepValidateVerify: in-place validation differs from the copied one\n");
				}
			}
		}
		result = reference;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepValidateInPlaceRanges, stats.inspected);
	if (stats.locked != 0) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepValidateInPlaceLocked, stats.locked);
	}
	return result;
}

// Certificate modes:
// - log (default): no coherence transition logged since the preparation began may touch a read
//   or digest range, and every range must be clean now; bytes are neither compared nor re-hashed.
//   When a transition did touch a range (or the log interval could not be read), the value check
//   below decides instead, so log mode refuses exactly what value mode refuses.
//   It relies on the log being complete for emulator-side changes and treats unsynchronized guest
//   CPU writes as races (they are: only a fence orders them against a draw, and a fence commits
//   the window before its handler runs). Audited with KYTY_DRAW_PREP_CERT=value
//   KYTY_DRAW_PREP_LOG_AUDIT=1 (DrawPrepLogMissed counts the draws log mode would have accepted
//   with changed bytes: 0 in all five Run A scenes).
// - value: every coalesced read range must be clean for a backing read now and hold the recorded
//   bytes, and every digest range must hash to the recorded digest. Sound on its own (readSet.h).
// Per-reason fallback counts for the periodic console line (PrintDrawPrepSummary).
std::array<std::atomic<uint64_t>, static_cast<size_t>(Failure::Mismatch) + 1u> g_fallback_reasons;

// KYTY_DRAW_PREP_CLEAN_SINCE (live, default 0): the "every range clean now" part of a log-mode
// certificate whose log check found no transition touching the ranges since the preparation
// began (coherence_generation) takes page verdicts found since then
// (LibKernel::Memory::IsGpuCleanForReadSince) instead of the exact predicates for every page. The
// per-generation verdict cache that IsGpuCleanForRead uses drops every verdict at any transition
// anywhere, which happens between almost all commits, so at the Sky Garden the exact checks were
// about 4.7% of the command processor's time. 1 on; verify runs both, uses the exact result and
// counts the reads the shortcut would have called clean that are not; exit stops at the first.
static int64_t ParseCleanSince(const char* value) {
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
		return 0;
	}
	if (std::strcmp(value, "verify") == 0) {
		return 2;
	}
	return std::strcmp(value, "exit") == 0 ? 3 : 1;
}
static Live::Switch g_clean_since("KYTY_DRAW_PREP_CLEAN_SINCE", ParseCleanSince);
static std::atomic<uint64_t> g_clean_since_checks {0};
static std::atomic<uint64_t> g_clean_since_mismatches {0};

static bool AllCleanAfterLog(const PreparedDraw& prepared) {
	const auto mode = g_clean_since.Get();
	if (mode == 0) {
		return prepared.reads.AllClean(LibKernel::Memory::IsGpuCleanForRead);
	}
	const auto since = prepared.coherence_generation;
	const auto shortcut = [since](uint64_t address, uint64_t size) {
		return LibKernel::Memory::IsGpuCleanForReadSince(address, size, since);
	};
	if (mode == 1) {
		return prepared.reads.AllClean(shortcut);
	}
	const bool fast  = prepared.reads.AllClean(shortcut);
	const bool exact = prepared.reads.AllClean(LibKernel::Memory::IsGpuCleanForRead);
	g_clean_since_checks.fetch_add(1, std::memory_order_relaxed);
	if (fast && !exact) {
		const auto count = g_clean_since_mismatches.fetch_add(1, std::memory_order_relaxed) + 1;
		if (count <= 16) {
			LOGF("DrawPrepCleanSince: a read the page verdicts called clean is not (%zu ranges, since "
			     "generation %" PRIu64 ")\n",
			     prepared.reads.Ranges().size(), since);
		}
		if (mode == 3) {
			EXIT("DrawPrepCleanSince: a read the page verdicts called clean is not\n");
		}
	}
	if ((g_clean_since_checks.load(std::memory_order_relaxed) & 0xfffffu) == 0) {
		std::printf("DrawPrepCleanSince verify: %" PRIu64 " checks, %" PRIu64 " mismatches\n",
		            g_clean_since_checks.load(std::memory_order_relaxed),
		            g_clean_since_mismatches.load(std::memory_order_relaxed));
	}
	return exact;
}

// The certificate part of Validate (draws) and ValidateDispatch: the shader map, then the log or
// value check of every read. False with `failure` set when it does not hold.
static bool CertificateHolds(PreparedDraw& prepared, Failure& failure) {
	const auto fail = [&](Failure reason) {
		failure = reason;
		return false;
	};
	if (ShaderMapGeneration() != prepared.shader_map_generation) {
		return fail(Failure::ShaderMap);
	}
	Profiler::ScopedFrameWait wait(Profiler::FrameWait::DrawPrepValidate);
	static thread_local std::vector<Coherence::Range> log_scratch;
	if (GetCertMode() == CertMode::Log) {
		const auto stats_start = CommitStats::StepsEnabled() ? CommitStats::Now() : 0;
		const auto cert_ranges = CommitCertificateRanges(prepared, log_scratch);
		const auto outcome =
		    Coherence::g_log.Check(prepared.coherence_generation, Coherence::Generation(),
		                           cert_ranges);
		if (stats_start != 0) {
			CommitStats::AddSub(CommitStats::Sub::ValLog, CommitStats::Now() - stats_start);
			CommitStats::AddItems(CommitStats::Item::LogEntries, outcome.entries);
			CommitStats::AddItems(CommitStats::Item::CertRanges, cert_ranges.size());
		}
		// The walk's length grows with the preparation-to-commit interval (KYTY_CP_SEQ=1).
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepLogChecks);
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepLogEntries, outcome.entries);
		if (outcome.result == Coherence::CheckResult::Clean) {
			const auto clean_start = CommitStats::StepsEnabled() ? CommitStats::Now() : 0;
			const bool clean       = AllCleanAfterLog(prepared);
			if (clean_start != 0) {
				CommitStats::AddSub(CommitStats::Sub::ValClean, CommitStats::Now() - clean_start);
			}
			if (!clean) {
				return fail(Failure::CertUnclean);
			}
		} else {
			// A logged transition touched a certified range, or the interval could not be read:
			// decide by value. Log mode then refuses exactly what the value certificate refuses;
			// it only skips the byte and digest comparisons where no transition happened.
			using E               = Profiler::FrameEvent;
			using Result          = Coherence::CheckResult;
			const auto log_result = outcome.result;
			Profiler::CountFrameEvent(log_result == Result::Conflict  ? E::DrawPrepLogConflicts
			                          : log_result == Result::Unknown ? E::DrawPrepLogUnknown
			                                                          : E::DrawPrepLogOverflows);
			switch (ValidateValues(prepared.reads)) {
				case ValidateResult::Ok:
					Profiler::CountFrameEvent(E::DrawPrepLogValueRescues);
					break;
				case ValidateResult::Unclean: return fail(Failure::CertUnclean);
				case ValidateResult::Changed: return fail(Failure::CertChanged);
			}
		}
	} else {
		const auto result = ValidateValues(prepared.reads);
		if (LogAuditEnabled()) {
			// Would the log-mode certificate have decided the same? (Unclean ranges fail both.)
			const auto audit =
			    Coherence::g_log.Check(prepared.coherence_generation, Coherence::Generation(),
			                           CommitCertificateRanges(prepared, log_scratch));
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepLogChecks);
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepLogEntries, audit.entries);
			const bool log_clean = audit.result == Coherence::CheckResult::Clean;
			if (log_clean && result == ValidateResult::Changed) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepLogMissed);
			} else if (!log_clean && result == ValidateResult::Ok) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepLogWouldReject);
			}
		}
		switch (result) {
			case ValidateResult::Ok: break;
			case ValidateResult::Unclean: return fail(Failure::CertUnclean);
			case ValidateResult::Changed: return fail(Failure::CertChanged);
		}
	}
	return true;
}

static void CountCertificate(const PreparedDraw& prepared) {
	const auto ranges = prepared.reads.Ranges();
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepCertRanges, ranges.size());
	uint64_t bytes = 0;
	for (const auto& range: ranges) {
		bytes += range.end - range.begin;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepCertBytes, bytes);
	uint64_t digest_bytes = 0;
	for (const auto& range: prepared.reads.DigestRanges()) {
		digest_bytes += range.end - range.begin;
	}
	if (digest_bytes != 0) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepCertDigestBytes, digest_bytes);
	}
}

bool Validate(PreparedDraw& prepared, bool pixel_active,
              std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping) {
	const auto fail = [&](Failure failure) {
		g_fallback_reasons[static_cast<size_t>(failure)].fetch_add(1, std::memory_order_relaxed);
		prepared.failure = failure;
		Profiler::CountFrameEvent(FallbackEvent(failure));
		g_totals.fallbacks.fetch_add(1, std::memory_order_relaxed);
		g_totals.last_failure.store(failure, std::memory_order_relaxed);
		return false;
	};
	if (!prepared.ok) {
		return fail(prepared.failure);
	}
	if (prepared.pixel_active != pixel_active ||
	    !SameMapping(prepared.target_export_mapping, target_export_mapping)) {
		return fail(Failure::Mismatch);
	}
	Failure failure = Failure::None;
	if (!CertificateHolds(prepared, failure)) {
		return fail(failure);
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepCommitted);
	OwnedCounters::Add(g_totals.committed, 1);
	CountCertificate(prepared);
	return true;
}

// KYTY_DRAW_PREP_DISPATCH: per-reason fallback counts of dispatch preparations (the 10 s line).
std::array<std::atomic<uint64_t>, static_cast<size_t>(Failure::Mismatch) + 1u> g_dispatch_fallback_reasons;

bool ValidateDispatch(PreparedDraw& prepared, const ShaderComputeInputInfo& input_info) {
	const auto fail = [&](Failure failure) {
		g_dispatch_fallback_reasons[static_cast<size_t>(failure)].fetch_add(1,
		                                                                    std::memory_order_relaxed);
		prepared.failure = failure;
		g_totals.dispatch_fallbacks.fetch_add(1, std::memory_order_relaxed);
		return false;
	};
	if (!prepared.ok) {
		return fail(prepared.failure);
	}
	// The same op arguments (MakeComputeInputInfo on both sides): a difference is a bug.
	const auto& info = prepared.compute_info;
	if (info.dispatch_thread_dimensions != input_info.dispatch_thread_dimensions ||
	    !std::equal(std::begin(info.workgroup_counts), std::end(info.workgroup_counts),
	                std::begin(input_info.workgroup_counts))) {
		return fail(Failure::Mismatch);
	}
	Failure failure = Failure::None;
	if (!CertificateHolds(prepared, failure)) {
		return fail(failure);
	}
	g_totals.dispatch_committed.fetch_add(1, std::memory_order_relaxed);
	CountCertificate(prepared);
	return true;
}

namespace {

bool SameSnapshot(const ShaderRecompiler::IR::ResourceSnapshot& a,
                  const ShaderRecompiler::IR::ResourceSnapshot& b) {
	return a.buffers == b.buffers && a.images == b.images && a.samplers == b.samplers &&
	       a.flattened_srt == b.flattened_srt && a.user_data == b.user_data &&
	       a.uniform_fill == b.uniform_fill;
}

bool SamePrep(const PipelineCache::StagePrep& a, const PipelineCache::StagePrep& b) {
	return SameSnapshot(a.resources, b.resources) && a.specialization == b.specialization &&
	       a.permutation == b.permutation;
}

bool SameVertexInfo(const ShaderVertexInputInfo& a, const ShaderVertexInputInfo& b) {
	std::vector<uint32_t> key_a;
	std::vector<uint32_t> key_b;
	BuildStageStaticKey(a, key_a);
	BuildStageStaticKey(b, key_b);
	if (key_a != key_b || a.logical_stage != b.logical_stage || a.buffers_num != b.buffers_num ||
	    a.fetch_external != b.fetch_external || a.stage.program != b.stage.program) {
		return false;
	}
	for (int i = 0; i < a.resources_num; i++) {
		if (std::memcmp(a.resources[i].fields, b.resources[i].fields,
		                sizeof(a.resources[i].fields)) != 0) {
			return false;
		}
	}
	for (int i = 0; i < a.buffers_num; i++) {
		const auto& x = a.buffers[i];
		const auto& y = b.buffers[i];
		if (x.addr != y.addr || x.stride != y.stride || x.num_records != y.num_records ||
		    x.fetch_index != y.fetch_index || x.attr_num != y.attr_num) {
			return false;
		}
		for (int j = 0; j < x.attr_num; j++) {
			if (x.attr_indices[j] != y.attr_indices[j] || x.attr_offsets[j] != y.attr_offsets[j]) {
				return false;
			}
		}
	}
	return true;
}

bool SamePixelInfo(const ShaderPixelInputInfo& a, const ShaderPixelInputInfo& b) {
	std::vector<uint32_t> key_a;
	std::vector<uint32_t> key_b;
	BuildStageStaticKey(a, key_a);
	BuildStageStaticKey(b, key_b);
	return key_a == key_b && a.stage.program == b.stage.program;
}

} // namespace

bool VerifyCommitted(bool pixel_active, const PipelineCache::GraphicsPrograms& programs,
                     const ShaderVertexInputInfo& vertex_info, const ShaderPixelInputInfo& pixel_info,
                     const PipelineCache::GraphicsStagePreps& preps,
                     const PipelineCache::GraphicsPrograms& serial_programs,
                     const ShaderVertexInputInfo&            serial_vertex_info,
                     const ShaderPixelInputInfo&             serial_pixel_info,
                     const PipelineCache::GraphicsStagePreps& serial_preps) {
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepVerifyChecks);
	const bool programs_equal = programs.pixel.id == serial_programs.pixel.id &&
	                            programs.vertex[0].id == serial_programs.vertex[0].id &&
	                            programs.vertex[1].id == serial_programs.vertex[1].id;
	// Without an active pixel shader neither path writes the pixel prep (the draw state keeps
	// an earlier draw's, which nothing reads).
	const bool pixel_equal = SamePixelInfo(pixel_info, serial_pixel_info) &&
	                         (!pixel_active || SamePrep(preps.pixel, serial_preps.pixel));
	const bool vertex_equal = SamePrep(preps.vertex[0], serial_preps.vertex[0]) &&
	                          SameVertexInfo(vertex_info, serial_vertex_info);
	if (programs_equal && pixel_equal && vertex_equal) {
		return true;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepVerifyMismatches);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("DrawPrepVerify: committed preparation differs from the serial one: programs=%d "
		     "pixel=%d vertex=%d ps=%" PRIu64 "/%" PRIu64 " vs=%" PRIu64 "/%" PRIu64 "\n",
		     programs_equal, pixel_equal, vertex_equal, programs.pixel.id, serial_programs.pixel.id,
		     programs.vertex[0].id, serial_programs.vertex[0].id);
	}
	if (VerifyMode() == 2) {
		EXIT("DrawPrepVerify: committed preparation differs from the serial path\n");
	}
	return false;
}

bool VerifyDispatch(const PreparedDraw& prepared, const ShaderProgram& program,
                    const ShaderComputeInputInfo& info, const PipelineCache::StagePrep& prep) {
	g_totals.dispatch_verify_checks.fetch_add(1, std::memory_order_relaxed);
	const auto& a = prepared.compute_info;
	std::vector<uint32_t> key_a;
	std::vector<uint32_t> key_b;
	BuildStageStaticKey(a, key_a);
	BuildStageStaticKey(info, key_b);
	const bool program_equal = prepared.compute_program.id == program.id &&
	                           a.stage.program == info.stage.program;
	const bool info_equal =
	    key_a == key_b && a.lds_storage == info.lds_storage &&
	    a.host_subgroup_size == info.host_subgroup_size &&
	    a.dispatch_thread_dimensions == info.dispatch_thread_dimensions &&
	    std::equal(std::begin(a.workgroup_counts), std::end(a.workgroup_counts),
	               std::begin(info.workgroup_counts));
	const bool prep_equal = SamePrep(prepared.compute_prep, prep);
	if (program_equal && info_equal && prep_equal) {
		return true;
	}
	g_totals.dispatch_verify_mismatches.fetch_add(1, std::memory_order_relaxed);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("DrawPrepVerify: dispatch preparation differs from the serial one: program=%d info=%d "
		     "prep=%d cs=%" PRIu64 "/%" PRIu64 "\n",
		     program_equal, info_equal, prep_equal, prepared.compute_program.id, program.id);
	}
	if (VerifyMode() == 2) {
		EXIT("DrawPrepVerify: dispatch preparation differs from the serial path\n");
	}
	return false;
}

// ---------------------------------------------------------------------------------------------
// Engine

struct Engine::Slot {
	DrawKind         kind      = DrawKind::Index;
	uint64_t         submit_id = 0;
	DrawIndexArgs    index_args {};
	DrawAutoArgs     auto_args {};
	DispatchArgs     dispatch_args {}; // DrawKind::Dispatch (KYTY_DRAW_PREP_DISPATCH)
	RegisterSnapshot registers;
	PreparedDraw     prepared;
	// KYTY_DRAW_PREP_BINDINGS (bindingPlan.h): computed with `prepared` by the preparing thread,
	// published with it by Complete; used at commit only if Validate accepts `prepared`. Reset
	// after every commit.
	BindingPlan      plan;
	bool             eligible        = false; // DrawReachesPrograms, decided at submission
	bool             worker_prepared = false;
	// P3b diagnostics: the first draw the sequencer published after one of its stops
	// (Engine::NoteStop: 1 a barrier, 2 another stop); classifies a commit wait on this slot
	// (DrawPrepCommitWaitsBarrier / DrawPrepCommitWaitsStart).
	uint8_t          after_stop = 0;
	// P3c (Engine::PublishSpeculative): published by a speculative parse; the packets and the hash
	// of the bytes it consumed from its wait to this draw (the adoption key).
	bool             speculative   = false;
	uint64_t         spec_packets  = 0;
	uint64_t         spec_inputs   = 0;
	// KYTY_CP_REPEAT_TRACE (repeatTrace.h): the draw's input hashes, filled by the preparing thread,
	// and the guest address of its draw packet.
	RepeatTrace::DrawRecord repeat;
	uint64_t                repeat_packet = 0;
	// KYTY_DRAW_RUN (drawRun.h): the structure key of the preparation, by the preparing thread.
	uint64_t run_key = 0;
};

namespace {

// After Prepare, on the preparing thread: the repeat trace's hashes of this draw's inputs, and the
// run key (KYTY_DRAW_RUN).
void HashForRepeatTrace(Engine::Slot& slot) {
	if (slot.kind == DrawKind::Dispatch) {
		slot.run_key = 0; // not a draw: no run, not in the draw repeat trace
		return;
	}
	slot.run_key = DrawRun::Enabled() ? DrawRun::StructureKey(slot.registers.context,
	                                                          slot.registers.user_config,
	                                                          slot.prepared)
	                                  : 0;
	if (!RepeatTrace::Enabled()) [[likely]] {
		return;
	}
	RepeatTrace::HashDraw(slot.registers, slot.kind == DrawKind::Index ? &slot.index_args : nullptr,
	                      slot.kind == DrawKind::Auto ? &slot.auto_args : nullptr, slot.eligible,
	                      slot.prepared, slot.repeat);
}

// After Prepare, on the preparing thread (KYTY_DRAW_PREP_BINDINGS): the slot's binding plan.
void PlanBindings(Engine::Slot& slot, const BindingPlanContext& context) {
	if (slot.kind == DrawKind::Dispatch) {
		return;
	}
	if (context.pipelines->PipelinePrefetchEnabled()) {
		PrefetchBindingPipeline(context, slot.registers,
		    slot.kind == DrawKind::Index ? &slot.index_args : nullptr, slot.prepared);
	}
	if (BindingParts() == 0) [[likely]] {
		return;
	}
	ComputeBindingPlan(context, slot.registers,
	                   slot.kind == DrawKind::Index ? &slot.index_args : nullptr, slot.prepared,
	                   slot.plan);
}

// A slot's preparation: a draw's programs (Prepare) or a dispatch's (PrepareDispatch).
void PrepareSlot(PipelineCache& pipeline_cache, Engine::Slot& slot, bool exact) {
	if (slot.kind == DrawKind::Dispatch) {
		PrepareDispatch(pipeline_cache, slot.registers, slot.dispatch_args, slot.eligible, exact,
		                slot.prepared);
	} else {
		Prepare(pipeline_cache, slot.registers, slot.eligible, exact, slot.prepared);
	}
}

BindingPlanContext MakeBindingPlanContext(RenderContext& renderer) {
	(void)BindingParts(); // read (and logged) here, before any worker needs it
	return {&renderer.GetPipelineCache(), &renderer.GetSamplerCache(), &renderer.GetGraphics(),
	        &renderer.GetRenderExecutor().GetTextureMemo()};
}

// The DrawPrep#k thread's number k (0 on other threads): its busy time is FrameWait
// DrawPrepWorker<k> (threads past the eighth share DrawPrepWorker8).
thread_local uint32_t t_worker_number = 0;

Profiler::FrameWait WorkerBusyWait(uint32_t number) {
	return static_cast<Profiler::FrameWait>(
	    static_cast<uint32_t>(Profiler::FrameWait::DrawPrepWorker1) + std::min(number, 8u) - 1u);
}

uint32_t EnvUnsigned(const char* name, uint32_t fallback, uint32_t low, uint32_t high) {
	const auto* value = EnvValue(name);
	if (value == nullptr) {
		return fallback;
	}
	const auto parsed = std::strtoul(value, nullptr, 10);
	return static_cast<uint32_t>(std::clamp<unsigned long>(parsed, low, high));
}

void CpuRelax() {
#if defined(_M_X64) || defined(__x86_64__)
	_mm_pause();
#else
	std::this_thread::yield();
#endif
}

uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// The command processor preparing a stolen slot runs as a DrawPrep worker for the duration:
// CommandScheduler::CheckActive stops the emulator if the preparation reaches the scheduler, as
// it does on the DrawPrep#k threads.
class WorkerThreadScope {
public:
	WorkerThreadScope() noexcept: m_previous(t_worker_thread) { t_worker_thread = true; }
	~WorkerThreadScope() { t_worker_thread = m_previous; }
	WorkerThreadScope(const WorkerThreadScope&)            = delete;
	WorkerThreadScope& operator=(const WorkerThreadScope&) = delete;

private:
	bool m_previous;
};

} // namespace

// The parallel-mode window and its DrawPrep#k threads. Workers only run Prepare on claimed
// slots (register snapshot in, preparation out); everything else stays on the GPU thread.
// Parking follows WorkerGate (workerGate.h): KYTY_DRAW_PREP_HOT workers spin as before, the
// others park until the unclaimed backlog reaches KYTY_DRAW_PREP_WAKE_BACKLOG.
struct Engine::Workers {
	Workers(PipelineCache& pipeline_cache, uint32_t window, uint32_t count, uint64_t spin_ns,
	        uint32_t hot, uint32_t wake_backlog, uint64_t cold_spin_ns, StealPolicy steal,
	        bool cold_token)
	    : pipeline_cache(pipeline_cache), window(window), spin_ns(spin_ns),
	      cold_spin_ns(cold_spin_ns), steal(steal), gate(count, hot, wake_backlog, cold_token) {
		for (uint32_t index = 0; index < count; index++) {
			threads.emplace_back([this, index] { Run(index); });
		}
	}

	~Workers() {
		stop.store(true, std::memory_order_seq_cst);
		gate.WakeAll();
		for (auto& thread: threads) {
			thread.join();
		}
	}

	Workers(const Workers&)            = delete;
	Workers& operator=(const Workers&) = delete;

	// Producer side, after Publish: wake parked hot workers, and now and then check whether the
	// backlog needs a cold one. Paired with the gate's park protocol (seq_cst sleepers count and
	// predicate reload): either the worker sees the new slot before parking, or this thread sees
	// it parking and changes the signal it waits on.
	void Wake() {
		if (gate.OnPublish([this] { return window.Unclaimed(); })) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepColdWakes);
		}
	}

	// A worker's preparation of a claimed slot. Also the command processor's for a stolen slot
	// (Engine::CommitHead), so the two cannot differ: the conservative clean hint (exact=false),
	// then the release store of Done that publishes the preparation to the commit.
	void PrepareClaimed(Slot& slot, uint64_t seq) {
		std::optional<Profiler::ScopedFrameWait> busy;
		if (t_worker_number != 0) {
			busy.emplace(WorkerBusyWait(t_worker_number));
		}
		// KYTY_DRAW_PREP_PHASES (common/prepPhases.h): the split of this preparation.
		namespace Phases = Common::PrepPhases;
		Phases::Begin();
		PrepareSlot(pipeline_cache, slot, false);
		HashForRepeatTrace(slot);
		Phases::Mark(Phases::Phase::RunKey);
		PlanBindings(slot, plan_context);
		Phases::Mark(Phases::Phase::Plan);
		slot.worker_prepared = true;
		window.Complete(seq);
		Phases::End(Phases::Phase::Complete);
	}

	void Run(uint32_t index) {
		char name[32];
		std::snprintf(name, sizeof(name), "DrawPrep#%u", index + 1u);
		Profiler::SetThreadName(name);
		t_worker_thread = true;
		uint32_t placement_count = 0; // placement samples (common/cpuPlacement.h), every 64th slot
		t_worker_number = index + 1u;
		RunPreparationWorker(
		    gate, window, index, spin_ns, cold_spin_ns, stop,
		    [this, &placement_count](Slot& slot, uint64_t seq) {
			    PrepareClaimed(slot, seq);
			    if ((++placement_count & 63u) == 0u) {
				    Common::SamplePlacement(Common::ThreadRole::Host);
			    }
		    },
		    [] { Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepColdWakes); });
	}

	PipelineCache&           pipeline_cache;
	BindingPlanContext       plan_context; // set by the engine before the first Submit
	Window<Slot>             window;
	uint64_t                 spin_ns      = 0;
	uint64_t                 cold_spin_ns = 0;
	StealPolicy              steal; // KYTY_DRAW_PREP_STEAL, KYTY_DRAW_PREP_STEAL_AFTER_US
	WorkerGate               gate;
	std::atomic<bool>        stop {false};
	std::vector<std::thread> threads;
};

Engine::Engine(RenderContext& renderer, std::function<void()> service_commands,
               std::function<void()> after_commit)
    : m_renderer(renderer), m_mode(GetMode()), m_service_commands(std::move(service_commands)),
      m_after_commit(std::move(after_commit)), m_inline_slot(std::make_unique<Slot>()) {
	if (RepeatTrace::Enabled()) {
		RepeatTrace::SetRenderContext(&m_renderer);
	}
	if (m_mode == Mode::Parallel) {
		const auto window  = EnvUnsigned("KYTY_DRAW_PREP_WINDOW", 32, 2, 1024);
		const auto workers = EnvUnsigned("KYTY_DRAW_PREP_WORKERS", 6, 1, 32);
		const auto spin_us = EnvUnsigned("KYTY_DRAW_PREP_SPIN_US", 200, 0, 1000000);
		// Workers that keep spinning; the rest park until the backlog needs them. A value of at
		// least KYTY_DRAW_PREP_WORKERS keeps every worker hot (the behaviour before the gate).
		// Wake a parked worker once 2 slots wait unclaimed (8 until U54: the CP then waited for
		// heads 2.5x as long after short bursts, DEEP-TRACE-U54 3.1; draw_prep_tests
		// --measure-worker-gate, heavy bursts: 270.4 -> 265.2 ms, no extra worker CPU).
		const auto hot          = EnvUnsigned("KYTY_DRAW_PREP_HOT", 2, 1, 32);
		const auto wake_backlog = EnvUnsigned("KYTY_DRAW_PREP_WAKE_BACKLOG", 2, 1, 1024);
		const auto cold_spin_us = EnvUnsigned("KYTY_DRAW_PREP_COLD_SPIN_US", 50, 0, 1000000);
		// Work stealing while a worker holds the head (Engine::CommitHead): the unclaimed backlog
		// that makes the command processor prepare slots itself (0 = never, the default), and how
		// long it first waits for the head. Off by default: in the measured workloads it removes
		// most of the CP's idle spin but not wall time (a stolen preparation that outlasts the
		// head delays the commits behind it).
		StealPolicy steal;
		steal.min_unclaimed  = EnvUnsigned("KYTY_DRAW_PREP_STEAL", 0, 0, 1024);
		const auto steal_us  = EnvUnsigned("KYTY_DRAW_PREP_STEAL_AFTER_US", 0, 0, 1000000);
		steal.after_ns       = uint64_t {steal_us} * 1000u;
		// KYTY_DRAW_PREP_COLD_TOKEN=1: the cold-wake protocol whose wakes cannot stick
		// (workerGate.h); with the default one the cold workers stop waking after a race.
		const bool cold_token = EnvUnsigned("KYTY_DRAW_PREP_COLD_TOKEN", 0, 0, 1) != 0;
		m_workers = std::make_unique<Workers>(m_renderer.GetPipelineCache(), window, workers,
		                                      uint64_t {spin_us} * 1000u, hot, wake_backlog,
		                                      uint64_t {cold_spin_us} * 1000u, steal, cold_token);
		m_workers->plan_context = MakeBindingPlanContext(m_renderer);
		LOGF("DrawPrep: parallel mode, window=%u workers=%u spin=%uus hot=%u wake_backlog=%u "
		     "cold_spin=%uus cold_token=%d steal=%u steal_after=%uus cert=%s verify=%d\n",
		     m_workers->window.Capacity(), workers, spin_us, m_workers->gate.HotCount(),
		     wake_backlog, cold_spin_us, cold_token ? 1 : 0, steal.min_unclaimed, steal_us,
		     GetCertMode() == CertMode::Log ? "log" : "value", VerifyMode());
	}
}

Engine::~Engine() {
	// Pending draws at teardown are dropped with the stream (shutdown only).
	m_workers.reset();
}

bool Engine::Pending() const noexcept {
	return m_workers != nullptr && !m_workers->window.Empty();
}

void Engine::FillSlot(Slot& slot, uint64_t submit_id, const DrawIndexArgs* index_args,
                      const DrawAutoArgs* auto_args, const HW::Context& context,
                      const HW::UserConfig& user_config, const HW::Shader& shaders) {
	slot.submit_id = submit_id;
	uint32_t count = 0;
	uint32_t instances = 0;
	if (index_args != nullptr) {
		slot.kind       = DrawKind::Index;
		slot.index_args = *index_args;
		count           = index_args->index_count;
		instances       = index_args->instance_count;
	} else {
		slot.kind      = DrawKind::Auto;
		slot.auto_args = *auto_args;
		count          = auto_args->vertex_count;
		instances      = auto_args->instance_count;
	}
	slot.registers.context     = context;
	slot.registers.user_config = user_config;
	slot.registers.shaders     = shaders;
	slot.eligible        = DrawReachesPrograms(context, user_config, shaders, count, instances);
	slot.worker_prepared = false;
	slot.after_stop      = 0;
	slot.speculative     = false;
	slot.repeat_packet   = RepeatTrace::Enabled() ? RepeatTrace::TakeDrawPacket() : 0;
}

bool Engine::Submit(uint64_t submit_id, const DrawIndexArgs* index_args,
                    const DrawAutoArgs* auto_args, const HW::Context& context,
                    const HW::UserConfig& user_config, const HW::Shader& shaders) {
	EXIT_IF(!GuestGpu::IsGpuThread());
	EXIT_IF((index_args == nullptr) == (auto_args == nullptr));
	if (m_mode == Mode::Off) {
		return false;
	}
	m_draws_since_fence++;
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepSubmitted);
	if (m_workers == nullptr) {
		// Inline: prepare on this thread from the snapshot, then commit at once.
		auto& slot = *m_inline_slot;
		FillSlot(slot, submit_id, index_args, auto_args, context, user_config, shaders);
		Prepare(m_renderer.GetPipelineCache(), slot.registers, slot.eligible, true, slot.prepared);
		m_renderer.GetPipelineCache().NoteProgramPrefetchWait(slot.prepared.program_compile_ns);
		HashForRepeatTrace(slot);
		PlanBindings(slot, MakeBindingPlanContext(m_renderer)); // tests: plans on this thread
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepSelfPrepared);
		Commit(slot);
		return true;
	}
	auto& window = m_workers->window;
	if (window.Full()) {
		CommitHead();
	}
	FillSlot(window.Reserve(), submit_id, index_args, auto_args, context, user_config, shaders);
	window.Publish();
	m_real_tail = window.Tail(); // no speculative slots outside thread mode
	m_workers->Wake();
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepPublished);
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepWindowOccupancy, window.Occupancy());
	return true;
}

uint64_t Engine::Publish(const DrawIndexArgs* index_args, const DrawAutoArgs* auto_args,
                         const HW::Context& context, const HW::UserConfig& user_config,
                         const HW::Shader& shaders, const std::function<bool()>& wait_for_space) {
	EXIT_IF(m_workers == nullptr);
	EXIT_IF((index_args == nullptr) == (auto_args == nullptr));
	auto& window = m_workers->window;
	// Speculative slots are adopted or dropped (TryAdopt, DropSpeculative) before a publish.
	EXIT_IF(m_real_tail != window.Tail());
	if (window.Full()) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::CpSeqWindowFull);
		if (!wait_for_space()) {
			return UINT64_MAX;
		}
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepSubmitted);
	const auto position = window.Tail();
	m_real_tail         = position + 1u;
	// The submission id is the resolver's (CommitPublished); the repeat trace is off in this mode.
	auto& slot = window.Reserve();
	FillSlot(slot, 0, index_args, auto_args, context, user_config, shaders);
	slot.after_stop = m_stop_pending;
	if (m_stop_pending == 1) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::CpSeqBarrierDrawBursts);
	}
	m_stop_pending = 0;
	window.Publish();
	m_workers->Wake();
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepPublished);
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepWindowOccupancy, window.Occupancy());
	return position;
}

bool Engine::WindowHasSpace() const noexcept {
	return m_workers != nullptr && !m_workers->window.Full();
}

bool Engine::PublishSpeculative(const DrawIndexArgs* index_args, const DrawAutoArgs* auto_args,
                                const HW::Context& context, const HW::UserConfig& user_config,
                                const HW::Shader& shaders, uint64_t packets, uint64_t inputs) {
	EXIT_IF(m_workers == nullptr);
	EXIT_IF((index_args == nullptr) == (auto_args == nullptr));
	auto& window = m_workers->window;
	if (window.Full()) {
		return false;
	}
	auto& slot = window.Reserve();
	FillSlot(slot, 0, index_args, auto_args, context, user_config, shaders);
	slot.speculative  = true;
	slot.spec_packets = packets;
	slot.spec_inputs  = inputs;
	window.Publish();
	m_workers->Wake();
	Profiler::CountFrameEvent(Profiler::FrameEvent::CpSeqPrefetchDraws);
	g_totals.prefetch_published.fetch_add(1, std::memory_order_relaxed);
	return true;
}

uint64_t Engine::SpeculativeSlots() const noexcept {
	return m_workers != nullptr ? m_workers->window.Tail() - m_real_tail : 0u;
}

Engine::Adoption Engine::TryAdopt(uint64_t packets, uint64_t inputs, uint64_t& position,
                                  uint64_t& dropped) {
	if (SpeculativeSlots() == 0) {
		return Adoption::None;
	}
	auto& window = m_workers->window;
	// The slot's key fields were written by this thread (the speculative publish).
	const auto& slot = window.PayloadAt(m_real_tail);
	if (slot.spec_packets == packets && slot.spec_inputs == inputs &&
	    CpSeq::PrefetchMode() != 2) {
		position = m_real_tail++;
		// Diagnostics: the first draw after the stop was prepared ahead (not a burst start).
		m_stop_pending = 0;
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepSubmitted);
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepPublished);
		Profiler::CountFrameEvent(Profiler::FrameEvent::CpSeqPrefetchAdopted);
		g_totals.prefetch_adopted.fetch_add(1, std::memory_order_relaxed);
		return Adoption::Adopted;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::CpSeqPrefetchMismatches);
	dropped = DropSpeculative();
	return Adoption::Dropped;
}

uint64_t Engine::DropSpeculative() {
	const auto count = SpeculativeSlots();
	if (count != 0) {
		m_real_tail = m_workers->window.Tail();
		Profiler::CountFrameEvent(Profiler::FrameEvent::CpSeqPrefetchSkipped, count);
		g_totals.prefetch_skipped.fetch_add(count, std::memory_order_relaxed);
	}
	return count;
}

void Engine::SkipPublished(uint64_t count) {
	EXIT_IF(!GuestGpu::IsGpuThread() || m_workers == nullptr);
	auto& window = m_workers->window;
	for (uint64_t i = 0; i < count; i++) {
		EXIT_IF(window.Empty());
		auto& slot = window.HeadPayload();
		if (!window.TryClaimHead()) {
			// A worker prepares it (or has): its preparation is discarded once it is done.
			while (!window.HeadDone()) {
				CpuRelax();
			}
		}
		// Claimed or done (acquired): the slot is this thread's until it retires.
		EXIT_IF(!slot.speculative);
		slot.plan.Reset();
		window.Retire();
	}
}

void Engine::CommitPublished(uint64_t position, uint64_t submit_id, uint32_t instance_count) {
	EXIT_IF(!GuestGpu::IsGpuThread());
	EXIT_IF(m_workers == nullptr || m_workers->window.Empty() ||
	        m_workers->window.Head() != position);
	const std::function<void(Slot&)> patch = [submit_id, instance_count](Slot& slot) {
		slot.submit_id = submit_id;
		if (instance_count != UINT32_MAX) {
			if (slot.kind == DrawKind::Index) {
				slot.index_args.instance_count = instance_count;
			} else {
				slot.auto_args.instance_count = instance_count;
			}
		}
	};
	CommitHead(&patch);
}

void Engine::CommitHead(const std::function<void(Slot&)>* patch) {
	auto& slot = ReadyHead();
	// A dispatch slot is only ever taken by its own op (ExecuteDispatch).
	EXIT_IF(slot.kind == DrawKind::Dispatch);
	if (patch != nullptr) {
		// The preparation (and the preparing thread's reads of the slot) is complete.
		(*patch)(slot);
	}
	Commit(slot);
	m_workers->window.Retire();
}

Engine::Slot& Engine::ReadyHead() {
	auto& window = m_workers->window;
	EXIT_IF(window.Empty());
	// Commits happen at packet boundaries, never inside a preparation: the recorder and the
	// per-thread scratch the preparations below use are idle here.
	EXIT_IF(Speculative());
	auto& slot = window.HeadPayload();
	CommitStats::StepMark(CommitStats::Step::GapDrawOp);
	const auto stats_start = CommitStats::Now();
	if (window.TryClaimHead()) {
		// No worker has started it: prepare it here, with the exact clean predicate.
		const CpGaps::Scope gap(CpGaps::Cat::HeadSelf);
		PrepareSlot(m_renderer.GetPipelineCache(), slot, true);
		HashForRepeatTrace(slot);
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepSelfPrepared);
		if (stats_start != 0) {
			CommitStats::NoteHead(CommitStats::Now() - stats_start, 0);
		}
	} else if (window.HeadDone()) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepReady);
	} else {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepCommitWaits);
		const CpGaps::Scope gap(CpGaps::Cat::HeadWait);
		// Diagnostics: what the wait began with (the window's depth, a waiting unclaimed slot).
		const auto occupancy = window.Occupancy();
		HangWatchdog::Scope wait("draw-prep-head", reinterpret_cast<uint64_t>(&window),
		                         HangWatchdog::Enabled() ? window.Head() : 0, occupancy);
		if (window.Unclaimed() != 0) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepCommitWaitsUnclaimed);
		}
		// A worker holds the head. KYTY_DRAW_PREP_STEAL: meanwhile this thread prepares the next
		// unclaimed slots exactly as a worker does (AwaitHead, workerGate.h): same function, the
		// workers' clean hint, CheckActive armed, Done published with a release store. They are
		// committed in order when they reach the head, through the same Validate as any worker's
		// preparation. Only a head this thread claims itself (above) uses the exact predicate: it
		// is committed at once. A stolen slot is committed after the draws before it, which change
		// what that predicate reads, so the certificate at its commit decides.
		const auto stats = AwaitHead(
		    m_workers->gate, window, m_workers->steal,
		    [this](Slot& claimed, uint64_t seq) {
			    Profiler::ScopedFrameWait steal_time(Profiler::FrameWait::DrawPrepSteal);
			    const WorkerThreadScope   as_worker;
			    m_workers->PrepareClaimed(claimed, seq);
			    m_renderer.GetPipelineCache().NoteProgramPrefetchWait(claimed.prepared.program_compile_ns);
		    },
		    [this](uint32_t spins, uint64_t spin_start) {
			    CpuRelax();
			    // Workers never dereference guest memory, so they cannot page fault into a
			    // request this thread must service. Only as a last-resort deadlock guard, after
			    // 2 ms of spinning, pending cross-thread commands are serviced here; such a
			    // command then observes the window partially committed, as if it had arrived
			    // just before these draws were parsed.
			    if ((spins & 1023u) == 1023u && m_service_commands &&
			        NowNs() - spin_start > 2'000'000u) {
				    m_service_commands();
			    }
		    },
		    [] { Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepColdWakes); });
		// One call per held head; its time is only the idle spin (steals are DrawPrepSteal).
		Profiler::AddFrameWait(Profiler::FrameWait::DrawPrepCommitWait, 1, stats.spin_ns);
		// Actual idle CP wait on a shader miss, capped by the compiler call duration. Steals
		// are accounted separately above. Whole-route flip intervals remain the hitch metric.
		m_renderer.GetPipelineCache().NoteProgramPrefetchWait(
		    std::min(stats.spin_ns, slot.prepared.program_compile_ns));
		// The head is done (acquired): its publication fields are visible.
		if (slot.speculative) {
			// P3c: an adopted slot whose speculative preparation had not finished yet.
			Profiler::CountFrameEvent(Profiler::FrameEvent::CpSeqPrefetchAdoptedWaits);
		}
		CpGaps::Relabel(slot.after_stop == 1   ? CpGaps::Cat::HeadWait
		                : slot.after_stop == 2 ? CpGaps::Cat::HeadWaitStart
		                : occupancy <= 2       ? CpGaps::Cat::HeadWaitShallow
		                                       : CpGaps::Cat::HeadWaitDeep);
		if (slot.after_stop == 1) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepCommitWaitsBarrier);
			Profiler::AddFrameWait(Profiler::FrameWait::DrawPrepCommitWaitBarrier, 1, stats.spin_ns);
		} else if (slot.after_stop == 2) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepCommitWaitsStart);
			Profiler::AddFrameWait(Profiler::FrameWait::DrawPrepCommitWaitStart, 1, stats.spin_ns);
		} else if (occupancy <= 2) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepCommitWaitsShallow);
			Profiler::AddFrameWait(Profiler::FrameWait::DrawPrepCommitWaitShallow, 1, stats.spin_ns);
		} else {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepCommitWaitsDeep);
			Profiler::AddFrameWait(Profiler::FrameWait::DrawPrepCommitWaitDeep, 1, stats.spin_ns);
		}
		if (stats.stolen != 0) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepSteals, stats.stolen);
		}
		if (stats_start != 0) {
			CommitStats::NoteHead(0, CommitStats::Now() - stats_start);
		}
	}
	CommitStats::StepMark(CommitStats::Step::GapHead);
	return slot;
}

bool Engine::IndirectEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DRAW_PREP_INDIRECT");
		return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

uint64_t Engine::PublishIndirect(bool indexed, const HW::Context& context,
                                 const HW::UserConfig& user_config, const HW::Shader& shaders,
                                 const std::function<bool()>& wait_for_space) {
	// The counts are GPU data: the preparation assumes the draw draws, as for an inherited
	// instance count (an unused preparation is dropped, DrawPrepUnused).
	if (indexed) {
		DrawIndexArgs args;
		args.index_count    = 1;
		args.instance_count = 1;
		return Publish(&args, nullptr, context, user_config, shaders, wait_for_space);
	}
	DrawAutoArgs args;
	args.vertex_count   = 1;
	args.instance_count = 1;
	return Publish(nullptr, &args, context, user_config, shaders, wait_for_space);
}

uint64_t Engine::PublishDispatch(const DispatchArgs& args, const HW::Context& context,
                                 const HW::UserConfig& user_config, const HW::Shader& shaders,
                                 const std::function<bool()>& wait_for_space) {
	EXIT_IF(m_workers == nullptr);
	auto& window = m_workers->window;
	// Speculative slots are dropped (DropSpeculative) before a dispatch is published.
	EXIT_IF(m_real_tail != window.Tail());
	if (window.Full()) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::CpSeqWindowFull);
		if (!wait_for_space()) {
			return UINT64_MAX;
		}
	}
	const auto position = window.Tail();
	m_real_tail         = position + 1u;
	auto& slot          = window.Reserve();
	slot.submit_id      = 0;
	slot.kind           = DrawKind::Dispatch;
	slot.dispatch_args  = args;
	slot.registers.context     = context;
	slot.registers.user_config = user_config;
	slot.registers.shaders     = shaders;
	// RenderExecutor::DispatchDirect returns before its program lookup for the others.
	slot.eligible        = shaders.GetCs().cs_regs.data_addr != 0 &&
	                (args.indirect || (args.x != 0 && args.y != 0 && args.z != 0));
	slot.worker_prepared = false;
	slot.after_stop      = m_stop_pending;
	slot.speculative     = false;
	slot.repeat_packet   = 0;
	m_stop_pending       = 0;
	window.Publish();
	m_workers->Wake();
	return position;
}

void Engine::ExecuteDispatch(uint64_t position, const std::function<void()>& dispatch) {
	EXIT_IF(!GuestGpu::IsGpuThread());
	EXIT_IF(m_workers == nullptr || m_workers->window.Empty() ||
	        m_workers->window.Head() != position);
	auto& slot     = ReadyHead();
	auto& executor = m_renderer.GetRenderExecutor();
	EXIT_IF(slot.kind != DrawKind::Dispatch);
	executor.m_prepared_dispatch = &slot.prepared;
	dispatch();
	if (executor.m_prepared_dispatch != nullptr) {
		// The dispatch returned before its program lookup (no shader, zero groups).
		executor.m_prepared_dispatch = nullptr;
		g_totals.dispatch_unused.fetch_add(1, std::memory_order_relaxed);
	}
	m_workers->window.Retire();
}

void Engine::ExecuteIndirect(uint64_t position, const std::function<void()>& draw) {
	EXIT_IF(!GuestGpu::IsGpuThread());
	EXIT_IF(m_workers == nullptr || m_workers->window.Empty() ||
	        m_workers->window.Head() != position);
	auto& slot     = ReadyHead();
	auto& executor = m_renderer.GetRenderExecutor();
	// KYTY_DRAW_RUN_INDIRECT (drawRun.h): an engine commit with the slot's run key and binding
	// plan, but not the plan's vertex ranges, which come from the placeholder counts. Otherwise only
	// the programs are offered and the draw is never a draw-run commit.
	const bool commit = DrawRun::IndirectRunsEnabled();
	if (commit && slot.plan.valid) {
		slot.plan.vertex_ranges.valid = false;
	} else {
		slot.plan.Reset();
	}
	executor.m_prepared_draw       = &slot.prepared;
	executor.m_binding_plan        = slot.plan.valid ? &slot.plan : nullptr;
	executor.m_binding_plan_active = false;
	executor.m_run_key             = commit ? slot.run_key : 0;
	executor.m_in_engine_commit    = commit;
	{
		const CpGaps::Scope gap(CpGaps::Cat::Commit);
		CommitStats::BeginDraw();
		draw();
		CommitStats::EndDraw();
	}
	executor.m_in_engine_commit = false;
	executor.m_run_key          = 0;
	if (slot.plan.valid) {
		CountCommittedPlan(executor.m_binding_plan_active);
		slot.plan.Reset();
	}
	if (executor.m_prepared_draw != nullptr) {
		// The draw returned before preparing its programs.
		executor.m_prepared_draw = nullptr;
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepUnused);
	}
	executor.m_binding_plan        = nullptr;
	executor.m_binding_plan_active = false;
	m_workers->window.Retire();
}

void Engine::Drain() {
	if (!Pending()) {
		return;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepDrains);
	g_totals.drains.fetch_add(1, std::memory_order_relaxed);
	while (Pending()) {
		CommitHead();
	}
}

// One console line every 10 s (command processor thread): draws committed from a worker's
// preparation against draws that fell back to the serial path, and why. Fallbacks run the whole
// program preparation on the command processor, so the ratio says how much of it is avoidable.
namespace {
void PrintDrawPrepSummary() {
	static uint64_t last_ns        = 0;
	static uint64_t last_committed = 0;
	static uint64_t last_fallbacks = 0;
	static std::array<uint64_t, static_cast<size_t>(Failure::Mismatch) + 1u> last_reasons {};
	// KYTY_CP_COMMIT=draws: the clock is read every 256th commit (the line is due every 10 s).
	static uint32_t calls = 0;
	if (CpCommit::Enabled(CpCommit::Part::Draws) && last_ns != 0 && (++calls & 255u) != 0) {
		return;
	}
	const auto now = NowNs();
	if (last_ns == 0) {
		last_ns = now;
		return;
	}
	if (now - last_ns < 10'000'000'000ull) {
		return;
	}
	const auto committed = g_totals.committed.load(std::memory_order_relaxed);
	const auto fallbacks = g_totals.fallbacks.load(std::memory_order_relaxed);
	static const char* const names[] = {"none",       "ineligible", "unclean",     "backing",
	                                    "overflow",   "inconsistent", "uncertified", "notpublished",
	                                    "shadermap",  "certunclean", "certchanged", "coherencelog",
	                                    "mismatch"};
	std::string reasons;
	for (size_t i = 1; i < last_reasons.size(); i++) {
		const auto value = g_fallback_reasons[i].load(std::memory_order_relaxed);
		if (value != last_reasons[i]) {
			reasons += " " + std::string(names[i]) + "=" + std::to_string(value - last_reasons[i]);
			last_reasons[i] = value;
		}
	}
	std::string dispatches;
	if (DispatchPrepEnabled()) {
		// KYTY_DRAW_PREP_DISPATCH: dispatch preparations taken, refused (and why), unused.
		static uint64_t last_dispatch[5] {};
		static std::array<uint64_t, static_cast<size_t>(Failure::Mismatch) + 1u> last_dispatch_reasons {};
		const uint64_t values[5] = {g_totals.dispatch_committed.load(std::memory_order_relaxed),
		                            g_totals.dispatch_fallbacks.load(std::memory_order_relaxed),
		                            g_totals.dispatch_unused.load(std::memory_order_relaxed),
		                            g_totals.dispatch_verify_checks.load(std::memory_order_relaxed),
		                            g_totals.dispatch_verify_mismatches.load(std::memory_order_relaxed)};
		std::string dispatch_reasons;
		for (size_t i = 1; i < last_dispatch_reasons.size(); i++) {
			const auto value = g_dispatch_fallback_reasons[i].load(std::memory_order_relaxed);
			if (value != last_dispatch_reasons[i]) {
				dispatch_reasons += " " + std::string(names[i]) + "=" +
				                    std::to_string(value - last_dispatch_reasons[i]);
				last_dispatch_reasons[i] = value;
			}
		}
		dispatches = " | dispatches: " + std::to_string(values[0] - last_dispatch[0]) +
		             " prepared, " + std::to_string(values[1] - last_dispatch[1]) + " serial" +
		             (dispatch_reasons.empty() ? std::string() : " (" + dispatch_reasons.substr(1) + ")") +
		             ", " + std::to_string(values[2] - last_dispatch[2]) + " unused; verify " +
		             std::to_string(values[3] - last_dispatch[3]) + " checks, " +
		             std::to_string(values[4] - last_dispatch[4]) + " mismatches";
		std::copy(std::begin(values), std::end(values), std::begin(last_dispatch));
	}
	if (g_clean_since.Get() >= 2) {
		dispatches += " | clean-since verify: " +
		              std::to_string(g_clean_since_checks.load(std::memory_order_relaxed)) +
		              " checks, " +
		              std::to_string(g_clean_since_mismatches.load(std::memory_order_relaxed)) +
		              " mismatches (total)";
	}
	std::printf("DrawPrep %.0fs: %" PRIu64 " committed from a prepared slot, %" PRIu64
	            " fell back to the serial path;%s%s\n",
	            static_cast<double>(now - last_ns) * 1e-9, committed - last_committed,
	            fallbacks - last_fallbacks, reasons.empty() ? " none" : reasons.c_str(),
	            dispatches.c_str());
	last_ns        = now;
	last_committed = committed;
	last_fallbacks = fallbacks;
}
} // namespace

void Engine::Commit(Slot& slot) {
	Profiler::ScopedFrameWait commit_time(Profiler::FrameWait::DrawPrepCommit);
	const CpGaps::Scope       gap(CpGaps::Cat::Commit);
	if (!CommitStats::IndirectOnly()) {
		CommitStats::BeginDraw();
	}
	auto&      scheduler = m_renderer.GetCommandScheduler();
	auto&      executor  = m_renderer.GetRenderExecutor();
	const auto previous  = scheduler.BindRegisters(slot.registers.context,
	                                               slot.registers.user_config,
	                                               slot.registers.shaders);
	executor.m_prepared_draw = &slot.prepared;
	// KYTY_DRAW_PREP_BINDINGS: the slot's plan, which RefreshShaders activates once Validate
	// accepted the preparation it was computed from.
	executor.m_binding_plan        = slot.plan.valid ? &slot.plan : nullptr;
	executor.m_binding_plan_active = false;
	// KYTY_DRAW_RUN (drawRun.h): the draw is an engine commit with this structure key.
	executor.m_run_key          = slot.run_key;
	executor.m_in_engine_commit = true;
	CommitStats::StepMark(CommitStats::Step::BindRegs);
	CommitStats::StepMark(CommitStats::Step::Calib);
	if (slot.kind == DrawKind::Index) {
		executor.DrawIndex(slot.submit_id, scheduler.Current(), slot.index_args);
	} else {
		executor.DrawAuto(slot.submit_id, scheduler.Current(), slot.auto_args);
	}
	executor.m_in_engine_commit = false;
	executor.m_run_key          = 0;
	if (slot.plan.valid) {
		CountCommittedPlan(executor.m_binding_plan_active);
		slot.plan.Reset();
	}
	executor.m_binding_plan        = nullptr;
	executor.m_binding_plan_active = false;
	const bool taken = executor.m_prepared_draw == nullptr;
	if (executor.m_prepared_draw != nullptr) {
		// The draw returned before preparing its programs (nothing to draw, a metadata
		// operation, no targets): the preparation is simply dropped.
		executor.m_prepared_draw = nullptr;
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepUnused);
	}
	if (RepeatTrace::Enabled()) {
		auto record      = slot.repeat;
		record.packet    = slot.repeat_packet;
		record.committed = taken && slot.prepared.ok && slot.prepared.failure == Failure::None;
		record.failure   = static_cast<uint8_t>(slot.prepared.failure);
		RepeatTrace::OnDraw(record);
	}
	scheduler.RestoreRegisters(previous);
	if (m_after_commit) {
		m_after_commit();
	}
	CommitStats::EndDraw();
	PrintDrawPrepSummary();
	DrawRun::PrintSummary();
	TexelImageLookup::PrintSummary();
	XFrameReuse::PrintSummary();
	CommitStats::StepMark(CommitStats::Step::GapCommitTail);
}

void Engine::NoteFence() {
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepFences);
	using E          = Profiler::FrameEvent;
	const auto draws = m_draws_since_fence;
	const auto event = draws == 0    ? E::DrawPrepFenceDraws0
	                   : draws == 1  ? E::DrawPrepFenceDraws1
	                   : draws < 4   ? E::DrawPrepFenceDraws2To3
	                   : draws < 8   ? E::DrawPrepFenceDraws4To7
	                   : draws < 16  ? E::DrawPrepFenceDraws8To15
	                   : draws < 32  ? E::DrawPrepFenceDraws16To31
	                   : draws < 64  ? E::DrawPrepFenceDraws32To63
	                                 : E::DrawPrepFenceDraws64Plus;
	Profiler::CountFrameEvent(event);
	m_draws_since_fence = 0;
}

void Engine::OnPacket(PacketClass packet_class, FenceKind fence_kind) {
	switch (packet_class) {
		case PacketClass::WindowSafe: break;
		case PacketClass::Draw:
			// Off mode (histogram only) counts here; inline/parallel count in Submit, which
			// also sees draws issued through other paths.
			if (m_mode == Mode::Off) {
				m_draws_since_fence++;
			}
			break;
		case PacketClass::Fence: {
			using E = Profiler::FrameEvent;
			static constexpr std::array<E, static_cast<size_t>(FenceKind::Count)> events {
			    E::DrawPrepFenceRegIndirect, E::DrawPrepFenceEventWrite,
			    E::DrawPrepFenceEndOfPipe,   E::DrawPrepFenceAcquireMem,
			    E::DrawPrepFenceWait,        E::DrawPrepFenceDataWrite,
			    E::DrawPrepFenceConstantEngine, E::DrawPrepFenceMarker,
			    E::DrawPrepFenceDispatch,    E::DrawPrepFenceIndirectDraw,
			    E::DrawPrepFenceContextControl, E::DrawPrepFenceOther,
			};
			if (static_cast<size_t>(fence_kind) < events.size()) {
				Profiler::CountFrameEvent(events[static_cast<size_t>(fence_kind)]);
			}
			NoteFence();
			Drain();
			break;
		}
	}
}

} // namespace Libs::Graphics::DrawPrep
