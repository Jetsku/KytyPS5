#include "graphics/host_gpu/renderer/drawPrep/drawPrep.h"
#include "common/staticOnce.h"

#include "graphics/host_gpu/renderer/drawPrep/ownedCounters.h"

#include "common/assert.h"
#include "common/prepPhases.h"
#include "common/coreBudget.h"
#include "common/cpuPlacement.h"
#include "common/hangWatchdog.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/helperPriority.h"
#include "common/spinGate.h"
#include "graphics/guest_gpu/command_processor/cpOps.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/coherenceLog.h"
#include "graphics/host_gpu/renderer/cache/texelImageLookup.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/cpCommit.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/drawPrep/adaptivePool.h"
#include "graphics/host_gpu/renderer/drawPrep/bindingPlan.h"
#include "graphics/host_gpu/renderer/drawPrep/commitStats.h"
#include "graphics/host_gpu/renderer/drawPrep/cpGaps.h"
#include "graphics/host_gpu/renderer/drawPrep/drawRun.h"
#include "graphics/host_gpu/renderer/drawPrep/imgSide.h"
#include "graphics/host_gpu/renderer/drawPrep/lean.h"
#include "graphics/host_gpu/renderer/drawPrep/repeatTrace.h"
#include "graphics/host_gpu/renderer/drawPrep/window.h"
#include "graphics/host_gpu/renderer/drawPrep/workerGate.h"
#include "graphics/host_gpu/renderer/drawPrep/xframeReuse.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/syncEpoch.h"
#include "graphics/host_gpu/renderer/renderDraw.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"

#include <algorithm>
#include <atomic>
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
		if (render_target_bound(rt) && render_target_mask_slot(ctx.GetRenderTargetMask(), slot) != 0 &&
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
	static constexpr auto mode_init = [] {
		const auto* value = EnvValue("KYTY_DRAW_PREP_VERIFY");
		if (value == nullptr || std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(mode_init())>> mode_once;
	const int& mode = mode_once.Get(mode_init);
	return mode;
}

CertMode GetCertMode() {
	static constexpr auto mode_init = [] {
		// Log by default (Run A audit, CP-ARCHITECTURE-20260927.md: DrawPrepLogMissed 0 in every
		// scene); KYTY_DRAW_PREP_CERT=value restores the byte and digest comparisons.
		const auto* value = EnvValue("KYTY_DRAW_PREP_CERT");
		return value != nullptr && std::strcmp(value, "value") == 0 ? CertMode::Value : CertMode::Log;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(mode_init())>> mode_once;
	const CertMode& mode = mode_once.Get(mode_init);
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
	static constexpr auto mode_init = [] {
		const auto* value = EnvValue("KYTY_DRAW_PREP_CERT_RANGES_VERIFY");
		if (value == nullptr || std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(mode_init())>> mode_once;
	const int& mode = mode_once.Get(mode_init);
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

static bool ReadsUntouched(PreparedDraw& prepared);

void Prepare(PipelineCache& pipeline_cache, const RegisterSnapshot& registers, bool eligible,
             bool exact, PreparedDraw& prepared) {
	prepared.program_compile_ns = 0;
	prepared.ok         = false;
	prepared.failure    = Failure::None;
	prepared.programs   = {};
	prepared.pixel_info = {};
	prepared.reads.Reset();
	prepared.reads.SetCoalesce(DirectReadsEnabled());
	prepared.reads.SetLean(Lean::On(Lean::ReadSet)); // KYTY_DRAW_PREP_LEAN read-set part
	prepared.certificate.clear();
	prepared.certificate_built = false;
	prepared.reads_untouched   = false;
	prepared.touched_valid     = false;
	prepared.vertex_stages     = 1;
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
	recorder.code_epoch = t_code_epoch; // KYTY_DRAW_PREP_LEAN digest cache (0: not used)
	RecordScope scope(recorder);
	// KYTY_DRAW_PREP_TESS: a tessellated draw's three vertex stages (otherwise Ineligible).
	const bool tess = PipelineCache::TessellationActive(registers.user_config) &&
	                  PipelineCache::DrawPrepTessEnabled();
	if (tess) {
		prepared.vertex_stages = 3;
	}
	const auto result =
	    tess ? pipeline_cache.PrepareTessellatedProgramsSpeculative(
	               sh_ctx.GetVs(), sh_ctx.GetPs(), ctx.GetShaderRegisters(), ctx,
	               registers.user_config, prepared.target_export_mapping, prepared.pixel_active,
	               {&prepared.vertex_info, &prepared.tess_info[0], &prepared.tess_info[1]},
	               prepared.pixel_info,
	               {&prepared.vertex_prep, &prepared.tess_prep[0], &prepared.tess_prep[1]},
	               prepared.pixel_prep, prepared.programs, &prepared.program_compile_ns)
	         : pipeline_cache.PrepareGraphicsProgramsSpeculative(
	               sh_ctx.GetVs(), sh_ctx.GetPs(), ctx.GetShaderRegisters(), ctx,
	               registers.user_config, prepared.target_export_mapping, prepared.pixel_active,
	               prepared.vertex_info, prepared.pixel_info, prepared.vertex_prep,
	               prepared.pixel_prep, prepared.programs, &prepared.program_compile_ns);
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
	prepared.reads_untouched = ReadsUntouched(prepared);
	Common::PrepPhases::Mark(Common::PrepPhases::Phase::Certificate);
	prepared.ok = true;
}

// KYTY_SERVE_IN_WAITS (live, default 0; based on chenxiao07/KytyPS5 6801ddc13, 6e19833bc,
// 3658131e6): a mask of the command-processor waits that also run queued service commands (a game
// thread's readback, an unmap). 1: the commit thread's wait for a head a worker holds (CommitHead),
// which before ran them only after 2 ms. The resolver's wait for the sequencer and the blocked
// queues' wait already run them (graphicsRun.cpp).
static Live::Switch g_serve_in_waits("KYTY_SERVE_IN_WAITS", [](const char* value) -> int64_t {
	return value != nullptr ? static_cast<int64_t>(std::strtoul(value, nullptr, 0) & 1u) : 0;
});

bool ServeInHeadWait() {
	return (g_serve_in_waits.Get() & 1) != 0;
}

// KYTY_SPEC_SEGMENTS (live, default 0; segment.h): 0 off, 1 splice, 2 / verify.
static Live::Switch g_spec_segments("KYTY_SPEC_SEGMENTS", [](const char* value) -> int64_t {
	return static_cast<int64_t>(Segment::ParseMode(value));
});
static Segment::Counters g_segment_counters;

Segment::Mode SpecSegmentsMode() {
	return static_cast<Segment::Mode>(g_spec_segments.Get());
}

Segment::Counters& SegmentCounters() {
	return g_segment_counters;
}

// KYTY_BK_* (live, default 0; bookkeep.h): 0 off, 1 on, 2 / verify.
static int64_t ParseBookkeepMode(const char* value) {
	return static_cast<int64_t>(Bookkeep::ParseMode(value));
}
static Live::Switch       g_bk_handover("KYTY_BK_HANDOVER", ParseBookkeepMode);
static Live::Switch       g_bk_adopt("KYTY_BK_ADOPT", ParseBookkeepMode);
static Live::Switch       g_bk_record("KYTY_BK_RECORD", ParseBookkeepMode);
static Live::Switch       g_bk_lean("KYTY_BK_LEAN", ParseBookkeepMode);
static Bookkeep::Counters g_bk_counters;

Bookkeep::Mode BkHandoverMode() {
	return static_cast<Bookkeep::Mode>(g_bk_handover.Get());
}
Bookkeep::Mode BkAdoptMode() {
	return static_cast<Bookkeep::Mode>(g_bk_adopt.Get());
}
Bookkeep::Mode BkRecordMode() {
	return static_cast<Bookkeep::Mode>(g_bk_record.Get());
}
Bookkeep::Mode BkLeanMode() {
	return static_cast<Bookkeep::Mode>(g_bk_lean.Get());
}
Bookkeep::Counters& BkCounters() {
	return g_bk_counters;
}

static std::atomic<bool (*)()> g_service_commands_pending {nullptr};

void SetServiceCommandsPending(bool (*pending)()) {
	g_service_commands_pending.store(pending, std::memory_order_release);
}

bool ServiceCommandsPending() {
	auto* pending = g_service_commands_pending.load(std::memory_order_acquire);
	return pending != nullptr && pending();
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

// KYTY_DRAW_PREP_LEAN (lean.h): live; every part is valid at any time (the digest cache's epochs
// are maintained whether it is on or not).
// A change also makes the next commit print the PrepStats / DrawPrep lines (closing the A/B slot).
static std::atomic<bool> g_stats_due {false};
static Live::Switch g_lean(
    "KYTY_DRAW_PREP_LEAN", [](const char* value) -> int64_t { return Lean::ParseParts(value); },
    [](int64_t, int64_t) { g_stats_due.store(true, std::memory_order_relaxed); });
static std::atomic<uint64_t> g_lean_verify_checks {0};
static std::atomic<uint64_t> g_lean_verify_mismatches {0};

namespace Lean {

uint32_t Parts() noexcept {
	return static_cast<uint32_t>(g_lean.Get());
}

int DigestVerifyMode() noexcept {
	static const int mode = [] {
		const auto* value = EnvValue("KYTY_DRAW_PREP_LEAN_VERIFY");
		if (value == nullptr || std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	}();
	return mode;
}

void NoteDigest(uint32_t hits, uint32_t misses, uint32_t uncacheable) noexcept {
	auto& local = PrepStats::t_local;
	local.digest_hits += hits;
	local.digest_misses += misses;
	local.digest_uncacheable += uncacheable;
}

void NoteDigestVerify(bool mismatch) noexcept {
	g_lean_verify_checks.fetch_add(1, std::memory_order_relaxed);
	if (mismatch) {
		g_lean_verify_mismatches.fetch_add(1, std::memory_order_relaxed);
	}
}

} // namespace Lean

bool DirectReadsEnabled() {
	return g_direct_reads.Get() != 0;
}

// KYTY_CLEAN_UNTOUCHED (live, default 0; needs KYTY_CP_SEQ=1, which maintains the sticky
// GPU-touched pages, gpuTouchedPages.h): a guest range none of whose pages any GPU-side transition
// ever touched (buffer GPU-dirty ranges, tracker GPU marks, GPU-modified images, backing
// publications: the only state that can make the exact clean predicate refuse a range) is clean
// without running that predicate. The pages are marked inside Coherence::Log::Append before the
// transition becomes observable and are never cleared, so "never touched" read after a state
// change was observable cannot miss it.
// - A preparation records whether all its read and digest ranges were untouched when it finished
//   (reads_untouched, checked on the preparing thread). A log-mode certificate whose log check finds
//   no transition touching those ranges since the preparation began then skips the "every range
//   clean now" pass: an unclean range would need a transition after the generation the
//   preparation loaded (the log check would have seen it) or before it (the preparing thread's
//   check, made after that load, would have seen its mark).
// - The clean-verdict query (kernel/memory.cpp) and the workers' read hint answer untouched ranges
//   directly.
// - The certificate's log walk first tests each entry against the hull of the certificate ranges.
// verify: every shortcut also runs the exact check, uses its answer and counts the ranges the
// shortcut called clean that are not; exit stops at the first.
static int64_t ParseUntouched(const char* value) {
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
		return 0;
	}
	if (std::strcmp(value, "verify") == 0) {
		return 2;
	}
	return std::strcmp(value, "exit") == 0 ? 3 : 1;
}
static Live::Switch g_clean_untouched("KYTY_CLEAN_UNTOUCHED", ParseUntouched);

int UntouchedMode() {
	return static_cast<int>(g_clean_untouched.Get());
}

// KYTY_CLEAN_UNTOUCHED_FINE (live, default 0): the clean-untouched verdicts (here and in
// kernel/memory.cpp) use the 1 KiB marks instead of the 4 KiB ones (gpuTouchedPages.h). Both maps
// are maintained from the start, so it can change at any time; both verdicts are exact.
static Live::Switch g_clean_untouched_fine("KYTY_CLEAN_UNTOUCHED_FINE", Live::ParseDefaultOff);

bool UntouchedFine() {
	return g_clean_untouched_fine.On();
}

static bool RangesUntouched(std::span<const Coherence::Range> ranges) {
	const bool fine = g_clean_untouched_fine.On();
	for (const auto& range: ranges) {
		if (GpuTouched::AnyTouched(fine, range.begin, range.end)) {
			return false;
		}
	}
	return true;
}

// KYTY_CLEAN_UNTOUCHED_DIAG=1 (gpuTouchedPages.h): per touched certificate, the set of marking
// sources whose pages it touches, whether 1 KiB marks would have called it untouched, and the
// pages hit most often (first touched page of each touched certificate). Racy counters, console
// line every 2^18 certificates.
namespace UntouchedDiag {
struct HotPage {
	std::atomic<uint64_t> page {0};
	std::atomic<uint64_t> hits {0};
	std::atomic<uint32_t> mask {0};
	std::atomic<uint64_t> bytes {0};
};
static std::atomic<uint64_t>                 g_certs {0};
static std::atomic<uint64_t>                 g_untouched {0};
static std::atomic<uint64_t>                 g_fine_untouched {0};
static std::array<std::atomic<uint64_t>, 16> g_masks {};
static std::atomic<uint64_t>                 g_ranges {0};
static std::atomic<uint64_t>                 g_touched_ranges {0};
static std::array<HotPage, 512>              g_hot {};

static uint32_t SourceMask(uint64_t begin, uint64_t end) {
	uint32_t mask = 0;
	for (uint32_t s = 0; s < static_cast<uint32_t>(GpuTouched::DiagSource::Count); s++) {
		if (GpuTouched::g_diag_pages[s].AnyTouched(begin, end)) {
			mask |= 1u << s;
		}
	}
	return mask;
}

static void NoteHot(uint64_t page, uint32_t mask, uint64_t bytes) {
	for (uint64_t probe = 0; probe < 8; probe++) {
		auto&    slot = g_hot[(page * 0x9E3779B97F4A7C15ull >> 55u) + probe & (g_hot.size() - 1u)];
		uint64_t key  = slot.page.load(std::memory_order_relaxed);
		if (key == 0 && slot.page.compare_exchange_strong(key, page, std::memory_order_relaxed)) {
			key = page;
		}
		if (key == page) {
			slot.hits.fetch_add(1, std::memory_order_relaxed);
			slot.mask.fetch_or(mask, std::memory_order_relaxed);
			slot.bytes.fetch_add(bytes, std::memory_order_relaxed);
			return;
		}
	}
}

// Commit side (the CP thread only): log entries walked per certificate check, and how many checks
// came from untouched preparations and found the log clean.
static uint64_t g_commit_checks  = 0;
static uint64_t g_commit_entries = 0;
static uint64_t g_commit_untouched_clean = 0;
static uint64_t g_commit_log_conflicts   = 0;
static std::array<uint64_t, 8> g_commit_entry_hist {}; // 0, 1-15, 16-63, 64-255, 256-1023, 1K-4K, 4K-16K, more

static void NoteCommit(uint64_t entries, bool untouched, bool clean) {
	g_commit_checks++;
	g_commit_entries += entries;
	g_commit_untouched_clean += (untouched && clean) ? 1u : 0u;
	g_commit_log_conflicts += clean ? 0u : 1u;
	const size_t bucket = entries == 0      ? 0
	                      : entries < 16    ? 1
	                      : entries < 64    ? 2
	                      : entries < 256   ? 3
	                      : entries < 1024  ? 4
	                      : entries < 4096  ? 5
	                      : entries < 16384 ? 6
	                                        : 7;
	g_commit_entry_hist[bucket]++;
	if ((g_commit_checks & 0x3ffffu) == 0) {
		const double c = static_cast<double>(g_commit_checks) / 100.0;
		std::printf("CleanUntouchedDiag commits: %" PRIu64 " checks, %.1f log entries/check, untouched+clean "
		            "%.1f%%, log not clean %.2f%%; entries hist [0 1-15 16-63 64-255 256-1K 1K-4K 4K-16K "
		            ">16K] %%: %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n",
		            g_commit_checks, static_cast<double>(g_commit_entries) / static_cast<double>(g_commit_checks),
		            static_cast<double>(g_commit_untouched_clean) / c,
		            static_cast<double>(g_commit_log_conflicts) / c, g_commit_entry_hist[0] / c,
		            g_commit_entry_hist[1] / c, g_commit_entry_hist[2] / c, g_commit_entry_hist[3] / c,
		            g_commit_entry_hist[4] / c, g_commit_entry_hist[5] / c, g_commit_entry_hist[6] / c,
		            g_commit_entry_hist[7] / c);
		std::fflush(stdout);
	}
}

static void Print(uint64_t certs) {
	const double c = static_cast<double>(certs) / 100.0;
	std::printf("CleanUntouchedDiag: %" PRIu64 " certs, untouched %.1f%%, 1KiB-untouched %.1f%% (of all), "
	            "ranges touched %.1f%%; touched by source mask [buf trk img pub]:",
	            certs, static_cast<double>(g_untouched.load()) / c,
	            static_cast<double>(g_fine_untouched.load()) / c,
	            100.0 * static_cast<double>(g_touched_ranges.load()) /
	                static_cast<double>(std::max<uint64_t>(1, g_ranges.load())));
	for (uint32_t m = 0; m < 16; m++) {
		if (const auto n = g_masks[m].load(); n != 0) {
			std::printf(" %c%c%c%c=%.1f%%", (m & 1u) ? 'b' : '-', (m & 2u) ? 't' : '-', (m & 4u) ? 'i' : '-',
			            (m & 8u) ? 'p' : '-', static_cast<double>(n) / c);
		}
	}
	std::printf("\n");
	std::array<std::pair<uint64_t, size_t>, 512> order {};
	for (size_t i = 0; i < g_hot.size(); i++) {
		order[i] = {g_hot[i].hits.load(), i};
	}
	std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
	for (size_t i = 0; i < 12 && order[i].first != 0; i++) {
		const auto& slot = g_hot[order[i].second];
		const auto  m    = slot.mask.load();
		std::printf("CleanUntouchedDiag:   page 0x%" PRIx64 " %.1f%% of certs, mask %c%c%c%c, avg range %" PRIu64
		            " B\n",
		            slot.page.load() << 12u, static_cast<double>(order[i].first) / c, (m & 1u) ? 'b' : '-',
		            (m & 2u) ? 't' : '-', (m & 4u) ? 'i' : '-', (m & 8u) ? 'p' : '-',
		            slot.bytes.load() / std::max<uint64_t>(1, order[i].first));
	}
	std::fflush(stdout);
}

static void Note(std::span<const Coherence::Range> a, std::span<const Coherence::Range> b, bool untouched) {
	const auto certs = g_certs.fetch_add(1, std::memory_order_relaxed) + 1;
	if (untouched) {
		g_untouched.fetch_add(1, std::memory_order_relaxed);
		g_fine_untouched.fetch_add(1, std::memory_order_relaxed);
	} else {
		uint32_t mask     = 0;
		bool     fine     = true;
		bool     first    = true;
		uint64_t ranges   = 0;
		uint64_t touched  = 0;
		for (const auto list: {a, b}) {
			for (const auto& range: list) {
				ranges++;
				if (!GpuTouched::g_pages.AnyTouched(range.begin, range.end)) {
					continue;
				}
				touched++;
				const auto m = SourceMask(range.begin, range.end);
				mask |= m;
				fine = fine && !GpuTouched::g_fine_pages.AnyTouched(range.begin, range.end);
				if (first) {
					// The first touched page of this range.
					for (auto page = range.begin >> 12u; page <= (range.end - 1u) >> 12u; page++) {
						if (GpuTouched::g_pages.AnyTouched(page << 12u, (page + 1u) << 12u)) {
							NoteHot(page, m, range.end - range.begin);
							break;
						}
					}
					first = false;
				}
			}
		}
		g_ranges.fetch_add(ranges, std::memory_order_relaxed);
		g_touched_ranges.fetch_add(touched, std::memory_order_relaxed);
		g_masks[mask & 15u].fetch_add(1, std::memory_order_relaxed);
		if (fine) {
			g_fine_untouched.fetch_add(1, std::memory_order_relaxed);
		}
	}
	if ((certs & 0x3ffffu) == 0) {
		Print(certs);
	}
}
} // namespace UntouchedDiag

// KYTY_CLEAN_UNTOUCHED_RANGES (live, default 0; with KYTY_CLEAN_UNTOUCHED=1): a preparation whose
// ranges were not all untouched records the touched ones (touched_reads), and a commit whose log
// check found no transition touching the certificate since the preparation began runs the clean
// predicate on those only. The reads_untouched argument holds per range: an untouched range turned
// unclean only by a transition after the generation the preparation loaded (the log check would
// have seen it) or before it (the preparing thread's check, made after that load, would have seen
// its mark). At the pirate island about half the certificates touch a few GPU-written pages (4-byte
// reads of buffers written by the GPU and published back), and 79% of their ranges are untouched.
// 2 (verify): the full pass decides; a touched-only "clean" the full pass refuses is counted.
static Live::Switch g_clean_untouched_ranges("KYTY_CLEAN_UNTOUCHED_RANGES", [](const char* value) -> int64_t {
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
		return 0;
	}
	return std::strcmp(value, "verify") == 0 || std::strcmp(value, "2") == 0 ? 2 : 1;
});

static void CollectTouched(PreparedDraw& prepared) {
	const bool fine = g_clean_untouched_fine.On();
	prepared.touched_reads.clear();
	for (const auto list: {prepared.reads.Ranges(), prepared.reads.DigestRanges()}) {
		for (const auto& range: list) {
			if (GpuTouched::AnyTouched(fine, range.begin, range.end)) {
				prepared.touched_reads.push_back(range);
			}
		}
	}
	prepared.touched_valid = true;
}

// The preparing thread, after the read set is final (and its certificate built, if any).
static bool ReadsUntouched(PreparedDraw& prepared) {
	prepared.touched_valid = false;
	if (g_clean_untouched.Get() == 0 || !GpuTouched::Enabled()) {
		return false;
	}
	if (g_clean_untouched_ranges.Get() != 0) {
		CollectTouched(prepared);
		const bool untouched = prepared.touched_reads.empty();
		if (GpuTouched::DiagEnabled()) [[unlikely]] {
			UntouchedDiag::Note(prepared.reads.Ranges(), prepared.reads.DigestRanges(), untouched);
		}
		return untouched;
	}
	bool untouched = false;
	if (prepared.certificate_built) {
		untouched = RangesUntouched(prepared.certificate);
	} else {
		untouched = RangesUntouched(prepared.reads.Ranges()) && RangesUntouched(prepared.reads.DigestRanges());
	}
	if (GpuTouched::DiagEnabled()) [[unlikely]] {
		if (prepared.certificate_built) {
			UntouchedDiag::Note(prepared.certificate, {}, untouched);
		} else {
			UntouchedDiag::Note(prepared.reads.Ranges(), prepared.reads.DigestRanges(), untouched);
		}
	}
	return untouched;
}

// KYTY_CERT_PREFETCH (live, default 0): when the commit thread has the head ready, it issues cache
// prefetches for the next position's certificate (the PreparedDraw line holding the generation,
// the certificate vector and the untouched flag, then the certificate ranges; 2 also the read and
// digest range lists the exact clean pass walks), if that preparation is already done. Those lines
// were written by a worker on another core, so the commit's log check otherwise starts with
// cross-core misses. A hint only: nothing is decided from it.
static Live::Switch g_cert_prefetch("KYTY_CERT_PREFETCH", [](const char* value) -> int64_t {
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
		return 0;
	}
	return std::strcmp(value, "2") == 0 ? 2 : 1;
});

static inline void PrefetchLine(const void* address) {
#if defined(__clang__) || defined(__GNUC__)
	__builtin_prefetch(address, 0, 3);
#elif defined(_M_X64) || defined(_M_IX86)
	_mm_prefetch(static_cast<const char*>(address), _MM_HINT_T0);
#endif
}

template <typename T>
static void PrefetchSpan(std::span<const T> items, size_t max_lines) {
	const auto* begin = reinterpret_cast<const uint8_t*>(items.data());
	const auto  bytes = items.size_bytes();
	for (size_t offset = 0, lines = 0; offset < bytes && lines < max_lines; offset += 64, lines++) {
		PrefetchLine(begin + offset);
	}
}

// The PreparedDraw lines a commit's certificate check reads first (no dependent loads).
static void PrefetchHeader(const PreparedDraw& prepared, int64_t mode) {
	PrefetchLine(&prepared.coherence_generation);
	PrefetchLine(&prepared.reads_untouched);
	if (mode == 2) {
		PrefetchLine(&prepared.reads);
		PrefetchLine(reinterpret_cast<const uint8_t*>(&prepared.reads) + 128);
	}
}

// The range lists (reads the header lines, prefetched one commit earlier).
static void PrefetchCertificate(const PreparedDraw& prepared, int64_t mode) {
	if (prepared.certificate_built) {
		PrefetchSpan(std::span<const Coherence::Range>(prepared.certificate), 8);
	}
	if (mode == 2 || !prepared.reads_untouched) {
		PrefetchLine(&prepared.reads);
		PrefetchSpan(prepared.reads.Ranges(), 8);
		PrefetchSpan(prepared.reads.DigestRanges(), 4);
	}
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
	prepared.reads.SetLean(Lean::On(Lean::ReadSet)); // KYTY_DRAW_PREP_LEAN read-set part
	prepared.certificate.clear();
	prepared.certificate_built = false;
	prepared.reads_untouched   = false;
	prepared.touched_valid     = false;
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
	recorder.code_epoch = t_code_epoch; // KYTY_DRAW_PREP_LEAN digest cache (0: not used)
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
	prepared.reads_untouched = ReadsUntouched(prepared);
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

static std::atomic<uint64_t> g_untouched_checks {0};
static std::atomic<uint64_t> g_untouched_hits {0};
static std::atomic<uint64_t> g_untouched_mismatches {0};
static std::atomic<uint64_t> g_untouched_ranges_checks {0};
static std::atomic<uint64_t> g_untouched_ranges_mismatches {0};

static bool AllCleanAfterLog(const PreparedDraw& prepared) {
	if (const auto untouched = g_clean_untouched.Get(); untouched != 0) {
		if (untouched == 1) {
			if (prepared.reads_untouched) {
				return true;
			}
			if (prepared.touched_valid) {
				// KYTY_CLEAN_UNTOUCHED_RANGES: only the ranges that were touched.
				const auto since    = prepared.coherence_generation;
				const bool shortcut = g_clean_since.Get() == 1;
				const auto is_clean = [&](uint64_t address, uint64_t size) {
					return shortcut ? LibKernel::Memory::IsGpuCleanForReadSince(address, size, since)
					                : LibKernel::Memory::IsGpuCleanForRead(address, size);
				};
				bool touched_clean = true;
				for (const auto& range: prepared.touched_reads) {
					if (!is_clean(range.begin, range.end - range.begin)) {
						touched_clean = false;
						break;
					}
				}
				if (g_clean_untouched_ranges.Get() != 2) {
					return touched_clean;
				}
				const bool exact = prepared.reads.AllClean(LibKernel::Memory::IsGpuCleanForRead);
				const auto checks = g_untouched_ranges_checks.fetch_add(1, std::memory_order_relaxed) + 1;
				if (touched_clean && !exact) {
					const auto count = g_untouched_ranges_mismatches.fetch_add(1, std::memory_order_relaxed) + 1;
					if (count <= 16) {
						std::printf("CleanUntouchedRanges: the touched ranges are clean but not all ranges "
						            "(%zu ranges, %zu touched)\n",
						            prepared.reads.Ranges().size(), prepared.touched_reads.size());
					}
				}
				if ((checks & 0x3ffffu) == 0) {
					std::printf("CleanUntouchedRanges verify: %" PRIu64 " checks, %" PRIu64 " mismatches\n",
					            checks, g_untouched_ranges_mismatches.load(std::memory_order_relaxed));
					std::fflush(stdout);
				}
				return exact;
			}
		} else {
			// verify: the exact pass decides; count how often the shortcut applies and errs.
			const bool exact = prepared.reads.AllClean(LibKernel::Memory::IsGpuCleanForRead);
			const auto checks = g_untouched_checks.fetch_add(1, std::memory_order_relaxed) + 1;
			if (prepared.reads_untouched) {
				g_untouched_hits.fetch_add(1, std::memory_order_relaxed);
				if (!exact) {
					const auto count = g_untouched_mismatches.fetch_add(1, std::memory_order_relaxed) + 1;
					if (count <= 16) {
						std::printf("CleanUntouched: a certificate with untouched ranges is not clean "
						            "(%zu ranges)\n",
						            prepared.reads.Ranges().size());
					}
					if (untouched == 3) {
						EXIT("CleanUntouched: a certificate with untouched ranges is not clean\n");
					}
				}
			}
			if ((checks & 0x3ffffu) == 0) {
				std::printf("CleanUntouched verify: %" PRIu64 " certificates, %" PRIu64
				            " untouched, %" PRIu64 " mismatches\n",
				            checks, g_untouched_hits.load(std::memory_order_relaxed),
				            g_untouched_mismatches.load(std::memory_order_relaxed));
			}
			return exact;
		}
	}
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

// KYTY_CERT_LOG_SUMMARY (live, default 0): the log-mode certificate first asks the commit thread's
// Coherence::Summary (every log entry ingested once into newest-generation-per-64-KiB-block),
// and walks the log (Coherence::Log::Check) only when the summary cannot vouch for the interval.
// The walk read every transition since the preparation began, for every commit: with a deep
// preparation window most entries were read by many commits (about 1.6% of the pirate island CP).
// 2 (verify): both run; the walk decides, and a summary "clean" the walk does not confirm is
// counted and logged.
static Live::Switch g_cert_log_summary("KYTY_CERT_LOG_SUMMARY", [](const char* value) -> int64_t {
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
		return 0;
	}
	return std::strcmp(value, "verify") == 0 || std::strcmp(value, "2") == 0 ? 2 : 1;
});
static std::atomic<uint64_t> g_summary_checks {0};
static std::atomic<uint64_t> g_summary_clean {0};
static std::atomic<uint64_t> g_summary_mismatches {0};

static Coherence::CheckOutcome SummaryCheck(uint64_t from, std::span<const Coherence::Range> ranges,
                                            int64_t mode) {
	// Commits run on the GPU thread only (Validate, ValidateDispatch).
	static Coherence::Summary summary(Coherence::g_log);
	const auto                to = Coherence::Generation();
	if (!GuestGpu::IsGpuThread()) {
		return Coherence::g_log.Check(from, to, ranges, true);
	}
	const bool clean = summary.Clean(from, to, ranges);
	const auto checks = g_summary_checks.fetch_add(1, std::memory_order_relaxed) + 1;
	if (clean) {
		g_summary_clean.fetch_add(1, std::memory_order_relaxed);
	}
	if (mode == 1) {
		if (clean) {
			return {};
		}
		return Coherence::g_log.Check(from, to, ranges, true);
	}
	const auto exact = Coherence::g_log.Check(from, to, ranges, true);
	if (clean && exact.result == Coherence::CheckResult::Conflict) {
		const auto count = g_summary_mismatches.fetch_add(1, std::memory_order_relaxed) + 1;
		if (count <= 16) {
			std::printf("CertLogSummary: the summary called an interval clean that conflicts (from %" PRIu64
			            " to %" PRIu64 ", %zu ranges)\n",
			            from, to, ranges.size());
		}
	}
	if ((checks & 0x3ffffu) == 0) {
		std::printf("CertLogSummary verify: %" PRIu64 " checks, %" PRIu64 " clean by summary, %" PRIu64
		            " mismatches\n",
		            checks, g_summary_clean.load(std::memory_order_relaxed),
		            g_summary_mismatches.load(std::memory_order_relaxed));
		std::fflush(stdout);
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
		Coherence::CheckOutcome outcome;
		if (const auto summary = g_cert_log_summary.Get(); summary != 0) {
			outcome = SummaryCheck(prepared.coherence_generation, cert_ranges, summary);
		} else {
			outcome = Coherence::g_log.Check(prepared.coherence_generation, Coherence::Generation(),
			                                 cert_ranges, g_clean_untouched.Get() != 0);
		}
		if (stats_start != 0) {
			CommitStats::AddSub(CommitStats::Sub::ValLog, CommitStats::Now() - stats_start);
			CommitStats::AddItems(CommitStats::Item::LogEntries, outcome.entries);
			CommitStats::AddItems(CommitStats::Item::CertRanges, cert_ranges.size());
		}
		if (GpuTouched::DiagEnabled()) [[unlikely]] {
			UntouchedDiag::NoteCommit(outcome.entries, prepared.reads_untouched,
			                          outcome.result == Coherence::CheckResult::Clean);
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

// KYTY_CERT_COUNT_LAZY (live, default 0): CountCertificate walks the read and digest range lists
// only when frame events are counted at all (Profiler::FrameEventsOn). Otherwise its sums are
// thrown away, and for an untouched certificate (KYTY_CLEAN_UNTOUCHED) that walk was the commit's
// only touch of those lists, cross-core misses included (about 0.8% of the Sky Garden CP).
static Live::Switch g_cert_count_lazy("KYTY_CERT_COUNT_LAZY", Live::ParseDefaultOff);

static void CountCertificate(const PreparedDraw& prepared) {
	if (g_cert_count_lazy.On() && !Profiler::FrameEventsOn()) {
		return;
	}
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

// KYTY_DRAW_PREP_VERIFY: draws compared with the serial path, and differences (DrawPrep 10 s line).
std::atomic<uint64_t> g_draw_verify_checks {0};
std::atomic<uint64_t> g_draw_verify_mismatches {0};

bool VerifyCommitted(bool pixel_active, const PipelineCache::GraphicsPrograms& programs,
                     const ShaderVertexInputInfo& vertex_info, const ShaderPixelInputInfo& pixel_info,
                     const PipelineCache::GraphicsStagePreps& preps,
                     const PipelineCache::GraphicsPrograms& serial_programs,
                     const ShaderVertexInputInfo&            serial_vertex_info,
                     const ShaderPixelInputInfo&             serial_pixel_info,
                     const PipelineCache::GraphicsStagePreps& serial_preps) {
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepVerifyChecks);
	g_draw_verify_checks.fetch_add(1, std::memory_order_relaxed);
	const bool programs_equal = programs.pixel.id == serial_programs.pixel.id &&
	                            programs.vertex[0].id == serial_programs.vertex[0].id &&
	                            programs.vertex[1].id == serial_programs.vertex[1].id &&
	                            programs.vertex[2].id == serial_programs.vertex[2].id;
	// Without an active pixel shader neither path writes the pixel prep (the draw state keeps
	// an earlier draw's, which nothing reads).
	const bool pixel_equal = SamePixelInfo(pixel_info, serial_pixel_info) &&
	                         (!pixel_active || SamePrep(preps.pixel, serial_preps.pixel));
	// KYTY_DRAW_PREP_TESS: a tessellated draw's HS and TES preps too.
	const bool tess_equal = !serial_programs.vertex[1] ||
	                        (SamePrep(preps.vertex[1], serial_preps.vertex[1]) &&
	                         SamePrep(preps.vertex[2], serial_preps.vertex[2]));
	const bool vertex_equal = SamePrep(preps.vertex[0], serial_preps.vertex[0]) &&
	                          SameVertexInfo(vertex_info, serial_vertex_info) && tess_equal;
	if (programs_equal && pixel_equal && vertex_equal) {
		return true;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepVerifyMismatches);
	g_draw_verify_mismatches.fetch_add(1, std::memory_order_relaxed);
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
	// KYTY_DRAW_RUN_TARGETS: the registers' part of the key (0 when the key is 0).
	uint64_t run_targets_key = 0;
	// KYTY_DRAW_RUN_PIPESWAP: the key without the vertex program.
	uint64_t run_swap_key = 0;
	// KYTY_DRAW_RUN_TEXDELTA: the key without the T#/S# words (texDelta.h).
	uint64_t run_texfree_key = 0;
	// KYTY_DRAW_RUN_KEY_DIAG: the key's part hashes (only while the switch is on).
	DrawRun::KeyParts run_parts {};
	bool              run_parts_valid = false;
	// PrepStats: the TSC when the preparing thread completed the slot.
	uint64_t done_tsc = 0;
	// KYTY_DRAW_PREP_LEAN digest cache: the publication epoch when the slot was filled
	// (syncEpoch.h CurrentPublish, read by the publishing thread).
	uint64_t code_epoch = 0;
	// KYTY_SPEC_SEGMENTS (segment.h): the draw the preparing thread encoded (miss NotEncoded while
	// the switch was off when it prepared the slot).
	Segment::EncodedDraw segment;
};

namespace {
// The epoch a preparation of `slot` may use for the code digest cache: none for speculative slots
// (published before their wait completed).
uint64_t SlotCodeEpoch(const Engine::Slot& slot) {
	return slot.speculative ? 0 : slot.code_epoch;
}

// Sets t_code_epoch for one preparation on this thread.
class CodeEpochScope {
public:
	explicit CodeEpochScope(uint64_t epoch) noexcept { t_code_epoch = epoch; }
	~CodeEpochScope() { t_code_epoch = 0; }
	CodeEpochScope(const CodeEpochScope&)            = delete;
	CodeEpochScope& operator=(const CodeEpochScope&) = delete;
};
} // namespace

namespace {

// KYTY_SPEC_SEGMENTS (segment.h), after the run key on the preparing thread: the slot's encoded
// draw. Reads only the slot (its arguments, registers and preparation).
void EncodeSlotSegment(Engine::Slot& slot) {
	if (SpecSegmentsMode() == Segment::Mode::Off) [[likely]] {
		slot.segment.miss = Segment::Miss::NotEncoded;
		return;
	}
	Segment::EncodeInputs in;
	in.indexed = slot.kind == DrawKind::Index;
	if (in.indexed) {
		const auto& args   = slot.index_args;
		const auto& vs     = slot.prepared.vertex_info;
		const auto  prim   = slot.registers.user_config.GetPrimType();
		in.prepared_ok     = slot.prepared.ok;
		in.run_key         = slot.run_key;
		in.offsets_from_state   = args.offset_source == DrawOffsetSource::DrawState;
		in.gpu_instance_address = args.gpu_instance_address;
		in.plain_primitive = prim != Prospero::PrimitiveType::kQuadListLegacy &&
		                     (prim == Prospero::PrimitiveType::kPointList || prim == Prospero::PrimitiveType::kLineList ||
		                      prim == Prospero::PrimitiveType::kLineStrip || prim == Prospero::PrimitiveType::kTriList ||
		                      prim == Prospero::PrimitiveType::kTriFan || prim == Prospero::PrimitiveType::kTriStrip ||
		                      prim == Prospero::PrimitiveType::kRectList || prim == Prospero::PrimitiveType::kPolygon ||
		                      prim == Prospero::PrimitiveType::kRectListLegacy || prim == Prospero::PrimitiveType::kPatch);
		in.index_type      = args.index_type_and_size;
		in.vertex_stage    = !vs.fetch_embedded || static_cast<bool>(vs.stage);
		in.index_count     = args.index_count;
		in.instance_count  = args.instance_count;
		in.base_vertex     = args.base_vertex;
		in.index_address   = reinterpret_cast<uint64_t>(args.index_addr);
		if (in.prepared_ok && in.vertex_stage) {
			const auto [vertex_offset, instance_offset] =
			    ResolveDrawOffsets(slot.registers.user_config.GetIndexOffset(), vs);
			in.resolved_vertex_offset   = vertex_offset;
			in.resolved_instance_offset = instance_offset;
		}
	}
	slot.segment = Segment::Encode(in);
	auto& counters = SegmentCounters();
	if (slot.segment.miss == Segment::Miss::None) {
		counters.encoded.fetch_add(1, std::memory_order_relaxed);
	} else {
		counters.misses[static_cast<size_t>(slot.segment.miss)].fetch_add(1, std::memory_order_relaxed);
	}
}

// After Prepare, on the preparing thread: the repeat trace's hashes of this draw's inputs, and the
// run key (KYTY_DRAW_RUN).
void HashForRepeatTrace(Engine::Slot& slot) {
	if (slot.kind == DrawKind::Dispatch) {
		slot.run_key = 0; // not a draw: no run, not in the draw repeat trace
		return;
	}
	slot.run_targets_key = 0;
	slot.run_swap_key    = 0;
	slot.run_texfree_key = 0;
	slot.run_key = DrawRun::Enabled() ? DrawRun::StructureKey(slot.registers.context,
	                                                          slot.registers.user_config,
	                                                          slot.prepared, &slot.run_targets_key,
	                                                          &slot.run_swap_key,
	                                                          DrawRun::TexDeltaMode() != 0 ? &slot.run_texfree_key : nullptr)
	                                  : 0;
	slot.run_parts_valid = slot.run_key != 0 && DrawRun::KeyDiagEnabled();
	if (slot.run_parts_valid) [[unlikely]] {
		DrawRun::StructureKeyParts(slot.registers.context, slot.registers.user_config, slot.prepared,
		                           slot.run_parts);
	}
	EncodeSlotSegment(slot);
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
// KYTY_DRAW_PREP_ADAPT (adaptivePool.h): the demand-sized pool's settings.
struct AdaptConfig {
	uint64_t spin_ns     = 0;
	uint32_t min_cap     = 1;
	uint32_t max_cap     = 0; // 0: affinity processors - 1 (AdaptiveProcessorBound)
	uint64_t interval_ns = 250'000'000;
};

namespace {
// The logical processors the process may run on (its affinity mask), not the machine's.
uint32_t ProcessAffinityProcessors() {
	return Common::ProcessAffinityProcessorCount();
}

// The adaptive pool's last cap decision, for the PrepStats line (publisher thread writes).
std::atomic<uint32_t> g_adapt_cap {0};
std::atomic<uint32_t> g_adapt_lps {0};
std::atomic<uint64_t> g_adapt_busy_milli {0}; // busy cores x 1000 of the last interval

// The engine's worker pool (one parallel engine per process) and its mode setter, for the live switch
// below (Engine::Workers is private to the engine).
std::atomic<void*> g_live_workers {nullptr};
void (*g_live_set_adapt)(void*, bool) = nullptr;
void (*g_live_set_active)(void*, uint32_t) = nullptr;
void ApplyAdaptMode(int64_t previous, int64_t value);
void ApplyActiveWorkers(int64_t previous, int64_t value);
} // namespace

// KYTY_DRAW_PREP_ACTIVE=<n> (live, default unset or 0 = every worker): only the first n draw-prep
// workers run the hot/cold gate loop (KYTY_DRAW_PREP_HOT of them hot, the rest cold); the others park
// until the value changes. With the six default workers, n = 3 is KYTY_DRAW_PREP_WORKERS=3 (2 hot +
// 1 cold) switchable inside one boot. A worker that is retired while parked cold leaves when its
// next cold wake reaches it, so that one wake is lost (the commit thread prepares an unclaimed head
// itself, as always). No effect while KYTY_DRAW_PREP_ADAPT is on (its cap decides); applies again
// when the gate loop resumes.
static Live::Switch g_active_workers(
    "KYTY_DRAW_PREP_ACTIVE",
    [](const char* value) -> int64_t {
	    return value != nullptr ? static_cast<int64_t>(std::min(std::strtoul(value, nullptr, 10), 32ul)) : 0;
    },
    [](int64_t previous, int64_t value) { ApplyActiveWorkers(previous, value); });

// KYTY_DRAW_PREP_ADAPT=1 (live, default 0): the adaptive pool instead of the hot/cold gate. Both
// exist from the start; a change moves the workers from one loop to the other (on the command
// processor thread at a flip): the new loop's leave flag is cleared and the mode set before the old
// loop's leave flag is set and its sleepers woken, so a worker leaving one loop finds the other.
// Publishes in between may wake the side the workers just left; the commit thread then prepares
// such a head itself, as any unclaimed head.
// Unset follows KYTY_CORE_BUDGET_PROFILE (common/coreBudget.h); a profile change moves the workers
// the same way while the switch is unset.
static Live::Switch g_adapt_mode("KYTY_DRAW_PREP_ADAPT", Common::ParseTriState,
                                 [](int64_t previous, int64_t value) { ApplyAdaptMode(previous, value); });

namespace {
bool AdaptOn() {
	return Common::EffectiveSwitch(g_adapt_mode.Get());
}
[[maybe_unused]] const bool g_adapt_profile_listener = [] {
	Common::OnCoreBudgetProfileChange([](bool /*on*/) {
		if (g_adapt_mode.Get() < 0) {
			ApplyAdaptMode(-1, -1);
		}
	});
	return true;
}();
} // namespace

struct Engine::Workers {
	Workers(PipelineCache& pipeline_cache, uint32_t window, uint32_t count, uint64_t spin_ns,
	        uint32_t hot, uint32_t wake_backlog, uint64_t cold_spin_ns, StealPolicy steal,
	        bool cold_token, const AdaptConfig& config)
	    : pipeline_cache(pipeline_cache), window(window), spin_ns(spin_ns),
	      cold_spin_ns(cold_spin_ns), steal(steal), gate(count, hot, wake_backlog, cold_token),
	      adapt_config(config), threads_count(std::min(count, AdaptivePool::MaxWorkers)) {
		// KYTY_SPIN_BUDGET (common/spinGate.h): the commit thread may park on a held head; the
		// workers' Complete then wakes it. Decided before any worker starts.
		if (Common::SpinGate::BudgetAvailable()) {
			this->window.EnableHeadWake();
		}
		const auto lps = ProcessAffinityProcessors();
		// Start at the bound; the first interval's busy time sizes it.
		const auto cap = AdaptiveCap(1000.0, count, lps, config.min_cap, config.max_cap);
		adapt          = std::make_unique<AdaptivePool>(count, cap);
		g_adapt_cap.store(cap, std::memory_order_relaxed);
		g_adapt_lps.store(lps, std::memory_order_relaxed);
		const bool on = AdaptOn();
		adapt_on.store(on, std::memory_order_seq_cst);
		leave_adapt.store(!on, std::memory_order_seq_cst);
		leave_gate.store(on, std::memory_order_seq_cst);
		const auto active_value = static_cast<uint32_t>(g_active_workers.Get());
		active.store(active_value == 0 ? count : std::min(active_value, count), std::memory_order_seq_cst);
		PublishLeaves();
		for (uint32_t index = 0; index < count; index++) {
			threads.emplace_back([this, index] { Run(index); });
		}
		g_live_set_adapt  = [](void* workers, bool on) { static_cast<Workers*>(workers)->SetAdapt(on); };
		g_live_set_active = [](void* workers, uint32_t n) { static_cast<Workers*>(workers)->SetActive(n); };
		g_live_workers.store(this, std::memory_order_seq_cst);
	}

	~Workers() {
		g_live_workers.store(nullptr, std::memory_order_seq_cst);
		stop.store(true, std::memory_order_seq_cst);
		leave_gate.store(true, std::memory_order_seq_cst);
		leave_adapt.store(true, std::memory_order_seq_cst);
		PublishLeaves();
		gate.WakeAll();
		adapt->WakeAll();
		NotifyActive();
		for (auto& thread: threads) {
			thread.join();
		}
	}

	// KYTY_DRAW_PREP_ACTIVE: each worker's leave flag for the gate loop = the loop's own leave flag,
	// or the worker is beyond the active count (command processor thread, and the constructor).
	void PublishLeaves() {
		const bool all = leave_gate.load(std::memory_order_seq_cst);
		const auto n   = active.load(std::memory_order_seq_cst);
		for (uint32_t index = 0; index < threads_count; index++) {
			leave_worker[index].store(all || index >= n, std::memory_order_seq_cst);
		}
	}
	// Wakes the workers parked beyond the active count, so they re-read it (and the mode, and stop).
	void NotifyActive() {
		active_word.fetch_add(1, std::memory_order_seq_cst);
		active_word.notify_all();
	}
	// KYTY_DRAW_PREP_ACTIVE live change (command processor thread). 0 = every worker.
	void SetActive(uint32_t n) {
		n = n == 0 ? threads_count : std::min(n, threads_count);
		active.store(n, std::memory_order_seq_cst);
		PublishLeaves();
		NotifyActive();
	}

	Workers(const Workers&)            = delete;
	Workers& operator=(const Workers&) = delete;

	// KYTY_DRAW_PREP_ADAPT live change (command processor thread).
	void SetAdapt(bool on) {
		if (on) {
			leave_adapt.store(false, std::memory_order_seq_cst);
			adapt_on.store(true, std::memory_order_seq_cst);
			leave_gate.store(true, std::memory_order_seq_cst);
			PublishLeaves();
			gate.WakeAll();
			NotifyActive();
		} else {
			// WakeAll left the cold token in its shutdown state; no worker is parked in the gate now.
			gate.Rearm();
			leave_gate.store(false, std::memory_order_seq_cst);
			PublishLeaves();
			adapt_on.store(false, std::memory_order_seq_cst);
			leave_adapt.store(true, std::memory_order_seq_cst);
			adapt->WakeAll();
		}
	}

	// Producer side, after Publish: wake parked hot workers, and now and then check whether the
	// backlog needs a cold one. Paired with the gate's park protocol (seq_cst sleepers count and
	// predicate reload): either the worker sees the new slot before parking, or this thread sees
	// it parking and changes the signal it waits on.
	// KYTY_DRAW_PREP_ADAPT: wake one parked worker when nobody spins (adaptivePool.h), and
	// recompute the active cap about every 250 ms.
	void Wake() {
		if (adapt_on.load(std::memory_order_relaxed)) {
			if (adapt->OnPublish()) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepColdWakes);
			}
			if (adapt->CapDue(adapt_config.interval_ns)) {
				const auto busy = adapt->TakeBusyCores();
				const auto lps  = ProcessAffinityProcessors();
				const auto cap  = AdaptiveCap(busy, adapt->Workers(), lps, adapt_config.min_cap,
				                              adapt_config.max_cap);
				adapt->SetCap(cap);
				g_adapt_cap.store(cap, std::memory_order_relaxed);
				g_adapt_lps.store(lps, std::memory_order_relaxed);
				g_adapt_busy_milli.store(static_cast<uint64_t>(busy * 1000.0), std::memory_order_relaxed);
			}
			return;
		}
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
		{
			const CodeEpochScope epoch(SlotCodeEpoch(slot));
			PrepareSlot(pipeline_cache, slot, false);
		}
		HashForRepeatTrace(slot);
		Phases::Mark(Phases::Phase::RunKey);
		PlanBindings(slot, plan_context);
		Phases::Mark(Phases::Phase::Plan);
		slot.worker_prepared = true;
		slot.done_tsc        = PrepStats::Tsc();
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
		const auto prepare = [this, &placement_count](Slot& slot, uint64_t seq) {
			Common::HelperPriority::Check(0); // KYTY_HELPER_PRIORITY (measurement)
			PrepareClaimed(slot, seq);
			if ((++placement_count & 63u) == 0u) {
				Common::SamplePlacement(Common::ThreadRole::Host);
			}
		};
		// KYTY_DRAW_PREP_LEAN claim part: two slots per claim-counter update in a backlog.
		const auto batch = [] { return Lean::On(Lean::Claim) ? 2u : 1u; };
		while (!stop.load(std::memory_order_acquire)) {
			if (adapt_on.load(std::memory_order_seq_cst)) {
				if (leave_adapt.load(std::memory_order_seq_cst)) {
					WorkerRelax(); // a switch in progress
					continue;
				}
				RunAdaptiveWorker(*adapt, window, index, adapt_config.spin_ns, leave_adapt, prepare, batch);
			} else {
				if (leave_gate.load(std::memory_order_seq_cst)) {
					WorkerRelax();
					continue;
				}
				// KYTY_DRAW_PREP_ACTIVE: beyond the active count, park until it (or the mode) changes.
				if (index >= active.load(std::memory_order_seq_cst)) {
					const auto observed = active_word.load(std::memory_order_seq_cst);
					if (index >= active.load(std::memory_order_seq_cst) &&
					    !adapt_on.load(std::memory_order_seq_cst) && !stop.load(std::memory_order_seq_cst)) {
						active_word.wait(observed, std::memory_order_seq_cst);
					}
					continue;
				}
				// KYTY_SPIN_BUDGET (common/spinGate.h): a short idle spin while the core budget is
				// exceeded; the idle periods' spin and parks go to the site counters.
				struct BudgetHooks {
					[[nodiscard]] uint64_t SpinLimit(uint64_t wanted) const {
						return Common::SpinGate::SpinNs(Common::SpinGate::Site::Worker, wanted);
					}
					void Spun(uint64_t tsc, bool parked) const {
						Common::SpinGate::NoteSpin(Common::SpinGate::Site::Worker, tsc);
						if (parked) {
							Common::SpinGate::NotePark(Common::SpinGate::Site::Worker);
						}
					}
				};
				RunPreparationWorkerBatched(
				    gate, window, index, spin_ns, cold_spin_ns, leave_worker[index], prepare,
				    [] { Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepColdWakes); }, batch,
				    BudgetHooks {});
			}
		}
	}
	PipelineCache&           pipeline_cache;
	BindingPlanContext       plan_context; // set by the engine before the first Submit
	Window<Slot>             window;
	uint64_t                 spin_ns      = 0;
	uint64_t                 cold_spin_ns = 0;
	StealPolicy              steal; // KYTY_DRAW_PREP_STEAL, KYTY_DRAW_PREP_STEAL_AFTER_US
	WorkerGate               gate;
	AdaptConfig              adapt_config;
	std::unique_ptr<AdaptivePool> adapt; // KYTY_DRAW_PREP_ADAPT
	// KYTY_DRAW_PREP_ADAPT live: which loop the workers run, and each loop's leave flag (also set at
	// shutdown).
	std::atomic<bool> adapt_on {false};
	std::atomic<bool> leave_gate {false};
	std::atomic<bool> leave_adapt {true};
	std::atomic<bool>        stop {false};
	// KYTY_DRAW_PREP_ACTIVE: the workers that run the gate loop, the parked others' wake word, and
	// each worker's leave flag for that loop (PublishLeaves).
	const uint32_t                                       threads_count;
	std::atomic<uint32_t>                                active {0};
	std::atomic<uint32_t>                                active_word {0};
	std::array<std::atomic<bool>, AdaptivePool::MaxWorkers> leave_worker {};
	std::vector<std::thread> threads;
};

namespace {
void ApplyAdaptMode(int64_t /*previous*/, int64_t value) {
	// Command processor thread only. Unset (-1) follows KYTY_CORE_BUDGET_PROFILE.
	const bool on = Common::EffectiveSwitch(value);
	if (auto* workers = g_live_workers.load(std::memory_order_seq_cst); workers != nullptr) {
		g_live_set_adapt(workers, on);
	}
	g_stats_due.store(true, std::memory_order_relaxed); // closes the A/B slot's PrepStats period
}
void ApplyActiveWorkers(int64_t /*previous*/, int64_t value) {
	// Command processor thread only.
	if (auto* workers = g_live_workers.load(std::memory_order_seq_cst); workers != nullptr) {
		g_live_set_active(workers, static_cast<uint32_t>(value));
	}
	g_stats_due.store(true, std::memory_order_relaxed);
}
} // namespace

Engine::Engine(RenderContext& renderer, std::function<void()> service_commands,
               void (*after_commit)(void*), void* after_commit_context)
    : m_renderer(renderer), m_mode(GetMode()), m_service_commands(std::move(service_commands)),
      m_after_commit(after_commit), m_after_commit_context(after_commit_context),
      m_inline_slot(std::make_unique<Slot>()) {
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
		// KYTY_DRAW_PREP_ADAPT=1 (adaptivePool.h): the demand-sized pool instead of the hot/cold
		// gate (KYTY_DRAW_PREP_HOT, _SPIN_US, _COLD_SPIN_US, _WAKE_BACKLOG are then unused).
		AdaptConfig adapt;
		adapt.spin_ns     = uint64_t {EnvUnsigned("KYTY_DRAW_PREP_ADAPT_SPIN_US", 10, 0, 1000000)} * 1000u;
		adapt.min_cap     = EnvUnsigned("KYTY_DRAW_PREP_ADAPT_MIN", 1, 1, 32);
		adapt.max_cap     = EnvUnsigned("KYTY_DRAW_PREP_ADAPT_MAX", 0, 0, 32);
		adapt.interval_ns = uint64_t {EnvUnsigned("KYTY_DRAW_PREP_ADAPT_INTERVAL_MS", 250, 10, 10000)} * 1000000u;
		m_workers = std::make_unique<Workers>(m_renderer.GetPipelineCache(), window, workers,
		                                      uint64_t {spin_us} * 1000u, hot, wake_backlog,
		                                      uint64_t {cold_spin_us} * 1000u, steal, cold_token, adapt);
		m_workers->plan_context = MakeBindingPlanContext(m_renderer);
		LOGF("DrawPrep: parallel mode, window=%u workers=%u spin=%uus hot=%u wake_backlog=%u "
		     "cold_spin=%uus cold_token=%d steal=%u steal_after=%uus cert=%s verify=%d adapt=%d "
		     "adapt_spin=%" PRIu64 "us adapt_min=%u adapt_max=%u cap=%u lps=%u\n",
		     m_workers->window.Capacity(), workers, spin_us, m_workers->gate.HotCount(),
		     wake_backlog, cold_spin_us, cold_token ? 1 : 0, steal.min_unclaimed, steal_us,
		     GetCertMode() == CertMode::Log ? "log" : "value", VerifyMode(), AdaptOn() ? 1 : 0,
		     adapt.spin_ns / 1000u, adapt.min_cap, adapt.max_cap,
		     g_adapt_cap.load(std::memory_order_relaxed), g_adapt_lps.load(std::memory_order_relaxed));
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
	slot.code_epoch      = SyncEpoch::CurrentPublish();
	slot.segment.miss    = Segment::Miss::NotEncoded; // (until its preparing thread encodes it)
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
	const HeadPatch patch {submit_id, instance_count};
	CommitHead(&patch);
}

void Engine::CommitHead(const HeadPatch* patch) {
	auto& slot = ReadyHead();
	// A dispatch slot is only ever taken by its own op (ExecuteDispatch).
	EXIT_IF(slot.kind == DrawKind::Dispatch);
	if (patch != nullptr) {
		// The preparation (and the preparing thread's reads of the slot) is complete.
		slot.submit_id = patch->submit_id;
		if (patch->instance_count != UINT32_MAX) {
			if (slot.kind == DrawKind::Index) {
				slot.index_args.instance_count = patch->instance_count;
			} else {
				slot.auto_args.instance_count = patch->instance_count;
			}
		}
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
	// KYTY_BK_HANDOVER (bookkeep.h): a head inside the ready run was seen finished already.
	const auto handover = static_cast<Bookkeep::Mode>(g_bk_handover.Get());
	m_bk_run_head       = false;
	if (handover != Bookkeep::Mode::Off) {
		g_bk_counters.Add(g_bk_counters.heads);
		if (m_ready_run.Covers(window.Head())) {
			bool ready = true;
			if (handover == Bookkeep::Mode::Verify) {
				// The per-draw protocol: the head's state word says Done for this position.
				g_bk_counters.Add(g_bk_counters.handover_checks);
				ready = window.HeadDone();
				if (!ready) {
					Bookkeep::ReportMismatch(g_bk_counters.handover_mismatches, "handover", "head not done");
					m_ready_run.Reset();
				}
			}
			if (ready) {
				auto& prep_stats = PrepStats::g_commit;
				prep_stats.heads++;
				prep_stats.ready++;
				prep_stats.occupancy += window.Occupancy();
				Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepReady);
				g_bk_counters.Add(g_bk_counters.run_heads);
				m_bk_run_head = true;
				CommitStats::StepMark(CommitStats::Step::GapHead);
				PrefetchAhead();
				return slot;
			}
		}
	}
	const auto stats_start = CommitStats::Now();
	auto&      prep_stats  = PrepStats::g_commit;
	const auto head_tsc    = PrepStats::Tsc();
	prep_stats.heads++;
	prep_stats.occupancy += window.Occupancy();
	// KYTY_HANDOFF_LEAN: a head a worker has finished is seen with a plain acquire load, without
	// the locked compare-exchange that would fail on it anyway.
	const bool head_done = Common::HandoffLean::On() && window.HeadDone();
	if (!head_done && window.TryClaimHead()) {
		// No worker has started it: prepare it here, with the exact clean predicate.
		const CpGaps::Scope gap(CpGaps::Cat::HeadSelf);
		{
			const CodeEpochScope epoch(SlotCodeEpoch(slot));
			PrepareSlot(m_renderer.GetPipelineCache(), slot, true);
		}
		HashForRepeatTrace(slot);
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepSelfPrepared);
		if (stats_start != 0) {
			CommitStats::NoteHead(CommitStats::Now() - stats_start, 0);
		}
		prep_stats.self++;
		prep_stats.self_tsc += PrepStats::Tsc() - head_tsc;
	} else if (head_done || window.HeadDone()) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepReady);
		prep_stats.ready++;
		const auto lead = head_tsc > slot.done_tsc ? head_tsc - slot.done_tsc : 0;
		prep_stats.lead_tsc += lead;
		// Buckets in TSC ticks at ~4 GHz: <2, <10, <50, <200, <1000 us.
		const uint64_t bounds[5] = {8'000, 40'000, 200'000, 800'000, 4'000'000};
		size_t         bucket    = 0;
		while (bucket < 5 && lead >= bounds[bucket]) {
			bucket++;
		}
		prep_stats.lead_hist[bucket]++;
	} else {
		prep_stats.waited++;
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
		// KYTY_SPIN_BUDGET (common/spinGate.h): while the core budget is exceeded this thread spins
		// only briefly, then sleeps on the head's state word until the worker's Complete wakes it
		// (Window::ParkHead; the head wake is enabled at startup when the switch is available).
		using Common::SpinGate::Site;
		const uint64_t gated      = Common::SpinGate::SpinNs(Site::CommitHead, UINT64_MAX);
		const uint64_t park_after = window.HeadWakeEnabled() ? gated : UINT64_MAX;
		const auto     wait_ns0   = NowNs();
		uint64_t       parked_tsc = 0;
		uint64_t       parks      = 0;
		const auto stats = AwaitHead(
		    m_workers->gate, window, m_workers->steal,
		    [this](Slot& claimed, uint64_t seq) {
			    Profiler::ScopedFrameWait steal_time(Profiler::FrameWait::DrawPrepSteal);
			    const WorkerThreadScope   as_worker;
			    m_workers->PrepareClaimed(claimed, seq);
			    m_renderer.GetPipelineCache().NoteProgramPrefetchWait(claimed.prepared.program_compile_ns);
		    },
		    [this, &window, park_after, &parked_tsc, &parks](uint32_t spins, uint64_t spin_start) {
			    CpuRelax();
			    if (park_after != UINT64_MAX && (spins & 15u) == 15u && NowNs() - spin_start >= park_after) {
				    const auto park_tsc = PrepStats::Tsc();
				    if (window.ParkHead()) {
					    parks++;
				    }
				    parked_tsc += PrepStats::Tsc() - park_tsc;
			    }
			    // Workers never dereference guest memory, so they cannot page fault into a
			    // request this thread must service. Only as a last-resort deadlock guard, after
			    // 2 ms of spinning, pending cross-thread commands are serviced here; such a
			    // command then observes the window partially committed, as if it had arrived
			    // just before these draws were parsed.
			    // KYTY_SERVE_IN_WAITS bit 1 (based on chenxiao07/KytyPS5 6801ddc13, 6e19833bc):
			    // the same, every 64th look instead of after 2 ms, so a game thread's readback
			    // does not queue behind the head's preparation.
			    if ((spins & 63u) != 63u || !m_service_commands) {
				    return;
			    }
			    // (The clock only at a guard look, as before.)
			    const uint64_t waited  = (spins & 1023u) == 1023u ? NowNs() - spin_start : 0;
			    const auto     service = HeadWaitServiceAt(spins, waited, ServeInHeadWait(),
			                                           [] { return ServiceCommandsPending(); });
			    if (service == HeadWaitService::Serve) {
				    Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepServedInWait);
				    static uint64_t served = 0; // (the commit thread only)
				    if ((++served & 0xfffu) == 1) {
					    std::printf("ServeInWaits: %" PRIu64 " head waits served service commands\n", served);
					    std::fflush(stdout);
				    }
			    }
			    if (service != HeadWaitService::None) {
				    m_service_commands();
			    }
		    },
		    [] { Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepColdWakes); });
		{
			// The spin share of this wait in TSC ticks: its spin time (steals excluded) at the wait's
			// own TSC rate, minus the time parked.
			const auto wait_tsc = PrepStats::Tsc() - head_tsc;
			const auto wait_ns  = std::max<uint64_t>(NowNs() - wait_ns0, 1);
			const auto spin_tsc = static_cast<uint64_t>(static_cast<double>(stats.spin_ns) *
			                                            static_cast<double>(wait_tsc) /
			                                            static_cast<double>(wait_ns));
			Common::SpinGate::NoteSpin(Site::CommitHead, spin_tsc > parked_tsc ? spin_tsc - parked_tsc : 0);
			if (parks != 0) {
				Common::SpinGate::NotePark(Site::CommitHead, parks);
			}
		}
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
		prep_stats.wait_tsc += PrepStats::Tsc() - head_tsc;
	}
	if (handover != Bookkeep::Mode::Off) {
		// The head is ready: note the finished positions after it (one acquire load each), up to
		// a segment's cap (32 draws).
		constexpr uint32_t ReadyRunCap = 32;
		const auto positions = m_ready_run.Extend(window.Head(), window.Tail(), ReadyRunCap,
		                                          [&window](uint64_t seq) { return window.DoneAt(seq); });
		if (positions != 0) {
			g_bk_counters.Add(g_bk_counters.runs);
			g_bk_counters.Add(g_bk_counters.run_positions, positions);
		}
	}
	CommitStats::StepMark(CommitStats::Step::GapHead);
	PrefetchAhead();
	return slot;
}

void Engine::PrefetchAhead() {
	auto& window = m_workers->window;
	if (const auto mode = g_cert_prefetch.Get(); mode != 0) {
		// KYTY_CERT_PREFETCH, a two-step pipeline: the position after next gets its state word and
		// certificate header lines prefetched; the next one (whose lines the previous commit
		// prefetched), when its preparation is done, gets its range lists prefetched.
		const auto head = window.Head();
		const auto tail = window.Tail();
		if (head + 2u < tail) {
			PrefetchLine(window.StateAddress(head + 2u));
			PrefetchHeader(window.PayloadAt(head + 2u).prepared, mode);
		}
		if (head + 1u < tail && window.DoneAt(head + 1u)) {
			PrefetchCertificate(window.PayloadAt(head + 1u).prepared, mode);
		}
	}
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
	executor.m_run_targets_key     = commit ? slot.run_targets_key : 0;
	executor.m_run_swap_key        = commit ? slot.run_swap_key : 0;
	executor.m_run_texfree_key     = commit ? slot.run_texfree_key : 0;
	executor.m_run_parts           = commit && slot.run_parts_valid ? &slot.run_parts : nullptr;
	executor.m_in_engine_commit    = commit;
	{
		const CpGaps::Scope gap(CpGaps::Cat::Commit);
		CommitStats::BeginDraw();
		draw();
		CommitStats::EndDraw();
	}
	executor.m_in_engine_commit = false;
	executor.m_run_key          = 0;
	executor.m_run_targets_key  = 0;
	executor.m_run_swap_key     = 0;
	executor.m_run_texfree_key  = 0;
	executor.m_run_parts        = nullptr;
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

bool Engine::FenceOverlapEnabled() {
	static const bool enabled = [] {
		const auto* value = EnvValue("KYTY_DRAW_PREP_FENCE_OVERLAP");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

void Engine::NoteDeferredFence(FenceKind fence_kind) {
	Profiler::CountFrameEvent(fence_kind == FenceKind::Dispatch
	                              ? Profiler::FrameEvent::DrawPrepFenceDispatch
	                              : Profiler::FrameEvent::DrawPrepFenceIndirectDraw);
	NoteFence();
}

void Engine::DrainBefore(uint64_t position) {
	EXIT_IF(!GuestGpu::IsGpuThread() || m_workers == nullptr);
	auto& window = m_workers->window;
	if (window.Empty() || window.Head() == position) {
		return;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepDrains);
	g_totals.drains.fetch_add(1, std::memory_order_relaxed);
	while (!window.Empty() && window.Head() != position) {
		CommitHead();
	}
	EXIT_IF(window.Empty());
}

void Engine::CommitOne() {
	EXIT_IF(!GuestGpu::IsGpuThread() || m_workers == nullptr || m_workers->window.Empty());
	CommitHead();
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
// PrepStats (prepStats.h): one line per 10 s, the deltas since the previous line. TSC ticks are
// converted with the TSC rate measured between two lines.
void PrintPrepStats(uint64_t now_ns) {
	static uint64_t last_ns  = 0;
	static uint64_t last_tsc = 0;
	struct W {
		uint64_t slots = 0, busy = 0, spin = 0, park = 0, parks = 0;
	};
	static W                                         last_w[PrepStats::MaxWorkers] {};
	static PrepStats::Commit                         last_c {};
	const auto                                       tsc = PrepStats::Tsc();
	if (last_ns == 0) {
		last_ns  = now_ns;
		last_tsc = tsc;
		last_c   = PrepStats::g_commit;
		std::printf("PrepStats sizes: slot %zu registers %zu prepared %zu plan %zu reads %zu\n",
		            sizeof(Engine::Slot), sizeof(RegisterSnapshot), sizeof(PreparedDraw),
		            sizeof(BindingPlan), sizeof(ReadSet));
		return;
	}
	const double ticks_per_us = static_cast<double>(tsc - last_tsc) / (static_cast<double>(now_ns - last_ns) * 1e-3);
	const double secs         = static_cast<double>(now_ns - last_ns) * 1e-9;
	const auto   ms           = [&](uint64_t t) { return static_cast<double>(t) / ticks_per_us * 1e-3 / secs; };
	const auto&  c            = PrepStats::g_commit;
	const auto   heads        = c.heads - last_c.heads;
	const auto   ready        = c.ready - last_c.ready;
	std::string  workers;
	W            sum;
	for (uint32_t i = 0; i < PrepStats::MaxWorkers; i++) {
		const auto& g = PrepStats::g_workers[i];
		W           v {g.slots.load(std::memory_order_relaxed), g.busy.load(std::memory_order_relaxed),
                       g.spin.load(std::memory_order_relaxed), g.park.load(std::memory_order_relaxed),
                       g.parks.load(std::memory_order_relaxed)};
		W           d {v.slots - last_w[i].slots, v.busy - last_w[i].busy, v.spin - last_w[i].spin,
                       v.park - last_w[i].park, v.parks - last_w[i].parks};
		last_w[i] = v;
		if (d.slots == 0 && d.spin == 0 && d.park == 0) {
			continue;
		}
		sum.slots += d.slots; sum.busy += d.busy; sum.spin += d.spin; sum.park += d.park; sum.parks += d.parks;
		char text[160];
		std::snprintf(text, sizeof(text), " w%u:%.0f/s,%.2fus,b%.0f,s%.0f,p%.0f,n%.0f", i,
		              static_cast<double>(d.slots) / secs,
		              d.slots != 0 ? static_cast<double>(d.busy) / ticks_per_us / static_cast<double>(d.slots) : 0.0,
		              ms(d.busy), ms(d.spin), ms(d.park), static_cast<double>(d.parks) / secs);
		workers += text;
	}
	std::printf("PrepStats %.0fs: tsc %.0f MHz | heads %.0f/s ready %.1f%% waited %.1f%% (%.2f ms/s) self %.1f%% "
	            "(%.2f ms/s) | lead avg %.1f us hist %" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64
	            "/%" PRIu64 " | occupancy %.1f | workers %.0f slots/s %.2f us/slot busy %.0f spin %.0f park %.0f "
	            "ms/s parks %.0f/s |%s\n",
	            secs, ticks_per_us, static_cast<double>(heads) / secs,
	            heads != 0 ? 100.0 * static_cast<double>(ready) / static_cast<double>(heads) : 0.0,
	            heads != 0 ? 100.0 * static_cast<double>(c.waited - last_c.waited) / static_cast<double>(heads) : 0.0,
	            ms(c.wait_tsc - last_c.wait_tsc),
	            heads != 0 ? 100.0 * static_cast<double>(c.self - last_c.self) / static_cast<double>(heads) : 0.0,
	            ms(c.self_tsc - last_c.self_tsc),
	            ready != 0 ? static_cast<double>(c.lead_tsc - last_c.lead_tsc) / ticks_per_us / static_cast<double>(ready) : 0.0,
	            c.lead_hist[0] - last_c.lead_hist[0], c.lead_hist[1] - last_c.lead_hist[1],
	            c.lead_hist[2] - last_c.lead_hist[2], c.lead_hist[3] - last_c.lead_hist[3],
	            c.lead_hist[4] - last_c.lead_hist[4], c.lead_hist[5] - last_c.lead_hist[5],
	            heads != 0 ? static_cast<double>(c.occupancy - last_c.occupancy) / static_cast<double>(heads) : 0.0,
	            static_cast<double>(sum.slots) / secs,
	            sum.slots != 0 ? static_cast<double>(sum.busy) / ticks_per_us / static_cast<double>(sum.slots) : 0.0,
	            ms(sum.busy), ms(sum.spin), ms(sum.park), static_cast<double>(sum.parks) / secs, workers.c_str());
	static PrepStats::CpIdle last_i {};
	const auto&              i = PrepStats::g_cp_idle;
	std::printf("PrepStats cp idle %.0fs: starved %.2f ms/s (%.0f/s) | outside slices: empty %.2f blocked %.2f "
	            "fence %.2f nowork %.2f ms/s\n",
	            secs, ms(i.starved - last_i.starved), static_cast<double>(i.starved_n - last_i.starved_n) / secs,
	            ms(i.empty - last_i.empty), ms(i.blocked - last_i.blocked), ms(i.fence - last_i.fence),
	            ms(i.nowork - last_i.nowork));
	last_i = i;
	static uint64_t last_seq = 0, last_rec = 0;
	const auto      seq_spin = PrepStats::g_spin.sequencer.load(std::memory_order_relaxed);
	const auto      rec_spin = PrepStats::g_spin.recorder.load(std::memory_order_relaxed);
	std::printf("PrepStats spin %.0fs: workers %.0f sequencer %.0f recorder %.0f ms/s\n", secs, ms(sum.spin),
	            ms(seq_spin - last_seq), ms(rec_spin - last_rec));
	last_seq = seq_spin;
	last_rec = rec_spin;
	// KYTY_DRAW_PREP_LEAN: the parts on now, and the digest cache's lookups (workers).
	static uint64_t last_hits = 0, last_misses = 0, last_unc = 0;
	uint64_t        hits = 0, misses = 0, unc = 0;
	for (const auto& g: PrepStats::g_workers) {
		hits += g.digest_hits.load(std::memory_order_relaxed);
		misses += g.digest_misses.load(std::memory_order_relaxed);
		unc += g.digest_uncacheable.load(std::memory_order_relaxed);
	}
	std::printf("PrepStats lean %.0fs: parts 0x%x | digest hits %.0f/s misses %.0f/s uncacheable %.0f/s | verify "
	            "%" PRIu64 " checks %" PRIu64 " mismatches (total) | adapt cap %u of %u lps, busy %.2f cores\n",
	            secs, Lean::Parts(), static_cast<double>(hits - last_hits) / secs,
	            static_cast<double>(misses - last_misses) / secs, static_cast<double>(unc - last_unc) / secs,
	            g_lean_verify_checks.load(std::memory_order_relaxed),
	            g_lean_verify_mismatches.load(std::memory_order_relaxed),
	            g_adapt_cap.load(std::memory_order_relaxed), g_adapt_lps.load(std::memory_order_relaxed),
	            static_cast<double>(g_adapt_busy_milli.load(std::memory_order_relaxed)) / 1000.0);
	last_hits   = hits;
	last_misses = misses;
	last_unc    = unc;
	std::fflush(stdout);
	last_ns  = now_ns;
	last_tsc = tsc;
	last_c   = c;
}

void PrintDrawPrepSummary() {
	static uint64_t last_ns        = 0;
	static uint64_t last_committed = 0;
	static uint64_t last_fallbacks = 0;
	static std::array<uint64_t, static_cast<size_t>(Failure::Mismatch) + 1u> last_reasons {};
	// KYTY_CP_COMMIT=draws: the clock is read every 256th commit (the line is due every 10 s).
	static uint32_t calls = 0;
	// A live change of KYTY_DRAW_PREP_LEAN closes the period at once (same-boot A/B slots).
	const bool forced = g_stats_due.load(std::memory_order_relaxed) &&
	                    g_stats_due.exchange(false, std::memory_order_relaxed);
	if (!forced && CpCommit::Enabled(CpCommit::Part::Draws) && last_ns != 0 && (++calls & 255u) != 0) {
		return;
	}
	const auto now = NowNs();
	if (last_ns == 0) {
		last_ns = now;
		return;
	}
	// KYTY_DRAW_PREP_STATS_PERIOD_S (default 10): the lines' period.
	static const uint64_t period_ns = uint64_t {EnvUnsigned("KYTY_DRAW_PREP_STATS_PERIOD_S", 10, 1, 1000000)} *
	                                  1'000'000'000ull;
	if (!forced && now - last_ns < period_ns) {
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
	if (VerifyMode() != 0) {
		dispatches += " | draw verify: " +
		              std::to_string(g_draw_verify_checks.load(std::memory_order_relaxed)) +
		              " checks, " +
		              std::to_string(g_draw_verify_mismatches.load(std::memory_order_relaxed)) +
		              " mismatches (total)";
	}
	if (BindingsVerifyMode() != 0) {
		auto& totals = GetBindingTotals();
		dispatches += " | binding verify: " +
		              std::to_string(totals.verify_checks.load(std::memory_order_relaxed)) + " checks, " +
		              std::to_string(totals.verify_mismatches.load(std::memory_order_relaxed)) +
		              " mismatches (total)";
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
	if (const auto mode = SpecSegmentsMode(); mode != Segment::Mode::Off) {
		// KYTY_SPEC_SEGMENTS: totals since the start (segment.h FormatLine).
		std::printf("%s\n", Segment::FormatLine(g_segment_counters, mode == Segment::Mode::Verify ? "verify" : "on").c_str());
	}
	{
		// KYTY_BK_* (bookkeep.h): totals since the start, while any part is on or has counted.
		const auto handover = BkHandoverMode();
		const auto adopt    = BkAdoptMode();
		const auto record   = BkRecordMode();
		const auto lean     = BkLeanMode();
		const auto& c       = g_bk_counters;
		if (handover != Bookkeep::Mode::Off || adopt != Bookkeep::Mode::Off || record != Bookkeep::Mode::Off ||
		    lean != Bookkeep::Mode::Off || c.heads.load(std::memory_order_relaxed) != 0 ||
		    c.adopted.load(std::memory_order_relaxed) != 0 || c.record_checks.load(std::memory_order_relaxed) != 0 ||
		    c.lean_rendering.load(std::memory_order_relaxed) != 0) {
			std::printf("%s\n", Bookkeep::FormatLine(c, handover, adopt, record, lean).c_str());
		}
	}
	if (BindingParts() != 0) {
		// KYTY_DRAW_PREP_BINDINGS: how many committed plans carried a pipeline the commit could use.
		auto&           totals = GetBindingTotals();
		static uint64_t last_plans = 0, last_used = 0, last_pipes = 0, last_abstain = 0,
		                last_busy = 0, last_fallback = 0;
		const uint64_t  plans    = totals.plans.load(std::memory_order_relaxed);
		const uint64_t  used     = totals.used.load(std::memory_order_relaxed);
		const uint64_t  pipes    = totals.pipelines_used.load(std::memory_order_relaxed);
		const uint64_t  abstain  = totals.pipeline_abstains.load(std::memory_order_relaxed);
		const uint64_t  busy     = totals.pipeline_busy.load(std::memory_order_relaxed);
		const uint64_t  fallback = totals.pipeline_fallbacks.load(std::memory_order_relaxed);
		std::printf("DrawPrep plans %.0fs: %" PRIu64 " plans committed, %" PRIu64
		            " used; pipelines: %" PRIu64 " taken, %" PRIu64 " abstained, %" PRIu64
		            " busy, %" PRIu64 " refused at commit\n",
		            static_cast<double>(now - last_ns) * 1e-9, plans - last_plans, used - last_used,
		            pipes - last_pipes, abstain - last_abstain, busy - last_busy,
		            fallback - last_fallback);
		last_plans = plans; last_used = used; last_pipes = pipes; last_abstain = abstain;
		last_busy = busy; last_fallback = fallback;
		static std::array<uint64_t, 6> last_reasons {};
		std::string reasons;
		static const char* const reason_names[6] = {"topology", "index", "restart",
		                                            "targets",  "key",   "absent"};
		for (size_t i = 0; i < last_reasons.size(); i++) {
			const auto value = totals.pipeline_abstain_reasons[i].load(std::memory_order_relaxed);
			reasons += " " + std::string(reason_names[i]) + "=" + std::to_string(value - last_reasons[i]);
			last_reasons[i] = value;
		}
		std::printf("DrawPrep plan abstains %.0fs:%s\n", static_cast<double>(now - last_ns) * 1e-9,
		            reasons.c_str());
	}
	PrintPrepStats(now);
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
	executor.m_run_targets_key  = slot.run_targets_key;
	executor.m_run_swap_key     = slot.run_swap_key;
	executor.m_run_texfree_key  = slot.run_texfree_key;
	executor.m_run_parts        = slot.run_parts_valid ? &slot.run_parts : nullptr;
	executor.m_in_engine_commit = true;
	CommitStats::StepMark(CommitStats::Step::BindRegs);
	CommitStats::StepMark(CommitStats::Step::Calib);
	// KYTY_SPEC_SEGMENTS: the slot's encoded draw, for the continuation kernel (segment.h).
	const bool segments = slot.kind == DrawKind::Index && SpecSegmentsMode() != Segment::Mode::Off;
	executor.m_segment           = segments ? &slot.segment : nullptr;
	executor.m_segment_validated = false;
	if (slot.kind == DrawKind::Index) {
		executor.DrawIndex(slot.submit_id, scheduler.Current(), slot.index_args);
	} else {
		executor.DrawAuto(slot.submit_id, scheduler.Current(), slot.auto_args);
	}
	if (m_bk_run_head && executor.m_bk_kernel_taken) {
		g_bk_counters.Add(g_bk_counters.run_kernel);
	}
	executor.m_bk_kernel_taken = false;
	if (segments) {
		static Segment::SegmentTracker tracker; // (the commit thread only)
		if (executor.m_segment_validated) {
			if (tracker.Note(executor.m_run.key ^ executor.m_run.rendering_serial)) {
				g_segment_counters.segments.fetch_add(1, std::memory_order_relaxed);
			}
		} else {
			tracker.Break();
		}
	}
	executor.m_segment           = nullptr;
	executor.m_segment_validated = false;
	executor.m_segment_splice    = nullptr;
	executor.m_in_engine_commit = false;
	executor.m_run_key          = 0;
	executor.m_run_targets_key  = 0;
	executor.m_run_swap_key     = 0;
	executor.m_run_texfree_key  = 0;
	executor.m_run_parts        = nullptr;
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
	if (m_after_commit != nullptr) {
		m_after_commit(m_after_commit_context);
	}
	CommitStats::EndDraw();
	// The 10 s lines. KYTY_BK_HANDOVER: their four period checks every 256th commit (or at once when
	// a live change closes the A/B slot's period), not after every draw.
	static uint32_t printer_calls = 0; // (the commit thread only)
	if (static_cast<Bookkeep::Mode>(g_bk_handover.Get()) == Bookkeep::Mode::Off ||
	    (++printer_calls & 255u) == 0 || g_stats_due.load(std::memory_order_relaxed)) {
		PrintDrawPrepSummary();
		DrawRun::PrintSummary();
		TexelImageLookup::PrintSummary();
		XFrameReuse::PrintSummary();
		ImgSide::PrintSummary();
	}
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
