#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

class Value;

using SrtMemoryReader = bool (*)(void* userdata, uint64_t address, std::span<uint32_t> values);
using SrtReadObserver = void (*)(void* userdata, uint64_t address,
                                std::span<const uint32_t> values, bool success);
using SrtMappedRange  = bool (*)(uint64_t address, uint64_t size);

struct SrtRuntime {
	std::span<const uint32_t> user_data;
	uint64_t                  shader_base                = 0;
	SrtMemoryReader           read_memory                = nullptr;
	void*                     userdata                   = nullptr;
	SrtMemoryReader           read_specialization_memory = nullptr;
	// Optional exact backing probe. False must leave output untouched and must not
	// read, synchronize, fault, record missing ranges or otherwise mutate guest state.
	// Success must include fresh GPU-dirty, pending-publication and mapping checks.
	SrtMemoryReader           try_read_clean_backing = nullptr;
	// Internal capture override selected by CleanRuntime; null reuses the probe above.
	SrtMemoryReader           try_read_specialization_backing = nullptr;
	// Observes actual returned memory bytes, independently of reader/capture userdata.
	// A failed semantic read is reported; a failed speculative backing probe is not.
	SrtReadObserver           observe_read = nullptr;
	void*                     observer_userdata = nullptr;
	// Share successful strict evaluations with an ordinary evaluator in this same
	// materialization only. Never import ordinary results into a strict evaluator.
	bool share_clean_values = false;
	// Optional: whether [address, address + size) is guest memory the GPU can read. The in-place
	// read (read_memory null, no successful probe) reads 0 outside it instead of touching the
	// address. Without it, only addresses that are never mapped read 0 (the first 64 KiB and
	// non-canonical addresses), where the in-place read would fault with nothing to resolve it.
	SrtMappedRange is_guest_mapped = nullptr;
	// Dispatch workgroup counts (compute). No descriptor this planner tracks reads them yet.
	std::span<const uint32_t> workgroup_counts;
};

// Addresses no guest or host mapping can contain: the first 64 KiB (never mapped on Windows, Linux
// or the PS5) and non-canonical ones. A read of one faults with nothing to resolve it.
constexpr bool NeverMappedAddress(uint64_t address, uint64_t size) {
	constexpr uint64_t NullRegionEnd  = 0x10000;
	constexpr uint64_t CanonicalLimit = uint64_t {1} << 47u;
	return address < NullRegionEnd || size > CanonicalLimit || address > CanonicalLimit - size;
}

inline void ObserveSrtRead(const SrtRuntime& runtime, uint64_t address,
                           std::span<const uint32_t> values, bool success) {
	if (runtime.observe_read != nullptr) {
		runtime.observe_read(runtime.observer_userdata, address, values, success);
	}
}

enum class RuntimeValueType { Any, Integer };

// Collects reachable ReadConst values. Immediate offsets receive compact flat-buffer slots;
// dynamic offsets remain explicit and are never assigned a fake slot. With variant_reads
// (KYTY_SRT_VARIANT_READS), a read whose address is not a valid runtime value (a loop-carried
// pointer, data the GPU produces) is a runtime read as well: a flat slot is evaluated before the
// dispatch, when that address does not exist yet.
void BuildSrtPlan(Program& program, bool variant_reads = false);
// Compile bounded adjacent flat-read runs after cloning and clean-slot discovery.
void BuildSrtReadRuns(ResourcePlan& program);
// Opt-in immutable operand decoding; never evaluates or reads guest values.
void BuildSrtEvaluationRecipes(ResourcePlan& program);
// Opt-in linear pure regions; guest reads and control flow stay evaluator boundaries.
void BuildSrtArithmeticTapes(ResourcePlan& program);
// Assigns every remaining memo slot of an extracted plan. Evaluation of a sealed plan
// never writes to it, so any number of threads may walk it with their own scratch.
void SealEvaluationIndices(ResourcePlan& program);
// This thread's scratch, used by callers that do not supply their own.
EvaluationScratch& ThreadEvaluationScratch();
bool ValidateRuntimeValue(const ResourcePlan& program, Value value,
                          RuntimeValueType type = RuntimeValueType::Any);
// Whether a flat SRT read's address depends on a phi ResolveInvariantPhi cannot reduce (a
// loop-carried pointer such as a BVH traversal's instance record): no evaluation before the
// dispatch can produce it, so the plan never materializes. `pc` receives the first such read's
// guest pc. Walks the plan's IR; meant for a failed materialization, not for every one.
bool FindVariantFlatRead(const ResourcePlan& program, uint32_t& pc);
// Uses the strict reader for values that affect shader specialization.
SrtRuntime CleanRuntime(SrtRuntime runtime);

class SrtWalker;
struct SrtTrace;
struct SrtTraceCall;

// KYTY_SRT_TRACE (live, 0 | 1 | verify | exit; default 1): replay traces of a plan's resource
// refresh. A refresh (MaterializeResources) evaluates the plan's SRT, descriptor and condition
// values through two memoized, recursive walkers (clean and ordinary). A trace records one successful refresh as
// a flat list of the evaluations it performed, in order, each with the operands it consumed
// (earlier entries or immediates), plus the requests the refresh made and the entries that answer
// each. A request is a whole walker operation (the flat buffer refresh, one descriptor, the active
// sources) or one value. A later refresh of the same plan on the same thread replays the list in a
// loop: no memo, no recursion, no recipe or tape dispatch. The values themselves (user data, guest
// words) may differ from the recording: the replay recomputes every entry with the walker's own
// operations (EvaluateInstWithOperands for pure instructions, ComputeRawAddress and ReadRawWord /
// the clean probe for guest reads).
//
// Every data-dependent decision of the recording is a guard: every entry must succeed (only
// refreshes without a failed evaluation are recorded), a select must take the recorded arm, every
// condition of the active-source walk must have the recorded truth, and the refresh must make the
// same requests in the same order. A flat read run is replayed as its words, which must lie in the
// buffer; how the walk batches them (by page ends) does not matter. A guard that does not hold
// abandons the trace for the rest of the refresh: the walkers' memo was never touched, so
// evaluation just continues through them from scratch (reads the replay already made are made
// again, which only adds identical ranges to a read certificate). A plan keeps up to eight traces
// (variants, most recently served first): when the first request does not match one, the next is
// tried; a refresh that matches none, or a variant that fails later, makes the next refresh record.
// A plan whose traces keep failing stops recording for a while.
//
// Traces are per thread (nothing shared between the draw-prep workers), keyed by the plan's
// address and sealing identity and by the runtime's reader configuration. The trace sits above the
// recipes, tapes and read runs (KYTY_SRT_COMPILED_RECIPES / _ARITHMETIC_TAPES / _READ_RUNS), which
// remain the evaluator a recording observes (each recipe node is one entry; tapes are bypassed while
// recording, as they compute the same node values; a flat read run is one entry plus its members)
// and the path of every refresh not served; a replay uses none of them.
// verify: every refresh that used a trace is evaluated again without it and compared field by
// field; differences are counted and logged (exit: stops at the first). KYTY_SRT_TRACE_STATS=1
// (or verify) prints how refreshes went with the abandon reasons.
class SrtTraceSession {
public:
	// `allowed`: this refresh may use traces (no captured specialization reads).
	SrtTraceSession(const ResourcePlan& program, SrtWalker& clean, SrtWalker& walker,
	                const SrtRuntime& runtime, bool allowed);
	~SrtTraceSession();
	SrtTraceSession(const SrtTraceSession&)            = delete;
	SrtTraceSession& operator=(const SrtTraceSession&) = delete;

	// The refresh succeeded: a recording becomes the plan's trace.
	void Succeeded() { m_succeeded = true; }
	// At least one request was answered from the trace.
	[[nodiscard]] bool UsedTrace() const { return m_served != 0; }
	// Turns traces off on this thread (verification re-runs).
	static bool& Suppressed();
	// 0 off, 1 on, 2 verify, 3 verify and stop at the first difference.
	static int Mode();
	// verify: one refresh compared; `field` names the first difference (nullptr: equal).
	static void NoteVerified(const ResourcePlan& program, const char* field);
	// Refreshes served from a trace from start to end (counted with verify or
	// KYTY_SRT_TRACE_STATS=1 only), and the statistics line.
	static uint64_t ServedRefreshes();
	static void     PrintStats();

private:
	friend class SrtWalker;
	enum class State : uint8_t { Off, Record, Serve, Passthrough };
	// Requests: one value (Wide, Operand), or a whole walker operation (FlatBuffer: every flat
	// slot; Descriptor: one descriptor's dwords; Active: the active sources). Inside a whole
	// operation (a coarse request) the values it evaluates are its results, not requests.
	enum class CallKind : uint8_t { None, Wide, Operand, FlatBuffer, Descriptor, Active };
	struct Event {
		int32_t  ref   = 0; // >= 0: an entry; < 0: immediate -1 - ref
		uint64_t value = 0;
	};
	struct Frame {
		Event    events[6] {};
		uint32_t count = 0;
	};
	struct CallId {
		CallKind    kind   = CallKind::None;
		uint8_t     walker = 0;
		Value       value;          // Wide
		const void* id     = nullptr; // Operand
		uint32_t    a      = 0;       // Descriptor: the source
	};

	// Entry points (SrtWalker's wrappers).
	bool Wide(SrtWalker& walker, Value value, uint64_t& result);
	bool Operand(SrtWalker& walker, const ResourcePlan::EvaluationOperand& operand,
	             uint64_t& result);
	bool FlatBuffer(SrtWalker& walker, std::vector<uint32_t>& flat);
	bool Descriptor(SrtWalker& walker, uint32_t source, DescriptorValue& result);
	std::span<const uint8_t> Active(SrtWalker& walker);
	// Recording hooks.
	void PushImmediate(uint64_t bits);
	void PushMemo(const SrtWalker& walker, uint32_t index);
	void Open();
	void CloseInst(const SrtWalker& walker, const Inst& inst, uint32_t index, bool ok,
	               uint64_t value);
	void CloseRecipe(const SrtWalker& walker, const ResourcePlan::EvaluationRecipe& recipe,
	                 uint32_t index, bool ok, uint64_t value);
	void BeginFlat();
	void CloseFlat(const SrtWalker& walker, const Inst& inst, uint32_t first, uint32_t end,
	               uint32_t words, bool probed, bool stop_data, uint32_t consumed);
	void Abort(const char* reason);

	void Push(const Event& event);
	void BeginCoarse(const CallId& id);
	bool EndCoarse(uint32_t expected);
	const SrtTraceCall* NextCall(const CallId& id);
	[[nodiscard]] uint64_t Result(int32_t ref) const;
	bool RunOps(uint32_t end);
	bool SelectVariant(uint32_t from);
	bool Retry(const char* reason);
	void Abandon(const char* reason);
	void Detach();
	static std::vector<Frame>& Frames();

	const ResourcePlan& m_program;
	SrtWalker*          m_walkers[2];
	void*               m_slot      = nullptr; // the thread's cache entry for the plan
	State               m_state     = State::Off;
	bool                m_succeeded = false;
	uint32_t            m_served    = 0;
	// Recording.
	SrtTrace*           m_recording = nullptr;
	CallId              m_pending;
	bool                m_coarse = false; // inside a coarse request
	const char*         m_abort  = nullptr;
	// Serving.
	const SrtRuntime*   m_runtime = nullptr;
	const SrtTrace*     m_trace   = nullptr;
	uint32_t            m_variant = 0;
	const char*         m_fail    = nullptr;
	uint64_t*           m_values  = nullptr;
	uint32_t*           m_stash   = nullptr;
	uint32_t            m_cursor  = 0;
	uint32_t            m_next_op = 0;
};

// One memoized evaluation session shared by the entire shader resource refresh.
class SrtWalker {
public:
	// Uses the clean evaluator's scratch, or this thread's scratch without one.
	SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
	          std::span<const uint8_t> clean_flat_slots = {}, SrtWalker* clean_evaluator = nullptr,
	          Value active_mask = {});
	// Walkers sharing a scratch must be destroyed in reverse construction order.
	SrtWalker(const ResourcePlan& program, EvaluationScratch& scratch, const SrtRuntime& runtime,
	          std::span<const uint8_t> clean_flat_slots = {}, SrtWalker* clean_evaluator = nullptr,
	          Value active_mask = {});
	~SrtWalker();
	SrtWalker(const SrtWalker&)            = delete;
	SrtWalker& operator=(const SrtWalker&) = delete;

	bool Evaluate(Value value, uint32_t& result);
	bool EvaluateDescriptor(uint32_t source, DescriptorValue& result);
	// An empty span means that all sources are active.
	std::span<const uint8_t> FindActiveSources();
	bool RefreshFlatBuffer(std::vector<uint32_t>& flat);

private:
	// KYTY_SRT_TRACE: the public operations above are served or recorded whole by a session.
	bool EvaluateDescriptorImpl(uint32_t source, DescriptorValue& result);
	std::span<const uint8_t> FindActiveSourcesImpl();
	bool RefreshFlatBufferImpl(std::vector<uint32_t>& flat);

private:
	static ResourcePlan::EvaluationContext& AcquireContext(EvaluationScratch& scratch);
	static float Float32(uint64_t bits);
	uint32_t MemoIndex(const Inst& inst) const;
	friend class SrtTraceSession;
	bool EvaluateWide(Value value, uint64_t& result) {
		if (m_trace != nullptr) [[unlikely]] {
			return m_trace->Wide(*this, value, result);
		}
		return EvaluateWideImpl(value, result);
	}
	bool EvaluateWideImpl(Value value, uint64_t& result);
	bool EvaluateRoot(Value value, const ResourcePlan::EvaluationOperand& root, uint32_t& result);
	const Inst* FlatReadInstruction(uint32_t slot, uint32_t& memo_index) const;
	bool EvaluateOperand(const ResourcePlan::EvaluationOperand& operand, uint64_t& result) {
		if (m_trace != nullptr) [[unlikely]] {
			return m_trace->Operand(*this, operand, result);
		}
		return EvaluateOperandImpl(operand, result);
	}
	bool EvaluateOperandImpl(const ResourcePlan::EvaluationOperand& operand, uint64_t& result);
	bool EvaluateRecipeNode(uint32_t index, uint64_t& result);
	bool BorrowCleanValue(uint32_t index, uint64_t& result);
	bool EvaluateRecipe(const ResourcePlan::EvaluationRecipe& recipe, uint64_t& result);
	bool EvaluateArithmeticTape(const ResourcePlan::ArithmeticTape& tape, uint64_t& result);
	bool ResolveRecipeReadAddress(const ResourcePlan::EvaluationRecipe& recipe,
	                              uint64_t& address, uint64_t& available);
	bool Arg(const Inst& inst, size_t index, uint64_t& result);
	bool EvaluatePhi(const Inst& inst, uint64_t& result);
	bool EvaluateExtract(const Inst& inst, uint64_t& result);
	bool ResolveRawReadAddress(const Inst& inst, uint64_t& address, uint64_t& available);
	bool ReadRawWord(uint64_t address, uint64_t& result, bool allow_probe = true);
	[[nodiscard]] bool InPlaceReadable(uint64_t address);
	void NoteUnmappedRead(uint64_t address) const;
	bool EvaluateRawRead(const Inst& inst, uint64_t& result);
	bool EvaluateFlatRun(uint32_t first, uint32_t end, std::vector<uint32_t>& flat,
	                     uint32_t& consumed);
	bool EvaluateInst(const Inst& inst, uint64_t& result,
	                  const ResourcePlan::EvaluationRecipe* recipe = nullptr);
	template <typename ReadOperand>
	bool EvaluateInstWithOperands(const Inst& inst, uint64_t& result, ReadOperand&& arg);

	const ResourcePlan&              m_program;
	EvaluationScratch&               m_scratch;
	SrtRuntime                      m_runtime;
	std::span<const uint8_t>         m_clean_flat_slots;
	SrtWalker*                      m_clean_evaluator = nullptr;
	Value                           m_active_mask;
	ResourcePlan::EvaluationContext& m_context;
	// The last 4 KiB page is_guest_mapped confirmed: mappings do not change during one walk
	// (unmaps run on the GPU thread, which is walking), and flat reads cluster in a few tables.
	uint64_t                        m_mapped_page = UINT64_MAX;
	bool                            m_count_recipes = false;
	uint64_t                        m_compiled_nodes = 0;
	uint64_t                        m_fallback_nodes = 0;
	uint64_t                        m_memo_hits = 0;
	uint64_t                        m_tape_executions = 0;
	uint64_t                        m_tape_operations = 0;
	uint64_t                        m_tape_boundary_calls = 0;
	bool                            m_share_clean_values = false;
	// KYTY_SRT_TRACE: the session serving or recording this walker's requests (null: none), and
	// m_trace_record when it records (the evaluator's hooks report to it).
	SrtTraceSession*                m_trace        = nullptr;
	SrtTraceSession*                m_trace_record = nullptr;
	uint8_t                         m_trace_id     = 0; // 0 the clean walker, 1 the ordinary one
	uint64_t                        m_shared_clean_hits = 0;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_ */
