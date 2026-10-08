#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <span>

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
	static ResourcePlan::EvaluationContext& AcquireContext(EvaluationScratch& scratch);
	static float Float32(uint64_t bits);
	uint32_t MemoIndex(const Inst& inst) const;
	bool EvaluateWide(Value value, uint64_t& result);
	bool EvaluateRoot(Value value, const ResourcePlan::EvaluationOperand& root, uint32_t& result);
	const Inst* FlatReadInstruction(uint32_t slot, uint32_t& memo_index) const;
	bool EvaluateOperand(const ResourcePlan::EvaluationOperand& operand, uint64_t& result);
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
	uint64_t                        m_shared_clean_hits = 0;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_ */
