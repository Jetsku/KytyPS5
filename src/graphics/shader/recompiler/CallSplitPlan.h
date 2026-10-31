#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_CALLSPLITPLAN_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_CALLSPLITPLAN_H_

#include "graphics/shader/recompiler/CallSplit.h"
#include "graphics/shader/recompiler/ShaderCalls.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// The planner of call-split programs (CallSplit.h): the shape of a program's call loops, the
// codes of its split programs, their register layouts (from planning translations on a
// background thread) and, per dispatch, the record -> callee map of each site's table.
namespace Libs::Graphics::ShaderRecompiler::CallSplit {

// One call loop: guest pcs [a, b), the call at call_pc, the waterfall's record-index VGPR.
struct Site {
	uint32_t  a           = 0;
	uint32_t  b           = 0;
	uint32_t  call_pc     = 0;
	uint32_t  source_reg  = 0;
	uint32_t  return_reg  = 0;
	uint32_t  record_vgpr = 0;
	CallTable table;
};

struct Shape {
	std::vector<Site>       sites;     // by pc
	std::vector<uint32_t>   canonical; // S_SWAPPC_B64 -> S_GETPC_B64, cut after the last S_ENDPGM
	std::array<uint64_t, 2> digest {};
	uint32_t                vgprs = 0;
};

// The program's call loops; false with `failure` when it does not have the supported shape
// (every call alone in a waterfall loop no other loop encloses, one wave per workgroup).
bool AnalyzeShape(std::span<const uint32_t> code, const CompileOptions& options, Shape& shape,
                  std::string& failure);

enum class Kind { Resume, Callee, Probe };
// The code of a split program. Resume with site 0 is R_0; Callee needs the original code, its
// guest base and the callee; liveness: a planning translation.
bool BuildCode(const Shape& shape, const Header& layouts, Kind kind, uint32_t site, bool liveness,
               std::span<const uint32_t> original, uint64_t base, uint64_t callee,
               const Decoder::ShaderCodeReader& read, std::vector<uint32_t>& code,
               std::string& failure);

// A split program's code and its digest (the program key's expansion id).
struct Code {
	std::vector<uint32_t> words;
	uint64_t              digest = 0;
};

// What a dispatch runs: R_0, R_ak per site, and per site the callee programs of the table's
// targets (null: not usable or not planned yet) and each record's target slot (NoSlot: none).
struct Run {
	static constexpr uint32_t NoSlot = 0xffffffffu;
	struct SiteRun {
		std::shared_ptr<const Code>              resume;
		std::vector<std::shared_ptr<const Code>> callees;
		std::vector<bool>                        pending; // per callee: not planned yet (null code)
		std::vector<uint64_t>                    targets; // per callee: its guest address (status line)
		std::vector<uint32_t>                    record_slots;
		uint32_t                                 exec_lo_slot  = 0; // uniform dwords
		uint32_t                                 exec_hi_slot  = 0;
		uint32_t                                 record_dword  = 0; // the record VGPR's lane 0
	};
	uint64_t                    kernel = 0;
	uint64_t                    hash   = 0; // the guest shader (status line)
	std::shared_ptr<const Code> start;
	std::vector<SiteRun>        sites;
	uint32_t                    wave_dwords = 0;
	// Diagnostics (KYTY_SHADER_CALLS_SPLIT_DUMP): the site layouts and targets.
	std::vector<SiteLayout>            layouts;
	std::vector<std::vector<uint64_t>> targets;
};

class Planner {
public:
	Planner();
	~Planner();
	Planner(const Planner&)            = delete;
	Planner& operator=(const Planner&) = delete;

	enum class Result { NotSplit, Pending, Ready };
	// A dispatch of `code` (guest address `base`): NotSplit when the program has no supported call
	// loops (or splitting failed), Pending while its plan is built, Ready with `run`.
	Result Plan(std::span<const uint32_t> code, uint64_t base, const CompileOptions& options,
	            const ShaderComputeInputInfo& info, const Decoder::ShaderCodeReader& read, Run& run);

	// What the planner holds right now (the RT status line): split kernels still planning, ready
	// or failed, and over the ready ones' sites the callee targets whose code is built, still being
	// planned, or unusable (their lanes resume without the call).
	struct Status {
		uint32_t planning = 0, ready = 0, failed = 0;
		uint32_t targets = 0, built = 0, pending = 0, unusable = 0;
		uint32_t jobs = 0; // planning jobs queued
	};
	Status GetStatus();

	static Planner& Instance();
	static bool     Enabled();
	// Tests: KYTY_SHADER_CALLS_SPLIT as given (-1: the environment).
	static void SetEnabledOverride(int enabled);

private:
	struct TargetState {
		std::shared_ptr<const Code> code; // null: pending or unusable
		bool                        unusable = false;
	};
	struct SitePlan {
		std::vector<uint64_t>                  targets; // slot order
		std::unordered_map<uint64_t, uint32_t> slot;
		std::vector<TargetState>               state;
		std::shared_ptr<const Code>            resume;
		uint64_t                               table_digest = 0;
		std::vector<uint32_t>                  record_slots;
		uint32_t                               scanned_at   = 0;
		CallTableView                          view {};
	};
	struct Kernel {
		enum class State { Planning, Ready, Failed } state = State::Planning;
		Shape                       shape;
		Header                      layouts; // final layouts (entry_site 0, no flags)
		std::vector<uint32_t>       original;
		uint64_t                    base = 0;
		std::vector<uint32_t>       user_data;
		ShaderComputeInputInfo      info {};
		uint64_t                    hash = 0;
		uint32_t                    user_data_base = 0;
		std::shared_ptr<const Code> start;
		std::vector<SitePlan>       sites;
		uint32_t                    dispatches = 0;
		std::string                 failure;
	};
	struct Job {
		uint64_t              kernel = 0;
		uint32_t              site   = 0; // 0: the whole plan
		uint64_t              target = 0;
	};

	void Work();
	void PlanKernel(uint64_t id);
	void PlanTarget(uint64_t id, uint32_t site, uint64_t target);
	bool Translate(const Kernel& kernel, std::span<const uint32_t> code, Report& report);

	std::mutex                          m_mutex;
	std::condition_variable             m_wake;
	std::deque<Job>                     m_jobs;
	std::unordered_map<uint64_t, Kernel> m_kernels;
	std::vector<std::thread>            m_threads;
	bool                                m_stop = false;
	Decoder::ShaderCodeReader           m_read;
	std::vector<uint32_t>               m_table_words;
};

} // namespace Libs::Graphics::ShaderRecompiler::CallSplit

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_CALLSPLITPLAN_H_ */
