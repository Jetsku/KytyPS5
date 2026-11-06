#pragma once

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <optional>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct ReadLaneStats {
	uint32_t rewritten_reads = 0;
	// Reads of lanes 32-63 resolved under unreachable_high_lanes.
	uint32_t high_lane_reads = 0;
};

// unreachable_high_lanes (KYTY_WAVE64_HIGH_LANES): the host runs this wave64 program on 32-lane
// subgroups with no link between them (VS/PS), so lanes 32-63 of a VGPR exist only as V_WRITELANE
// values. A constant V_READLANE of such a lane then takes the value its write chain gives where
// one exists, and 0 (defined) on paths that never wrote the lane, instead of a host shuffle of a
// lane the subgroup lacks.
[[nodiscard]] ReadLaneStats EliminateReadLane(Program& program, uint32_t wave_size,
                                              bool unreachable_high_lanes = false);

// A wave reduction as RDNA compilers write it: with every lane enabled, four DPP row_shr steps
// (1, 2, 4, 8) of one operation scan each row of 16 lanes, and v_readlane takes a row's last lane
// (lanes 16r+15); or V_PERMLANEX16 first adds the other row's last lane, and v_readlane takes the
// last lane of a row pair (lanes 32h+31). A host subgroup can lack lanes of the guest wave (a
// partly filled fragment warp); then the scan cannot read them and v_readlane may name one. On the
// guest such a lane holds the operation's identity (the shader selects it for lanes outside its
// own EXEC), so the reduction over the lanes the host has is the guest's value.
// (Ported from Senaxx's wolverine branch; KYTY_LANE_REDUCTIONS.)
struct LaneReduction {
	ValueOpcode operation = ValueOpcode::Void;
	Value       source; // the scanned value, before the first step
	uint32_t    first_lane = 0;
	uint32_t    lanes      = 0; // 16 or 32
};

// The reduction a ReadLane reads, if it reads the last lane of such a scan.
[[nodiscard]] std::optional<LaneReduction> MatchLaneReduction(const Inst& read_lane,
                                                              uint32_t    wave_size);
[[nodiscard]] uint32_t ReductionIdentity(ValueOpcode operation);

} // namespace Libs::Graphics::ShaderRecompiler::IR
