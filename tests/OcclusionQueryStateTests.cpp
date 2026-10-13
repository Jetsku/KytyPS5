// OcclusionQueryState (occlusionQueryState.h): the occlusion counter's query bookkeeping, driven
// through the sequences that made OcclusionCounter::Prepare exit with
// "condition (m_active || m_pending >= QueryCapacity)" (rocket_hourglass, accurate occlusion on).
#include "graphics/host_gpu/renderer/occlusionQueryState.h"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

using Libs::Graphics::OcclusionQueryState;

static void Require(bool value, const char* what) {
	if (!value) {
		std::fprintf(stderr, "OcclusionQueryStateTests: failed: %s\n", what);
		std::exit(1);
	}
}

namespace {
constexpr uint32_t Capacity = 1024;

// A model of OcclusionCounter with its command buffer: queries recorded per index, the rendering
// instance, and the pending batched barriers whose flush ends the instance (FlushBarriers()).
struct Model {
	OcclusionQueryState   state {Capacity};
	bool                  rendering        = false;
	bool                  pending_barriers = false;
	std::vector<int>      begun = std::vector<int>(Capacity, 0);
	std::vector<int>      ended = std::vector<int>(Capacity, 0);
	uint32_t              reductions = 0;
	uint32_t              stale_ends = 0;
	uint32_t              dropped    = 0;
	std::function<void()> on_end_record; // runs while an end is recorded (a barrier flush)

	void Reduce() {
		for (uint32_t i = 0; i < state.Pending(); ++i) {
			Require(begun[i] == 1 && ended[i] == 1, "every reduced query was begun and ended once");
			begun[i] = ended[i] = 0;
		}
		state.Reduced();
		++reductions;
	}
	void End() {
		const auto query = state.End();
		if (query == OcclusionQueryState::NoQuery) return;
		if (on_end_record) on_end_record();
		Require(begun[query] == 1 && ended[query] == 0, "a query ends once, after its begin");
		ended[query] = 1;
	}
	void Accumulate() {
		if (state.Full()) Reduce();
	}
	void EndRendering() {
		if (!rendering) return;
		End();
		rendering = false;
		Accumulate();
	}
	void FlushBarriers() {
		if (!pending_barriers) return;
		if (rendering) EndRendering();
		pending_barriers = false;
	}
	// OcclusionCounter::Prepare with the recovery: never exits.
	bool Prepare() {
		const auto recovery = state.BeforePrepare();
		if (recovery.stale_query != OcclusionQueryState::NoQuery) {
			Require(begun[recovery.stale_query] == 1, "the stale query was begun");
			ended[recovery.stale_query] = 1;
			++stale_ends;
		}
		if (recovery.reduce) Reduce();
		if (!state.CanBegin()) {
			++dropped;
			return false;
		}
		return true;
	}
	void BeginRendering() {
		if (rendering) EndRendering();
		const bool counted = Prepare();
		rendering          = true;
		if (counted) begun[state.Begin()] = 1;
	}
	// A split dump (KYTY_OCCLUSION_SPLIT): ends the instance's query, keeps the instance, begins
	// the next query in it. `sink_flush` reproduces the old recording through Sink(), which
	// recorded the pending batched barriers first and so ended the instance in the middle.
	void SplitDump(bool sink_flush) {
		if (sink_flush) on_end_record = [this] { FlushBarriers(); };
		End();
		on_end_record = nullptr;
		const bool split = rendering && !state.Full();
		if (!split) EndRendering();
		if (split && state.CanBegin()) begun[state.Begin()] = 1;
	}
};
} // namespace

int main() {
	{
		// A scope begun without an end: the next Prepare ends it and continues.
		Model model;
		model.BeginRendering();
		Require(model.state.Active() && model.state.Pending() == 0, "the scope is open");
		model.rendering = false; // its end was skipped
		model.BeginRendering();
		Require(model.stale_ends == 1, "the open query was ended once");
		Require(model.state.Active() && model.state.Pending() == 1, "the new scope uses the next index");
		model.EndRendering();
		Require(!model.state.Active() && model.state.Pending() == 2, "both queries are pending");
	}
	{
		// 1024+ queries: the pool is reduced, never overrun, whichever path fills it.
		Model model;
		for (uint32_t i = 0; i < 5 * Capacity + 7; ++i) {
			model.BeginRendering();
			model.EndRendering();
			Require(model.state.Pending() < Capacity, "the pool never stays full");
		}
		Require(model.reductions == 5 && model.dropped == 0, "a reduction per 1024 queries");
		// A full pool left behind (its reduction skipped) is reduced by the next Prepare.
		OcclusionQueryState state(Capacity);
		for (uint32_t i = 0; i < Capacity; ++i) {
			Require(state.CanBegin() && state.Begin() == i, "indices in order");
			Require(state.End() == i, "the open index ends");
		}
		Require(state.Full() && !state.CanBegin(), "the full pool refuses a begin");
		const auto recovery = state.BeforePrepare();
		Require(recovery.stale_query == OcclusionQueryState::NoQuery && recovery.reduce, "reduce first");
		state.Reduced();
		Require(state.CanBegin() && state.Pending() == 0, "the pool is usable again");
	}
	{
		// The rocket_hourglass sequence: a counted instance with pending batched barriers, split
		// dumps. With the end's barrier flush ending the instance re-entrantly, the old counter ended
		// two indices (the second never begun) and ended rendering a second time as a no-op, so the
		// pool reached 1024 without a reduction. The state ends one index per open query; the model
		// then also shows the recording without the flush (StateSink) keeping the instance.
		{
			Model model;
			for (uint32_t frame = 0; frame < 3 * Capacity; ++frame) {
				model.BeginRendering();
				model.pending_barriers = true; // a sunk barrier kept pending in the instance
				model.SplitDump(false);
				Require(model.rendering, "the split dump keeps the instance");
				Require(model.state.Pending() < Capacity, "no pool overrun after a dump");
				model.pending_barriers = false;
				model.EndRendering();
				Require(model.state.Pending() < Capacity, "no pool overrun after the instance");
			}
			Require(model.stale_ends == 0 && model.dropped == 0, "no recovery needed");
			Require(model.reductions >= 5, "the full pool was reduced");
		}
		{
			// With the barrier flush during the end (the old recording), the state still ends one
			// index; the instance ends, the dump does not split, and the pool stays consistent.
			Model model;
			for (uint32_t i = 0; i < 1021; ++i) {
				model.BeginRendering();
				model.EndRendering();
			}
			model.BeginRendering();
			model.pending_barriers = true;
			model.SplitDump(true);
			Require(!model.rendering && !model.state.Active(), "the flush ended the instance");
			Require(model.state.Pending() == 1022, "exactly one index ended (was two before)");
			model.BeginRendering();
			Require(model.stale_ends == 0 && model.state.Pending() == 1022, "no stale query");
			model.EndRendering();
			model.BeginRendering();
			model.EndRendering();
			Require(model.state.Pending() == 0 && model.reductions == 1, "reduced at 1024");
		}
		// Re-entrant end directly: one index per open query.
		OcclusionQueryState state(Capacity);
		state.Begin();
		uint32_t nested = 0;
		const auto outer = state.End();
		nested           = state.End(); // what the flush's EndRendering reached
		Require(outer == 0 && nested == OcclusionQueryState::NoQuery && state.Pending() == 1,
		        "a re-entrant end ends nothing");
	}
	std::puts("OcclusionQueryStateTests: passed");
	return 0;
}
