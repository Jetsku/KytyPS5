#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_SUBMITDEFER_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_SUBMITDEFER_H_

#include <cstdint>

// KYTY_PASS_SUBMIT_DEFER: the command processor's decisions that keep rendering instances out of
// optional submission boundaries (commandProcessor.h), as plain functions over its state so they
// can be tested without a device (tests/CpSequencerTests.cpp).
namespace Libs::Graphics::SubmitDefer {

struct State {
	bool     submit = false; // an optional submission waits until no rendering instance is active
	bool     yield  = false; // a slice yield waits likewise
	uint64_t carry  = 0;     // recording tick a completed graphics submission left open (0: none)
};

// An optional submission (end-of-pipe batch or its packet bound) or, with yield, a slice yield is
// requested. It is deferred (true) when the switch is on and a rendering instance is active.
inline bool Defer(State& state, bool enabled, bool in_rendering, bool yield) {
	if (!enabled || !in_rendering) {
		return false;
	}
	(yield ? state.yield : state.submit) = true;
	return true;
}

struct Take {
	bool yield  = false;
	bool submit = false;
};

// At an op or packet boundary: what was deferred and is due now that no instance is active. The
// caller submits for take.submit (Submitted clears it) and ends the slice for take.yield.
inline Take AtBoundary(State& state, bool in_rendering) {
	if (in_rendering || (!state.submit && !state.yield)) {
		return {};
	}
	const Take take {state.yield, state.submit};
	state.yield = false;
	return take;
}

// A graphics submission completed with its recording at current_tick: true when the recording
// stays open for the next graphics slice (switch on, rendering instance active).
inline bool Carry(State& state, bool enabled, bool in_rendering, uint64_t current_tick) {
	if (!enabled || !in_rendering) {
		return false;
	}
	state.carry = current_tick;
	return true;
}

// The carried recording is still unsubmitted: no submission (of any queue) took it since.
inline bool CarryPending(State& state, uint64_t current_tick) {
	if (state.carry != 0 && state.carry != current_tick) {
		state.carry = 0;
	}
	return state.carry != 0;
}

// A slice of this processor begins: a yield belongs to its own slice; a carried recording is
// submitted at the new slice's first boundary outside a rendering instance.
inline void SliceStart(State& state, uint64_t current_tick) {
	state.yield = false;
	if (CarryPending(state, current_tick)) {
		state.submit = true;
	}
}

// The current recording was submitted.
inline void Submitted(State& state) {
	state.submit = false;
	state.carry  = 0;
}

} // namespace Libs::Graphics::SubmitDefer

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_SUBMITDEFER_H_
