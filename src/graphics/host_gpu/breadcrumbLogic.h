#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_BREADCRUMBLOGIC_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_BREADCRUMBLOGIC_H_

#include <cstdint>

// GPU breadcrumbs (KYTY_GPU_BREADCRUMBS, gpuBreadcrumbs.h): the pure parts, tested without a device
// (tests/DeviceCompatTests.cpp).
//
// Before each guest operation the queue writes the operation's 32-bit marker twice into a
// host-visible buffer (VK_AMD_buffer_marker): at the top of the pipe ("begun": the command
// processor reached the operation) and at the bottom of the pipe ("done before": every command
// recorded before it, on that queue, has completed). After a device loss the two words of a queue
// bound the operations that may still have been running: those from "done before" to "begun".
namespace Libs::Graphics::Breadcrumbs {

// The 64-bit breadcrumb sequence a 32-bit marker stands for, given the newest sequence issued so
// far (markers are the low 32 bits; a marker above the newest one's low bits belongs to the
// previous 2^32 window). 0 means no marker was written.
[[nodiscard]] constexpr uint64_t SequenceFromMarker(uint32_t marker, uint64_t newest) noexcept {
	if (marker == 0 || newest == 0) {
		return 0;
	}
	const uint64_t candidate = (newest & ~uint64_t {0xffffffffu}) | marker;
	if (candidate <= newest) {
		return candidate;
	}
	return candidate >= (uint64_t {1} << 32u) ? candidate - (uint64_t {1} << 32u) : 0;
}

struct InFlight {
	uint64_t first = 0; // oldest operation that may not have completed (0: none known)
	uint64_t last  = 0; // newest operation the queue began
	[[nodiscard]] constexpr bool Empty() const noexcept { return first == 0 || last < first; }
	[[nodiscard]] constexpr uint64_t Count() const noexcept { return Empty() ? 0 : last - first + 1; }
};

// The operations of one queue that may still have been running. `done_before` == `begun` means
// everything before `begun` completed and `begun` itself started; an end marker (a command buffer's
// last breadcrumb, written only at the bottom) with done_before == begun means the queue was idle.
[[nodiscard]] constexpr InFlight QueueInFlight(uint64_t begun, uint64_t done_before) noexcept {
	if (begun == 0) {
		return {};
	}
	if (done_before == 0 || done_before > begun) {
		// No bottom marker yet (or one from an older window): everything up to `begun` is open.
		return {.first = done_before == 0 ? 1 : begun, .last = begun};
	}
	return {.first = done_before, .last = begun};
}

} // namespace Libs::Graphics::Breadcrumbs

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_BREADCRUMBLOGIC_H_ */
