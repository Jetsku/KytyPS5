#pragma once

#include <cstdint>

// Runs fn(context) and, on Windows, turns a structured exception raised inside it (an access
// violation in a driver, say) into a return value instead of ending the process. 0: fn returned
// normally; otherwise the exception code (0xC0000005 for an access violation). Elsewhere it just
// calls fn and returns 0.
//
// For driver calls whose failure may be skipped past, such as building a pipeline for an optional
// ray-tracing kernel. The driver's state after such a fault is not known: the caller stops using
// whatever it was doing (it marks the feature failed) but does not rely on the driver for it again.
// fn's frame is left without running destructors, so it must hold no object that needs one: pass
// a plain function and a context struct, and keep C++ exception handling inside fn.
namespace Common {

uint32_t CallCatchingStructuredException(void (*fn)(void*), void* context);

} // namespace Common
