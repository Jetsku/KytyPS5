#include "graphics/host_gpu/renderer/occlusionReset.h"
#include "common/liveSwitch.h"

#include <array>
#include <cstdio>
#include <cstdlib>

using Libs::Graphics::OcclusionResetWindow;

static void Require(bool value) {
    if (!value) { std::fputs("occlusion reset invariant failed\n", stderr); std::abort(); }
}

int main() {
    Live::Switch enabled("KYTY_OCCLUSION_RESET_BATCH", Live::ParseDefaultOff);
    OcclusionResetWindow live_window;
    for (const auto* value: {"1", "0", "1", "", "1"}) {
        Live::Testing::StageText(std::string("KYTY_OCCLUSION_RESET_BATCH=") + value + "\n");
        Live::OnCpFlip();
        live_window.Reduced();
        Require(live_window.Prepare(0, 1024, enabled.On() ? 64u : 1u).count ==
                (enabled.On() ? 64u : 1u));
    }
    // Simulate query availability, including a partial reduction, pool exhaustion, and live
    // on/off changes. Consumed queries may not be reset before the reduction reads their results.
    for (uint32_t capacity: {1u, 17u, 64u, 65u, 1024u}) {
        OcclusionResetWindow window;
        std::array<bool, 1024> reset {};
        uint32_t seed = 1;
        for (uint32_t epoch = 0; epoch != 100; ++epoch) {
            reset.fill(false);
            window.Reduced();
            const auto used = 1u + epoch % capacity;
            for (uint32_t next = 0; next != used; ++next) {
                // KYTY_OCCLUSION_SPLIT begins a query at `next` without a reset only when the
                // window says it was reset and not consumed since.
                if (window.IsReset(next)) Require(reset[next]);
                seed = seed * 1664525u + 1013904223u;
                const auto range = window.Prepare(next, capacity, seed & 1u ? 64u : 1u);
                Require(range.first == next && range.count <= capacity - next);
                for (uint32_t i = range.first; i < range.first + range.count; ++i) reset[i] = true;
                Require(reset[next]);
                reset[next] = false; // begin/end consumes the reset
                for (uint32_t i = 0; i <= next; ++i) Require(!reset[i]);
            }
        }
    }
    OcclusionResetWindow window;
    Require(window.Prepare(0, 1024, 64).count == 64);
    for (uint32_t i = 1; i != 64; ++i) Require(window.Prepare(i, 1024, 64).count == 0);
    Require(window.Prepare(64, 1024, 64).count == 64);
    Require(window.Prepare(1024, 1024, 64).count == 0);
    Require(window.IsReset(127) && !window.IsReset(128));
    window.Reduced();
    Require(!window.IsReset(0));
    Require(window.Prepare(0, 1024, 64).count == 64);
    Require(window.IsReset(1) && window.IsReset(63) && !window.IsReset(64));
    // Legacy mode always resets exactly the next query, even after a wider reset.
    Require(window.Prepare(1, 1024, 1).count == 1);
    std::puts("occlusion reset windows passed");
}
