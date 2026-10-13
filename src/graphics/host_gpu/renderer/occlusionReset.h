#ifndef KYTY_RENDERER_OCCLUSION_RESET_H_
#define KYTY_RENDERER_OCCLUSION_RESET_H_

#include <algorithm>
#include <cstdint>

namespace Libs::Graphics {

// Query indices are consumed monotonically until their results have been reduced. Reset only
// unused indices; a reduction starts a fresh epoch even when it stays in the same command buffer.
class OcclusionResetWindow {
public:
    struct Range { uint32_t first; uint32_t count; };

    [[nodiscard]] Range Prepare(uint32_t next, uint32_t capacity, uint32_t batch) noexcept {
        if (next >= capacity) return {next, 0};
        if (batch > 1 && next < m_end) return {next, 0};
        const auto count = std::min(std::max(batch, 1u), capacity - next);
        m_end = std::max(m_end, next + count);
        return {next, count};
    }

    void Reduced() noexcept { m_end = 0; }

    // The unused index next was reset since the last reduction (a query may begin there without
    // a reset, e.g. inside a rendering instance).
    [[nodiscard]] bool IsReset(uint32_t next) const noexcept { return next < m_end; }

private:
    uint32_t m_end = 0;
};

} // namespace Libs::Graphics

#endif
