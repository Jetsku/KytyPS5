#pragma once

#include <filesystem>
#include <vector>

namespace Common {

// Host fonts that cover CJK (Simplified/Traditional Chinese, Japanese, Korean) glyphs.
// KYTY_CJK_FONT, when set, is tried first. Only existing regular files are returned,
// ordered by preference. TrueType collections (.ttc) are included; use face index 0.
std::vector<std::filesystem::path> FindCjkFontFiles();

} // namespace Common
