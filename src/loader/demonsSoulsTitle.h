#pragma once

#include <string_view>

namespace Loader::DemonsSouls {
// Demon's Souls in the regions seen (PPSA01340, PPSA01341, PPSA01342): each code patch finds its
// code by the signature of the builds seen (loader/guestCode.h), so other titles are left alone.
constexpr bool IsSupportedTitle(std::string_view title) {
	return title == "PPSA01340" || title == "PPSA01341" || title == "PPSA01342";
}
// The running game's TITLE_ID (param.sfo) is one of them.
bool IsSupportedGame();
} // namespace Loader::DemonsSouls
