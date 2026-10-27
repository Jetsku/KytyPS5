#include "loader/demonsSoulsTitle.h"

#include "loader/systemContent.h"

#include <string>

namespace Loader::DemonsSouls {

bool IsSupportedGame() {
	std::string title;
	return SystemContentParamSfoGetString("TITLE_ID", &title) && IsSupportedTitle(title);
}

} // namespace Loader::DemonsSouls
