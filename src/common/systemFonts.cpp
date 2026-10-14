#include "common/systemFonts.h"

#include <cstdlib>
#include <system_error>

namespace Common {

namespace {

void AppendIfFile(std::vector<std::filesystem::path>& out, const std::filesystem::path& path) {
	if (path.empty()) {
		return;
	}
	std::error_code ec;
	if (std::filesystem::is_regular_file(path, ec) && !ec) {
		out.push_back(path);
	}
}

} // namespace

std::vector<std::filesystem::path> FindCjkFontFiles() {
	std::vector<std::filesystem::path> result;

	if (const char* env = std::getenv("KYTY_CJK_FONT"); env != nullptr && *env != '\0') {
		AppendIfFile(result, std::filesystem::path(env));
	}

#if defined(_WIN32)
	std::filesystem::path fonts_dir = "C:\\Windows\\Fonts";
	if (const char* windir = std::getenv("WINDIR"); windir != nullptr && *windir != '\0') {
		fonts_dir = std::filesystem::path(windir) / "Fonts";
	}
	for (const char* name: {"msyh.ttc", "msyh.ttf", "Deng.ttf", "simhei.ttf", "simsun.ttc",
	                        "msjh.ttc", "malgun.ttf", "YuGothM.ttc", "meiryo.ttc"}) {
		AppendIfFile(result, fonts_dir / name);
	}
#elif defined(__APPLE__)
	for (const char* path: {"/System/Library/Fonts/PingFang.ttc",
	                        "/System/Library/Fonts/Hiragino Sans GB.ttc",
	                        "/System/Library/Fonts/STHeiti Medium.ttc",
	                        "/System/Library/Fonts/STHeiti Light.ttc",
	                        "/Library/Fonts/Arial Unicode.ttf"}) {
		AppendIfFile(result, path);
	}
#else
	for (const char* path: {"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
	                        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
	                        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
	                        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
	                        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
	                        "/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc",
	                        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
	                        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"}) {
		AppendIfFile(result, path);
	}
#endif

	return result;
}

} // namespace Common
