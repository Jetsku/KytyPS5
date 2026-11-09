// kyty_shaderlist_export: a window that turns the shader journals a player's Kyty recorded into a
// shareable shader list, for players who do not use a command line. It runs kyty_pipeline_list (next
// to this executable) export, verify and info, and keeps the list only when verify finds no shader
// code in it. The list holds where each shader sits in the game files and its hashes, never code.
// The game is a folder or a .zar archive of one (the launcher's "Compress game").
#ifndef UNICODE
#define UNICODE
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "gameFiles.h"

#include <windows.h>
#include <commctrl.h>
#include <shellapi.h>
#include <shlobj.h>
#include <shobjidl.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#pragma comment(linker, "\"/manifestdependency:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

namespace {

namespace fs = std::filesystem;

constexpr UINT WM_APP_TEXT = WM_APP + 1; // lParam: std::wstring* to append (the window frees it)
constexpr UINT WM_APP_DONE = WM_APP + 2; // wParam: 1 when the list was written

enum Control : int { KytyEdit = 101, KytyBrowse, GameEdit, GameBrowse, GameZarBrowse, ExportButton, OpenButton, Log };

HWND     g_window = nullptr;
HFONT    g_font   = nullptr;
fs::path g_written;
bool     g_busy = false;

fs::path Here() {
	std::wstring path(MAX_PATH, L'\0');
	for (;;) {
		const auto n = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
		if (n < path.size()) {
			path.resize(n);
			break;
		}
		path.resize(path.size() * 2);
	}
	return fs::path(path).parent_path();
}

void Post(const std::wstring& text) {
	PostMessageW(g_window, WM_APP_TEXT, 0, reinterpret_cast<LPARAM>(new std::wstring(text)));
}

std::wstring Widen(const std::string& text) {
	if (text.empty()) return {};
	const int    n = MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
	std::wstring out(static_cast<size_t>(n), L'\0');
	MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), out.data(), n);
	return out;
}

std::wstring Quote(const fs::path& path) {
	return L"\"" + path.wstring() + L"\"";
}

// Runs the tool with `arguments`, streaming its output to the log; its exit code (-1: not started).
int Run(const std::wstring& arguments) {
	SECURITY_ATTRIBUTES security {sizeof(security), nullptr, TRUE};
	HANDLE              read = nullptr, write = nullptr;
	if (!CreatePipe(&read, &write, &security, 0)) return -1;
	SetHandleInformation(read, HANDLE_FLAG_INHERIT, 0);
	STARTUPINFOW startup {};
	startup.cb         = sizeof(startup);
	startup.dwFlags    = STARTF_USESTDHANDLES;
	startup.hStdOutput = write;
	startup.hStdError  = write;
	startup.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);
	PROCESS_INFORMATION process {};
	std::wstring        command = Quote(Here() / L"kyty_pipeline_list.exe") + L" " + arguments;
	const BOOL          started = CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr,
	                                             nullptr, &startup, &process);
	CloseHandle(write);
	if (!started) {
		CloseHandle(read);
		return -1;
	}
	std::string pending;
	char        buffer[4096];
	DWORD       got = 0;
	while (ReadFile(read, buffer, sizeof(buffer), &got, nullptr) && got != 0) {
		pending.append(buffer, got);
		const auto end = pending.rfind('\n');
		if (end == std::string::npos) continue;
		std::string lines = pending.substr(0, end + 1);
		pending.erase(0, end + 1);
		std::wstring text;
		for (const auto c: Widen(lines)) {
			if (c == L'\n') text += L'\r';
			text += c;
		}
		Post(text);
	}
	if (!pending.empty()) Post(Widen(pending) + L"\r\n");
	CloseHandle(read);
	WaitForSingleObject(process.hProcess, INFINITE);
	DWORD code = 0;
	GetExitCodeProcess(process.hProcess, &code);
	CloseHandle(process.hProcess);
	CloseHandle(process.hThread);
	return static_cast<int>(code);
}

// A game folder or .zar archive with eboot.bin and a readable sce_sys/param.json.
bool ReadGame(const fs::path& game, std::string& title, std::string& version) {
	const auto files = PipelineListTool::GameFiles::Open(game);
	return files != nullptr && files->HasFile(L"eboot.bin") && PipelineListTool::ReadParam(*files, title, version);
}

// A Kyty folder: holds kyty_emulator.exe or _PipelineCache.
bool IsKytyFolder(const fs::path& folder) {
	std::error_code error;
	return fs::is_directory(folder / L"_PipelineCache", error) || fs::exists(folder / L"kyty_emulator.exe", error);
}

// This program's folder or its parent (the zip unpacked into the Kyty folder as its own folder).
fs::path FindKytyFolder() {
	for (const auto& folder: {Here(), Here().parent_path()}) {
		if (IsKytyFolder(folder)) return folder;
	}
	return {};
}

// The titles with both journals in the Kyty folder's _PipelineCache.
std::set<std::string> RecordedTitles(const fs::path& kyty) {
	std::set<std::string> titles;
	std::error_code       error;
	const auto            cache = kyty / L"_PipelineCache";
	for (fs::directory_iterator it(cache, error), end; !error && it != end; it.increment(error)) {
		const auto name   = it->path().filename().string();
		const auto suffix = std::string(".shaders.journal");
		if (name.size() <= suffix.size() || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
		const auto title = name.substr(0, name.size() - suffix.size());
		if (fs::exists(cache / (title + ".pipelines.journal"), error)) titles.insert(title);
	}
	return titles;
}

std::string Trim(std::string text) {
	while (!text.empty() && (text.front() == ' ' || text.front() == '"')) text.erase(text.begin());
	while (!text.empty() && (text.back() == ' ' || text.back() == '"' || text.back() == '\r')) text.pop_back();
	return text;
}

// Folders the launcher knows (Kyty.ini next to it, else ProgramData\Kyty\Kyty.ini): its game
// configurations and its game list folders.
std::vector<fs::path> LauncherFolders(const fs::path& kyty) {
	std::vector<fs::path> inis = {kyty / L"Kyty.ini"};
	PWSTR                 program_data = nullptr;
	if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_ProgramData, 0, nullptr, &program_data))) {
		inis.push_back(fs::path(program_data) / L"Kyty" / L"Kyty.ini");
		CoTaskMemFree(program_data);
	}
	std::vector<fs::path> folders;
	for (const auto& ini: inis) {
		std::ifstream in(ini, std::ios::binary);
		if (!in) continue;
		std::string line;
		while (std::getline(in, line)) {
			const auto equals = line.find('=');
			if (equals == std::string::npos) continue;
			const auto key   = line.substr(0, equals);
			const auto value = line.substr(equals + 1);
			const auto ends  = [&](const char* suffix) {
				const std::string s(suffix);
				return key.size() >= s.size() && key.compare(key.size() - s.size(), s.size(), s) == 0;
			};
			if (ends("basedir") || ends("game_path")) {
				if (!Trim(value).empty()) folders.emplace_back(Widen(Trim(value)));
			} else if (key == "game_dirs" || key == "game_dir") {
				std::stringstream list(value);
				std::string       item;
				while (std::getline(list, item, ',')) {
					if (!Trim(item).empty()) folders.emplace_back(Widen(Trim(item)));
				}
			}
		}
		if (!folders.empty()) break; // the launcher reads Kyty.ini next to it before ProgramData's
	}
	return folders;
}

struct FoundGame {
	fs::path    folder;
	std::string title, version;
};

// Game folders and .zar archives of the launcher's list (also in its folders and their direct
// subfolders) whose title has journals.
std::vector<FoundGame> DetectGames(const fs::path& kyty) {
	const auto             titles = RecordedTitles(kyty);
	std::vector<FoundGame> found;
	if (titles.empty()) return found;
	std::set<fs::path> seen;
	const auto         consider = [&](const fs::path& folder) {
		std::error_code error;
		const auto      canonical = fs::weakly_canonical(folder, error);
		if (error || !seen.insert(canonical).second) return;
		FoundGame game {canonical, {}, {}};
		if (ReadGame(canonical, game.title, game.version) && titles.contains(game.title)) {
			found.push_back(std::move(game));
		}
	};
	const auto is_zar = [](const fs::directory_entry& entry) {
		std::error_code error;
		return PipelineListTool::IsZarPath(entry.path()) && entry.is_regular_file(error);
	};
	for (const auto& folder: LauncherFolders(kyty)) {
		consider(folder); // a game folder or a .zar
		std::error_code error;
		for (fs::directory_iterator it(folder, error), end; !error && it != end; it.increment(error)) {
			if (is_zar(*it)) {
				consider(it->path());
			} else if (it->is_directory(error)) {
				consider(it->path());
				// The launcher also lists the archives inside its game folders' subfolders.
				std::error_code inner_error;
				for (fs::directory_iterator inner(it->path(), inner_error); !inner_error && inner != end;
				     inner.increment(inner_error)) {
					if (is_zar(*inner)) consider(inner->path());
				}
			}
		}
	}
	return found;
}

void Detect() {
	const HWND combo = GetDlgItem(g_window, GameEdit);
	std::wstring kyty(static_cast<size_t>(GetWindowTextLengthW(GetDlgItem(g_window, KytyEdit))) + 1, L'\0');
	GetDlgItemTextW(g_window, KytyEdit, kyty.data(), static_cast<int>(kyty.size()));
	kyty.resize(wcslen(kyty.c_str()));
	// Only for another Kyty folder: leaving the field unchanged keeps a game folder picked by hand.
	static std::wstring detected;
	if (kyty == detected) return;
	detected = kyty;
	SendMessageW(combo, CB_RESETCONTENT, 0, 0);
	if (kyty.empty()) return;
	const auto titles = RecordedTitles(kyty);
	if (titles.empty()) {
		Post(L"No recorded games found in " + kyty + L"\\_PipelineCache yet.\r\n");
		return;
	}
	const auto games = DetectGames(kyty);
	for (const auto& game: games) {
		SendMessageW(combo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(game.folder.wstring().c_str()));
		Post(L"Found " + Widen(game.title) + L" " + Widen(game.version) + L": " + game.folder.wstring() + L"\r\n");
	}
	if (!games.empty()) {
		SendMessageW(combo, CB_SETCURSEL, 0, 0);
		if (games.size() > 1) Post(L"Several games found: pick one in the list.\r\n");
	} else {
		std::wstring names;
		for (const auto& title: titles) names += (names.empty() ? L"" : L", ") + Widen(title);
		Post(L"Recorded: " + names + L". Its game folder or .zar file was not found in the launcher's list: use Folder... or "
		     L".zar file....\r\n");
	}
}

void Export(fs::path kyty, fs::path game) {
	const auto fail = [](const std::wstring& text) {
		Post(L"\r\nERROR: " + text + L"\r\n");
		PostMessageW(g_window, WM_APP_DONE, 0, 0);
	};
	if (!fs::exists(Here() / L"kyty_pipeline_list.exe")) return fail(L"kyty_pipeline_list.exe must be next to this program.");
	const auto cache = kyty / L"_PipelineCache";
	if (!fs::is_directory(cache)) return fail(L"There is no _PipelineCache folder in the Kyty folder. Pick the folder you ran the game from.");
	std::string title, version;
	if (!ReadGame(game, title, version)) {
		return fail(PipelineListTool::IsZarPath(game) ? L"The .zar file cannot be read or has no eboot.bin or sce_sys/param.json."
		                                              : L"The game folder has no eboot.bin or sce_sys\\param.json.");
	}
	const auto shaders   = cache / (Widen(title) + L".shaders.journal");
	const auto pipelines = cache / (Widen(title) + L".pipelines.journal");
	if (!fs::exists(shaders) || !fs::exists(pipelines)) {
		return fail(L"There are no " + Widen(title) +
		            L" journals in _PipelineCache. Kyty writes them while you play with its default preset.");
	}
	const auto name = Widen(title) + L"-" + Widen(version) + L"-shaderlist";
	const auto out  = Here() / (name + L".kyplst");
	Post(L"Game " + Widen(title) + L" version " + Widen(version) + L"\r\nWriting " + out.wstring() + L"\r\n\r\n");
	if (Run(L"export --out " + Quote(out) + L" --game " + Quote(game) + L" --shaders " + Quote(shaders) + L" --pipelines " +
	        Quote(pipelines) + L" --producer \"kyty player export\"") != 0) {
		return fail(L"The export failed (see above).");
	}
	Post(L"\r\nChecking that the list holds no shader code...\r\n");
	if (Run(L"verify " + Quote(out) + L" --game " + Quote(game) + L" --shaders " + Quote(shaders)) != 0) {
		std::error_code error;
		fs::remove(out, error);
		return fail(L"The check found shader code in the list, so the list was deleted.");
	}
	Post(L"\r\n");
	(void)Run(L"info " + Quote(out));
	g_written = out;
	Post(L"\r\nDone. Send this one file:\r\n  " + out.wstring() +
	     L"\r\n\r\nDo NOT send the _PipelineCache folder, the .journal files or any game files.\r\n");
	PostMessageW(g_window, WM_APP_DONE, 1, 0);
}

std::wstring Text(int id) {
	const HWND   control = GetDlgItem(g_window, id);
	std::wstring text(static_cast<size_t>(GetWindowTextLengthW(control)) + 1, L'\0');
	GetWindowTextW(control, text.data(), static_cast<int>(text.size()));
	text.resize(wcslen(text.c_str()));
	// Pasted paths often come quoted.
	while (!text.empty() && (text.front() == L'"' || text.front() == L' ')) text.erase(text.begin());
	while (!text.empty() && (text.back() == L'"' || text.back() == L' ')) text.pop_back();
	return text;
}

// A folder, or with zar a .zar file.
void PickFolder(int edit, const wchar_t* title, bool zar = false) {
	IFileOpenDialog* dialog = nullptr;
	if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog)))) return;
	DWORD options = 0;
	dialog->GetOptions(&options);
	dialog->SetOptions(options | (zar ? FOS_FILEMUSTEXIST : FOS_PICKFOLDERS) | FOS_FORCEFILESYSTEM);
	if (zar) {
		const COMDLG_FILTERSPEC types[] = {{L"Compressed game (*.zar)", L"*.zar"}};
		dialog->SetFileTypes(1, types);
	}
	dialog->SetTitle(title);
	if (SUCCEEDED(dialog->Show(g_window))) {
		IShellItem* item = nullptr;
		if (SUCCEEDED(dialog->GetResult(&item))) {
			PWSTR path = nullptr;
			if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &path))) {
				SetDlgItemTextW(g_window, edit, path);
				CoTaskMemFree(path);
			}
			item->Release();
		}
	}
	dialog->Release();
}

void Append(const std::wstring& text) {
	const HWND log = GetDlgItem(g_window, Log);
	const int  end = GetWindowTextLengthW(log);
	SendMessageW(log, EM_SETSEL, end, end);
	SendMessageW(log, EM_REPLACESEL, FALSE, reinterpret_cast<LPARAM>(text.c_str()));
}

HWND Add(const wchar_t* type, const wchar_t* text, DWORD style, int x, int y, int w, int h, int id) {
	const HWND control = CreateWindowExW(type == std::wstring(L"EDIT") ? WS_EX_CLIENTEDGE : 0, type, text,
	                                     WS_CHILD | WS_VISIBLE | style, x, y, w, h, g_window,
	                                     reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)), nullptr, nullptr);
	SendMessageW(control, WM_SETFONT, reinterpret_cast<WPARAM>(g_font), TRUE);
	return control;
}

void Layout(int width, int height) {
	const int margin = 12, button = 90, row = 24;
	const int edit_w = width - 3 * margin - button;
	MoveWindow(GetDlgItem(g_window, KytyEdit), margin, 34, edit_w, row, TRUE);
	MoveWindow(GetDlgItem(g_window, KytyBrowse), width - margin - button, 34, button, row, TRUE);
	MoveWindow(GetDlgItem(g_window, GameEdit), margin, 90, edit_w - margin - button, 200, TRUE); // with its drop-down list
	MoveWindow(GetDlgItem(g_window, GameBrowse), width - 2 * margin - 2 * button, 90, button, row, TRUE);
	MoveWindow(GetDlgItem(g_window, GameZarBrowse), width - margin - button, 90, button, row, TRUE);
	MoveWindow(GetDlgItem(g_window, ExportButton), margin, 128, 160, 30, TRUE);
	MoveWindow(GetDlgItem(g_window, OpenButton), margin + 172, 128, 160, 30, TRUE);
	MoveWindow(GetDlgItem(g_window, Log), margin, 170, width - 2 * margin, height - 170 - margin, TRUE);
}

LRESULT CALLBACK WindowProc(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
	switch (message) {
		case WM_CREATE: {
			g_window = window;
			NONCLIENTMETRICSW metrics {sizeof(metrics)};
			SystemParametersInfoW(SPI_GETNONCLIENTMETRICS, sizeof(metrics), &metrics, 0);
			g_font = CreateFontIndirectW(&metrics.lfMessageFont);
			Add(L"STATIC", L"Kyty folder (the one with kyty_emulator.exe and _PipelineCache):", 0, 12, 12, 600, 20, 0);
			Add(L"EDIT", L"", ES_AUTOHSCROLL, 0, 0, 0, 0, KytyEdit);
			Add(L"BUTTON", L"Browse...", BS_PUSHBUTTON, 0, 0, 0, 0, KytyBrowse);
			Add(L"STATIC", L"Game folder or .zar file (the folder is the one with eboot.bin and sce_sys):", 0, 12, 68, 600, 20, 0);
			Add(L"COMBOBOX", L"", CBS_DROPDOWN | CBS_AUTOHSCROLL | WS_VSCROLL | WS_TABSTOP, 0, 0, 0, 0, GameEdit);
			Add(L"BUTTON", L"Folder...", BS_PUSHBUTTON, 0, 0, 0, 0, GameBrowse);
			Add(L"BUTTON", L".zar file...", BS_PUSHBUTTON, 0, 0, 0, 0, GameZarBrowse);
			Add(L"BUTTON", L"Make shader list", BS_DEFPUSHBUTTON, 0, 0, 0, 0, ExportButton);
			Add(L"BUTTON", L"Show the file", BS_PUSHBUTTON | WS_DISABLED, 0, 0, 0, 0, OpenButton);
			Add(L"EDIT",
			    L"This makes a small shader list from what Kyty recorded while you played. The list holds no game "
			    L"code and no game files: only where each shader is in your eboot.bin and a hash. It is checked "
			    L"for shader code before it is kept.\r\n",
			    ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | WS_VSCROLL, 0, 0, 0, 0, Log);
			SendDlgItemMessageW(window, Log, EM_SETLIMITTEXT, 0, 0);
			// This program placed in the Kyty folder (or a folder inside it) fills both folders in.
			if (const auto kyty = FindKytyFolder(); !kyty.empty()) {
				SetDlgItemTextW(window, KytyEdit, kyty.c_str());
				Detect();
			}
			return 0;
		}
		case WM_SIZE: Layout(LOWORD(lparam), HIWORD(lparam)); return 0;
		case WM_GETMINMAXINFO: {
			auto* info             = reinterpret_cast<MINMAXINFO*>(lparam);
			info->ptMinTrackSize.x = 520;
			info->ptMinTrackSize.y = 360;
			return 0;
		}
		case WM_COMMAND:
			switch (LOWORD(wparam)) {
				case KytyBrowse:
					PickFolder(KytyEdit, L"Kyty folder");
					Detect();
					return 0;
				case KytyEdit:
					if (HIWORD(wparam) == EN_KILLFOCUS) Detect();
					return 0;
				case GameBrowse: PickFolder(GameEdit, L"Game folder"); return 0;
				case GameZarBrowse: PickFolder(GameEdit, L"Game .zar file", true); return 0;
				case OpenButton: {
					if (g_written.empty()) return 0;
					const std::wstring arguments = L"/select,\"" + g_written.wstring() + L"\"";
					ShellExecuteW(window, nullptr, L"explorer.exe", arguments.c_str(), nullptr, SW_SHOWNORMAL);
					return 0;
				}
				case ExportButton: {
					if (g_busy) return 0;
					const auto kyty = Text(KytyEdit), game = Text(GameEdit);
					if (kyty.empty() || game.empty()) {
						MessageBoxW(window, L"Pick the Kyty folder and the game folder or .zar file first.", L"Shader list", MB_ICONINFORMATION);
						return 0;
					}
					g_busy = true;
					g_written.clear();
					EnableWindow(GetDlgItem(window, ExportButton), FALSE);
					EnableWindow(GetDlgItem(window, OpenButton), FALSE);
					SetDlgItemTextW(window, Log, L"");
					std::thread(Export, fs::path(kyty), fs::path(game)).detach();
					return 0;
				}
			}
			break;
		case WM_APP_TEXT: {
			auto* text = reinterpret_cast<std::wstring*>(lparam);
			Append(*text);
			delete text;
			return 0;
		}
		case WM_APP_DONE:
			g_busy = false;
			EnableWindow(GetDlgItem(window, ExportButton), TRUE);
			EnableWindow(GetDlgItem(window, OpenButton), wparam != 0);
			return 0;
		case WM_CLOSE:
			if (g_busy && MessageBoxW(window, L"The export is still running. Close anyway?", L"Shader list",
			                          MB_YESNO | MB_ICONQUESTION) != IDYES) {
				return 0;
			}
			break;
		case WM_DESTROY:
			PostQuitMessage(0);
			return 0;
	}
	return DefWindowProcW(window, message, wparam, lparam);
}

} // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE /*previous*/, PWSTR /*command_line*/, int show) {
	(void)CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
	INITCOMMONCONTROLSEX controls {sizeof(controls), ICC_STANDARD_CLASSES};
	InitCommonControlsEx(&controls);
	WNDCLASSW window_class {};
	window_class.lpfnWndProc   = WindowProc;
	window_class.hInstance     = instance;
	window_class.hCursor       = LoadCursorW(nullptr, IDC_ARROW);
	window_class.hIcon         = LoadIconW(nullptr, IDI_APPLICATION);
	window_class.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_BTNFACE + 1);
	window_class.lpszClassName = L"KytyShaderListExport";
	RegisterClassW(&window_class);
	const HWND window = CreateWindowExW(0, window_class.lpszClassName, L"Kyty shader list export", WS_OVERLAPPEDWINDOW,
	                                    CW_USEDEFAULT, CW_USEDEFAULT, 760, 560, nullptr, nullptr, instance, nullptr);
	ShowWindow(window, show);
	MSG message;
	while (GetMessageW(&message, nullptr, 0, 0) > 0) {
		if (IsDialogMessageW(window, &message)) continue;
		TranslateMessage(&message);
		DispatchMessageW(&message);
	}
	CoUninitialize();
	return 0;
}
