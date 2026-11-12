#include "graphics/presentation/renderDoc.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <vector>
#include <cstdio>
#include <filesystem>
#include <renderdoc_app.h>
#include <string>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace Libs::Graphics {

enum class RenderDocState : uint32_t {
	Idle,
	Requested,
	Capturing,
};

static RENDERDOC_API_1_6_0*        g_api             = nullptr;
static std::atomic<RenderDocState> g_state           = RenderDocState::Idle;
static uint32_t                    g_captured_flips  = 0;
static std::atomic_bool            g_unavailable_log = false;

static void CaptureStatus(const char* message) {
	std::printf("RenderDoc: %s\n", message);
	std::fflush(stdout);
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

static bool BindRenderDocApi(HMODULE module) {
	auto* get_api = reinterpret_cast<pRENDERDOC_GetAPI>(GetProcAddress(module, "RENDERDOC_GetAPI"));
#else
static bool BindRenderDocApi(void* module) {
	auto* get_api = reinterpret_cast<pRENDERDOC_GetAPI>(::dlsym(module, "RENDERDOC_GetAPI"));
#endif
	if (get_api == nullptr) {
		return false;
	}

	void* api = nullptr;
	if (get_api(eRENDERDOC_API_Version_1_6_0, &api) != 1 || api == nullptr) {
		return false;
	}

	g_api = static_cast<RENDERDOC_API_1_6_0*>(api);
	g_api->SetCaptureKeys(nullptr, 0);
	g_api->UnloadCrashHandler();
	CaptureStatus("API 1.6.0 bound");
	return true;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

void RenderDocInit() {
	if (g_api != nullptr) {
		return;
	}

	auto* module = GetModuleHandleA("renderdoc.dll");
	if (module == nullptr) {
		HKEY key = nullptr;
		if (RegOpenKeyExW(HKEY_LOCAL_MACHINE,
		                  L"SOFTWARE\\Classes\\RenderDoc.RDCCapture.1\\DefaultIcon\\", 0, KEY_READ,
		                  &key) != ERROR_SUCCESS) {
			return;
		}

		std::array<wchar_t, MAX_PATH> path_buffer {};
		DWORD      path_size = static_cast<DWORD>(path_buffer.size() * sizeof(wchar_t));
		const auto result    = RegQueryValueExW(
		    key, L"", nullptr, nullptr, reinterpret_cast<LPBYTE>(path_buffer.data()), &path_size);
		RegCloseKey(key);
		if (result != ERROR_SUCCESS) {
			return;
		}

		auto path = std::filesystem::path(path_buffer.data()).parent_path() / "renderdoc.dll";
		module    = LoadLibraryW(path.c_str());
		if (module == nullptr) {
			return;
		}
	}

	if (!BindRenderDocApi(module)) {
		CaptureStatus("API 1.6.0 is unavailable");
	}
}

#else

void RenderDocInit() {
	if (g_api != nullptr) {
		return;
	}

	auto* module = ::dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
	if (module == nullptr) {
		module = ::dlopen("librenderdoc.so", RTLD_NOW);
	}
	if (module == nullptr) {
		return;
	}

	if (!BindRenderDocApi(module)) {
		CaptureStatus("API 1.6.0 is unavailable");
		::dlclose(module);
	}
}

#endif

void RenderDocRequestCapture() {
	if (g_api == nullptr) {
		if (!g_unavailable_log.exchange(true)) {
			CaptureStatus("capture requested, but RenderDoc is unavailable");
		}
		return;
	}

	RenderDocState expected = RenderDocState::Idle;
	if (g_state.compare_exchange_strong(expected, RenderDocState::Requested)) {
		CaptureStatus("capture requested");
	}
}

static void StartCapture() {
	if (g_api->IsFrameCapturing() != 0) {
		g_state.store(RenderDocState::Idle, std::memory_order_release);
		CaptureStatus("capture request ignored because a capture is already active");
		return;
	}

	const auto capture_id   = std::chrono::duration_cast<std::chrono::microseconds>(
	                              std::chrono::system_clock::now().time_since_epoch())
	                              .count();
	const auto capture_path = "_RenderDoc/kyty_" + std::to_string(capture_id);
	g_api->SetCaptureFilePathTemplate(capture_path.c_str());
	g_api->StartFrameCapture(nullptr, nullptr);
	if (g_api->IsFrameCapturing() == 0) {
		g_state.store(RenderDocState::Idle, std::memory_order_release);
		CaptureStatus("capture failed to start");
		return;
	}
	g_captured_flips = 0;
	g_state.store(RenderDocState::Capturing, std::memory_order_release);
	CaptureStatus("capture started");
}

// KYTY_RENDERDOC_CAPTURE_FLIPS=<n>[,<n>...] (default none): request a capture at these guest flip
// counts, as F1 would, for unattended runs.
static void AutoCaptureOnFlip() {
	static const std::vector<uint64_t> flips = [] {
		std::vector<uint64_t> list;
		const auto* value = std::getenv("KYTY_RENDERDOC_CAPTURE_FLIPS");
		for (const char* p = value; p != nullptr && *p != '\0';) {
			char* end = nullptr;
			const auto n = std::strtoull(p, &end, 0);
			if (end == p) break;
			list.push_back(n);
			p = *end == ',' ? end + 1 : end;
		}
		return list;
	}();
	// KYTY_RENDERDOC_CAPTURE_FILE=<path> (default unset): every 8th guest flip the file is checked;
	// when it exists it is deleted and a capture is requested, so a test runner can capture at an
	// event (a KYTY_EVENT line) instead of a guessed flip number.
	static const char* trigger = [] {
		const auto* value = std::getenv("KYTY_RENDERDOC_CAPTURE_FILE");
		return value != nullptr && *value != '\0' ? value : nullptr;
	}();
	static uint64_t flip = 0;
	++flip;
	if (trigger != nullptr && (flip % 8u) == 0u) {
		std::error_code error;
		if (std::filesystem::exists(trigger, error) && std::filesystem::remove(trigger, error)) {
			std::printf("RenderDoc: capture requested by KYTY_RENDERDOC_CAPTURE_FILE at guest flip %llu\n",
			            static_cast<unsigned long long>(flip));
			RenderDocRequestCapture();
		}
	}
	if (flips.empty()) return;
	for (const auto n: flips) {
		if (n == flip) {
			std::printf("RenderDoc: automatic capture at guest flip %llu (KYTY_RENDERDOC_CAPTURE_FLIPS)\n",
			            static_cast<unsigned long long>(flip));
			RenderDocRequestCapture();
		}
	}
}

void RenderDocOnGuestFlip(RenderContext& renderer) {
	AutoCaptureOnFlip();
	const auto state = g_state.load(std::memory_order_acquire);
	if (state == RenderDocState::Idle) {
		return;
	}
	if (state == RenderDocState::Capturing) {
		// LOGF does not evaluate its arguments when printf output is Silent.
		// KYTY_RENDERDOC_CAPTURE_LENGTH=<n> (default 2, diagnostic): guest flips per capture.
		static const uint32_t length = [] {
			const auto* value = std::getenv("KYTY_RENDERDOC_CAPTURE_LENGTH");
			const auto  n     = value != nullptr ? std::strtoul(value, nullptr, 10) : 2ul;
			return static_cast<uint32_t>(std::clamp(n, 1ul, 64ul));
		}();
		++g_captured_flips;
		std::printf("RenderDoc: captured guest flip %u/%u\n", g_captured_flips, length);
		std::fflush(stdout);
		if (g_captured_flips < length) {
			return;
		}
	}

	// Capture boundaries follow presentation and exclude concurrent queue access.
	RenderLockGuard render_lock(renderer.GetMutex());
	// KYTY_CP_RECORDER: every recording submitted so far reaches the queue or the broker first
	// (before queue_mutex, which the recorder may need).
	auto& scheduler = renderer.GetCommandScheduler();
	scheduler.WaitRecorded(scheduler.CurrentTick() - 1, false);
	Common::LockGuard queue_lock(renderer.GetGraphics().queue_mutex);
	renderer.GetGraphics().submission_queue.DrainPendingLocked();
	if (state == RenderDocState::Requested) {
		StartCapture();
	} else {
		const auto ok = g_api->EndFrameCapture(nullptr, nullptr);
		g_state.store(RenderDocState::Idle, std::memory_order_release);
		CaptureStatus(ok != 0 ? "capture finished" : "capture failed");
	}
}

} // namespace Libs::Graphics
