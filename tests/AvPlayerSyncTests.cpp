#include "libs/avPlayerSync.h"

#include <cstdio>
#include <cstdlib>

namespace {
using Libs::Audio::AvPlayer::VideoFrameIsDue;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "AvPlayerSyncTests: FAILED: %s\n", message);
		std::exit(1);
	}
}
}

int main() {
	Check(VideoFrameIsDue(3266, false, true, true, 3264, 13315),
	      "last video frame must drain after audio ends (BumperUnreal log)");
	Check(!VideoFrameIsDue(3266, false, true, false, 3264, 13315),
	      "active or queued audio remains the synchronization clock");
	Check(VideoFrameIsDue(3250, false, true, false, 3264, 1000),
	      "video follows active audio even when wall clock is behind");
	Check(VideoFrameIsDue(3250, false, true, true, 3264, 1000),
	      "audio end must preserve frames already due under the audio clock");
	Check(!VideoFrameIsDue(4000, false, true, true, 3264, 3500),
	      "audio end must not deliver future video frames early");
	Check(VideoFrameIsDue(4000, false, true, true, 3264, 4000),
	      "longer video continues on playback clock after shorter audio");
	Check(!VideoFrameIsDue(1000, false, false, false, 0, 999), "video-only clock waits");
	Check(VideoFrameIsDue(1000, false, false, false, 0, 1000), "video-only clock delivers");
	Check(VideoFrameIsDue(1000, false, false, false, 0, 0), "preserve initial frame behavior");
	Check(VideoFrameIsDue(9000, true, true, false, 0, 0),
	      "seek preview and non-synchronized mode deliver immediately");
	std::puts("AvPlayerSyncTests: passed");
	return 0;
}
