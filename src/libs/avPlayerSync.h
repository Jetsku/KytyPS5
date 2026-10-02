#ifndef KYTY_LIBS_AV_PLAYER_SYNC_H_
#define KYTY_LIBS_AV_PLAYER_SYNC_H_

#include <cstdint>

namespace Libs::Audio::AvPlayer {

inline bool VideoFrameIsDue(uint64_t frame_ms, bool immediate, bool has_audio,
                            bool audio_drained, uint64_t last_audio_ms, uint64_t clock_ms) {
	if (immediate) {
		return true;
	}
	// Once the decoder is done and no audio frames remain, its final timestamp
	// cannot advance. Video may legitimately have a later final presentation time.
	if (has_audio) {
		if (frame_ms <= last_audio_ms) {
			return true;
		}
		if (!audio_drained) {
			return false;
		}
	}
	return clock_ms == 0 || frame_ms <= clock_ms;
}

} // namespace Libs::Audio::AvPlayer

#endif
