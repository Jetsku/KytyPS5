# KytyPS5 U59 int19.1

int19.1 is int19 plus two fixes and a faster build; int19's notes are in docs/release-notes-history.md and on the int19 release page. The first launch rebuilds the shader and program caches once, so the first minutes can stutter.

- **Fixed:** the emulator no longer closes (`occlusion.cpp:405`) with "Accurate occlusion queries" on, seen in rocket_hourglass.
- **Faster:** the release build's profile-guided optimisation profile was retrained on the int19 code, about +8% fps in Sky Garden on an RTX 3090 / Ryzen 9 7950X3D.
- **Known issue:** rare GPU device loss (`ErrorDeviceLost`) on NVIDIA is under investigation; none reported by players so far. Workaround: `KYTY_DEFER_PROTECTS=0` in `u59-preset.json`. To report, reproduce once with `KYTY_DEVICE_FAULT_DIAGNOSTICS=1` and Printf output set to File, then attach `_kyty.txt` and the `_kyty` folder to an issue.
- **AMD, Intel and Linux users:** their fixes are in the int19 notes.
