# Sky Garden: combined EXTRQ-v4 and tracker parking experiment

## What is uploaded, and what is not enabled

The contributor requested that maintainers receive the actual code associated with the measured FPS improvement, together with the tested hardware.

The compiled source changes in this draft PR remain limited to tracker futex parking. `linux-extrq-v4-reference.patch` supplies the additional native EXTRQ implementation, loader integration, native regressions and narrow Linux compilation prerequisite as an inert review artifact. It is not applied to the source tree, added to CMake, enabled by a preset, or executed by this PR. No native-performance claim applies to building this PR alone.

The reference patch targets the tracker-parking commit `1964bbda5b6f08071fc28233cce5fbeacd15cfe3` on upstream int15 baseline `58a7f743c445714050e6e1cd3288bfc2cb38c032`. Its native implementation and test files are byte-identical to the actual locally tested combined performance source. The patch contains nine source/build/test paths, and a scratch-index apply check against its parent passed. No binaries, games, guest dumps, game patches, saves, private configuration, credentials or private absolute paths are included.

Reference artifact SHA-256:

    35902ebe376a4737ef616e9650543489a3779600301f641726fd9db8299601df

This is deliberately a rejected, not merge-ready native prototype provided for technical inspection. Do not apply it to a user's runtime or enable it by default. Publishing the artifact does not reverse its failed native correctness review.

## Tested system

Verified from local hardware/package information and the tested emulator's renderer caption on October 6, 2026:

- CPU: Intel Core i5-14600K, 14 cores / 20 logical CPUs.
- GPU used by the game: AMD Radeon RX 6800, RADV NAVI21.
- OS-visible VRAM reported by amdgpu: 15.98 GiB.
- OS-usable system RAM: 31.06 GiB. DIMM capacity/speed were not independently read; do not infer a verified module specification from this figure.
- OS: Fedora Linux 44, KDE Plasma Desktop Edition, x86-64.
- Kernel: 7.2.8-200.fc44.x86_64.
- Mesa Vulkan drivers: 26.2.3-1.fc44.
- Compiler used for the build: Clang 22.1.8.
- Build: Release with ThinLTO; no trained PGO profile.
- Game test: ASTRO BOT 01.018.000, Sky Garden; 1280x720, Mailbox presentation, 60-Hz vblank configuration.

The CPU was previously an i5-12400 and was replaced by the i5-14600K. Historical measurements on the old CPU are not software-only evidence. The comparisons below are from the same current i5-14600K system.

## Measured gameplay evidence

The user confirmed arrival at the comparison location, and was asked to keep the camera fixed. Camera identity was not independently observed. Runs were separate rather than a randomized same-session experiment.

| Configuration | Capture | Mean of samples | Range |
| --- | --- | --- | --- |
| Original int15 | 40 seconds, periodic emulator FPS telemetry | 9.75 FPS | 9.5–9.9 |
| Native EXTRQ-v4 + tracker futex parking together | 40 seconds, periodic emulator FPS telemetry | 21.68 FPS | 20.8–22.1 |
| Later diagnostic build, diagnostic gate OFF, same two performance controls | 30 seconds, 601 KWin caption samples | 20.10 FPS | 19–21 |

The first combined mean is 21.675 before rounding, approximately 2.22 times the 9.75 baseline (+122.3%). This is evidence for the combined local experiment in this scene, not separate attribution to EXTRQ or futex. The later caption mean is sample-weighted and is not a precise frame-time statistic.

Both successful combined runs used `KYTY_LINUX_NATIVE_CPU=extrq-v4` and `KYTY_TRACKER_LOCK_PARK=1`. These are historical configuration values, not recommended reproduction instructions. The readback-prefill probe produced no additional proven FPS improvement and is excluded.

## Verification already performed on the local experiment

The previously built combined candidate passed 11 selected integrated CTests, including tracker parking/original/publication, native instruction/neighbor, emulator CLI, memory tracker, PageManager, shader precompile and pipeline fast-first cases. Native straight-line differential tests covered 6,094,848 fixtures. These checks were selected, not the full emulator suite, and did not establish complete control-flow, stack-fault, asynchronous-signal or gameplay safety.

The current source-only PR's tracker implementation was reviewed separately and passed that source review and selected lightweight startup/publication checks. The native reference artifact did not pass its correctness review. No new complete native runtime build or gameplay run was performed to publish these documentation/reference files.

## Blocking native review findings

Independent source review found concrete unresolved issues:

1. Decoder failures and unknown indirect transfers do not globally veto native rewriting. Cross-executable-segment incoming targets and undiscovered entries are not completely proven safe before overwriting EXTRQ/neighbor bytes.
2. Earlier-starting alternate-decode interval rejection is applied to the neighbor-relocating modes but not plain `extrq` mode.
3. The replacement can write below the original guest red zone, down to RSP-160, introducing faults for stack mappings where the original register-only instruction would not fault. Transient RSP/SIMD state and recovery are not certified.
4. Native-only activation adds use of the inherited EH decoder without a complete loaded-readable-range proof for malformed or indirect metadata. The parser problem predates this contribution, but the additional activation path expands exposure.

Required follow-up includes conservative incoming-flow/overlap refusal across all modes, bounded metadata validation, stack/fault recovery, actual loader allocation/sealing/unload and combined-pass regressions. Passing the straight-line emitter fixtures is not a substitute for these proofs.

## Stability observations

A local combined run ended in a guest allocation-failure assertion, a later diagnostic run was associated with an owner-reported whole-system freeze requiring forced restart, and a diagnostic-OFF retry ended with a Havok Worker access fault. Their cause and relationship to each optimization remain unresolved. The previous-boot journal contained no explicit OOM or GPU-timeout record; that is not evidence of safety. A later 30-second measurement stayed running but does not certify stability.

Keep the PR Draft. The uploaded reference code is intended to help the maintainer inspect what was actually measured and why it must not yet be integrated as an enabled native optimization.
