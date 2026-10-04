# Early graphics pipeline compilation

`KYTY_PIPELINE_PREFETCH=1` starts graphics pipeline creation before a draw needs to bind its
pipeline. It defaults to **off**. `KYTY_PIPELINE_PREFETCH_THREADS` selects 1–4 compiler workers
(default 2). At most 128 pending requests are retained.
When the CP waits for a queued job, it moves that job ahead of speculative work. Compiles already
running complete normally.

Draw preparation predicts the ordinary complete pipeline key from its immutable register and
shader-interface snapshot. The CP also requests the pipeline after resolving its targets, before
preparing bindings and uploads. A request copies the interface arrays and keeps references only
to immutable compiled shader metadata and shader modules owned by the program cache.

The ordered draw looks up its actual key. If that key has a pending compilation, it **waits until
compilation completes**, regardless of whether the draw writes color, depth, storage images or
buffers. A prediction that differs from the actual key cannot substitute a different pipeline.
No draw skipping or draw deferral mode is provided. Queue saturation, unsupported predictions,
rect-list/tessellation pipelines and debug dumps use ordinary synchronous creation.

The pool drains and joins before the driver cache, shader modules or layouts are destroyed.
Unused speculative pipelines and their layout references are released at shutdown. Worker
compilation uses the normal monolithic create path and an internally synchronized Vulkan cache.
This preserves the renderer's ordering of resource transitions, commands and guest-visible writes.
Completed predictions that are never consumed retain a pending slot until shutdown. A workload
with many mismatched predictions can fill the 128 slots and revert later misses to synchronous
compilation; this bounds speculative memory, but limits how much stutter the pool can hide.

This mechanism overlaps driver compilation with CPU preparation; it cannot hide a long compile
when there is insufficient lead time. An entirely new shader still needs translation, emission
and module creation. A worker pool alone cannot guarantee a stutter-free first launch while
preserving every draw. Persistent pipeline recipes would also need source regeneration after
codegen updates; replaying old SPIR-V after an incompatible update is not a valid solution.

Both prefetch switches support live A/B changes. Turning pipeline prefetch off prevents new
requests; a draw still consumes and waits for a matching request already in flight.
The compiler pool is allocated only at startup. `KYTY_PIPELINE_PREFETCH_POOL=1` (default off)
creates parked workers when prefetch starts off, allowing a warm-cache same-process A/B test.
The worker count remains a startup setting. A pool created during boot can be disabled and
enabled again without changing shader or pipeline objects.

`KYTY_PIPELINE_LIBRARY=1` remains a separate existing experiment and takes precedence over
prefetch. It requires advertised GPL support and fast linking. The startup log now reports driver
support separately from enabled features; `enabled: false` is expected when GPL was not requested.
The existing device setup enables pipeline creation cache control only with GPL, so its enabled
state can also be false on hardware that advertises the feature.
The library path can change floating-point results in the last bits while fast-linked pipelines
are in use; see `pipelineLibrary.h` for its invariance restrictions.

`KYTY_PIPELINE_PREFETCH_PROGRAMS=1` separately allows draw-preparation workers to translate,
emit and create missing shader modules from clean snapshots (default off). It uses the existing
program cache's in-flight deduplication and immutable source/permutation publication. A failed
probe declines preparation; the draw's existing read certificate still decides whether the
prepared result can be consumed, and the CP retries normally if it cannot. Resource-reuse mode
and tessellation remain ineligible. Program compile records have `+prefetch` in their detail.
Compilation owns a clean copy of guest ISA and checks its content hash against the source key
before publishing anything. Declared hashes that do not equal the content hash, merged back-code
and debug dumps decline speculative compilation. The original guest base remains in resource
evaluation; only translation and disk encoding read the owned code copy.
CP shader-wait attribution includes direct CP speculative misses and idle head waits on a miss,
capped by that head's compiler-call duration; flip intervals also include other preparation work.

`KYTY_PIPELINE_COLD_SALT=N` is a **diagnostic**, default 0. It renames SPIR-V IDs immediately before
shader-module creation, changing driver input bytes without changing shader instructions or
literal operands. Application program-cache records stay unsalted. Use a fresh application cache
and a previously unused salt for cold measurements; verify coldness from actual driver compile
durations. Do not use this option for normal play.

Hang traces count worker compiles in `compile_gfx_pipelines` / `compile_gfx_pipeline_us` at worker
completion (`compiles.csv`, detail `prefetch-worker`). `compile_stall_us` and
`compile_stall_max_us` remain the time the consuming CP actually spends waiting, including its
shader work. Total worker time and total CP stall time are different quantities.

`KYTY_HANG_TRACE_FRAME_TIMES=1` additionally writes `frames.csv` with microsecond timestamps
and CP flip intervals, allowing whole-route worst-frame and percentile measurements. It is
default off, and these intervals describe emulator flips rather than OS presentation events.

Tests: `spirv_cache_salt`, `shader_code_snapshot`, `pipeline_compile_queue`, `pipeline_prefetch_rendering`, `pipeline_prefetch_programs` and
`pipeline_prefetch_draw_run`. Rendering checks use GPU readbacks, blend accumulation, depth and
texture sampling through the existing draw-prep and draw-run suites, with verification enabled.
