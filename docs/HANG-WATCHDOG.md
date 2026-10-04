# Hang watchdog

`KYTY_HANG_WATCHDOG=1` is the default. After both a guest flip and a renderer submission have
occurred, the watchdog writes one report if neither counter advances for five seconds. It checks
once a second, so detection can take approximately five to six seconds. It does not recover,
time out, reorder, or skip guest or GPU work. A long legitimate loading stall can also trigger it.

The report is `watchdog.txt` in the active hang-trace directory. With tracing disabled, it is in
`_HangTrace/watchdog-<timestamp>-pid<PID>/` beside the emulator on Windows. An explicit
`KYTY_HANG_TRACE_DIR` also works without `KYTY_HANG_TRACE`. The report does not depend on console
or guest printf redirection. If an explicit directory is unwritable, the executable-relative
directory is tried. Wait metadata is flushed before thread stacks are collected.

Normal operation records bounded atomic metadata and polls two counters once a second. There
are no watchdog file writes, stack walks or additional Vulkan queries until a stall fires.
The watchdog never takes guest, renderer, event-owner or submission locks to take its snapshot.
Event publishers use a separate short registry lock. On Windows, each process-owned thread is
briefly suspended to copy its context and at most 64 KiB of its stack, then resumed **before**
unwinding or writing. Native module offsets require the matching executable and PDB/link map.
Guest module bases are copied at load time, independently of hang tracing.

The report contains:

- Nested active wait/operation scopes, host thread IDs, guest thread objects and names, CP queue
  and admission sequence. Guest mutex owner objects can be matched to thread `guest` fields.
- Suspended `WAIT_REG_MEM` operands (address, reference, observed value, mask, comparison and
  width), frame-fence state, pending EOP labels and their timeline ticks.
- Guest event queue registrations and active waits; condition, rwlock, semaphore, event-flag,
  sync-on-address and join waits. Observations are captured by the owning thread, not read live.
- In-flight shader hashes/stages, native graphics/compute pipeline creation, background shader
  validation/checks, pipeline optimization and pipeline-cache serialization.
- Recorder, sequencer, draw-prep, broker, readback, priority-callback and master timeline waits.
  Contended tracking-region locks include the last owner host thread ID; resource parking locks
  include their lock address and last state.
  Returned master errors and device-loss diagnostic calls remain visible if fatal reporting blocks.
- The last 32 PM4/typed packets per internal queue and last 256 native submission bundles,
  including semaphore waits/signals and pipeline stage masks. A native bundle means the host
  was about to submit; it does not prove the driver call returned. Zero semaphore values may
  identify binary semaphores. Timeline values are last observations, not watchdog queries.
- Active APR file reads and guest copies, with file ID, offset, destination and byte count.
- Windows native module inventory and thread contexts/stacks, including waits inside uninstrumented library calls.
- The first fatal report before emergency shutdown, if one occurred, so cleanup waits do not
  obscure the original failure when console output is unavailable.

Fatal emergency shutdown also stops the profiler. Before it stops the monitor, an enabled
watchdog saves the terminal error, active scopes and native contexts immediately if no report
has fired. That file is marked `trigger=terminal-error`; a five-second stall report is marked
`trigger=stopped-progress`. Neither trigger changes normal execution or recovers failed GPU work.

Internal queue 0 is graphics. Internal compute queue `q` maps to guest queue `q + 31`.
Common scope operands can be read as follows (generic scope values are printed in hexadecimal):

| Scope | Address/resource | Expected | Observed | Aux |
| --- | --- | --- | --- | --- |
| `master-dispatch` | Timeline semaphore | Required tick | Last host-dispatched tick | Current recording tick |
| `master-gpu` | Timeline semaphore | Required tick | Last known completion tick | Current recording tick |
| `master-counter-query` | Timeline semaphore | Current recording tick | Unused | Unused |
| `master-*-error` | Timeline semaphore | Required/current tick | Returned signed VkResult as 64 bits | Unused |
| `program-compile` / `program-finish` | Guest shader hash | Shader stage enum | Unused | Unused |
| `program-in-flight` | Requested guest shader hash | Shader stage enum | In-flight compile count | Program cache object |
| `graphics-pipeline` | Guest vertex shader hash | Vertex program ID | Pixel program ID | Unused |
| `compute-pipeline` | Guest compute shader hash | Compute program ID | Unused | Unused |
| `readback-publication` | Guest page/window begin | Completion tick/value | Page/window end | Eager copy flag |
| `buffer-readback` | Guest address | Requested bytes | Unused | Write fault flag |
| `guest-mutex` | Mutex object | Requesting guest thread object | Last owner guest thread object | Unused |
| `guest-condition` | Condition object | Associated mutex object | Unused | Unused |
| `guest-equeue` | Queue object | Requested event count | Unused | Timeout in microseconds |
| `tracker-owner` | Tracking lock | Requesting host thread ID | Last owner host thread ID | Unused |
| `resource-parking-lock` | Parking lock | Free state (0) | Last lock state | Unused |

An active `master-dispatch` with observed below expected means the host has not finished handing
that tick to the driver. An active `master-gpu` has passed that host-dispatch check. An active
counter query instead means the host API has not returned; its scope cannot report a fresh GPU
value. Native stacks and the first fatal report help distinguish driver calls from teardown waits.

Typed packet opcodes use `0x10000 + OpKind`; their type names identify the operation. Typed
snapshot payloads may start with host register-snapshot addresses. Raw PM4 records retain the
header and first four payload words. No guest memory is dereferenced while reporting.

Storage is bounded (512 registered threads, 16 nested scopes, 57 CP queues, 1,024 event/label
registrations, 64 timeline slots, 64 guest modules). `overflow` reports capacity/publication
loss. Snapshots are observations made across threads, not a simultaneous global stop. Native
stacks can stop at generated/guest frames without unwind metadata. A report narrows the blocked
call; it cannot by itself prove a GPU shader or driver defect.

The graphics parser and resolver can publish packets concurrently. Packet slots have nonblocking
writer guards; a collision or a publisher overtaken after preemption increments `overflow`
instead of waiting or overwriting a newer packet. Their shared history carries monotonic IDs.

`KYTY_HANG_WATCHDOG=auto` (the release preset) keeps it off until the GPU is selected, then turns it on only for
NVIDIA RTX 50 (Blackwell) GPUs and logs `Kyty hang watchdog: on/off`.

`KYTY_HANG_WATCHDOG=0` disables monitoring and metadata publication. It is a live switch for
same-process A/B; a process started disabled needs `KYTY_LIVE_FILE` to allow later activation.
`KYTY_HANG_WATCHDOG_MS` changes the startup threshold (1,000–600,000 ms; default 5,000).

For a timing perturbation, use the default-off `KYTY_HANG_DELAY_SITE`:

| Setting | Purpose |
| --- | --- |
| `queue` | Delay the CP before servicing a selected internal queue. |
| `submit` | Delay a broker/recorder native submission. |
| `compile` | Delay a program compilation after releasing the program registry lock. |
| `label` | Delay the CP's deferred label write. |
| `readback` | Delay a readback publisher before acquiring its publication mutex. |
| `eager-issue` | Delay after publishing an eager readback, before flushing its producer tick. |

`KYTY_HANG_DELAY_KEY` optionally selects an internal queue ID, timeline tick, shader hash or guest
address respectively; `eager-issue` selects a tracker page address (decimal or `0x...`). `KYTY_HANG_DELAY_AFTER_MS` defaults to 30,000;
`KYTY_HANG_DELAY_MS` defaults to 6,000 and is capped at 30,000. Only one matching delay occurs
per process. An intentional `debug-delay-*` scope identifies it in the report. These controls
preserve ordering and introduce a finite delay; recovery afterwards is not reproduction of a
permanent freeze.

With `--graphics-debug-dump true`, matched-input JSON, IR and SPIR-V snapshots also cover
compute `305afd0aa0f66b9a` and vertices `cf1834bb2d5ac83d`/`d9cc5c62178518fa`. JSON records
the compiling CP queue/admission sequence and direct compute buffer descriptor operands; these
are compile-permutation inputs, not a complete record of later dispatches.

Build `hang_watchdog_tests`, then run CTest `hang_watchdog` and `hang_watchdog_fire`. They cover
arming/progress/once-only decisions, nested in-flight metadata, publication consistency during
concurrent reads, packet-ring wrap, event removal, native dependencies, file output with tracing
disabled, no output before arming, and Windows stack capture. Include this target in `kyty_tests`.
For Windows diagnosis a Release linker map (`/MAP`) can preserve symbol addresses without
changing shader codegen compile flags; keep it alongside the exact executable used for the test.

Resolve emulator frame offsets offline with the map from the **exact same executable**:

```text
python tools/symbolize_watchdog.py watchdog.txt kyty_emulator.map --output watchdog-symbolized.txt
```

This adds the nearest function symbol and displacement, preserves the original report, and
requires no debugger/symbol server. It does not resolve driver DLL offsets or inlined source
lines. Use `--module` if the executable was renamed; do not substitute a different build map.

`KYTY_SHADER_WRITE_RETICK=1` enables a separate, default-off readback correctness candidate.
Writable storage buffers are reserved while bindings are prepared. If a later upload submits
that recording before its draw/dispatch is emitted, the candidate retags those buffers with
the final producer tick in `CommitBindings`. A side readback then recognizes an unsubmitted
writer and drains it instead of publishing older GPU contents. Dirty/protected ranges do not
expand, and no extra submission is introduced by retagging itself.

CTest `buffer_cache_late_storage_*` forces this timing window on Vulkan: controls observe the
old value, the candidate observes the producer, and disabling side copies also removes the
race. The fixed/control pair also runs through the queued CP recorder. This proves a stale
readback mechanism; it does not establish that a particular game freeze has that cause.

The existing `_device_fault.nv-gpudmp` contains a Vulkan vendor-binary header before the
vendor payload. If an installed NVIDIA Aftermath reader rejects it, preserve the original
and extract a separate payload first:

```text
python tools/unwrap_device_fault.py _device_fault.nv-gpudmp vendor-payload.nv-gpudmp
```

This offline helper validates the version/length, prints header metadata, and refuses to
overwrite an existing output. It neither installs software nor changes driver settings.
