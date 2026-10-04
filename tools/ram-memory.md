# Windows system memory diagnostics

These experimental switches are off by default. Set them before launching the emulator;
changing the environment of a running emulator does not change their behavior.

| Switch | Behavior |
|---|---|
| `KYTY_RAM_STATS=1` | Prints allocation addresses and capacities for guest backing, guest stacks, mapped Vulkan buffers, command rings, draw preparation slots, and driver cache input. With VRAM stats enabled, also reports retained translation estimates and pipeline counts. |
| `KYTY_GUEST_BACKING_LAZY_COMMIT=1` | Reserves the Windows guest backing section and commits physical offsets when they are mapped, rather than committing its entire capacity at creation. |
| `KYTY_GUEST_BACKING_LAZY_ZERO=1` | With lazy commitment enabled, leaves fresh flexible-memory pages demand-zero. Previously committed pages are explicitly cleared on reuse. |
| `KYTY_RAM_SMALL_UPLOAD_RING=1` | Uses a 128 MiB persistent upload ring instead of 512 MiB. Oversized image uploads use temporary transfer/storage buffers kept alive until GPU completion. |

The guest backing switches preserve direct-memory contents across unmap/remap and preserve
alias visibility. Section commitment is a process-lifetime high-water mark: Windows does
not allow decommitting pages of a reserved paging-file section. Lazy zeroing only skips
clearing pages that have never been committed; it does not discard live guest contents.
Linux backing behavior is unchanged.

Run the read-only sampler alongside the PID started by the game harness:

```powershell
./tools/Sample-GameRam.ps1 -GameProcessId 1234 -OutputDir ./ram-samples
```

It samples private bytes and working set each second and writes full address-space CSVs
every ten seconds. To request a checkpoint snapshot, write a line such as
`title|2026-10-03T12:00:00Z` to `checkpoint-request.txt` in the output directory.
`checkpoint-snapshots.txt` records the corresponding region filename. The sampler exits
when the target exits, without suspending its threads or reading guest data.

Private bytes measure private commit, not total physical RAM. A paging-file-backed guest
section consumes system commit outside that counter. Guest views and the host backing
alias refer to the same section; their committed extents and working-set estimates are
not additive physical memory. Device-local Vulkan memory remains device memory even when
it has a CPU mapping. Use the system-heap lines from `KYTY_VRAM_STATS=1` to distinguish it
from upload/download allocations.

Region state, type, protection, and address columns are hexadecimal. `region_age_ms`
indicates how old cached region totals are; private bytes and working set are fresh on
each sample. Resident estimates sample one page per 64 KiB in large regions. `-1` means
unsampled, rather than zero residency. Estimates do not deduplicate shared aliases.
`peak_working_set` and `peak_commit` are process high-water counters returned by the same
counter call, including peaks between samples. Peak commit uses `PeakPagefileUsage` and
does not include separate shared-section commitment.

Relevant tests are `virtual_memory_allocation`, `virtual_memory_allocation_lazy_backing`,
`virtual_memory_allocation_lazy_zero`, `stream_buffer_ring`,
`stream_buffer_ring_small_upload`, and `stream_buffer_ring_small_upload_queued`.
Memory tests cover fresh zeroing, untouched neighboring pages, remaps, aliases and dirty
flexible reuse. Upload tests cover ring wrap/fences and temporary buffer transfer and
storage-shader access, including queued submission.

Validate startup switches using interleaved runs of the same executable and cache state,
with identical sampler settings and screenshots of the same scene. Commitment savings
alone do not establish that a title fits in a 16 GiB machine: resident peaks, driver
allocations, operating-system headroom, and performance under memory pressure also matter.
