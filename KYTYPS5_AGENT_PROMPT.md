# Task for the KytyPS5 agent: evaluate the PS5_Vulkan projects and how they can help KytyPS5

You are working on **KytyPS5**, a PlayStation 5 emulator for PC. Your job in this
task is one evaluation, not an implementation: find out what the external
PS5_Vulkan project already knows about real PS5 GPU behaviour, what that is worth
to KytyPS5, and where the two should meet. Then report, so a human decides.

Do not start changing KytyPS5 code in this task. The deliverable is a written
evaluation.

## 1. What the external projects are

- **PS5_Vulkan** — <https://github.com/mihawk-99/PS5_Vulkan>. A Vulkan driver that
  runs *on the PS5 itself* (Mesa RADV with a PS5 winsys, plus `ps5vk`, an earlier
  Mesa-derived driver), together with the probe harness used to establish what the
  console's GPU actually does. It runs real applications on retail hardware:
  vkQuake, PS5 RetroArch, Dolphin, PPSSPP, and the Khronos CTS.
- **PS5_Mesa** — <https://github.com/mihawk-99/PS5_Mesa>, branch `main`, built
  `-Dradv-winsys=ps5`. The Mesa fork carrying the port.

These are not emulators and not game tools. They are a source of *measured
hardware behaviour* — which is the same thing KytyPS5's graphics layer has to
guess at on PC.

Read first, in this order:

1. PS5_Vulkan `README.md` — what it is, what it explicitly is not.
2. `docs/HARDWARE_FINDINGS.md` — the append-only record. This is the valuable
   file. Every entry names the build or run that established it.
3. `docs/AGC_ENTRY_POINTS.md` — the AGC function table, derived from KytyPS5's own
   NID list.
4. `docs/VULKAN_PROBE_ACTIVE.md` — what is live right now.
5. `docs/CTS_GAPS.md` and `docs/BLOCKERS.md` — what is *not* done, with owners.
6. `docs/PROBE_MILESTONES.md`, `docs/M5_REFERENCE.md`, `docs/TESTING.md`,
   `docs/DEPLOYMENT.md` — the harness, as needed.

## 2. What to compare it against inside KytyPS5

The relevant code is the guest GPU model and the pieces it feeds:

- `src/graphics/guest_gpu/` — `pm4.*`, `tile.cpp`, `gpu_format.*`, `gpu_defs.h`,
  `command_processor/` (register state, command decoding, tiling, formats).
- `src/libs/agc.cpp`, `src/libs/libAgcDriver.cpp`, `src/libs/agcRegisterDefaults.inc`.
- `src/graphics/shader/recompiler/` — the shader path.
- `src/graphics/host_gpu/` — the Vulkan side that consumes all of the above.

For each area, the question is the same: **is this a place where KytyPS5 is
guessing, and can the console settle the guess?**

Useful starting signals for "guessing":

- `EXIT_NOT_IMPLEMENTED`, `UNIMPLEMENTED`, TODO and "unknown" comments in the
  guest GPU paths.
- Values pulled from the AMD register database or copied from another emulator
  and never checked against PS5 hardware.
- Divergences between what KytyPS5 assumes and what `agcRegisterDefaults.inc`
  declares.
- Anything that behaves differently on AMD vs NVIDIA vs MoltenVK — a portability
  symptom that may really be a wrong model.

Do not assume the external findings are right because they are confident. Check
each claimed fact against its cited build or run before repeating it.

The counterpart to this list on the probe side is in section 4: `tools/pm4_decode.py`
for the command packets, `tools/format_audit.py` and `tools/limits_audit.py` for
the format and limit tables, `HARDWARE_FINDINGS.md` for the layout and register
facts. A finding that matches a KytyPS5 assumption is a confirmation; one that
contradicts it is the finding worth reporting.

## 3. How that project's evidence works — read it correctly

This matters more than the findings themselves. The project's whole value is that
its claims are tied to runs, and an evaluation that repeats a claim without its
run has destroyed the value it was reporting.

**Every hardware finding names the build, commit or run that established it.**
A statement with no run behind it is a hypothesis, not a finding. When you cite
something, carry the identifier with it.

**Claims are tagged where they are uncertain:**
`[VERIFIED]` (read from source or measured), `[PUBLIC]` (known external fact),
`[INFERRED]` (a reasoned hypothesis — the document says to attack these first),
`[UNKNOWN]` (open question needing an experiment). If you find an `[INFERRED]`
item presented elsewhere as settled, say so.

**Failures are asserted alongside successes.** An evidence record asserts what is
known to fail, so it goes red when that failure is fixed. A record that asserts
only successes has to be silently rewritten when behaviour changes.

**Raw captures are not the deliverable.** What is committed is a small
machine-readable record — a capture plus the expectation it is compared against —
so pass and fail are decidable without a console. Replayable on the host:
`tools/evidence.py compare`.

**Findings are append-only.** New entries go at the end; existing ones are never
rewritten. If a claim was wrong, the correction is a new entry. So read the end of
`HARDWARE_FINDINGS.md`, not just the top.

**Its own claims about itself are bounded.** RADV on PS5 reports Vulkan 1.4 and
the CTS runs there, but `conformanceVersion` stays `0.0.0.0` until the pinned CTS
passes in full, and every remaining gap has an owner. Report the boundary, not the
headline. "1,569,390 pass" and "22 extensions and 40 features upstream RADV has
that this port does not" are both true and both belong in your report.

## 4. The instruments — by name

The workflow below is only meaningful with these, so here they are. All paths are
inside the PS5_Vulkan checkout; `tools/` there is the entry point for everything.

**The kernel log (`klog`) is the only trusted witness.** The console streams it
over a TCP port, not as a file — you connect and read. It is captured to
`Klog_Logs/klog-<time>.log`. Two records matter inside it: a `runner_summary`
line means a run completed, and a `signal:` or crash record means it faulted. The
title's own trace and the klog say different things and both are needed — the
trace stops at the last write, the fatal signal arrives after it.

**`ps5vkctl` is why runs are autonomous.** A resident control agent on the console
(`payload/ps5vkctl`, built by `tools/build-ps5vkctl.sh`, port `9111`, logging to
`/data/ps5vkctl.log`). A title cannot launch itself: the system refuses to launch
an application that is already running, and a title a probe has just faulted the
GPU inside cannot be trusted to close itself. So `ps5vkctl` starts and closes the
title the console is set up to run, over one command per connection. Its job is
what makes a battery a single unattended command instead of a person pressing
buttons.

**`tools/ps5_console.py` drives the console.** The subcommands are the workflow:

| Subcommand | Does |
| --- | --- |
| `klog` | connect, capture the klog, summarise the next run (`--follow` keeps capturing) |
| `battery <TITLE_ID> <queue.txt>` | upload a job queue, launch the title, run it, summarise — one unattended run |
| `push-jobs --replace <TITLE_ID> <queue.txt>` | upload the cases for a deployed title |
| `summary <klog file>` | summarise the runs already saved in a klog |
| `payload` | ask the resident agent what it is doing (idle, or which app and title) |
| `deploy-payload` | upload `ps5vkctl.elf` |

**The runner title executes the cases.** `PPSA99988` is the test runner: it loads a
job queue, runs the cases in order, and reports each verdict into the klog. `PPSA99015`
is the Khronos CTS running as a title. A deployable title needs
`tools/build-*-title.sh` (there is one per probe family) plus the app the queue drives.

**A job file is a case, not a script.** `jobs/<name>/queue.txt`, roughly ten lines:
a comment stating what the run is for and why it is safe or risky, then directives
like `capture` (record every frame's command stream) and `hold 60`, then the case
name to run. That layout is the method made concrete — one question, its reason,
and the case that answers it, in a file a person can read in five seconds.

**Analysers that turn captures into hardware facts.** These are the tools that
produce what `HARDWARE_FINDINGS.md` records:

| Tool | Does |
| --- | --- |
| `tools/pm4_decode.py` | decode the PM4 command packets AGC emits — the direct counterpart to KytyPS5's `guest_gpu/pm4.*` |
| `tools/format_audit.py` | format coverage, device row by row |
| `tools/limits_audit.py` | limits coverage, required vs reported |
| `tools/command_audit.py --check` | every Vulkan command is implemented, refused by name, or runtime-supplied (a gate) |
| `tools/golden.py` | replay recorded captures against their goldens on the host, no console needed |
| `tools/collect-device-report.py` | capture the device's properties, limits, features and queue families |
| `tools/check-tile-equations.py` | re-derive tiling equations and check them |
| `tools/run-cts.py`, `tools/cts-results.py` | run the conformance suite and reduce its results |
| `tools/check-driver.sh` | driver check: loader, host runs, link runs, negative tests |
| `tools/doctor.sh` | environment diagnosis before spending a run |

**Evidence tooling.** `tools/evidence.py distil <capture> --step <name>` turns a
capture into a committed record; `tools/evidence.py compare evidence/` replays every
record against its expectation on the host. `tools/fetch-trace.py` pulls a title's
own trace back; `tools/symbolize-crash.py` symbolizes it — and only against the
build that crashed.

**Host-side gates, so console time is not spent on what a PC can answer.**
`make check` (lint, host tests, full build), `make test` (host unit and
integration), `make deploy PS5_HOST=...` (FTP folder deployment, title closed
first), `make undeploy`.

**The full setup is written down** in that repo's `docs/DEPLOYMENT.md`, section
"Autonomous run loop" — including how `ps5vkctl` is built and uploaded. Read it
rather than inferring the arrangement from this summary.

## 5. The workflow that produced those findings

This is the loop the project iterates with, and the thing most worth borrowing.
It is designed around an agent that has a console it cannot block on.

**One question, one case.** A hardware question becomes a small test with a
pre-registered expectation, run on the console. Nothing is concluded from a full
run when a targeted case answers it.

**Never block on a run — poll.** Launch detached, then check every 20–30 s with one
cheap command and do other work in between. Judging completion is done from the
console's own kernel log, which streams live over TCP: a runner summary means
completion, a signal or crash record means a fault. **A stall is a result** — no
new log bytes for ~90 s means a wedged title to retry once from a clean launch,
not to wait out. Bound a run by how many cases answer the question, never by a
long timeout, so a timeout can never truncate the evidence.

**One console job at a time.** Two runs restart the same title and fight over it,
and the loser's log reads like a crash. Serialise every console action.

**Attribute the log before trusting it.** Every title's process has the same name,
so a capture is only attributable if the listener was started first and its
position in the stream recorded; every verdict comes from lines after that mark.
A title's own trace is append-mode, so it holds every run since deployment and the
run you want is at the **end** — a distillation that keeps the first N lines
records the oldest run.

**Read two records, not one.** The title's trace says what the application did;
the kernel log says what happened to the process. A trace ends at the last write,
and the fatal signal arrives after it.

**Separate what a host can check from what only hardware can.** Host tests are
fast, deterministic, never touch a console, and are safe to run alongside console
work. Anything about real behaviour goes to the console. A finding is only a
finding once hardware produced it.

**Install the debug channel before drawing anything.** The driver delivers
refusals through `VK_EXT_debug_utils` to a messenger the *application* installs;
nothing else carries the reason. A program that cannot say why it was refused
turns every unsupported path into a wasted run. Check `vkEndCommandBuffer`'s
result — that is where a refusal surfaces.

**Symbolize only against the build that crashed.** A map from another link gives
confident wrong names, which is worse than no names.

## 6. Working rules for an agent in that project

Carried here because they change what a session costs, and because they are the
reason the record above is trustworthy.

- **Append, don't edit.** The agent's context is cached by exact prefix, so
  appending is cheap and editing already-read text invalidates everything after
  it. Batch documentation edits at the end of a session.
- **A fixed read order.** A frozen instruction file, then a static plan, then one
  small volatile state file read last. Never put a date or version line at the top
  of a file that is read first.
- **One fact, one home.** Never copy status into a plan; link to it. Keep the
  volatile file small — its size is paid every session.
- **Never read a log end to end to answer a status question.** The active file
  says what passed.
- **Progress never goes in a plan.** Progress goes in the active file or an
  append-only phase log.
- **Match the evidence to the claim.** A sentence in a document or a commit is a
  claim until a run backs it.
- **Write a workaround so it can be deleted** — registered, with the conventional
  code that replaces it named, and a test keeping the two in step. An
  unregistered workaround is permanent.
- **Distinguish "the API can be used with different state" from "the application
  cannot work at all".** Only the first is a choice worth writing down; the second
  needs the missing capability, and the honest output is a request plus the
  measurement showing what it blocks.

## 7. What to report back

A single evaluation, short enough that a maintainer reads it and decides. Required
contents:

1. **The three to five highest-value things** the console work can give KytyPS5
   right now. For each: the KytyPS5 file or subsystem it touches, what is
   currently assumed there, what the console established instead, and the run or
   commit behind it.

2. **The items that are cheap to check and expensive to get wrong.** Wrong tiling,
   channel order or registers look exactly like game bugs, so these are worth
   naming even when they turn out correct.

3. **What is explicitly not offered.** No driver for a PC GPU, no game content,
   no conformance claim, nothing that requires the console to be modified. State
   this so nobody expects a shortcut that does not exist.

4. **The one integration worth considering**, with its cost. The honest version:
   KytyPS5 emits SPIR-V and already uses a real Vulkan device on every platform it
   supports, and the PS5 project ships a standards-conformant Vulkan implementation
   for the console — so KytyPS5 running *on* the PS5 is plausible with no new
   graphics backend, and the cost is not graphics: it is x86-64 guest execution,
   FreeBSD syscall emulation, process and memory constraints inside an unprivileged
   console process. Say which of those you can actually verify in this repository
   and which are unknown.

5. **A "do nothing" verdict if that is the honest answer.** If the overlap is
   smaller than it looks, say so and say why. That outcome is worth more than a
   padded list.

6. **A short list of concrete questions to send back** — the specific things
   KytyPS5 is unsure about that a targeted console case could answer. This is the
   cheapest possible next step and it should be the last section, so it survives
   even if the rest is judged not worth pursuing.

## 8. Rules for this evaluation

- **Do not repeat a hardware claim without its run identifier.** This is the one
  rule that matters most.
- **Do not present `[INFERRED]` items as established.**
- **Do not invent a use case the projects do not have.** They are a driver and a
  probe, not an emulator and not a compatibility layer.
- **Do not modify the external repositories** — read only, by design; they are
  maintained separately.
- **Say when you could not verify something.** "Not checked" is a useful result.
- **Report the boundary as loudly as the progress.** The pass count and the
  extension/feature gap belong in the same paragraph.
