#!/usr/bin/env python3
"""Per-thread profile from tools/thread_sampler.exe samples.

Usage:
  python analyze_samples.py SAMPLES.csv --exe kyty_emulator.exe [--log GUEST.log] [--top 14]

- Emulator code (kyty_emulator.exe) is resolved to functions with llvm-symbolizer and the PDB.
- Windows DLL code (ntdll, kernelbase, ...) is named by the nearest exported symbol (marked ~).
- Anything outside a loaded module is guest code; with --log it is attributed to the guest module
  (eboot.bin, libc.prx, ...) from the "Loading:" / "base_vaddr" lines.
- Thread names come from the "Pthread run begin: NAME ... os_thread_id = N" lines of --log.
- With the call stacks of the sampler (column "stack"), each thread also gets an INCLUSIVE view (how many
  samples pass through a function, wherever it is in the stack) and its most frequent call paths.
Pure standard library.
"""
import argparse
import bisect
import collections
import csv
import os
import re
import struct
import subprocess
import sys

SYMBOLIZER = r"C:\Program Files\LLVM\bin\llvm-symbolizer.exe"

EXCEPTION_PATTERNS = re.compile(
    r"KiUserExceptionDispatch|RtlDispatchException|RtlRaiseException|RtlpExecuteHandler|RtlCaptureContext|"
    r"RtlRestoreContext|NtContinue|ZwContinue|RtlVirtualUnwind|RtlLookupFunctionEntry|RtlpLookup|RtlUnwind|"
    r"RtlpGetStackLimits|KiRaiseUserExceptionDispatcher", re.I)
PROTECT_PATTERNS = re.compile(r"NtProtectVirtualMemory|ZwProtectVirtualMemory|VirtualProtect", re.I)
WAIT_PATTERNS = re.compile(r"NtWaitFor|ZwWaitFor|WaitForSingleObject|WaitForMultiple|SleepEx|NtDelayExecution|"
                           r"RtlSleepConditionVariable|RtlWaitOnAddress|NtWaitForAlertByThreadId|SwitchToThread|NtYieldExecution", re.I)


def parse_exports(path):
    """Sorted [(rva, name)] of a PE file's export table (named exports only)."""
    try:
        data = open(path, "rb").read()
    except OSError:
        return []
    if data[:2] != b"MZ":
        return []
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        return []
    nsec, = struct.unpack_from("<H", data, pe + 6)
    opt_size, = struct.unpack_from("<H", data, pe + 20)
    opt = pe + 24
    magic, = struct.unpack_from("<H", data, opt)
    dd = opt + (112 if magic == 0x20B else 96)
    export_rva, export_size = struct.unpack_from("<II", data, dd)
    if export_rva == 0:
        return []
    sections = []
    sec = opt + opt_size
    for i in range(nsec):
        vsize, vaddr, rawsize, rawptr = struct.unpack_from("<IIII", data, sec + 40 * i + 8)
        sections.append((vaddr, max(vsize, rawsize), rawptr))

    def to_off(rva):
        for vaddr, size, rawptr in sections:
            if vaddr <= rva < vaddr + size:
                return rva - vaddr + rawptr
        return None

    off = to_off(export_rva)
    if off is None:
        return []
    num_funcs, num_names, addr_funcs, addr_names, addr_ords = struct.unpack_from("<IIIII", data, off + 20)
    funcs_off, names_off, ords_off = to_off(addr_funcs), to_off(addr_names), to_off(addr_ords)
    if None in (funcs_off, names_off, ords_off):
        return []
    result = []
    for i in range(num_names):
        name_rva, = struct.unpack_from("<I", data, names_off + 4 * i)
        ordinal, = struct.unpack_from("<H", data, ords_off + 2 * i)
        func_rva, = struct.unpack_from("<I", data, funcs_off + 4 * ordinal)
        name_off = to_off(name_rva)
        if name_off is None:
            continue
        end = data.index(b"\0", name_off)
        result.append((func_rva, data[name_off:end].decode("ascii", "replace")))
    result.sort()
    return result


def nearest(exports, offset):
    if not exports:
        return None
    index = bisect.bisect_right([e[0] for e in exports], offset) - 1
    if index < 0:
        return None
    rva, name = exports[index]
    return name if offset - rva < 0x4000 else None


def load_log(path):
    names, guest_modules = {}, []
    if not path or not os.path.exists(path):
        return names, guest_modules
    pending = None
    thread_re = re.compile(r"Pthread run begin: (.+?), id = \d+, os_thread_id = (\d+)")
    load_re = re.compile(r"^Loading: .*[\\/]([^\\/]+)$")
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            if "Pthread run begin:" in line:
                m = thread_re.search(line)
                if m:
                    names[int(m.group(2))] = m.group(1)
            elif line.startswith("Loading: "):
                m = load_re.match(line.rstrip("\r\n"))
                pending = m.group(1) if m else None
            elif pending and line.startswith("base_vaddr"):
                base = int(line.split("=")[1].strip(), 16)
                guest_modules.append([base, 0, pending])
            elif pending and line.startswith("base_size"):
                guest_modules[-1][1] = int(line.split("=")[1].strip(), 16)
                pending = None
    guest_modules.sort()
    return names, guest_modules


def load_modules(path):
    modules = []
    mpath = path + ".modules"
    if os.path.exists(mpath):
        for r in csv.DictReader(open(mpath)):
            modules.append((int(r["base"], 16), int(r["size"], 16), r["name"]))
    return sorted(modules)


def module_of(modules, address):
    for base, size, name in modules:
        if base <= address < base + size:
            return name, address - base
    return None, 0


def guest_label(guest_modules, address):
    for base, size, name in guest_modules:
        if base <= address < base + max(size, 0x1000):
            return "guest %s+0x%x" % (name, address - base)
    return "guest/unknown 0x%x" % address


def symbolize(exe, addresses):
    if not addresses:
        return {}
    unique = sorted(set(addresses))
    proc = subprocess.run([SYMBOLIZER, "--obj=" + exe, "--demangle", "--no-inlines"],
                          input="\n".join("0x%x" % a for a in unique) + "\n", capture_output=True, text=True)
    blocks = proc.stdout.strip("\n").split("\n\n")
    result = {}
    for address, block in zip(unique, blocks):
        first = block.split("\n")[0].strip()
        result[address] = first if first and first != "??" else "(no symbol) 0x%x" % address
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--exe", required=True)
    parser.add_argument("--log")
    parser.add_argument("--top", type=int, default=14)
    parser.add_argument("--paths", type=int, default=6)
    parser.add_argument("--children", default="",
                        help="comma-separated function-name substrings: print what each one calls (its direct callees)")
    parser.add_argument("--thread", type=int, default=0, help="only this tid")
    parser.add_argument("--callers", default="",
                        help="comma-separated function-name substrings: print who calls each one (the frame above it)")
    parser.add_argument("--blame", default="",
                        help="comma-separated leaf-name substrings (e.g. a wait inside a driver DLL): print the first "
                             "emulator function on the stack of every sample that stops there")
    args = parser.parse_args()

    names, guest_modules = load_log(args.log)
    modules = load_modules(args.csv)
    # A sampler killed together with its target leaves a truncated last line: skip incomplete rows.
    rows = [r for r in csv.DictReader(open(args.csv))
            if r.get("tid") and r.get("rip") and r.get("module") and r.get("offset")]
    exe_name = os.path.basename(args.exe).lower()
    sysdir = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    exports = {}

    def frames_of(row):
        """[(address, module_name, offset)] leaf first, then the callers."""
        leaf = int(row["rip"], 16)
        result = [(leaf, row["module"], int(row["offset"], 16))]
        for token in (row.get("stack") or "").split(";"):
            if token:
                address = int(token, 16)
                name, off = module_of(modules, address)
                result.append((address, name or "-", off))
        return result

    parsed = [(int(r["tid"]), frames_of(r)) for r in rows]
    exe_addresses = [a for _, frames in parsed for a, m, _ in frames if m.lower() == exe_name]
    exe_symbols = symbolize(args.exe, exe_addresses)

    def label(address, module, offset):
        if module.lower() == exe_name:
            return exe_symbols.get(address, "(emulator)"), "emulator"
        if module == "-":
            return guest_label(guest_modules, address), "guest code"
        if module not in exports:
            exports[module] = parse_exports(os.path.join(sysdir, module))
        symbol = nearest(exports[module], offset)
        text = "%s!%s%s" % (module, "~" if symbol else "", symbol or "+0x%x" % offset)
        if EXCEPTION_PATTERNS.search(text):
            return text, "windows: exception dispatch"
        if PROTECT_PATTERNS.search(text):
            return text, "windows: page protection"
        if WAIT_PATTERNS.search(text):
            return text, "windows: waiting"
        return text, "windows dll"

    by_thread = collections.defaultdict(list)
    for tid, frames in parsed:
        by_thread[tid].append(frames)

    for tid, samples in sorted(by_thread.items(), key=lambda kv: -len(kv[1])):
        if args.thread and tid != args.thread:
            continue
        total = len(samples)
        categories = collections.Counter()
        leaves = collections.Counter()
        inclusive = collections.Counter()
        paths = collections.Counter()
        all_labels = []
        all_categories = []
        has_stacks = False
        for frames in samples:
            labels = []
            frame_categories = []
            for index, (address, module, offset) in enumerate(frames):
                text, category = label(address, module, offset)
                labels.append(text)
                frame_categories.append(category)
                if index == 0:
                    categories[category] += 1
                    leaves[text] += 1
            if len(labels) > 1:
                has_stacks = True
            all_labels.append(labels)
            all_categories.append(frame_categories)
            for text in set(labels):
                inclusive[text] += 1
            short = []
            for text in labels:
                if not short or short[-1] != text:
                    short.append(text)
            paths[" <- ".join(t[:60] for t in short[:6])] += 1
        print("=== thread %d  %s  (%d samples)" % (tid, names.get(tid, "(host thread, no guest name)"), total))
        print("  by category: " + "   ".join("%s %.0f%%" % (c, 100.0 * n / total) for c, n in categories.most_common()))
        print("  where the CPU is (leaf):")
        for text, count in leaves.most_common(args.top):
            print("   %5.1f%%  %s" % (100.0 * count / total, text[:130]))
        if has_stacks:
            print("  who owns the time (inclusive: samples passing through the function):")
            for text, count in inclusive.most_common(args.top):
                print("   %5.1f%%  %s" % (100.0 * count / total, text[:130]))
            print("  most frequent call paths (leaf <- caller <- ...):")
            for text, count in paths.most_common(args.paths):
                print("   %5.1f%%  %s" % (100.0 * count / total, text[:230]))
        for needle in [n.strip() for n in args.children.split(",") if n.strip()]:
            under = collections.Counter()
            hits = 0
            for labels in all_labels:
                # labels run leaf first; the function's direct callee is the frame just before it
                index = next((i for i, text in enumerate(labels) if needle in text), None)
                if index is None:
                    continue
                hits += 1
                under[labels[index - 1] if index > 0 else "(self: leaf)"] += 1
            if hits:
                print("  inside %s  (%.1f%% of samples); direct callees:" % (needle, 100.0 * hits / total))
                for text, count in under.most_common(args.top):
                    print("   %5.1f%%  of its time   %s" % (100.0 * count / hits, text[:140]))
        for needle in [n.strip() for n in args.callers.split(",") if n.strip()]:
            above = collections.Counter()
            hits = 0
            for labels in all_labels:
                # labels run leaf first; the caller is the next distinct frame after the function
                indexes = [i for i, text in enumerate(labels) if needle in text]
                if not indexes:
                    continue
                hits += 1
                last = indexes[-1]
                above[labels[last + 1] if last + 1 < len(labels) else "(stack ends here)"] += 1
            if hits:
                print("  %s  (%.1f%% of samples); called by:" % (needle, 100.0 * hits / total))
                for text, count in above.most_common(args.top):
                    print("   %5.1f%%  of them   %s" % (100.0 * count / hits, text[:140]))
        for needle in [n.strip() for n in args.blame.split(",") if n.strip()]:
            blamed = collections.Counter()
            hits = 0
            for labels, frame_categories in zip(all_labels, all_categories):
                if needle not in labels[0]:
                    continue
                hits += 1
                # the first emulator frame is the code that called into the DLL (the driver, the kernel)
                index = next((i for i, category in enumerate(frame_categories) if category == "emulator"), None)
                blamed[labels[index][:140] if index is not None else "(no emulator frame on the stack)"] += 1
            if hits:
                print("  samples stopped in %s  (%.1f%% of samples); first emulator function:" % (needle, 100.0 * hits / total))
                for text, count in blamed.most_common(args.top):
                    print("   %5.1f%%  of them   %s" % (100.0 * count / hits, text))
        print()


if __name__ == "__main__":
    sys.exit(main())
