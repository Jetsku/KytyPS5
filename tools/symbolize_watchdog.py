"""Annotate native emulator RVAs using the linker map from the exact same build.

Usage: python tools/symbolize_watchdog.py watchdog.txt kyty_emulator.map
No symbol server, debugger, packages, or running process are required. The nearest
function symbol is an aid to source lookup; it is not a source-line stack trace.
"""

import argparse
import bisect
import re
from pathlib import Path


def read_symbols(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    base = re.search(r"Preferred load address is\s+([0-9a-fA-F]+)", text)
    if not base:
        raise ValueError("map has no preferred load address")
    preferred = int(base.group(1), 16)
    symbols = {}
    # lld's MSVC-compatible map omits the f marker. Its CODE section inventory
    # still excludes data/absolute symbols; MSVC maps have the same inventory.
    code_sections = {
        int(match.group(1), 16)
        for match in re.finditer(
            r"^\s*([0-9a-fA-F]{4}):[0-9a-fA-F]+\s+[0-9a-fA-F]+H\s+\S+\s+CODE\s*$",
            text, re.MULTILINE)
    }
    pattern = re.compile(r"^\s*([0-9a-fA-F]{4}):[0-9a-fA-F]+\s+(\S+)\s+([0-9a-fA-F]+)(?:\s|$)")
    for line in text.splitlines():
        match = pattern.match(line)
        if match and int(match.group(1), 16) in code_sections:
            rva = int(match.group(3), 16) - preferred
            if rva >= 0:
                symbols.setdefault(rva, match.group(2))
    if not symbols:
        raise ValueError("map has no code symbols")
    addresses = sorted(symbols)
    return addresses, symbols


def annotate(text, module, addresses, symbols):
    pattern = re.compile(r"^(\s*)" + re.escape(module) + r"\+0x([0-9a-fA-F]+)\s*$", re.IGNORECASE)
    lines = []
    resolved = 0
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            rva = int(match.group(2), 16)
            index = bisect.bisect_right(addresses, rva) - 1
            if index >= 0:
                start = addresses[index]
                line += f"  [nearest: {symbols[start]}+0x{rva - start:x}]"
                resolved += 1
        lines.append(line)
    return "\n".join(lines) + "\n", resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("watchdog", type=Path)
    parser.add_argument("map", type=Path)
    parser.add_argument("--module", help="module basename (default: map stem plus .exe)")
    parser.add_argument("--output", type=Path, help="write annotated copy instead of stdout")
    args = parser.parse_args()
    try:
        addresses, symbols = read_symbols(args.map)
        result, resolved = annotate(args.watchdog.read_text(encoding="utf-8", errors="replace"),
                                    args.module or args.map.stem + ".exe", addresses, symbols)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not resolved:
        parser.error("no matching emulator frames; check module name and matching map")
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
