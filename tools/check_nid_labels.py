#!/usr/bin/env python3
"""Check that the AGC import labels are the names their NIDs hash to.

Usage:
  python tools/check_nid_labels.py [--verbose]

A NID is the guest's import identifier: the first 11 characters of the base64 of the
SHA-1 of the symbol name plus a fixed 16-byte suffix, first 8 bytes reversed, '/' written
as '-'. For every LIB_FUNC("<nid>", <namespace>::<Label>) in src/libs/libAgcDriver.cpp the
check computes the NID of "sce<Label>" and reports the entries whose label is not that
name. The entries below are known exceptions: their real names are not known (most are
probably C++ symbols, whose NID hashes a mangled name). The check fails when a label
stops matching, when a new unexplained mismatch appears, or when an exception is stale.
Pure standard library.
"""
import argparse
import base64
import hashlib
import os
import re
import sys

NID_SUFFIX = bytes([0x51, 0x8D, 0x64, 0xA6, 0x35, 0xDE, 0xD8, 0xC1,
                    0xE6, 0xB0, 0x39, 0xB1, 0xC3, 0xE5, 0x52, 0x30])

AGC_SOURCE = os.path.join("src", "libs", "libAgcDriver.cpp")

# NID -> why its label is not its name.
KNOWN_EXCEPTIONS = {
    # Labelled Unknown in the source: behaviour known, name not.
    "dolOmWH+huQ": "labelled Unknown",
    "fd5Bp5tGTgo": "labelled Unknown",
    "nApJjpKNBl4": "labelled Unknown",
    "Ikfdt-rIqCE": "labelled Unknown",
    "-KRzWekV120": "labelled Unknown",
    "U9ueyEhSkF4": "labelled Unknown",
    # Descriptive labels: what the function does, not what it is called.
    "23LRUSvYu1M": "descriptive label",
    "HV4j+E0MBHE": "descriptive label",
    "dbOlWdppb4o": "descriptive label",
    "V++UgBtQhn0": "descriptive label",
    "qj7QZpgr9Uw": "descriptive label",
    "k-JpyR2dYAM": "descriptive label",
    "3ZWa3AoyWZQ": "descriptive label",
    # A second NID bound to the handler of a named entry point.
    "AhGvpITrf4M": "second NID of sceAgcDriverSubmitDcb's handler",
    "+T8Xo6LtFJI": "second NID of sceAgcDriverSubmitMultiDcbs's handler",
}

LIB_FUNC = re.compile(r'LIB_FUNC\("([^"]+)",\s*(?:[\w]+::)*(\w+)\)')


def compute_nid(name):
    digest = hashlib.sha1(name.encode("ascii") + NID_SUFFIX).digest()
    return base64.b64encode(digest[:8][::-1]).decode("ascii")[:11].replace("/", "-")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verbose", action="store_true", help="list every exception")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, AGC_SOURCE), encoding="utf-8") as source:
        entries = LIB_FUNC.findall(source.read())
    if not entries:
        print(f"check_nid_labels: no LIB_FUNC entries in {AGC_SOURCE}", file=sys.stderr)
        return 1

    errors = []
    matched = 0
    excepted = 0
    seen = set()
    for nid, label in entries:
        seen.add(nid)
        hashes = compute_nid("sce" + label) == nid
        if hashes:
            matched += 1
            if nid in KNOWN_EXCEPTIONS:
                errors.append(f"{nid} {label}: now hashes, remove it from KNOWN_EXCEPTIONS")
        elif nid not in KNOWN_EXCEPTIONS:
            errors.append(f"{nid} {label}: sce{label} hashes to {compute_nid('sce' + label)}")
        else:
            excepted += 1
            if args.verbose:
                print(f"  {nid} {label}: {KNOWN_EXCEPTIONS[nid]}")
    for nid in sorted(set(KNOWN_EXCEPTIONS) - seen):
        errors.append(f"{nid}: in KNOWN_EXCEPTIONS but not in {AGC_SOURCE}")

    print(f"check_nid_labels: {len(entries)} entries, {matched} labels hash to their NID, "
          f"{excepted} known exceptions, {len(errors)} errors")
    for error in errors:
        print(f"check_nid_labels: error: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
