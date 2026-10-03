"""Summarize ProgramCachePreloadTests --benchmark CSV using only the standard library.

UTF8 (with or without BOM) and BOM-marked UTF16 LE/BE are accepted. Exit codes:
0: matching record counts; 1: integrity mismatch; 2: malformed input or I/O error.

Example:
    python tools/analyze_preload_benchmark.py benchmark.csv --expect-sources 1038 \
        --expect-permutations 1128 --expect-file-bytes 114782071 --output analysis.json
"""

import argparse
import codecs
import csv
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


COUNTS = ("file_bytes", "sources", "permutations", "rejected")
INTEGERS = ("threads", "run", *COUNTS, "workers")
TIMINGS = ("load_ms", "read_ms", "validate_ms", "index_ms")
NOTES = (
    "Variation in read_ms is not evidence of faster parallel validation; compare validate_ms "
    "separately and keep load_ms as the observed total.",
    "The benchmark copies the original input to a temporary file before loading it. "
    "Those writes warm the OS page cache; cold storage performance was not measured.",
    "The benchmark extracts header identity from its input. Successful parsing does not "
    "confirm compatibility with the current emulator's codegen, configuration or device.",
)


def read_samples(path):
    raw = path.read_bytes()
    encoding = "utf-16" if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)) else "utf-8-sig"
    reader = csv.DictReader(io.StringIO(raw.decode(encoding)))
    required = set(INTEGERS + TIMINGS)
    fields = reader.fieldnames or []
    missing = required.difference(fields)
    if missing:
        raise ValueError("missing CSV columns: " + ", ".join(sorted(missing)))
    if len(fields) != len(set(fields)):
        raise ValueError("duplicate CSV column names")
    samples = []
    for line, row in enumerate(reader, 2):
        if None in row or any(row.get(field) is None for field in required):
            raise ValueError(f"row {line}: incorrect number of CSV fields")
        sample = {"row": line}
        for field in INTEGERS:
            try:
                number = int(row[field])
            except ValueError:
                raise ValueError(f"row {line}: {field} must be an integer") from None
            if number < 0:
                raise ValueError(f"row {line}: {field} must be nonnegative")
            sample[field] = number
        for field in TIMINGS:
            try:
                number = float(row[field])
            except ValueError:
                raise ValueError(f"row {line}: {field} must be a finite nonnegative number") from None
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"row {line}: {field} must be a finite nonnegative number")
            sample[field] = number
        samples.append(sample)
    if not samples:
        raise ValueError("CSV contains no benchmark samples")
    return samples


def summarize(samples, explicit_expected):
    # When no external counts are supplied, this checks consistency with the first sample.
    # Rejections must always be zero; a consistently damaged input must not pass unnoticed.
    expected = {field: samples[0][field] for field in COUNTS}
    expected["rejected"] = 0
    expected.update(explicit_expected)
    errors = []
    seen = set()
    grouped = defaultdict(list)
    for sample in samples:
        label = f"row {sample['row']} (threads={sample['threads']}, run={sample['run']})"
        for field, number in expected.items():
            if sample[field] != number:
                errors.append(f"{label}: {field}={sample[field]}, expected {number}")
        key = (sample["threads"], sample["run"])
        if key in seen:
            errors.append(f"{label}: duplicate sample for this configuration and run")
        seen.add(key)
        grouped[sample["threads"]].append(sample)
    groups = []
    for threads, rows in sorted(grouped.items()):
        group = {
            "threads": threads,
            "workers": sorted({row["workers"] for row in rows}),
            "samples": len(rows),
        }
        for timing in TIMINGS:
            values = [row[timing] for row in rows]
            group[timing] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
        groups.append(group)
    return {
        "integrity": {
            "ok": not errors,
            "expected": expected,
            "explicit_expected": explicit_expected,
            "baseline": "first sample for unspecified counts; rejected must be zero",
            "errors": errors,
        },
        "groups": groups,
        "notes": list(NOTES),
    }


def print_report(report):
    integrity = report["integrity"]
    state = "PASS" if integrity["ok"] else "FAIL"
    counts = ", ".join(f"{field}={value}" for field, value in integrity["expected"].items())
    print(f"Integrity {state}: {counts}")
    print("Unspecified expected counts come from the first sample; supply --expect-* for known totals.")
    print("threads workers n " + " ".join(f"{timing:^24}" for timing in TIMINGS))
    print("                  " + " ".join(f"{'median [min, max]':^24}" for _ in TIMINGS))
    for group in report["groups"]:
        workers = "/".join(map(str, group["workers"]))
        cells = []
        for timing in TIMINGS:
            metric = group[timing]
            cells.append(f"{metric['median']:.3f} [{metric['min']:.3f}, {metric['max']:.3f}]")
        print(f"{group['threads']:7} {workers:>7} {group['samples']:2} " + " ".join(f"{cell:>24}" for cell in cells))
    for error in integrity["errors"]:
        print("Integrity mismatch: " + error)
    for note in report["notes"]:
        print(note)


def nonnegative_integer(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("expected count must be nonnegative")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="benchmark CSV (UTF8 or BOM-marked UTF16)")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    parser.add_argument("--expect-file-bytes", type=nonnegative_integer)
    parser.add_argument("--expect-sources", type=nonnegative_integer)
    parser.add_argument("--expect-permutations", type=nonnegative_integer)
    args = parser.parse_args()
    explicit = {
        field: getattr(args, "expect_" + field)
        for field in COUNTS if field != "rejected" and getattr(args, "expect_" + field) is not None
    }
    try:
        report = summarize(read_samples(args.csv), explicit)
        report["input"] = str(args.csv)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print_report(report)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))
    return 0 if report["integrity"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
