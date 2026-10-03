"""Analyze opt-in BDA counters and PrepareBda CPU timing; standard library only.

python tools/analyze_bda_sync.py sync.csv --cpu cpu.csv --start-ms 60000 --output report.json
Files are read once. Keep each run in its own directory; no game input is generated.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

SYNC_COUNTERS = (
    "registers", "unregisters", "maps", "unmaps", "calls", "passes", "epoch_skips",
    "submission_skips", "epoch_changed", "submission_changed", "structure_changed",
    "full_scans", "hot_passes", "dirty_log_passes", "incremental_skips",
)
SYNC_FIELDS = ("begin_ms", "end_ms", *SYNC_COUNTERS)
CPU_FIELDS = ("begin_ms", "end_ms", "calls", "total_ns", "max_ns")
PATH_COUNTERS = ("full_scan_ns", "hot_pass_ns", "dirty_log_ns",
                 "hot_range_runs_checked", "hot_range_runs_merged", "structure_log_passes",
                 "new_buffer_passes", "new_buffer_ns")


def read_intervals(path, fields, start_ms, end_ms, optional=()):
    if not math.isfinite(start_ms) or start_ms < 0:
        raise ValueError("start_ms must be finite and nonnegative")
    if end_ms is not None and (not math.isfinite(end_ms) or end_ms <= start_ms):
        raise ValueError("end_ms must be finite and greater than start_ms")
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8-sig")
    tail_excluded = bool(text) and not text.endswith("\n")
    if tail_excluded:
        text = text[:text.rfind("\n") + 1]
    reader = csv.DictReader(io.StringIO(text))
    header = reader.fieldnames or []
    missing = set(fields).difference(header)
    if missing:
        raise ValueError("missing CSV columns: " + ", ".join(sorted(missing)))
    if len(header) != len(set(header)):
        raise ValueError("duplicate CSV columns")
    optional = tuple(field for field in optional if field in header)
    fields = (*fields, *optional)
    rows = []
    previous_end = None
    complete_rows = 0
    for line, source in enumerate(reader, 2):
        if None in source or any(source.get(field) is None for field in fields):
            raise ValueError(f"row {line}: incorrect number of CSV fields")
        row = {}
        for field in fields:
            try:
                number = float(source[field]) if field.endswith("_ms") else int(source[field])
            except ValueError:
                raise ValueError(f"row {line}: invalid {field}") from None
            if number < 0 or (field.endswith("_ms") and not math.isfinite(number)):
                raise ValueError(f"row {line}: invalid {field}")
            if not field.endswith("_ms") and number > 2**64 - 1:
                raise ValueError(f"row {line}: {field} exceeds uint64")
            row[field] = number
        if row["end_ms"] <= row["begin_ms"]:
            raise ValueError(f"row {line}: interval must have positive duration")
        if previous_end is not None and row["begin_ms"] < previous_end:
            raise ValueError(f"row {line}: overlapping intervals or clock reset")
        previous_end = row["end_ms"]
        complete_rows += 1
        if row["begin_ms"] >= start_ms and (end_ms is None or row["end_ms"] <= end_ms):
            rows.append(row)
    return rows, {
        "path": str(Path(path).resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw), "complete_rows": complete_rows,
        "unterminated_tail_excluded": tail_excluded,
        "optional_columns": list(optional),
    }


def aggregate(rows, source, counters):
    seconds = sum(row["end_ms"] - row["begin_ms"] for row in rows) / 1000
    totals = {field: sum(row[field] for row in rows) for field in counters}
    return {
        "source": source, "intervals": len(rows), "observed_s": seconds,
        "first_begin_ms": rows[0]["begin_ms"] if rows else None,
        "last_end_ms": rows[-1]["end_ms"] if rows else None,
        "totals": totals,
        "rates_per_s": {field: value / seconds if seconds else None
                        for field, value in totals.items()},
    }


def fraction(value, denominator):
    return value / denominator if denominator else None


def analyze_sync(path, start_ms=0, end_ms=None):
    rows, source = read_intervals(path, SYNC_FIELDS, start_ms, end_ms, PATH_COUNTERS)
    result = aggregate(rows, source, (*SYNC_COUNTERS, *source["optional_columns"]))
    totals = result["totals"]
    result.update({
        "skip_fraction": fraction(totals["epoch_skips"] + totals["submission_skips"], totals["calls"]),
        "structure_changed_fraction": fraction(totals["structure_changed"], totals["calls"]),
        "full_scan_fraction_of_passes": fraction(totals["full_scans"], totals["passes"]),
    })
    path_cpu = {}
    for name, duration, calls in (("full_scan", "full_scan_ns", "full_scans"),
                                  ("hot_pass", "hot_pass_ns", "hot_passes"),
                                  ("dirty_log", "dirty_log_ns", "dirty_log_passes"),
                                  ("new_buffer", "new_buffer_ns", "new_buffer_passes")):
        path_cpu[name] = None if duration not in totals else {
            "total_ms": totals[duration] / 1e6,
            "cpu_ms_per_s": fraction(totals[duration] / 1e6, result["observed_s"]),
            "mean_us_per_pass": fraction(totals[duration] / 1000, totals.get(calls, 0)),
        }
    result["path_cpu"] = path_cpu if any(value is not None for value in path_cpu.values()) else None
    return result


def analyze_cpu(path, start_ms=0, end_ms=None):
    rows, source = read_intervals(path, CPU_FIELDS, start_ms, end_ms)
    for row in rows:
        if row["max_ns"] > row["total_ns"] or (row["calls"] == 0 and row["total_ns"] != 0):
            raise ValueError("inconsistent CPU timing counters")
    result = aggregate(rows, source, ("calls", "total_ns"))
    totals = result["totals"]
    result.update({
        "mean_us_per_call": fraction(totals["total_ns"] / 1000, totals["calls"]),
        "max_us_per_call": max((row["max_ns"] for row in rows), default=0) / 1000 if totals["calls"] else None,
        "cpu_ms_per_s": fraction(totals["total_ns"] / 1e6, result["observed_s"]),
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sync", type=Path)
    parser.add_argument("--cpu", type=Path)
    parser.add_argument("--start-ms", type=float, default=0)
    parser.add_argument("--end-ms", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = {
            "sync": analyze_sync(args.sync, args.start_ms, args.end_ms),
            "notes": [
                "Whole reporting intervals only; elapsed time does not identify a game phase.",
                "Epoch/submission/structure mismatches overlap; do not add them as disjoint causes.",
                "Cross-thread counters may straddle intervals; use totals over a longer window.",
                "Passes include hot/dirty-log/empty incremental passes, not just full scans.",
                "New-buffer work precedes the main pass/skip decision; report it separately from passes.",
                "PrepareBda is CPU work, not GPU time or the cost of every draw.",
                "Instrumentation adds overhead. This report does not demonstrate an FPS gain.",
                "CPU and sync files are independent samples; confirm they belong to the same run.",
                "Path durations exclude epoch checks and dirty-log snapshot acquisition; they are not all of PrepareBda.",
            ],
        }
        if args.cpu:
            result["cpu"] = analyze_cpu(args.cpu, args.start_ms, args.end_ms)
        rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            # Refuse to overwrite an earlier report or an input CSV.
            with args.output.open("x", encoding="utf-8") as output:
                output.write(rendered)
        print(rendered, end="")
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
