"""Lightweight regression tests for preload benchmark analysis (standard library only)."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("analyze_preload_benchmark.py")
HEADER = (
    "threads,run,file_bytes,sources,permutations,rejected,workers,"
    "load_ms,read_ms,validate_ms,index_ms\n"
)
ROWS = (
    "1,0,1000,3,4,0,1,12,8,3,1\n"
    "2,0,1000,3,4,0,2,9,6,2,1\n"
    "2,1,1000,3,4,0,2,11,8,2,1\n"
    "1,1,1000,3,4,0,1,18,10,6,2\n"
    "1,2,1000,3,4,0,1,15,9,4,2\n"
)


class PreloadBenchmarkTests(unittest.TestCase):
    def run_analysis(self, csv_text=HEADER + ROWS, encoding="utf-8", extra=()):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "benchmark.csv"
            output = Path(directory) / "analysis.json"
            source.write_text(csv_text, encoding=encoding)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(source), "--output", str(output), *extra],
                capture_output=True,
                text=True,
                check=False,
            )
            report = json.loads(output.read_text(encoding="utf-8")) if output.exists() else None
            return result, report

    def test_interleaved_encodings_and_hand_checked_statistics(self):
        # A parser that treats UTF16 as UTF8, or groups consecutive rows instead of worker
        # configurations, loses samples or produces the wrong median.
        for encoding in ("utf-8", "utf-8-sig", "utf-16"):
            with self.subTest(encoding=encoding):
                result, report = self.run_analysis(encoding=encoding)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(report["integrity"]["ok"])
                one, two = report["groups"]
                self.assertEqual((one["threads"], one["workers"], one["samples"]), (1, [1], 3))
                self.assertEqual(one["load_ms"], {"median": 15.0, "min": 12.0, "max": 18.0})
                self.assertEqual(one["read_ms"]["median"], 9.0)
                self.assertEqual(one["validate_ms"]["median"], 4.0)
                self.assertEqual(one["index_ms"]["median"], 2.0)
                self.assertEqual((two["threads"], two["samples"]), (2, 2))
                self.assertEqual(two["load_ms"], {"median": 10.0, "min": 9.0, "max": 11.0})
                self.assertIn("cold storage", result.stdout)
                self.assertIn("compatibility", result.stdout)

    def test_big_endian_utf16_bom(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "benchmark.csv"
            source.write_bytes(b"\xfe\xff" + (HEADER + ROWS).encode("utf-16-be"))
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(source)], capture_output=True, text=True, check=False
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_known_expected_counts_reject_consistently_wrong_samples(self):
        result, report = self.run_analysis(extra=("--expect-sources", "1038"))
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(report["integrity"]["ok"])
        self.assertEqual(report["integrity"]["expected"]["sources"], 1038)
        self.assertTrue(any("sources" in error for error in report["integrity"]["errors"]))

    def test_every_count_mismatch_and_nonzero_rejection_fails_integrity(self):
        for old, new, field in (
            ("2,0,1000,3,4,0,2", "2,0,999,3,4,0,2", "file_bytes"),
            ("2,0,1000,3,4,0,2", "2,0,1000,2,4,0,2", "sources"),
            ("2,0,1000,3,4,0,2", "2,0,1000,3,3,0,2", "permutations"),
            ("2,0,1000,3,4,0,2", "2,0,1000,3,4,1,2", "rejected"),
        ):
            with self.subTest(field=field):
                result, report = self.run_analysis(csv_text=HEADER + ROWS.replace(old, new))
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertFalse(report["integrity"]["ok"])
                self.assertTrue(any(field in error for error in report["integrity"]["errors"]))

    def test_actual_workers_are_reported_separately_from_requested(self):
        # A resource-limited loader can create fewer helpers without changing cached records.
        result, report = self.run_analysis(csv_text=HEADER + ROWS.replace(
            "2,1,1000,3,4,0,2", "2,1,1000,3,4,0,1"
        ))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["groups"][1]["workers"], [1, 2])

    def test_malformed_or_nonfinite_input_is_not_reported_as_success(self):
        for invalid in (
            "", HEADER, HEADER.replace("index_ms", "unknown") + ROWS,
            HEADER + ROWS.replace("12,8,3,1", "nan,8,3,1"),
            HEADER + ROWS.replace("12,8,3,1", "12,-8,3,1"),
            HEADER + ROWS.replace("1000,3,4", "oops,3,4"),
            HEADER + ROWS.replace("12,8,3,1", "12,8,3"),
        ):
            with self.subTest(invalid=invalid):
                result, report = self.run_analysis(csv_text=invalid)
                self.assertEqual(result.returncode, 2)
                self.assertIsNone(report)
                self.assertIn("error", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
