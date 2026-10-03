import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("analyze_bda_sync.py")
SPEC = importlib.util.spec_from_file_location("analyze_bda_sync", MODULE_PATH)
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


class BdaAnalysisTests(unittest.TestCase):
    def write(self, directory, name, header, rows, terminated=True):
        text = ",".join(header) + "\n"
        text += "\n".join(",".join(str(row.get(column, 0)) for column in header) for row in rows)
        if terminated:
            text += "\n"
        path = Path(directory) / name
        path.write_text(text, encoding="utf-8-sig")
        return path

    def test_rates_use_elapsed_time_and_distinguish_passes_from_full_scans(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, "sync.csv", analyzer.SYNC_FIELDS, [
                dict(begin_ms=1000, end_ms=2000, calls=100, passes=70,
                     epoch_skips=20, submission_skips=10, full_scans=5,
                     hot_passes=60, incremental_skips=5, structure_changed=30,
                     registers=10, unregisters=8),
                dict(begin_ms=2000, end_ms=5000, calls=300, passes=150,
                     epoch_skips=100, submission_skips=50, full_scans=15,
                     hot_passes=100, dirty_log_passes=35, structure_changed=70),
            ])
            result = analyzer.analyze_sync(path)
            self.assertEqual(result["observed_s"], 4)
            self.assertEqual(result["rates_per_s"]["calls"], 100)
            self.assertEqual(result["rates_per_s"]["full_scans"], 5)
            self.assertEqual(result["skip_fraction"], 0.45)
            self.assertEqual(result["structure_changed_fraction"], 0.25)
            self.assertAlmostEqual(result["full_scan_fraction_of_passes"], 20 / 220)

    def test_cpu_average_weights_calls_instead_of_averaging_row_means(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, "cpu.csv", analyzer.CPU_FIELDS, [
                dict(begin_ms=1000, end_ms=2000, calls=1, total_ns=100_000, max_ns=100_000),
                dict(begin_ms=2000, end_ms=5000, calls=99, total_ns=900_000, max_ns=20_000),
            ])
            result = analyzer.analyze_cpu(path)
            self.assertEqual(result["mean_us_per_call"], 10)
            self.assertEqual(result["cpu_ms_per_s"], 0.25)
            self.assertEqual(result["max_us_per_call"], 100)

    def test_window_keeps_whole_intervals_without_prorating_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, "cpu.csv", analyzer.CPU_FIELDS, [
                dict(begin_ms=1000, end_ms=2000, calls=100, total_ns=100_000, max_ns=1000),
                dict(begin_ms=2000, end_ms=5000, calls=3, total_ns=9000, max_ns=3000),
            ])
            result = analyzer.analyze_cpu(path, start_ms=1500, end_ms=5000)
            self.assertEqual(result["totals"]["calls"], 3)
            self.assertEqual(result["observed_s"], 3)

    def test_live_partial_tail_is_excluded_even_when_last_field_looks_numeric(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, "sync.csv", analyzer.SYNC_FIELDS, [
                dict(begin_ms=0, end_ms=1000, calls=10),
                dict(begin_ms=1000, end_ms=2000, calls=999),
            ], terminated=False)
            result = analyzer.analyze_sync(path)
            self.assertEqual(result["totals"]["calls"], 10)
            self.assertTrue(result["source"]["unterminated_tail_excluded"])

    def test_missing_columns_are_errors_instead_of_zeros(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, "sync.csv", ["begin_ms", "end_ms", "calls"],
                              [dict(begin_ms=0, end_ms=1000, calls=5)])
            with self.assertRaisesRegex(ValueError, "missing"):
                analyzer.analyze_sync(path)

    def test_clock_reset_or_overlapping_intervals_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, "cpu.csv", analyzer.CPU_FIELDS, [
                dict(begin_ms=1000, end_ms=2000), dict(begin_ms=0, end_ms=1000),
            ])
            with self.assertRaisesRegex(ValueError, "overlap|reset"):
                analyzer.analyze_cpu(path)

    def test_nonfinite_negative_and_fractional_counters_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for field, value in (("begin_ms", "nan"), ("calls", -1), ("calls", 1.5),
                                 ("calls", 2**64), ("calls", 10**400)):
                row = dict(begin_ms=0, end_ms=1000, calls=5)
                row[field] = value
                path = self.write(directory, "sync.csv", analyzer.SYNC_FIELDS, [row])
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    analyzer.analyze_sync(path)

    def test_no_calls_does_not_claim_zero_microseconds_per_call(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, "cpu.csv", analyzer.CPU_FIELDS,
                              [dict(begin_ms=0, end_ms=1000)])
            self.assertIsNone(analyzer.analyze_cpu(path)["mean_us_per_call"])

    def test_path_cpu_times_distinguish_full_scans_from_dirty_log(self):
        with tempfile.TemporaryDirectory() as directory:
            header = (*analyzer.SYNC_FIELDS, "full_scan_ns", "dirty_log_ns",
                      "hot_range_runs_checked", "hot_range_runs_merged")
            path = self.write(directory, "sync.csv", header, [
                dict(begin_ms=0, end_ms=3000, calls=9, passes=6, full_scans=2,
                     dirty_log_passes=4, full_scan_ns=6_000_000, dirty_log_ns=2_000_000,
                     hot_range_runs_checked=30, hot_range_runs_merged=10),
            ])
            result = analyzer.analyze_sync(path)
            self.assertEqual(result["path_cpu"]["full_scan"]["mean_us_per_pass"], 3000)
            self.assertEqual(result["path_cpu"]["dirty_log"]["mean_us_per_pass"], 500)
            self.assertEqual(result["path_cpu"]["full_scan"]["cpu_ms_per_s"], 2)
            self.assertIsNone(result["path_cpu"]["hot_pass"])
            self.assertEqual(result["totals"]["hot_range_runs_merged"], 10)

    def test_legacy_csv_does_not_invent_zero_path_durations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, "sync.csv", analyzer.SYNC_FIELDS,
                              [dict(begin_ms=0, end_ms=1000, calls=20)])
            self.assertIsNone(analyzer.analyze_sync(path)["path_cpu"])

    def test_new_buffer_work_is_measured_even_when_the_main_pass_skips(self):
        with tempfile.TemporaryDirectory() as directory:
            header = (*analyzer.SYNC_FIELDS, "new_buffer_passes", "new_buffer_ns")
            path = self.write(directory, "sync.csv", header, [
                dict(begin_ms=0, end_ms=2000, calls=10, epoch_skips=10,
                     new_buffer_passes=4, new_buffer_ns=8_000_000),
            ])
            result = analyzer.analyze_sync(path)
            self.assertEqual(result["skip_fraction"], 1)
            self.assertEqual(result["totals"]["passes"], 0)
            self.assertEqual(result["path_cpu"]["new_buffer"]["mean_us_per_pass"], 2000)
            self.assertEqual(result["path_cpu"]["new_buffer"]["cpu_ms_per_s"], 4)

    def test_new_buffer_duration_without_count_does_not_invent_a_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            header = (*analyzer.SYNC_FIELDS, "new_buffer_ns")
            path = self.write(directory, "sync.csv", header,
                              [dict(begin_ms=0, end_ms=1000, new_buffer_ns=1_000_000)])
            result = analyzer.analyze_sync(path)
            self.assertEqual(result["path_cpu"]["new_buffer"]["total_ms"], 1)
            self.assertIsNone(result["path_cpu"]["new_buffer"]["mean_us_per_pass"])


if __name__ == "__main__":
    unittest.main()
