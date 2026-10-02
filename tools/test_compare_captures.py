"""Synthetic BMP/manifest regressions; no desktop or emulator access."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


def bmp(path, pixels, width=2, top_down=False):
    height = len(pixels) // width
    stride = (width * 3 + 3) & ~3
    rows = [pixels[y * width:(y + 1) * width] for y in range(height)]
    if not top_down:
        rows.reverse()
    body = b"".join(b"".join(bytes(p[::-1]) for p in row) + bytes(stride - width * 3) for row in rows)
    path.write_bytes(struct.pack('<2sIHHI', b'BM', 54 + len(body), 0, 0, 54) +
                     struct.pack('<IiiHHIIiiII', 40, width, -height if top_down else height,
                                 1, 24, 0, len(body), 0, 0, 0, 0) + body)


class CompareCapturesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pixels = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
        for name in ('a', 'b'):
            bmp(self.root / (name + '.bmp'), self.pixels, top_down=name == 'b')
            self.manifest(name)

    def manifest(self, name, **overrides):
        row = dict(requested_seconds=10, actual_seconds=10.25, completed_seconds=10.3,
                   status='ok', image=name + '.bmp')
        row.update(overrides)
        (self.root / (name + '.json')).write_text(json.dumps(dict(
            alignment='elapsed-time', run=name, captures=[row])))

    def compare(self, expected_code):
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('compare_captures.py')),
                                 str(self.root / 'a.json'), str(self.root / 'b.json')],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, expected_code, result.stdout + result.stderr)
        self.assertTrue(result.stdout.startswith('{'), result.stdout + result.stderr)
        return json.loads(result.stdout)

    def test_equal_pixels_with_row_padding_and_opposite_bmp_orientation(self):
        report = self.compare(0)
        self.assertEqual(report['comparisons'][0]['changed_fraction'], 0)
        self.assertEqual(report['comparisons'][0]['histogram_distance'], 0)

    def test_pixel_change_detected_even_with_identical_histogram(self):
        bmp(self.root / 'b.bmp', list(reversed(self.pixels)))
        row = self.compare(1)['comparisons'][0]
        self.assertEqual(row['changed_fraction'], 1)
        self.assertEqual(row['histogram_distance'], 0)
        self.assertAlmostEqual(row['mean_absolute_error'], 170)

    def test_missing_image_is_invalid(self):
        (self.root / 'b.bmp').unlink()
        self.assertEqual(self.compare(2)['comparisons'][0]['status'], 'invalid')

    def test_black_pair_is_invalid(self):
        for name in ('a', 'b'):
            bmp(self.root / (name + '.bmp'), [(0, 0, 0)] * 4)
        self.compare(2)

    def test_failed_capture_cannot_use_stale_image(self):
        self.manifest('b', status='timeout')
        self.compare(2)

    def test_late_or_long_capture_is_invalid(self):
        for overrides in ({'actual_seconds': 15, 'completed_seconds': 15.1},
                          {'completed_seconds': 15.0}, {'actual_seconds': float('nan')}):
            with self.subTest(overrides=overrides):
                self.manifest('b', **overrides)
                self.compare(2)

    def test_missing_schedule_entry_is_invalid(self):
        self.manifest('b', requested_seconds=20)
        self.assertEqual(len(self.compare(2)['comparisons']), 2)

    def test_empty_schedule_is_invalid(self):
        (self.root / 'b.json').write_text('{"alignment":"elapsed-time","captures":[]}')
        self.compare(2)

    def test_truncated_bmp_is_invalid(self):
        path = self.root / 'b.bmp'
        path.write_bytes(path.read_bytes()[:-1])
        self.compare(2)

    def test_size_mismatch_is_invalid(self):
        bmp(self.root / 'b.bmp', self.pixels, width=1)
        self.compare(2)

    def test_color_change_changes_histogram(self):
        bmp(self.root / 'b.bmp', [(255, 0, 0)] * 4)
        row = self.compare(1)['comparisons'][0]
        self.assertEqual(row['changed_fraction'], 0.75)
        self.assertEqual(row['histogram_distance'], 0.5)

    def test_near_black_is_invalid(self):
        bmp(self.root / 'b.bmp', [(4, 4, 4)] * 4)
        self.compare(2)

    def test_duplicate_schedule_is_invalid(self):
        path = self.root / 'b.json'
        manifest = json.loads(path.read_text())
        manifest['captures'] *= 2
        path.write_text(json.dumps(manifest))
        self.compare(2)

    def test_small_pixel_variations_fit_default_thresholds(self):
        bmp(self.root / 'b.bmp', [(254, 1, 1), (1, 254, 1), (1, 1, 254), (254, 254, 254)])
        self.assertEqual(self.compare(0)['comparisons'][0]['mean_absolute_error'], 1)


if __name__ == '__main__':
    unittest.main()
