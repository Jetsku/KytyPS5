"""Compare bench_boot capture manifests using only Python's standard library.

Usage: python tools/compare_captures.py captures-drain-1.json captures-gpu-1.json
Exit codes: 0 within thresholds, 1 visual difference (review), 2 invalid evidence.
Elapsed-time pairs do NOT imply deterministic game-state/frame alignment. Even
an identical sampled image cannot prove an object never disappears between samples.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import struct
import sys


def read_bmp(path):
    data = path.read_bytes()
    if len(data) < 54 or data[:2] != b'BM':
        raise ValueError('not a BMP image')
    offset = struct.unpack_from('<I', data, 10)[0]
    header, width, height, planes, bits, compression = struct.unpack_from('<IiiHHI', data, 14)
    if header < 40 or offset < 14 + header or width <= 0 or not height or planes != 1 or bits != 24 or compression:
        raise ValueError('expected uncompressed 24-bit BMP')
    stride = (width * 3 + 3) & ~3
    if offset + stride * abs(height) > len(data):
        raise ValueError('truncated BMP pixels')
    rows = range(abs(height)) if height < 0 else reversed(range(height))
    pixels = b''.join(data[offset + y * stride:offset + y * stride + width * 3] for y in rows)
    # Conservative rejection: nearly black frames are not useful visual evidence,
    # whether caused by startup, a dark scene or unsupported Vulkan PrintWindow.
    dark = sum(max(pixels[i:i + 3]) <= 4 for i in range(0, len(pixels), 3))
    if dark / (width * abs(height)) >= 0.999:
        raise ValueError('black/near-black image; capture or scene is inconclusive')
    return width, abs(height), pixels


def load_manifest(path):
    manifest = json.loads(path.read_text(encoding='utf-8-sig'))
    if manifest.get('alignment') != 'elapsed-time':
        raise ValueError('manifest must declare elapsed-time alignment')
    records = {}
    for row in manifest['captures']:
        second = float(row['requested_seconds'])
        if not math.isfinite(second) or second < 0 or second in records:
            raise ValueError('invalid or duplicate scheduled capture time')
        records[second] = row
    if not records:
        raise ValueError('empty capture schedule')
    return records


def capture_image(manifest, row, max_time_error):
    if row['status'] != 'ok':
        raise ValueError('capture status: ' + str(row['status']))
    requested = float(row['requested_seconds'])
    actual = float(row['actual_seconds'])
    completed = float(row['completed_seconds'])
    if (not all(map(math.isfinite, (actual, completed))) or actual < requested or
            completed < actual or completed - requested > max_time_error):
        raise ValueError('capture outside elapsed-time tolerance')
    return read_bmp(manifest.parent / row['image'])


def compare_pixels(left, right, pixel_tolerance):
    if left[:2] != right[:2]:
        raise ValueError('image dimensions differ')
    a, b = left[2], right[2]
    differences = bytes(abs(x - y) for x, y in zip(a, b))
    changed = sum(max(differences[i:i + 3]) > pixel_tolerance for i in range(0, len(a), 3))
    # Per-channel 16-bin histograms, normalized total variation (0..1).
    hist_a = Counter((i % 3, value // 16) for i, value in enumerate(a))
    hist_b = Counter((i % 3, value // 16) for i, value in enumerate(b))
    histogram = sum(abs(hist_a[k] - hist_b[k]) for k in hist_a.keys() | hist_b.keys()) / (2 * len(a))
    return dict(width=left[0], height=left[1], mean_absolute_error=sum(differences) / len(a),
                changed_fraction=changed / (len(a) // 3), histogram_distance=histogram)


def finite_range(low, high):
    def parse(text):
        value = float(text)
        if not math.isfinite(value) or not low <= value <= high:
            raise argparse.ArgumentTypeError(f'expected finite value between {low} and {high}')
        return value
    return parse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--pixel-tolerance', type=finite_range(0, 255), default=8)
    parser.add_argument('--max-changed-fraction', type=finite_range(0, 1), default=0.01)
    parser.add_argument('--max-mean-error', type=finite_range(0, 255), default=2)
    parser.add_argument('--max-histogram-distance', type=finite_range(0, 1), default=0.02)
    parser.add_argument('--max-time-error', type=finite_range(0, 3600), default=2)
    args = parser.parse_args()
    report = dict(alignment='elapsed-time',
                  note='Elapsed time from process start; not deterministic game-state alignment. Review paired BMPs.',
                  thresholds={key: value for key, value in vars(args).items() if key not in ('left', 'right')},
                  comparisons=[])
    code = 0
    try:
        left, right = load_manifest(args.left), load_manifest(args.right)
        for second in sorted(left.keys() | right.keys()):
            result = dict(requested_seconds=second)
            report['comparisons'].append(result)
            try:
                a, b = left[second], right[second]
                result.update(left_actual_seconds=a.get('actual_seconds'), right_actual_seconds=b.get('actual_seconds'))
                result.update(compare_pixels(capture_image(args.left, a, args.max_time_error),
                                             capture_image(args.right, b, args.max_time_error), args.pixel_tolerance))
                different = (result['changed_fraction'] > args.max_changed_fraction or
                             result['mean_absolute_error'] > args.max_mean_error or
                             result['histogram_distance'] > args.max_histogram_distance)
                result['status'] = 'different' if different else 'within-thresholds'
                code = max(code, int(different))
            except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
                result.update(status='invalid', error=str(error))
                code = 2
    except (OSError, ValueError, KeyError, TypeError) as error:
        report.update(status='invalid', error=str(error))
        code = 2
    # Report malformed NaN timing values as null, never nonstandard JSON numbers.
    for row in report['comparisons']:
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                row[key] = None
    print(json.dumps(report, indent=2, allow_nan=False))
    return code


if __name__ == '__main__':
    sys.exit(main())
