#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from aperiodic_eeg.sleep_edf import (  # noqa: E402
    discover_recording_pairs,
    find_missing_pairs,
    find_partial_downloads,
    total_size_bytes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a staged Sleep-EDF subset.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/sleep-edf/sleep-cassette"),
        help="Directory containing Sleep-EDF Sleep-Cassette EDF files.",
    )
    parser.add_argument(
        "--expected-pairs",
        type=int,
        default=None,
        help="Expected number of PSG/Hypnogram recording pairs. Omit to only report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root

    if not data_root.exists():
        print(f"ERROR: data root does not exist: {data_root}", file=sys.stderr)
        return 2

    pairs = discover_recording_pairs(data_root)
    missing = find_missing_pairs(data_root)
    partials = find_partial_downloads(data_root)
    all_edf_paths = sorted(data_root.glob("*.edf"))

    print(f"Data root: {data_root}")
    print(f"EDF files: {len(all_edf_paths)}")
    print(f"Recording pairs: {len(pairs)}")
    print(f"Total EDF size: {total_size_bytes(all_edf_paths) / (1024 ** 2):.1f} MiB")

    if pairs:
        print(f"First pair: {pairs[0].key}")
        print(f"Last pair: {pairs[-1].key}")

    if missing["missing_hypnograms"]:
        print(f"Missing hypnograms for: {missing['missing_hypnograms']}", file=sys.stderr)
    if missing["missing_psg"]:
        print(f"Missing PSG files for: {missing['missing_psg']}", file=sys.stderr)
    if partials:
        print(f"Partial aria2 downloads remain: {[p.name for p in partials]}", file=sys.stderr)

    ok = True
    if args.expected_pairs is not None:
        ok = ok and len(pairs) == args.expected_pairs
    ok = ok and not missing["missing_hypnograms"]
    ok = ok and not missing["missing_psg"]
    ok = ok and not partials

    if ok:
        print("Status: OK")
        return 0

    print("Status: FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
