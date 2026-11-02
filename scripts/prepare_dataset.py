#!/usr/bin/env python3
"""Mix the three dataset classes and split them into train / validation / test files.

    python scripts/prepare_dataset.py [--seed 0] [--val 0.1] [--test 0.1]

Exact duplicate lines are dropped first, so no position can leak from train to test.
The output directory (default: data/) is git-ignored: the files are too heavy to be pushed.
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from fen_parser import LABELS

DATASET_GLOB = "*/*.txt"


def read_unique_lines(dataset_dir):
    unique = {}  # a dict keeps the order and drops exact duplicates
    total = 0
    for path in sorted(Path(dataset_dir).glob(DATASET_GLOB)):
        with open(path, encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    total += 1
                    unique[line] = None
    return list(unique), total


def label_of(line):
    return next((field for field in line.split()[1:] if field in LABELS), "?")


def write_split(path, lines):
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(lines) + "\n")
    counts = Counter(label_of(line) for line in lines)
    summary = ", ".join(f"{label}: {counts[label]}" for label in LABELS)
    print(f"  {path.name:<16} {len(lines):>8} lines  ({summary})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default=str(ROOT / "dataset"))
    parser.add_argument("--out-dir", default=str(ROOT / "data"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val", type=float, default=0.1, help="validation ratio")
    parser.add_argument("--test", type=float, default=0.1, help="test ratio")
    args = parser.parse_args()
    if not (args.val >= 0 and args.test >= 0 and args.val + args.test < 1):
        parser.error("--val and --test must be >= 0 and sum to less than 1")

    lines, total = read_unique_lines(args.dataset_dir)
    if not lines:
        sys.exit(f"no dataset file found in {args.dataset_dir}/{DATASET_GLOB}")
    print(f"{total} lines read, {total - len(lines)} duplicates dropped, {len(lines)} kept")

    random.Random(args.seed).shuffle(lines)
    val_end = int(len(lines) * args.val)
    test_end = val_end + int(len(lines) * args.test)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"written in {out_dir}:")
    write_split(out_dir / "validation.txt", lines[:val_end])
    write_split(out_dir / "test.txt", lines[val_end:test_end])
    write_split(out_dir / "train.txt", lines[test_end:])


if __name__ == "__main__":
    main()
