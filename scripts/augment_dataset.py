#!/usr/bin/env python3
"""Augment chess dataset by vertically flipping the board and swapping colors."""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fen_parser import LABELS, flip_fen_line


def label_of(line: str) -> str:
    return next((field for field in line.split()[1:] if field in LABELS), "?")


def augment_lines(lines: list[str], mode: str = "combine", dedup: bool = False, shuffle: bool = False, seed: int = 42) -> tuple[list[str], int]:
    augmented: list[str] = []
    seen: set[str] = set()

    def add(l: str):
        if not l:
            return
        if dedup:
            if l not in seen:
                seen.add(l)
                augmented.append(l)
        else:
            augmented.append(l)

    total_flipped = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        flipped = flip_fen_line(stripped)
        total_flipped += 1

        if mode == "combine":
            add(stripped)
            add(flipped)
        else:
            add(flipped)

    if shuffle:
        random.Random(seed).shuffle(augmented)

    return augmented, total_flipped


def process_file(in_path: Path, out_path: Path, mode: str, dedup: bool, shuffle: bool, seed: int):
    with open(in_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    result, flipped_count = augment_lines(lines, mode=mode, dedup=dedup, shuffle=shuffle, seed=seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(result) + "\n")

    counts = Counter(label_of(l) for l in result)
    summary = ", ".join(f"{label}: {counts[label]}" for label in LABELS)
    print(f"  {in_path.name} -> {out_path.name}: {len(lines)} input lines, {flipped_count} flipped -> {len(result)} output lines ({summary})")


def process_directory(in_dir: Path, out_dir: Path, mode: str, dedup: bool, shuffle: bool, seed: int):
    files = sorted(in_dir.rglob("*.txt"))
    if not files:
        sys.exit(f"No .txt files found in {in_dir}")

    print(f"Processing {len(files)} files from {in_dir} into {out_dir} (mode={mode})...")
    for in_file in files:
        rel = in_file.relative_to(in_dir)
        target = out_dir / rel
        process_file(in_file, target, mode, dedup, shuffle, seed)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pos_input", nargs="?", default=None, help="Input file or directory (positional alternative to -i)")
    parser.add_argument("pos_output", nargs="?", default=None, help="Output file or directory (positional alternative to -o)")
    parser.add_argument("-i", "--input", dest="opt_input", default=None, help="Input dataset file or directory")
    parser.add_argument("-o", "--output", dest="opt_output", default=None, help="Output destination file or directory")
    parser.add_argument("--mode", choices=["combine", "flipped_only"], default="combine", help="Augmentation mode: combine (original + flipped) or flipped_only")
    parser.add_argument("--dedup", action="store_true", help="Remove duplicate lines in the output")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset rows randomly")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random shuffling")
    args = parser.parse_args()

    input_target = args.opt_input or args.pos_input
    output_target = args.opt_output or args.pos_output

    if not input_target:
        parser.error("An input file or directory must be specified.")

    return Path(input_target), Path(output_target) if output_target else None, args.mode, args.dedup, args.shuffle, args.seed


def main():
    in_path, out_path, mode, dedup, shuffle, seed = parse_args()

    if not in_path.exists():
        sys.exit(f"Input path does not exist: {in_path}")

    if in_path.is_file():
        if out_path is None:
            out_path = in_path.parent / f"{in_path.stem}_augmented{in_path.suffix}"
        process_file(in_path, out_path, mode, dedup, shuffle, seed)
    else:
        if out_path is None:
            out_path = in_path.parent / f"{in_path.name}_augmented"
        process_directory(in_path, out_path, mode, dedup, shuffle, seed)


if __name__ == "__main__":
    main()

