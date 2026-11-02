#!/usr/bin/env python3
"""Evaluate a network on a labelled chess file, through `my_torch_analyzer --predict`.

    python scripts/evaluate.py NETWORK CHESSFILE [--png confusion.png]

Prints the accuracy, the per-class recall and the confusion matrix (rows: expected, columns: predicted).
When the analyzer outputs colours ("Check White"), the exact-match accuracy is printed as well.
"""
import argparse
import subprocess
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from fen_parser import CHECKING_COLOR, LABEL_INDEX, LABELS

COLORS = tuple(CHECKING_COLOR.values())
CELL_WIDTH = 11


def read_expected(path):
    expected = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            fields = line.split()
            if not fields:
                continue
            suffix = [field for field in fields[1:] if field in LABELS or field in COLORS]
            if not suffix or suffix[0] not in LABELS:
                sys.exit(f"{path}: missing label in line: {line.strip()}")
            expected.append(" ".join(suffix))
    return expected


def predict(network, chessfile):
    result = subprocess.run([sys.executable, str(ROOT / "my_torch_analyzer"), "--predict", network, chessfile],
                            capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        sys.exit(result.returncode)
    return result.stdout.splitlines()


def confusion_matrix(expected, predicted):
    matrix = [[0] * len(LABELS) for _ in LABELS]
    for expected_output, predicted_output in zip(expected, predicted):
        matrix[LABEL_INDEX[expected_output.split()[0]]][LABEL_INDEX[predicted_output.split()[0]]] += 1
    return matrix


def print_report(matrix, expected, predicted):
    total = len(expected)
    correct = sum(row[i] for i, row in enumerate(matrix))
    print(f"positions: {total}")
    print(f"accuracy (label only): {correct / total:.4f}")
    if any(" " in output for output in predicted):
        exact = sum(got == want for got, want in zip(predicted, expected))
        print(f"accuracy (label + colour): {exact / total:.4f}")

    print("\nconfusion matrix (rows: expected, columns: predicted)")
    print(" " * CELL_WIDTH + "".join(f"{label:>{CELL_WIDTH}}" for label in LABELS) + "   recall")
    for i, (label, row) in enumerate(zip(LABELS, matrix)):
        row_total = sum(row)
        recall = row[i] / row_total if row_total else float("nan")
        print(f"{label:<{CELL_WIDTH}}" + "".join(f"{count:>{CELL_WIDTH}}" for count in row) + f"   {recall:.3f}")


def save_confusion_png(matrix, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap="Blues")
    ticks = range(len(LABELS))
    ax.set_xticks(ticks, LABELS)
    ax.set_yticks(ticks, LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("expected")
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            ax.text(j, i, count, ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("network")
    parser.add_argument("chessfile")
    parser.add_argument("--png", help="save the confusion matrix as an image")
    args = parser.parse_args()

    expected = read_expected(args.chessfile)
    predicted = predict(args.network, args.chessfile)
    if len(predicted) != len(expected):
        sys.exit(f"{len(predicted)} predictions for {len(expected)} positions")

    matrix = confusion_matrix(expected, predicted)
    print_report(matrix, expected, predicted)

    if args.png:
        save_confusion_png(matrix, args.png)
        print(f"\nconfusion matrix saved in {args.png}")


if __name__ == "__main__":
    main()
