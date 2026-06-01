#!/usr/bin/env python3
"""
analyze.py -- post-run analysis for HW3 autoresearch loop.

After each experiment (uv run test.py > run.log 2>&1), run:
    uv run analyze.py

Outputs:
  1. Per-epoch training loss curve  (parsed from run.log)
  2. Per-class accuracy             (from submission CSV + CIFAR-10 test labels)

No extra dependencies beyond what test.py already uses.
"""

import csv
import re
import sys
from pathlib import Path

import torch
from torchvision import datasets
import torchvision.transforms as trns

# ── Config ────────────────────────────────────────────────────────────────────
CSV_FILE  = "r14725055_submission.csv"
LOG_FILE  = "run.log"
DATA_ROOT = "./dataset/"
CLASSES   = ["airplane", "automobile", "bird", "cat", "deer",
             "dog",      "frog",       "horse", "ship", "truck"]
SEP = "-" * 60


# ── 1. Training loss curve ────────────────────────────────────────────────────

def parse_epoch_losses(log_path: str):
    """
    Extract per-epoch *average* loss from run.log.

    Epoch summary lines look like:
        Epoch 0 | Loss 2.3025
    Batch lines start with a tab and contain 'Batch', so they are skipped.
    """
    losses = []
    # Matches "Epoch N | Loss X.XXXX" at start of (stripped) line,
    # with nothing else after -- excludes batch lines that have "| Batch N |"
    pattern = re.compile(r"^Epoch\s+(\d+)\s*\|\s*Loss\s+([\d.]+)\s*$")
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = pattern.match(line.strip())
                if m:
                    losses.append((int(m.group(1)), float(m.group(2))))
    except FileNotFoundError:
        pass
    return losses


def print_loss_curve(losses):
    if not losses:
        print("  [no epoch-loss data found in run.log -- was the run completed?]")
        return

    vals     = [v for _, v in losses]
    hi, lo   = max(vals), min(vals)
    W        = 30   # bar width

    print(f"  {'Epoch':>5}  {'Avg Loss':>9}  {'bar  (higher = worse)':}")
    print(f"  {'-----':>5}  {'---------':>9}  {'-'*W}")
    for idx, (epoch, loss) in enumerate(losses):
        filled = int((loss - lo) / (hi - lo) * W) if hi > lo else W // 2
        bar    = "#" * filled + "." * (W - filled)
        if idx == 0:
            arrow = " "
        elif loss < vals[idx - 1] - 1e-4:
            arrow = "v"   # improving
        elif loss > vals[idx - 1] + 1e-4:
            arrow = "^"   # worsening (rare)
        else:
            arrow = "-"   # flat
        print(f"  {epoch:>5}  {loss:>9.4f}  [{bar}] {arrow}")

    drop = vals[0] - vals[-1]
    print(f"\n  Epoch 0 -> {losses[-1][0]}: {vals[0]:.4f} -> {vals[-1]:.4f}  "
          f"(total drop: {drop:+.4f})")

    # Convergence hint
    if len(vals) >= 2:
        last_drop = vals[-2] - vals[-1]
        if last_drop > 0.002:
            print("  [hint] Loss still falling at final epoch "
                  "-> model may benefit from more capacity / depth")
        else:
            print("  [hint] Loss nearly flat at final epoch "
                  "-> training has converged within 10 epochs")


# ── 2. Per-class accuracy ─────────────────────────────────────────────────────

def load_predictions(csv_path: str):
    """Returns list of predicted class indices ordered by ImageId (1-based)."""
    preds = {}
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                preds[int(row["ImageId"])] = int(row["Prediction"])
    except FileNotFoundError:
        return None
    return [preds[i] for i in sorted(preds)]


def load_true_labels():
    """Load CIFAR-10 test set labels (downloads dataset if needed)."""
    data_test = datasets.CIFAR10(
        root=DATA_ROOT, train=False,
        transform=trns.ToTensor(), download=True,
    )
    return [label for _, label in data_test]


def print_per_class_accuracy(preds, labels):
    n   = len(CLASSES)
    ok  = [0] * n
    tot = [0] * n
    for pred, true in zip(preds, labels):
        tot[true] += 1
        if pred == true:
            ok[true] += 1

    accs    = [ok[i] / tot[i] for i in range(n)]
    overall = sum(ok) / sum(tot)
    order   = sorted(range(n), key=lambda i: accs[i], reverse=True)  # best first
    W       = 24

    print(f"  {'Class':<12}  {'Acc':>6}  {'Correct/Total':>13}  bar")
    print(f"  {'-'*12}  {'-'*6}  {'-'*13}  {'-'*W}")
    for rank, i in enumerate(order):
        bar   = "#" * int(accs[i] * W) + "." * (W - int(accs[i] * W))
        tag   = "  <- best"  if rank == 0     else \
                "  <- worst" if rank == n - 1 else ""
        print(f"  {CLASSES[i]:<12}  {accs[i]:>6.1%}  "
              f"{ok[i]:>6}/{tot[i]:<5}  [{bar}]{tag}")

    best_i  = order[0]
    worst_i = order[-1]
    spread  = accs[best_i] - accs[worst_i]
    print(f"\n  Overall : {overall:.4f}  ({overall:.2%})")
    print(f"  Best    : {CLASSES[best_i]:<12} {accs[best_i]:.1%}")
    print(f"  Worst   : {CLASSES[worst_i]:<12} {accs[worst_i]:.1%}")
    print(f"  Spread  : {spread:.1%}  (smaller = more balanced across classes)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{SEP}")
    print("  1. Training loss curve  [run.log]")
    print(SEP)
    print_loss_curve(parse_epoch_losses(LOG_FILE))

    print(f"\n{SEP}")
    print("  2. Per-class accuracy  [submission CSV + CIFAR-10 test labels]")
    print(SEP)
    preds = load_predictions(CSV_FILE)
    if preds is None:
        print(f"  [{CSV_FILE} not found -- run test.py first]")
        sys.exit(1)
    labels = load_true_labels()
    if len(preds) != len(labels):
        print(f"  [size mismatch: {len(preds)} predictions vs {len(labels)} labels]")
        sys.exit(1)
    print_per_class_accuracy(preds, labels)
    print()


if __name__ == "__main__":
    main()
