"""Archive the top-K submissions for the HW3 autoresearch loop.

The training script (test.py) always overwrites r14725055_submission.{pt,csv}.
This helper snapshots a finished run into ./submissions/, keyed by test accuracy,
maintains ./submissions/leaderboard.tsv, and keeps only the top-K runs by accuracy.
Only models strictly under 1 MB qualify (the assignment's hard size cap).

Rationale: Kaggle allows multiple uploads and scores your best, so we keep the
top-K CSVs (each with its matching .pt) to hedge against overfitting the public split.

Usage:
    uv run archive_topk.py --acc 0.6234 --commit abc1234 --desc "head_dim 16, mlp 192" [--k 5]
"""

import argparse
import csv
import os
import shutil

STUDENT_ID = "r14725055"
SUB_DIR = "submissions"
ONE_MB = 1_048_576
PT_SRC = f"{STUDENT_ID}_submission.pt"
CSV_SRC = f"{STUDENT_ID}_submission.csv"
LEADERBOARD = os.path.join(SUB_DIR, "leaderboard.tsv")
FIELDS = ["accuracy", "size_mb", "size_bytes", "commit", "pt_file", "csv_file", "description"]


def load_board():
    if not os.path.exists(LEADERBOARD):
        return []
    with open(LEADERBOARD, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_board(rows):
    with open(LEADERBOARD, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in FIELDS})


def safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", type=float, required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--desc", default="")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(SUB_DIR, exist_ok=True)

    if not os.path.exists(PT_SRC):
        print(f"SKIP: {PT_SRC} not found")
        return
    size_bytes = os.path.getsize(PT_SRC)
    size_mb = size_bytes / ONE_MB
    if size_bytes >= ONE_MB:
        print(f"SKIP: {PT_SRC} is {size_bytes} bytes ({size_mb:.3f} MB) >= 1MB, not eligible")
        return

    board = load_board()
    # Drop any prior entry for the same commit (re-run) and remove its old files.
    kept = []
    for r in board:
        if r["commit"] == args.commit:
            safe_remove(r.get("pt_file"))
            safe_remove(r.get("csv_file"))
        else:
            kept.append(r)
    board = kept

    tag = f"acc{args.acc:.4f}_{args.commit}"
    pt_dst = os.path.join(SUB_DIR, f"{STUDENT_ID}_{tag}.pt")
    csv_dst = os.path.join(SUB_DIR, f"{STUDENT_ID}_{tag}.csv")
    shutil.copy2(PT_SRC, pt_dst)
    if os.path.exists(CSV_SRC):
        shutil.copy2(CSV_SRC, csv_dst)

    board.append({
        "accuracy": f"{args.acc:.4f}",
        "size_mb": f"{size_mb:.3f}",
        "size_bytes": str(size_bytes),
        "commit": args.commit,
        "pt_file": pt_dst,
        "csv_file": csv_dst,
        "description": args.desc.replace("\t", " "),
    })

    # Rank: highest accuracy first, smaller model as tie-break.
    board.sort(key=lambda r: (-float(r["accuracy"]), float(r["size_mb"])))

    survivors = board[: args.k]
    for r in board[args.k:]:
        safe_remove(r.get("pt_file"))
        safe_remove(r.get("csv_file"))

    write_board(survivors)

    in_top = any(r["commit"] == args.commit for r in survivors)
    print(f"{'ARCHIVED' if in_top else 'NOT IN TOP-' + str(args.k)}: acc={args.acc:.4f} size={size_mb:.3f}MB commit={args.commit}")
    print(f"--- top-{args.k} leaderboard ---")
    for i, r in enumerate(survivors, 1):
        print(f"{i}. acc={r['accuracy']} size={r['size_mb']}MB {r['commit']} {r['description']}")


if __name__ == "__main__":
    main()
