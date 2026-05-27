# autoresearch — HW3 Vision Transformer on CIFAR-10

This is an autonomous research loop, adapted from karpathy's `autoresearch`, for **HW3**.
The goal: search over the four editable parts of `test.py` to **maximize CIFAR-10 test
accuracy**, subject to the **hard constraint that the saved `.pt` model file is < 1 MB**.

You (the agent) iterate: edit `test.py` → train → measure accuracy + model size →
keep or discard → repeat, autonomously, until the human stops you.

## Background (the assignment)

- Task: classify CIFAR-10 (32×32 RGB, 10 classes) with a ViT.
- Grading (for context — your job is accuracy under the 1 MB cap):
  - Part 5 FFN correctness 20%, Part 6 attention 20% (+20% bonus for multi-head),
    Part 7 transformer 20%.
  - Model size 15%: 5% if `.pt` < **1 MB**, 10% by class ranking (smaller = better).
  - Accuracy 15%: 5% if test acc > **57%**, 10% by class ranking (higher = better).
  - Accuracy on a hidden "another dataset" 10%: by class ranking.
- The current baseline `.pt` is ~1.77 MB (OVER the limit) at ~60.8% accuracy. The single
  biggest size waste is `head_dim=64` with `hidden_dim=64` → `inner_dim=256`, which inflates
  the QKV/out projections ~4×. Getting under 1 MB is the first priority.

## Setup

To set up a new run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may28`). The branch
   `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current `main`.
3. **Read the in-scope files** for full context:
   - `2026_HW3_Instructions_Update2.pdf` / `HW3_Implementation_Guide.md` — the rules.
   - `test.py` — the file you edit. Model parts (Part 5/6/7), ViT (Part 8), hyperparameters
     (Part 9), training + eval (Part 12/13).
4. **Verify environment**: `uv run python -c "import torch; print(torch.cuda.is_available())"`
   must print `True` (GPU: RTX 3060 Ti). CIFAR-10 auto-downloads on the first run.
5. **Initialize `results.tsv`**: create it with just the header row. The baseline is recorded
   after the first run.
6. **Confirm and go.**

Once you get confirmation, kick off the experimentation.

## Experimentation

Run an experiment with: `uv run test.py > run.log 2>&1`
It trains for a **fixed 10 epochs** (Adam, lr 3e-4, batch 16 — all fixed), evaluates on the
CIFAR-10 **test set**, saves `r14725055_submission.pt` + `.csv`, and prints the param count,
model size, and accuracy.

**What you CAN modify** — ONLY these four `[MODIFIABLE]` regions of `test.py`:
- **Part 5 `myFFN`** — the position-wise feed-forward layer.
- **Part 6 `myAttention`** — scaled dot-product / multi-head attention.
- **Part 7 `myTransformer`** — number of layers, dropout, norm placement, residuals, etc.
- **Part 9 hyperparameters** — `hidden_dim`, `heads`, `head_dim`, `mlp_dim`.

**What you CANNOT do:**
- Touch any `[DO NOT MODIFY]` section: Part 1 imports, Part 2 globals (`batch_size=16`,
  `num_epoch=10`, `patch_size=4`, `input_size`, etc.), Part 3 dataloader/transforms (so **no
  data augmentation**), Part 4 `myTokenization`, Part 8 `myViT`, Part 10 loss, Part 11 Adam
  optimizer, Part 12 training loop, Part 13 evaluation.
- Use `torch.nn.MultiheadAttention()` — forbidden by the assignment.
- Break the **required math** (these are correctness-graded):
  - `myFFN` must implement `FFN(x) = max(0, xW1+b1)W2+b2` — i.e. two `nn.Linear` layers with
    **ReLU** in between. Adding `nn.Dropout` is fine; swapping ReLU for another activation is
    not (risks losing the 20%).
  - `myAttention` must be scaled dot-product attention `softmax(QKᵀ/√d_k)V`. Multi-head is
    encouraged (keeps the +20% bonus). Don't regress to something that isn't real attention.
- Install packages or add dependencies (only what's in `pyproject.toml`).

**The goal:** **maximize CIFAR-10 test accuracy** subject to the **hard constraint** that the
saved `.pt` file is **< 1 MB (1,048,576 bytes)**. A run whose `.pt` ≥ 1 MB is a FAIL regardless
of accuracy → status `discard`. Among runs under 1 MB, higher accuracy wins.

**Simplicity criterion**: all else equal, simpler is better. A tiny accuracy gain that adds
ugly complexity is not worth it; removing code for equal/better accuracy is a clear win.

**Overfitting caveat**: the keep/discard metric is test-set accuracy (per the user's choice),
and the class also grades a hidden "another dataset". Prefer robust, regularized designs
(sensible dropout, not absurdly large for 10 epochs) over configs that win by luck.

**The first run**: establish the baseline by running `test.py` as-is (no edits).

## Output format

`test.py` prints near the end:

```
參數量: 250,123
模型大小: 0.954 MB
...
Accuracy: 0.6234
```

Extract the metrics:

```
grep "Accuracy:\|參數量:\|模型大小:" run.log
```

**Authoritative size check** — the grading metric is the `.pt` FILE size on disk, not the
printed param-size estimate. After a run, verify:

```
ls -l r14725055_submission.pt        # bytes must be < 1048576
```

## Logging results

Append every experiment to `results.tsv` (tab-separated, NOT comma — commas break descriptions).
Header + 5 columns:

```
commit	accuracy	size_mb	status	description
```

1. git commit hash (short, 7 chars)
2. test accuracy achieved (e.g. `0.6234`) — use `0.0000` for crashes
3. `.pt` file size in MB, `.3f` (file_bytes / 1048576) — use `0.0` for crashes
4. status: `keep`, `discard`, or `crash`
5. short description of what this experiment tried (no commas, no tabs)

Example:

```
commit	accuracy	size_mb	status	description
a1b2c3d	0.6080	1.772	discard	baseline (over 1MB cap)
b2c3d4e	0.6010	0.512	keep	head_dim 64->16 so inner_dim==hidden_dim
c3d4e5f	0.6150	0.620	keep	mlp_dim 128->192 + dropout 0.1
d4e5f6g	0.0000	0.0	crash	einops pattern typo (h d mismatch)
```

Do **NOT** commit `results.tsv` — leave it untracked.

## The experiment loop

The loop runs on the dedicated branch `autoresearch/<tag>`.

LOOP FOREVER:

1. Look at the git state: current branch/commit, and the current best-kept accuracy.
2. Edit ONE of the four MODIFIABLE regions in `test.py` with an experimental idea.
3. `git commit` (only `test.py`).
4. Run: `uv run test.py > run.log 2>&1`
5. Read results: `grep "Accuracy:" run.log` and `ls -l r14725055_submission.pt`.
6. If the grep is empty, the run crashed. `tail -n 50 run.log` to read the traceback. If it's
   trivial (typo, einops pattern, shape off-by-one), fix and re-run. If the idea is
   fundamentally broken, log `crash` and move on.
7. Record the result in `results.tsv`.
7b. **Archive for top-K** (any run, not only "keep"): if the `.pt` is < 1 MB, snapshot it:
    `uv run archive_topk.py --acc <acc> --commit <hash> --desc "<short desc>"`
    This keeps the top-5 runs by test accuracy in `./submissions/` (each with its `.pt` +
    `.csv`) and maintains `submissions/leaderboard.tsv`, for multi-upload to Kaggle. A run can
    be `discard` for the git/keep decision but still earn a top-5 archive slot — archive it
    regardless of the keep/discard decision, as long as it is < 1 MB.
8. Decision (git branch advance):
   - If `.pt` < 1 MB **AND** accuracy > current best-kept accuracy → **KEEP** (advance the
     branch, keep the commit).
   - Otherwise → **DISCARD**: `git reset --hard <last-kept-commit>`.
   (The `.pt`/`.csv` are gitignored, so reset never touches them — the on-disk `.pt` is only
   used for the size check of the most recent run.)
9. Repeat.

**Timeout**: a run is ~a few minutes. If one exceeds ~20 minutes, kill it and treat it as a
failure (discard + revert).

**NEVER STOP**: once the loop has begun (after setup), do NOT pause to ask the human whether to
continue. The human may be away and expects you to keep working indefinitely until manually
stopped. If you run out of ideas, think harder: re-read the in-scope files, combine previous
near-misses, try more radical architecture changes. The loop runs until interrupted.

### Idea bank (non-exhaustive)

First priority — get under 1 MB (baseline is 1.77 MB):
- Set `head_dim ≈ hidden_dim // heads` so `inner_dim ≈ hidden_dim` (the baseline's
  `head_dim=64`, `hidden_dim=64` makes `inner_dim=256`, ~4× the QKV/out cost). This alone
  should drop well under 1 MB.
- Reduce `hidden_dim` (e.g. 48, 64) and/or `mlp_dim`.

Then climb accuracy within the budget:
- Number of transformer layers (`num_layers` in Part 7): try 3–8. Deeper isn't always better
  with only 10 epochs.
- `mlp_dim` as 1×–4× `hidden_dim`.
- Dropout rate in Part 7 (and optionally attention dropout in Part 6): 0.0–0.2.
- Pre-LN vs Post-LN; final LayerNorm on/off.
- `heads` count (4 vs 8) with `head_dim` tuned to keep `inner_dim` modest.
- Untie `head_dim` from `hidden_dim//heads` slightly to trade size for capacity.
- Combine the best near-misses; push size down to free budget for more depth/width.

Goal recap: smallest path to the highest test accuracy **under 1 MB**.
