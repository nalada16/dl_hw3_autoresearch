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
- Current best accuracy: **68.42%** (unseeded, `.pt` ~0.986 MB, comfortably under the 1 MB
  cap). The Kaggle public leaderboard top score is ~**77%**, so there is significant room to
  improve. The model is already budget-efficient; the focus is purely on accuracy gains.

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
6. **Update `size_check.py` baseline**: open `size_check.py` and update the `BASELINE` dict
   and `BASELINE_COMMIT` / `BASELINE_ACC` constants to reflect the current best-kept config.
   This ensures the comparison and scan outputs are anchored to the right starting point.
7. **Confirm and go.**

Once you get confirmation, kick off the experimentation.

## Tools

### `size_check.py` — pre-flight size estimator

**Use this BEFORE editing test.py** whenever a proposed change might affect model size.
It instantiates the model, saves to a temp `.pt`, and reports exact bytes + budget status
in under one second — far faster than running a full training to discover the model is over.

```bash
# Check a specific config (auto-compares vs current best)
uv run size_check.py --layers 9 --heads 16 --head_dim 4 --mlp 80

# See param count broken down by component
uv run size_check.py --layers 9 --heads 16 --head_dim 4 --mlp 80 --breakdown

# Scan a grid — shows all configs that fit under 1 MB
uv run size_check.py --scan
```

Key output fields:
- `.pt size` — exact file size in bytes (the authoritative grading metric)
- `left` — bytes remaining before hitting 1 MB cap (~= left // 4 params of headroom)
- `d_bytes` — difference vs current best (positive = larger)

**When to skip it**: changes that don't touch `layers`, `heads`, `head_dim`, `mlp_dim`,
or `hidden_dim` (e.g. pure dropout value changes, init changes) cost 0 bytes — no check needed.

**After a new KEEP**: update `BASELINE`, `BASELINE_COMMIT`, `BASELINE_ACC` in `size_check.py`
so future comparisons stay anchored to the new best.

### `archive_topk.py` — top-K submission archiver

```bash
uv run archive_topk.py --acc <acc> --commit <hash> --desc "<short desc>"
```

Snapshots `.pt` + `.csv` into `submissions/`, keeps top-5 by accuracy.
Run this for every sub-1MB result regardless of keep/discard.

---

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
- **Set any random seed** — `torch.manual_seed(...)`, `torch.cuda.manual_seed_all(...)`, or
  any equivalent. Seed-searching exploits test-set randomness; it is not a genuine
  architectural improvement and is forbidden. Every keep/discard decision must have a
  mechanistic justification. "Accuracy happened to be higher" is not sufficient.
- **Circumvent DO NOT MODIFY sections by any indirect means.** This includes: subclassing a
  DO NOT MODIFY class and swapping the instance, monkey-patching methods at runtime, calling
  a DO NOT MODIFY function from a MODIFIABLE section in a way that alters its behavior, or
  wrapping DO NOT MODIFY code with logic that changes its inputs or outputs. If you are
  unsure whether a change crosses this line, assume it does and don't do it.

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
2. Design the next experiment. **If it changes `layers`/`heads`/`head_dim`/`mlp_dim`/`hidden_dim`,
   run `uv run size_check.py --layers X --heads Y ...` first.** If the output shows `[!!] OVER`,
   abandon the idea immediately — do not edit, do not commit.
3. Edit ONE of the four MODIFIABLE regions in `test.py` with an experimental idea.
4. `git commit` (only `test.py`).
5. Run: `uv run test.py > run.log 2>&1`
6. Read results: `grep "Accuracy:" run.log` and `ls -l r14725055_submission.pt`.
7. If the grep is empty, the run crashed. `tail -n 50 run.log` to read the traceback. If it's
   trivial (typo, einops pattern, shape off-by-one), fix and re-run. If the idea is
   fundamentally broken, log `crash` and move on.
8. Record the result in `results.tsv`.
8b. **Archive for top-K** (any run, not only "keep"): if the `.pt` is < 1 MB, snapshot it:
    `uv run archive_topk.py --acc <acc> --commit <hash> --desc "<short desc>"`
    This keeps the top-5 runs by test accuracy in `./submissions/` (each with its `.pt` +
    `.csv`) and maintains `submissions/leaderboard.tsv`, for multi-upload to Kaggle. A run can
    be `discard` for the git/keep decision but still earn a top-5 archive slot — archive it
    regardless of the keep/discard decision, as long as it is < 1 MB.
9. Decision (git branch advance):
   - If `.pt` < 1 MB **AND** accuracy > current best-kept accuracy → **KEEP** (advance the
     branch, keep the commit). Also update `BASELINE` in `size_check.py`.
   - Otherwise → **DISCARD**: `git reset --hard <last-kept-commit>`.
   (The `.pt`/`.csv` are gitignored, so reset never touches them — the on-disk `.pt` is only
   used for the size check of the most recent run.)
10. Repeat.

**Timeout**: a run is ~a few minutes. If one exceeds ~20 minutes, kill it and treat it as a
failure (discard + revert).

**NEVER STOP**: once the loop has begun (after setup), do NOT pause to ask the human whether to
continue. The human may be away and expects you to keep working indefinitely until manually
stopped. If you run out of ideas, think harder: re-read the in-scope files, combine previous
near-misses, try more radical architecture changes. The loop runs until interrupted.

### Idea bank (non-exhaustive)

The model is already under 1 MB. The only goal is accuracy improvement within the budget.
Use `uv run size_check.py` before committing any change that affects model dimensions.

Architectural dimensions to explore:
- Number of transformer layers (`num_layers` in Part 7): current is 9. Verify budget before
  increasing.
- `mlp_dim`: current is 72. Higher values cost bytes; check budget first.
- `heads` and `head_dim`: current is 16 heads × 4 dim = inner 64. Alternative ratios may
  capture different attention patterns.
- `hidden_dim`: current is 64. Both directions carry risk — verify carefully.
- Dropout: `attn_dropout` (current 0.22) and residual dropout (current 0.05) in Part 7.
  Per-layer schedules, annealing, or asymmetric application are all fair game.
- Normalization placement: Pre-LN (current) vs Post-LN vs sandwich. The final LayerNorm
  in the transformer can be removed or moved.
- Attention internals (Part 6): weight initialization, projection structure, dropout placement.
- FFN internals (Part 5): regularization between the two Linear layers, weight initialization.
  Remember: `FFN(x) = max(0, xW1+b1)W2+b2` — the ReLU is required.
- Initialization strategy (Part 9): how `cls_token`, `pos_embedding`, QKV weights, and the
  classifier head are initialized can meaningfully affect convergence in 10 epochs.
- Combining multiple small changes that individually seem minor can sometimes compound.

Goal: highest test accuracy **under 1 MB** through genuine, mechanistically-justified changes.
