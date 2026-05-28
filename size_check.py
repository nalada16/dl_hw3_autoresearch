#!/usr/bin/env python3
"""
size_check.py — fast architecture size estimator for HW3 autoresearch loop.

Instantiates a model with proposed parameters, saves to a temporary .pt file,
and reports the EXACT file size + budget status — without running any training.
Use this BEFORE editing test.py to decide whether a proposed config fits < 1 MB.

Usage
-----
  # Check a specific config
  uv run size_check.py --layers 7 --heads 16 --head_dim 4 --mlp 128

  # Auto-compare against current best
  uv run size_check.py --layers 8 --heads 16 --head_dim 4 --mlp 128

  # Show param breakdown by component
  uv run size_check.py --layers 7 --heads 16 --head_dim 4 --mlp 128 --breakdown

  # Scan a grid of configs (prints all that fit under 1 MB)
  uv run size_check.py --scan
"""

import argparse
import os
import tempfile

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# ── Fixed globals (mirrors test.py Part 2) ───────────────────────────────────
ONE_MB      = 1_048_576
SIZE_LIMIT  = ONE_MB
HIDDEN_DIM  = 64
INPUT_SIZE  = (32, 32, 3)
PATCH_SIZE  = 4
NUM_CLASSES = 10

# Current best config — update this after each new "keep"
BASELINE = dict(layers=9, heads=16, head_dim=4, mlp_dim=72, hidden_dim=64)
BASELINE_COMMIT = "4332098"
BASELINE_ACC    = 0.6851


# ── Model classes (same as test.py, no training side-effects) ────────────────

class _Tokenization(nn.Module):
    def __init__(self, output_dim, patch_size, channels):
        super().__init__()
        patch_dim = patch_size * patch_size * channels
        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, output_dim),
        )
    def forward(self, x):
        return self.to_patch_tokens(x)


class _FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    def forward(self, x):
        return self.ffn(x)


class _Attention(nn.Module):
    def __init__(self, input_dim, heads, head_dim):
        super().__init__()
        self.heads     = heads
        inner_dim      = heads * head_dim
        self.scale     = head_dim ** -0.5
        self.attn_drop = nn.Dropout(0.0)   # mirrors myAttention.attn_drop (no params, affects .pt size)
        self.to_qkv    = nn.Linear(input_dim, inner_dim * 3, bias=False)
        self.to_out    = nn.Linear(inner_dim, input_dim)
    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        attn = self.attn_drop(
            (torch.einsum('bhid,bhjd->bhij', q, k) * self.scale).softmax(dim=-1))
        out  = torch.einsum('bhij,bhjd->bhid', attn, v)
        return self.to_out(rearrange(out, 'b h n d -> b n (h d)'))


class _Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(dim), _Attention(dim, heads, dim_head), nn.Dropout(0.0),
                nn.LayerNorm(dim), _FFN(dim, mlp_dim),               nn.Dropout(0.0),
            ])
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        for n1, attn, d1, n2, ffn, d2 in self.layers:
            x = d1(attn(n1(x))) + x
            x = d2(ffn(n2(x)))  + x
        return self.norm(x)


class _ViT(nn.Module):
    def __init__(self, input_size, patch_size, hidden_dim,
                 heads, head_dim, mlp_dim, num_classes, num_layers):
        super().__init__()
        h, w, c    = input_size
        num_patches = (h // patch_size) * (w // patch_size)
        self.tokenize     = _Tokenization(hidden_dim, patch_size, c)
        self.cls_token    = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embedding= nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))
        self.transformer  = _Transformer(hidden_dim, heads, head_dim, mlp_dim, num_layers)
        self.mlp_head     = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, x):
        x = self.tokenize(x)
        b, n, _ = x.shape
        x = torch.cat((repeat(self.cls_token, '() n d -> b n d', b=b), x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        return self.mlp_head(self.transformer(x)[:, 0])


# ── Core measurement (exact: saves to temp file) ─────────────────────────────

def measure(layers, heads, head_dim, mlp_dim, hidden_dim=HIDDEN_DIM):
    """Return (total_params, exact_pt_bytes) without any training."""
    model = _ViT(
        input_size  = INPUT_SIZE,
        patch_size  = PATCH_SIZE,
        hidden_dim  = hidden_dim,
        heads       = heads,
        head_dim    = head_dim,
        mlp_dim     = mlp_dim,
        num_classes = NUM_CLASSES,
        num_layers  = layers,
    )
    total_params = sum(p.numel() for p in model.parameters())

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmp = f.name
    try:
        torch.save(model, tmp)
        size_bytes = os.path.getsize(tmp)
    finally:
        os.unlink(tmp)

    return total_params, size_bytes


# ── Param breakdown by component ─────────────────────────────────────────────

def param_breakdown(layers, heads, head_dim, mlp_dim, hidden_dim=HIDDEN_DIM):
    c         = INPUT_SIZE[2]
    patch_dim = PATCH_SIZE * PATCH_SIZE * c
    num_patches = (INPUT_SIZE[0] // PATCH_SIZE) * (INPUT_SIZE[1] // PATCH_SIZE)
    inner_dim = heads * head_dim

    tok        = 2 * patch_dim + (patch_dim * hidden_dim + hidden_dim)
    embed      = hidden_dim + (num_patches + 1) * hidden_dim
    attn_params= hidden_dim * inner_dim * 3 + inner_dim * hidden_dim + hidden_dim
    attn_ln    = 2 * hidden_dim
    ffn_params = (hidden_dim * mlp_dim + mlp_dim) + (mlp_dim * hidden_dim + hidden_dim)
    ffn_ln     = 2 * hidden_dim
    per_layer  = attn_params + attn_ln + ffn_params + ffn_ln
    trans      = per_layer * layers + 2 * hidden_dim   # + final LN
    head_p     = 2 * hidden_dim + hidden_dim * NUM_CLASSES + NUM_CLASSES

    return {
        "tokenization"       : tok,
        "cls + pos_embedding": embed,
        "transformer (total)": trans,
        "  per layer"        : per_layer,
        "    attention QKV+out": attn_params,
        "    attention LN"   : attn_ln,
        "    FFN"            : ffn_params,
        "    FFN LN"         : ffn_ln,
        "mlp_head"           : head_p,
    }


# ── Display helpers ───────────────────────────────────────────────────────────

SEP = "-" * 60

def budget_line(size_bytes):
    if size_bytes < SIZE_LIMIT:
        left = SIZE_LIMIT - size_bytes
        return (f"[OK] UNDER  | left: {left:,} bytes  "
                f"~= {left // 4:,} params of headroom")
    else:
        over = size_bytes - SIZE_LIMIT
        return (f"[!!] OVER   | exceeds by: {over:,} bytes  "
                f"~= {over // 4:,} params too many")


def print_config(layers, heads, head_dim, mlp_dim, hidden_dim=HIDDEN_DIM,
                 label="", show_breakdown=False):
    params, size_bytes = measure(layers, heads, head_dim, mlp_dim, hidden_dim)
    overhead = size_bytes - params * 4

    print(f"\n{SEP}")
    if label:
        print(f"  {label}")
    print(f"  layers={layers}  heads={heads}  head_dim={head_dim}"
          f"  mlp={mlp_dim}  hidden={hidden_dim}")
    print(SEP)
    print(f"  Params          : {params:,}")
    print(f"  .pt size        : {size_bytes:,} bytes  ({size_bytes/ONE_MB:.4f} MB)")
    print(f"  Serialization+  : {overhead:,} bytes  (overhead beyond params*4)")
    print(f"  Budget          : {budget_line(size_bytes)}")

    if show_breakdown:
        print(f"\n  Param breakdown:")
        bd = param_breakdown(layers, heads, head_dim, mlp_dim, hidden_dim)
        for k, v in bd.items():
            bar = "#" * max(1, v * 30 // max(bd.values()))
            print(f"    {k:<26}: {v:>8,}  {bar}")

    return params, size_bytes


# ── Scan mode ────────────────────────────────────────────────────────────────

def scan_grid():
    print(f"\nScanning grid -- showing configs under 1 MB only\n")
    header = (f"  {'L':>3}  {'heads':>5}  {'hd':>4}  {'mlp':>5}  "
              f"{'params':>9}  {'MB':>7}  {'left bytes':>11}  note")
    print(header)
    print("  " + "-" * (len(header) - 2))

    bl = BASELINE
    for layers in [6, 7, 8]:
        for heads in [8, 16]:
            for head_dim in [4, 8]:
                for mlp_dim in [96, 112, 128, 144, 160]:
                    params, size_bytes = measure(layers, heads, head_dim, mlp_dim)
                    if size_bytes >= SIZE_LIMIT:
                        continue
                    left = SIZE_LIMIT - size_bytes
                    mb   = size_bytes / ONE_MB
                    note = ""
                    if (layers == bl['layers'] and heads == bl['heads'] and
                            head_dim == bl['head_dim'] and mlp_dim == bl['mlp_dim']):
                        note = f"* BEST ({BASELINE_COMMIT}: {BASELINE_ACC:.4f})"
                    print(f"  {layers:>3}  {heads:>5}  {head_dim:>4}  {mlp_dim:>5}  "
                          f"{params:>9,}  {mb:>7.4f}  {left:>11,}  {note}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Check model .pt size before running an autoresearch experiment.")
    ap.add_argument("--layers",    type=int, default=None)
    ap.add_argument("--heads",     type=int, default=None)
    ap.add_argument("--head_dim",  type=int, default=None)
    ap.add_argument("--mlp",       type=int, default=None)
    ap.add_argument("--hidden",    type=int, default=HIDDEN_DIM)
    ap.add_argument("--scan",      action="store_true",
                    help="Scan a grid of common configs")
    ap.add_argument("--breakdown", action="store_true",
                    help="Show param count by component")
    args = ap.parse_args()

    if args.scan:
        scan_grid()
        return

    # Fill in defaults from BASELINE for any unspecified args
    layers   = args.layers   or BASELINE['layers']
    heads    = args.heads    or BASELINE['heads']
    head_dim = args.head_dim or BASELINE['head_dim']
    mlp_dim  = args.mlp      or BASELINE['mlp_dim']
    hidden   = args.hidden

    proposed = dict(layers=layers, heads=heads, head_dim=head_dim,
                    mlp_dim=mlp_dim, hidden_dim=hidden)
    is_same_as_baseline = (proposed == BASELINE and hidden == BASELINE['hidden_dim'])

    pp, pb = print_config(
        layers, heads, head_dim, mlp_dim, hidden,
        label="Proposed config",
        show_breakdown=args.breakdown,
    )

    # Always show baseline for comparison when config differs
    if not is_same_as_baseline:
        bp, bb = measure(**BASELINE)
        print_config(
            **BASELINE,
            label=f"Current best  [{BASELINE_COMMIT}  acc={BASELINE_ACC:.4f}]",
        )
        dp = pp - bp
        db = pb - bb
        sp = "+" if dp >= 0 else ""
        sb = "+" if db >= 0 else ""
        print(f"\n  d_params : {sp}{dp:,}")
        print(f"  d_bytes  : {sb}{db:,}  ({'larger' if db > 0 else 'smaller'})\n")
    else:
        print(f"\n  (This is the current best config - {BASELINE_COMMIT})\n")


if __name__ == "__main__":
    main()
