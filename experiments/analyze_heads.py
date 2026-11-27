#!/usr/bin/env python
"""Is version drift organized by attention head?

The input to each layer's o_proj is the concatenation of H per-head attention
outputs (Qwen2-0.5B: H=14 heads x 64 dims). If a version update keeps each head
doing the same job, drift at o_proj should be explained by H small per-head maps
(block-diagonal). If the update re-assigns jobs among heads, a permutation of
heads plus per-head maps should explain it. If neither does, the heads mix and
a full map is required.

For every layer we fit, on 80% of the anchor tokens, and score on the held-out
20% (R^2 of predicting the M_v activation from the M_{v+1} activation, the
direction P_in uses):
  identity        no map (naive transfer's implicit assumption)
  full ridge      one d x d map (what the method uses today)
  block, same     per-head 64x64 maps, head i -> head i
  block, matched  per-head maps after matching heads by linear CKA (Hungarian)
We also report the residual-stream input (q_proj; k_proj/v_proj read the same
tensor) with identity / orthogonal / full-ridge maps.

Usage:
    python experiments/analyze_heads.py --config configs/qwen05b_heads.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptport import load_config
from adaptport.alignment import fit_map
from adaptport.cka import linear_cka


def split(n, frac=0.8, seed=0):
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    k = int(frac * n)
    return perm[:k], perm[k:]


def r2(pred, target, target_mean):
    """1 - SSE/SST on held-out rows; target_mean comes from the TRAIN rows."""
    sse = ((pred - target) ** 2).sum()
    sst = ((target - target_mean) ** 2).sum()
    return (1 - sse / sst).item()


def fit_apply(s_tr, t_tr, s_te, method, lam):
    """Fit P with s @ P.T ~ t on train (centered), predict held-out t."""
    p = fit_map(s_tr, t_tr, method, lam, center=True)
    return (s_te - s_tr.mean(0)) @ p.t() + t_tr.mean(0)


def score_map(s, t, tr, te, method, lam):
    return r2(fit_apply(s[tr], t[tr], s[te], method, lam), t[te], t[tr].mean(0))


def block_r2(s, t, tr, te, head_dim, pairing, lam):
    """Per-head ridge maps: v+1 head pairing[i] -> v head i. Total R^2 over all dims."""
    pred = torch.empty_like(t[te])
    for i, j in enumerate(pairing):
        ti, sj = slice(i * head_dim, (i + 1) * head_dim), slice(j * head_dim, (j + 1) * head_dim)
        pred[:, ti] = fit_apply(s[tr][:, sj], t[tr][:, ti], s[te][:, sj], "ridge", lam)
    return r2(pred, t[te], t[tr].mean(0))


def head_cka(xv, xv1, n_heads, head_dim, idx):
    m = torch.zeros(n_heads, n_heads)
    for i in range(n_heads):
        for j in range(n_heads):
            m[i, j] = linear_cka(xv[idx][:, i * head_dim:(i + 1) * head_dim],
                                 xv1[idx][:, j * head_dim:(j + 1) * head_dim])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--n-heads", type=int, default=14)
    args = ap.parse_args()
    cfg = load_config(args.config)
    acts_v = torch.load(cfg.out("cache", "acts_v.pt"))
    acts_v1 = torch.load(cfg.out("cache", "acts_v_next.pt"))
    H, lam = args.n_heads, args.lam

    layers = sorted({int(k.split(".")[2]) for k in acts_v})
    rows = []
    for L in layers:
        o = f"model.layers.{L}.self_attn.o_proj"
        q = f"model.layers.{L}.self_attn.q_proj"
        xv, xv1 = acts_v[o]["in"], acts_v1[o]["in"]
        hd = xv.shape[1] // H
        tr, te = split(xv.shape[0])

        cka = head_cka(xv, xv1, H, hd, tr)               # match on TRAIN rows only
        ri, ci = linear_sum_assignment(-cka.numpy())     # maximize total CKA
        matched = [int(c) for _, c in sorted(zip(ri, ci))]
        moved = sum(1 for i, j in enumerate(matched) if i != j)

        rv, rv1 = acts_v[q]["in"], acts_v1[q]["in"]
        rtr, rte = split(rv.shape[0], seed=1)
        row = {
            "layer": L,
            "o_proj": {
                "identity": r2(xv1[te], xv[te], xv[tr].mean(0)),
                "full_ridge": score_map(xv1, xv, tr, te, "ridge", lam),
                "block_same_head": block_r2(xv1, xv, tr, te, hd, list(range(H)), lam),
                "block_matched_head": block_r2(xv1, xv, tr, te, hd, matched, lam),
                "heads_moved_by_matching": moved,
                "matching": matched,
                "cka_same_head_mean": cka.diag().mean().item(),
                "cka_matched_mean": cka[torch.arange(H), torch.tensor(matched)].mean().item(),
                "cka_offdiag_max": (cka - torch.diag(torch.full((H,), 9.0))).max().item(),
            },
            "residual": {
                "identity": r2(rv1[rte], rv[rte], rv[rtr].mean(0)),
                "orthogonal": score_map(rv1, rv, rtr, rte, "procrustes", 0.0),
                "full_ridge": score_map(rv1, rv, rtr, rte, "ridge", lam),
            },
        }
        o_ = row["o_proj"]
        print(f"L{L:02d} o_proj R2: id {o_['identity']:+.3f} full {o_['full_ridge']:.3f} "
              f"block {o_['block_same_head']:.3f} matched {o_['block_matched_head']:.3f} "
              f"moved {moved:2d} | resid R2: id {row['residual']['identity']:+.3f} "
              f"orth {row['residual']['orthogonal']:.3f} ridge {row['residual']['full_ridge']:.3f}",
              flush=True)
        rows.append(row)

    n_tok = next(iter(acts_v.values()))["in"].shape[0]
    out = cfg.out("heads.json")
    out.write_text(json.dumps({"lambda": lam, "n_heads": H, "anchor_tokens": n_tok,
                               "split": "80/20 random rows, held-out R^2", "layers": rows},
                              indent=2))
    print(f"[heads] -> {out}")


if __name__ == "__main__":
    main()
