#!/usr/bin/env python
"""Ablation harness over the closed-form alignment knobs.

Sweeps: method (ridge/procrustes), ridge lambda, anchor-token budget,
per-layer vs single shared map, and module subset. For each config it refits
the maps and scores transplanted adapters -- WITHOUT re-extracting activations.

Two modes:
  real (default): load cached acts_{v,v_next}.pt (from run_cka_drift.py), refit,
                  transplant each task adapter, evaluate on task data.
  --synthetic   : planted multi-module drift + known adapters, scored by
                  functional recovery. Runs anywhere, no model download.

Run:
  python experiments/run_ablations.py --synthetic
  python experiments/run_ablations.py --config configs/qwen2_to_qwen25_1.5b.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptport.alignment import fit_map, fit_module_maps, transplant_adapter
from adaptport.cka import drift

# ---- sweep grid (edit here) ------------------------------------------------
GRID = {
    "method": ["ridge", "procrustes"],
    "lam": [1e-3, 1.0, 100.0],        # only varied for ridge
    "anchor_tokens": [100, 500, 4000],
    "map_scope": ["per_layer", "shared"],
}


# ---- generic helpers on cached activation dicts ----------------------------
def subsample(acts, n, seed=0):
    out = {}
    for k, v in acts.items():
        N = v["in"].shape[0]
        idx = torch.randperm(N, generator=torch.Generator().manual_seed(seed))[:min(n, N)]
        out[k] = {"in": v["in"][idx], "out": v["out"][idx]}
    return out


def filter_modules(acts, substrings):
    if not substrings:
        return acts
    return {k: v for k, v in acts.items() if any(s in k for s in substrings)}


def shared_module_maps(acts_v, acts_v1, method, lam, center):
    """One P_in/P_out fit on activations stacked across all (equal-dim) modules,
    then reused for every module."""
    in_v = torch.cat([a["in"] for a in acts_v.values()], 0)
    in_v1 = torch.cat([acts_v1[k]["in"] for k in acts_v], 0)
    out_v = torch.cat([a["out"] for a in acts_v.values()], 0)
    out_v1 = torch.cat([acts_v1[k]["out"] for k in acts_v], 0)
    P_in = fit_map(in_v1, in_v, method, lam, center)
    P_out = fit_map(out_v, out_v1, method, lam, center)
    return {k: {"P_in": P_in, "P_out": P_out} for k in acts_v}


def build_maps(acts_v, acts_v1, cfg):
    if cfg["map_scope"] == "shared":
        return shared_module_maps(acts_v, acts_v1, cfg["method"], cfg["lam"], True)
    return fit_module_maps(acts_v, acts_v1, cfg["method"], cfg["lam"], center=True)


def enumerate_configs():
    for scope in GRID["map_scope"]:
        for ntok in GRID["anchor_tokens"]:
            for method in GRID["method"]:
                lams = GRID["lam"] if method == "ridge" else [0.0]
                for lam in lams:
                    yield {"method": method, "lam": lam,
                           "anchor_tokens": ntok, "map_scope": scope}


# ---- synthetic problem (runs anywhere) -------------------------------------
def synthetic_problem(n_modules=4, d=64, r=4, n_anchor=4000, n_test=512, seed=0):
    torch.manual_seed(seed)

    def drift_mat():
        q1, _ = torch.linalg.qr(torch.randn(d, d))
        q2, _ = torch.linalg.qr(torch.randn(d, d))
        return q1 @ torch.diag(torch.empty(d).uniform_(0.7, 1.3)) @ q2.t()

    mods = [f"model.layers.{i}.self_attn.q_proj" for i in range(n_modules)]
    W = torch.randn(d, d) / d ** 0.5
    plant = {}                          # per-module drift + adapter + test data
    acts_v, acts_v1 = {}, {}
    for m in mods:
        G_in, G_out = drift_mat(), drift_mat()
        B, A = torch.randn(d, r) / r ** 0.5, torch.randn(r, d) / d ** 0.5
        X = torch.randn(n_anchor, d)
        acts_v[m] = {"in": X, "out": X @ W.t()}
        acts_v1[m] = {"in": X @ G_in.t(), "out": (X @ W.t()) @ G_out.t()}
        a_v = torch.randn(n_test, d)
        plant[m] = {"G_in": G_in, "G_out": G_out, "dW": B @ A, "a_v": a_v}
    return acts_v, acts_v1, plant


def synthetic_scorer(plant):
    def score(maps):
        recs = []
        for m, p in plant.items():
            if m not in maps:
                continue
            a_v1 = p["a_v"] @ p["G_in"].t()
            correct = (p["a_v"] @ p["dW"].t()) @ p["G_out"].t()
            naive = a_v1 @ p["dW"].t()
            dW_a = maps[m]["P_out"] @ p["dW"] @ maps[m]["P_in"]
            aligned = a_v1 @ dW_a.t()
            ne = (naive - correct).norm() / correct.norm()
            ae = (aligned - correct).norm() / correct.norm()
            recs.append((1 - ae / ne).item())
        return {"recovery": sum(recs) / len(recs)}      # mean error-reduction
    return score


# ---- real problem (needs cached acts + GPU for eval) -----------------------
def real_problem_and_scorer(cfg_path):
    from adaptport import load_config
    from adaptport.eval import evaluate
    from adaptport.models import load_base
    from adaptport.transfer import attach_adapter, load_lora_state, write_transplanted_adapter

    cfg = load_config(cfg_path)
    acts_v = torch.load(cfg.out("cache", "acts_v.pt"))
    acts_v1 = torch.load(cfg.out("cache", "acts_v_next.pt"))
    base, tok = load_base(cfg.pair.base_v_next, cfg.device, cfg.dtype)

    def score(maps):
        res = {}
        for task in cfg.tasks:
            state = transplant_adapter(load_lora_state(task.adapter_path), maps)
            out_dir = str(cfg.out("adapters_ablate", task.name))
            write_transplanted_adapter(task.adapter_path, state, out_dir)
            model = attach_adapter(base, out_dir)
            res[task.name] = evaluate(model, tok, task, cfg.device)["value"]
            del model
        return res
    return acts_v, acts_v1, score, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--modules", nargs="*", default=None,
                    help="restrict to modules containing these substrings")
    ap.add_argument("--out", default="runs/ablations.json")
    args = ap.parse_args()

    if args.synthetic or not args.config:
        acts_v, acts_v1, plant = synthetic_problem()
        scorer = synthetic_scorer(plant)
        out_path = Path(args.out)
    else:
        acts_v, acts_v1, scorer, cfg = real_problem_and_scorer(args.config)
        out_path = cfg.out("ablations.json")

    acts_v = filter_modules(acts_v, args.modules)
    acts_v1 = filter_modules(acts_v1, args.modules)

    mean_drift = sum(drift(acts_v[m]["in"], acts_v1[m]["in"]) for m in acts_v) / len(acts_v)
    print(f"[ablate] {len(acts_v)} modules, mean input drift={mean_drift:.4f}\n")

    rows = []
    header = f"{'method':>10}{'lam':>8}{'anchors':>9}{'scope':>11}   score"
    print(header)
    print("-" * len(header))
    for c in enumerate_configs():
        av = subsample(acts_v, c["anchor_tokens"])
        av1 = subsample(acts_v1, c["anchor_tokens"])
        maps = build_maps(av, av1, c)
        sc = scorer(maps)
        rows.append({**c, "score": sc})
        s = sc.get("recovery", next(iter(sc.values())))
        print(f"{c['method']:>10}{c['lam']:>8.0e}{c['anchor_tokens']:>9}"
              f"{c['map_scope']:>11}   {s:+.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\n[ablate] {len(rows)} configs -> {out_path}")


if __name__ == "__main__":
    main()
