#!/usr/bin/env python
"""Aggregate the main results table:
    naive | ridge-aligned | procrustes-aligned | oracle, per task,
    plus % of the oracle-naive gap recovered by alignment.

Reads the results_*.json produced by the other stages. Oracle results are
optional (drop results_oracle.json in the run dir to include the upper bound).

Usage:
    python experiments/evaluate.py --config configs/qwen2_to_qwen25_1.5b.yaml
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptport import load_config
from adaptport.eval import gap_recovered


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Path(cfg.output_dir)

    naive = _load(run / "results_naive.json")
    ridge = _load(run / "results_aligned_ridge.json")
    proc = _load(run / "results_aligned_procrustes.json")
    oracle = _load(run / "results_oracle.json")

    def v(d, t):
        return d.get(t, {}).get("value", float("nan"))

    rows = []
    print(f"{'task':<20}{'metric':>12}{'naive':>10}{'ridge':>10}{'procr':>10}{'oracle':>10}{'gap%(ridge)':>14}")
    for task in cfg.tasks:
        t = task.name
        higher = task.metric == "accuracy"
        n, r, p, o = v(naive, t), v(ridge, t), v(proc, t), v(oracle, t)
        gr = gap_recovered(n, r, o, higher_better=higher) * 100 if oracle else float("nan")
        print(f"{t:<20}{task.metric:>12}{n:>10.3f}{r:>10.3f}{p:>10.3f}{o:>10.3f}{gr:>14.1f}")
        rows.append({"task": t, "naive": n, "ridge": r, "procrustes": p,
                     "oracle": o, "gap_recovered_ridge": gr})

    out = run / "summary.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n[summary] -> {out}")


if __name__ == "__main__":
    main()
