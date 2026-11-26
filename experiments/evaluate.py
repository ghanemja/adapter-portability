#!/usr/bin/env python
"""Aggregate the main results table:
    naive | Cross-LoRA | procrustes-aligned | ridge-aligned | oracle, per task,
    plus the fraction of the oracle-naive gap each transfer method recovers.

Reads the results_*.json produced by the other stages. Oracle results are
optional (drop results_oracle.json in the run dir to include the upper bound).
Any extra sweep files (results_crosslora_r64.json, results_aligned_ridge_lam0.1.json,
...) are picked up automatically as additional conditions.

Usage:
    python experiments/evaluate.py --config configs/qwen2_to_qwen25_1.5b.yaml
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptport import load_config
from adaptport.eval import HIGHER_BETTER, gap_recovered

# fixed column order for the four reference conditions + the two bounds
CORE = [
    ("naive", "results_naive.json"),
    ("crosslora", "results_crosslora.json"),
    ("procrustes", "results_aligned_procrustes.json"),
    ("ridge", "results_aligned_ridge.json"),
    ("oracle", "results_oracle.json"),
]


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else {}


def discover_conditions(run: Path) -> list[tuple[str, Path]]:
    """Core conditions first, then any sweep variants found in the run dir."""
    conds = [(name, run / fn) for name, fn in CORE]
    known = {fn for _, fn in CORE}
    extras = sorted(p for p in run.glob("results_*.json") if p.name not in known
                    and p.name != "summary.json")
    for p in extras:
        name = p.stem[len("results_"):].replace("aligned_", "")
        conds.append((name, p))
    return conds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Path(cfg.output_dir)

    conds = discover_conditions(run)
    data = {name: _load(p) for name, p in conds}
    present = [n for n, _ in conds if data[n]]
    have_oracle = bool(data.get("oracle"))

    def v(name, t):
        return data.get(name, {}).get(t, {}).get("value", float("nan"))

    w = 15
    print(f"{'task':<16}{'metric':>22}" + "".join(f"{n[:w-1]:>{w}}" for n in present))
    rows = []
    for task in cfg.tasks:
        t = task.name
        higher = task.metric in HIGHER_BETTER
        vals = {n: v(n, t) for n in present}
        row = {"task": t, "metric": task.metric, **vals}
        print(f"{t:<16}{task.metric:>22}" + "".join(f"{vals[n]:>{w}.3f}" for n in present))
        if have_oracle:
            gaps = {}
            for n in present:
                if n in ("naive", "oracle") or math.isnan(vals[n]):
                    continue
                gaps[n] = gap_recovered(vals["naive"], vals[n], vals["oracle"],
                                        higher_better=higher) * 100
            row["gap_recovered_pct"] = gaps
            # backward-compatible scalar used by earlier notes/plots
            row["gap_recovered_ridge"] = gaps.get("ridge", float("nan"))
            print(f"{'  gap recovered %':<38}" + "".join(
                f"{gaps.get(n, float('nan')):>{w}.1f}" for n in present))
        rows.append(row)

    out = run / "summary.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n[summary] -> {out}")


if __name__ == "__main__":
    main()
