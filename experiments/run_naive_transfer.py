#!/usr/bin/env python
"""Baseline -- naive transfer: load each M_v adapter onto M_{v+1} unchanged.

Usage:
    python experiments/run_naive_transfer.py --config configs/qwen2_to_qwen25_1.5b.yaml
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptport import load_config
from adaptport.eval import evaluate
from adaptport.models import load_base
from adaptport.transfer import naive_transfer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    base, tok = load_base(cfg.pair.base_v_next, cfg.device, cfg.dtype)
    results = {}
    for task in cfg.tasks:
        model = naive_transfer(base, task.adapter_path)
        results[task.name] = evaluate(model, tok, task, cfg.device)
        model = model.unload() if hasattr(model, "unload") else None
        print(f"[naive] {task.name}: {results[task.name]}")

    out = cfg.out("results_naive.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"[naive] -> {out}")


if __name__ == "__main__":
    main()
