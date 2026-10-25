#!/usr/bin/env python
"""Method -- linear representational alignment.

Reuses cached anchor activations (run run_cka_drift.py first), fits per-module
P_in/P_out maps, transplants each task adapter onto M_{v+1}, evaluates.

Usage:
    python experiments/run_linear_alignment.py --config configs/qwen2_to_qwen25_1.5b.yaml
    python experiments/run_linear_alignment.py --config ... --method procrustes
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptport import load_config
from adaptport.alignment import fit_module_maps, transplant_adapter
from adaptport.eval import evaluate
from adaptport.models import load_base
from adaptport.transfer import (
    attach_adapter,
    load_lora_state,
    write_transplanted_adapter,
)
from run_cka_drift import cache_activations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", default=None, help="override align.method")
    args = ap.parse_args()
    cfg = load_config(args.config)
    method = args.method or cfg.align.method

    acts_v = cache_activations(cfg, "v", cfg.pair.base_v)
    acts_v_next = cache_activations(cfg, "v_next", cfg.pair.base_v_next)
    maps = fit_module_maps(acts_v, acts_v_next, method=method,
                           lam=cfg.align.ridge_lambda, center=cfg.align.center)
    print(f"[align] fit {len(maps)} module maps via {method}")

    base, tok = load_base(cfg.pair.base_v_next, cfg.device, cfg.dtype)
    results = {}
    for task in cfg.tasks:
        state = load_lora_state(task.adapter_path)
        new_state = transplant_adapter(state, maps)
        out_dir = str(cfg.out("adapters", f"{task.name}_{method}"))
        write_transplanted_adapter(task.adapter_path, new_state, out_dir)
        model = attach_adapter(base, out_dir)
        results[task.name] = evaluate(model, tok, task, cfg.device)
        print(f"[align:{method}] {task.name}: {results[task.name]}")
        del model
        torch.cuda.empty_cache()

    out = cfg.out(f"results_aligned_{method}.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"[align:{method}] -> {out}")


if __name__ == "__main__":
    main()
