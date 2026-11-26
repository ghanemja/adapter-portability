#!/usr/bin/env python
"""Baseline -- Cross-LoRA (Xia et al. 2025, arXiv 2508.05232): weight-only
SVD-subspace transfer. Uses the two base checkpoints and the adapter only; no
anchors, no activations. See src/adaptport/crosslora.py for the equations.

Writes results_crosslora.json (and, with --tag rK, results_crosslora_rK.json)
so evaluate.py can place it next to naive / procrustes / ridge / oracle.

Usage:
    python experiments/run_crosslora_transfer.py --config configs/qwen2_to_qwen25_1.5b.yaml
    python experiments/run_crosslora_transfer.py --config ... --rank 64 --tag r64
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptport import load_config
from adaptport.crosslora import (
    collect_target_weights,
    crosslora_transplant,
    fit_crosslora_projectors,
)
from adaptport.eval import evaluate
from adaptport.models import load_base
from adaptport.transfer import (
    attach_adapter,
    load_lora_state,
    write_transplanted_adapter,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rank", type=int, default=320,
                    help="SVD truncation rank k (Cross-LoRA paper, Table 5: 320)")
    ap.add_argument("--tag", default="",
                    help="suffix for output files, e.g. r64 -> results_crosslora_r64.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    name = "crosslora" + (f"_{args.tag}" if args.tag else "")

    # Source base weights (M_v): load, grab target modules, free.
    model_v, _ = load_base(cfg.pair.base_v, cfg.device, cfg.dtype)
    w_v = collect_target_weights(model_v, cfg.pair.target_modules)
    del model_v
    torch.cuda.empty_cache()

    base, tok = load_base(cfg.pair.base_v_next, cfg.device, cfg.dtype)
    w_v_next = collect_target_weights(base, cfg.pair.target_modules)
    projectors = fit_crosslora_projectors(w_v, w_v_next, rank=args.rank)
    print(f"[crosslora] fit {len(projectors)} module projectors at rank k={args.rank}")
    torch.save(projectors, cfg.out("cache", f"{name}_projectors.pt"))

    results = {}
    for task in cfg.tasks:
        state = load_lora_state(task.adapter_path)
        new_state = crosslora_transplant(state, projectors)
        out_dir = str(cfg.out("adapters", f"{task.name}_{name}"))
        write_transplanted_adapter(task.adapter_path, new_state, out_dir)
        model = attach_adapter(base, out_dir)
        results[task.name] = evaluate(model, tok, task, cfg.device)
        results[task.name]["svd_rank"] = args.rank
        print(f"[{name}] {task.name}: {results[task.name]}")
        model.unload()
        torch.cuda.empty_cache()

    out = cfg.out(f"results_{name}.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"[{name}] -> {out}")


if __name__ == "__main__":
    main()
