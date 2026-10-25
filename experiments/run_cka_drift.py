#!/usr/bin/env python
"""Stage 1 -- measure representational drift between M_v and M_{v+1}.

Extracts per-module anchor activations from both base checkpoints, caches them
(reused by run_linear_alignment.py), and writes per-module CKA drift to JSON.

Usage:
    python experiments/run_cka_drift.py --config configs/qwen2_to_qwen25_1.5b.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptport import load_config
from adaptport.anchors import load_anchor_loader
from adaptport.cka import drift
from adaptport.extract import extract_activations
from adaptport.models import load_base


def cache_activations(cfg, which: str, model_id: str):
    """Extract + cache activations for one base; return the activations dict."""
    cache = cfg.out("cache", f"acts_{which}.pt")
    if cache.exists():
        print(f"[cache] load {cache}")
        return torch.load(cache)
    model, tok = load_base(model_id, cfg.device, cfg.dtype)
    loader = load_anchor_loader(cfg.anchors, tok)
    acts = extract_activations(model, loader, cfg.pair.target_modules,
                               device=cfg.device, seed=cfg.anchors.seed)
    torch.save(acts, cache)
    del model
    torch.cuda.empty_cache()
    return acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    acts_v = cache_activations(cfg, "v", cfg.pair.base_v)
    acts_v_next = cache_activations(cfg, "v_next", cfg.pair.base_v_next)

    drifts = {
        name: drift(acts_v[name]["in"], acts_v_next[name]["in"])
        for name in acts_v if name in acts_v_next
    }
    out = cfg.out("drift.json")
    out.write_text(json.dumps(drifts, indent=2))
    mean = sum(drifts.values()) / len(drifts)
    print(f"[drift] mean={mean:.4f} over {len(drifts)} modules -> {out}")


if __name__ == "__main__":
    main()
