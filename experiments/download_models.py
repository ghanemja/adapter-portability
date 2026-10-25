#!/usr/bin/env python
"""Pre-download both base checkpoints for a config (run on the GPU box).

Pulls M_v and M_{v+1} from HF into the local cache so the pipeline runs offline.
Gated repos (Gemma) require a license accept + token:

    huggingface-cli login            # paste a token from hf.co/settings/tokens
    # or: export HF_TOKEN=hf_xxx

Usage:
    python experiments/download_models.py --config configs/qwen2_to_qwen25_1.5b.yaml
    python experiments/download_models.py --config configs/gemma3_to_gemma4_12b.yaml
"""
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptport import load_config

# weights + tokenizer + config only; skip the rest to save bandwidth
ALLOW = ["*.safetensors", "*.json", "*.model", "tokenizer*", "*.txt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    token = os.environ.get("HF_TOKEN")

    for repo in (cfg.pair.base_v, cfg.pair.base_v_next):
        print(f"[download] {repo} ...")
        path = snapshot_download(repo, allow_patterns=ALLOW, token=token)
        print(f"[download]   -> {path}")
    print("[download] done. Run check_tokenizer_compat.py next, then the pipeline.")


if __name__ == "__main__":
    main()
