#!/usr/bin/env python
"""Upper bound -- evaluate the ORACLE adapter (trained directly on M_{v+1}).

Writes results_oracle.json so evaluate.py can report the % of the oracle-naive
gap that alignment recovers. Adapter paths are taken from --oracle-dir/<task>.

Usage:
    python experiments/run_oracle.py --config <cfg> --oracle-dir runs/.../adapters_oracle
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptport import load_config
from adaptport.eval import evaluate
from adaptport.models import load_base
from adaptport.transfer import attach_adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--oracle-dir", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    base, tok = load_base(cfg.pair.base_v_next, cfg.device, cfg.dtype)
    results = {}
    for task in cfg.tasks:
        adapter = str(Path(args.oracle_dir) / task.name)
        model = attach_adapter(base, adapter)
        results[task.name] = evaluate(model, tok, task, cfg.device)
        print(f"[oracle] {task.name}: {results[task.name]}")

    out = cfg.out("results_oracle.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"[oracle] -> {out}")


if __name__ == "__main__":
    main()
