#!/usr/bin/env python
"""Plumbing validation on REAL transformers modules -- no weight download.

Exercises the code paths the synthetic test did not: models.target_linear_modules
discovery and extract.extract_activations forward hooks, on two tiny LlamaForCausalLM
instances standing in for M_v and M_{v+1}. Confirms activations are captured per
adapted module, drift is computed, maps fit, and a fake adapter transplants with
correct shapes.

Run:  python experiments/validate_extract.py
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptport.alignment import fit_module_maps, transplant_adapter
from adaptport.cka import drift
from adaptport.extract import extract_activations
from adaptport.models import target_linear_modules

torch.manual_seed(0)

VOCAB, D, LAYERS, HEADS = 256, 64, 2, 4
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def tiny_llama(seed):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=D, intermediate_size=128,
                      num_hidden_layers=LAYERS, num_attention_heads=HEADS,
                      num_key_value_heads=HEADS, max_position_embeddings=128)
    return LlamaForCausalLM(cfg).eval()


def fake_loader(n=64, seqlen=32, bs=8):
    ids = torch.randint(0, VOCAB, (n, seqlen))
    mask = torch.ones_like(ids)
    rows = [{"input_ids": ids[i], "attention_mask": mask[i]} for i in range(n)]
    return DataLoader(rows, batch_size=bs,
                      collate_fn=lambda r: {
                          "input_ids": torch.stack([x["input_ids"] for x in r]),
                          "attention_mask": torch.stack([x["attention_mask"] for x in r])})


def main():
    m_v, m_v1 = tiny_llama(1), tiny_llama(2)   # two "versions" = different weights
    print(f"[discover] target modules in M_v: {len(target_linear_modules(m_v, TARGETS))}")

    loader = fake_loader()
    acts_v = extract_activations(m_v, loader, TARGETS, device="cpu", max_tokens=2000, seed=0)
    acts_v1 = extract_activations(m_v1, loader, TARGETS, device="cpu", max_tokens=2000, seed=0)
    print(f"[extract] captured {len(acts_v)} modules; "
          f"example shape in={tuple(next(iter(acts_v.values()))['in'].shape)}")

    name = next(iter(acts_v))
    print(f"[drift] {name}: {drift(acts_v[name]['in'], acts_v1[name]['in']):.4f}")

    maps = fit_module_maps(acts_v, acts_v1, method="ridge", lam=1.0, center=True)
    print(f"[align] fit maps for {len(maps)} modules")

    # fake rank-r adapter over those modules; confirm transplant preserves shapes
    r = 4
    state = {}
    for mod in maps:
        d_in = acts_v[mod]["in"].shape[1]
        d_out = acts_v[mod]["out"].shape[1]
        state[f"base_model.model.{mod}.lora_A.weight"] = torch.randn(r, d_in)
        state[f"base_model.model.{mod}.lora_B.weight"] = torch.randn(d_out, r)
    ts = transplant_adapter(state, maps)
    ok = all(ts[k].shape == state[k].shape for k in state)
    print(f"[transplant] {len(state)} tensors; shapes preserved = {ok}")
    print("\nPASS if: modules discovered, activations captured, shapes preserved.")


if __name__ == "__main__":
    main()
