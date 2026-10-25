#!/usr/bin/env python
"""Controlled mechanism validation -- no model download required.

Proves the paper's central claim on synthetic-but-faithful data using the REAL
adaptport.cka and adaptport.alignment code:

  1. Plant a known representational drift (G_in, G_out) between version v and v+1
     activation spaces, plus noise.
  2. Plant a rank-r LoRA adapter ΔW = B A in v's geometry.
  3. Show NAIVE transfer (drop ΔW onto v+1) produces the wrong delta.
  4. Show ALIGNED transfer (ΔW' = P_out ΔW P_in, maps fit from anchors only)
     recovers the correct delta G_out·ΔW·a_v.
  5. Cross-check that transplant_adapter() on a peft-style state dict matches.

Run:  python experiments/validate_mechanism.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptport.alignment import fit_module_maps, transplant_adapter
from adaptport.cka import drift

torch.manual_seed(0)

D, R, N_ANCHOR, N_TEST = 64, 4, 4000, 512
MODULE = "model.layers.0.self_attn.q_proj"   # name the maps key off of


def random_drift(d, lo=0.7, hi=1.3):
    """Well-conditioned linear drift = rotation * anisotropic scaling.
    Non-orthogonal so it is visible to CKA (orthogonal drift would not be)."""
    q1, _ = torch.linalg.qr(torch.randn(d, d))
    q2, _ = torch.linalg.qr(torch.randn(d, d))
    s = torch.empty(d).uniform_(lo, hi)
    return q1 @ torch.diag(s) @ q2.t()


def rel_err(pred, target):
    return (pred - target).norm() / target.norm()


def main():
    # --- planted drift + base + adapter -------------------------------------
    G_in, G_out = random_drift(D), random_drift(D)
    W_base = torch.randn(D, D) / D ** 0.5
    B = torch.randn(D, R) / R ** 0.5
    A = torch.randn(R, D) / D ** 0.5
    dW = B @ A                                   # rank-r LoRA delta in v geometry

    # --- anchors: same inputs, two version geometries -----------------------
    X = torch.randn(N_ANCHOR, D)                 # v input activations
    noise = lambda: 0.02 * torch.randn(N_ANCHOR, D)
    acts_v = {MODULE: {"in": X,                "out": X @ W_base.t()}}
    acts_v1 = {MODULE: {"in": X @ G_in.t() + noise(),
                        "out": (X @ W_base.t()) @ G_out.t() + noise()}}

    # --- representational drift (real cka.py) -------------------------------
    d_in = drift(acts_v[MODULE]["in"], acts_v1[MODULE]["in"])
    d_out = drift(acts_v[MODULE]["out"], acts_v1[MODULE]["out"])
    print(f"[drift] input CKA-drift  = {d_in:.4f}")
    print(f"[drift] output CKA-drift = {d_out:.4f}")

    # --- fit alignment maps from anchors only (real alignment.py) ------------
    for method, lam in [("ridge", 1e-3), ("procrustes", 0.0)]:
        maps = fit_module_maps(acts_v, acts_v1, method=method, lam=lam, center=False)
        P_in, P_out = maps[MODULE]["P_in"], maps[MODULE]["P_out"]

        # held-out test inputs; "ground truth" delta expressed in v+1 geometry
        a_v = torch.randn(N_TEST, D)
        a_v1 = a_v @ G_in.t()                    # same inputs, v+1 input basis
        delta_v = a_v @ dW.t()
        correct = delta_v @ G_out.t()            # target delta in v+1 output basis

        naive = a_v1 @ dW.t()                    # T = identity
        dW_aligned = P_out @ dW @ P_in
        aligned = a_v1 @ dW_aligned.t()

        # cross-check the state-dict plumbing matches the matrix form
        state = {f"base_model.model.{MODULE}.lora_A.weight": A,
                 f"base_model.model.{MODULE}.lora_B.weight": B}
        ts = transplant_adapter(state, maps)
        dW_from_state = ts[f"base_model.model.{MODULE}.lora_B.weight"] @ \
                        ts[f"base_model.model.{MODULE}.lora_A.weight"]
        plumbing = rel_err(dW_from_state, dW_aligned).item()

        print(f"\n[{method}] naive   delta rel-err vs correct = {rel_err(naive, correct):.4f}")
        print(f"[{method}] aligned delta rel-err vs correct = {rel_err(aligned, correct):.4f}")
        print(f"[{method}] state-dict transplant vs matrix    = {plumbing:.2e}")
        recovered = 1 - rel_err(aligned, correct) / rel_err(naive, correct)
        print(f"[{method}] error reduction vs naive          = {recovered*100:.1f}%")

    print("\nPASS if: drift>0, aligned rel-err << naive rel-err, plumbing ~0.")


if __name__ == "__main__":
    main()
