"""Fit per-module alignment maps and transplant a LoRA adapter.

Convention (see Eq. (transplant) in the paper):
    P_in  maps an M_{v+1} input  activation -> the matching M_v input  activation
    P_out maps an M_v     output activation -> the matching M_{v+1} output activation

We fit each map M such that  source @ M.T ~= target :
    P_in  = fit_map(source=H_in[v+1],  target=H_in[v])      # (d_in_v   x d_in_{v+1})
    P_out = fit_map(source=H_out[v],   target=H_out[v+1])   # (d_out_{v+1} x d_out_v)

Then the transplanted adapter keeps rank r:
    A' = A  @ P_in        # (r, d_in_{v+1})
    B' = P_out @ B        # (d_out_{v+1}, r)
"""
from __future__ import annotations

import torch


def fit_map(
    source: torch.Tensor,
    target: torch.Tensor,
    method: str = "ridge",
    lam: float = 1.0,
    center: bool = True,
) -> torch.Tensor:
    """Return M with source @ M.T ~= target.

    source: (N, d_s), target: (N, d_t). Returns M: (d_t, d_s).
    method="ridge": ridge-regularized least squares (allows d_s != d_t).
    method="procrustes": nearest orthogonal map (requires d_s == d_t).
    """
    s = source.float()
    t = target.float()
    if center:
        s = s - s.mean(0, keepdim=True)
        t = t - t.mean(0, keepdim=True)

    if method == "ridge":
        d_s = s.shape[1]
        gram = s.t() @ s + lam * torch.eye(d_s, dtype=s.dtype)
        # M^T = (S^T S + lam I)^-1 S^T T  ->  M = (T^T S)(S^T S + lam I)^-1
        m = (t.t() @ s) @ torch.linalg.inv(gram)
        return m
    if method == "procrustes":
        if s.shape[1] != t.shape[1]:
            raise ValueError("procrustes requires equal feature dims")
        # min_{M orthogonal} ||T - S M^T||  ->  M = U V^T, SVD(S^T T) = U S V^T
        u, _, vt = torch.linalg.svd(s.t() @ t, full_matrices=False)
        return (u @ vt).t()
    if method == "identity":
        d = source.shape[1]
        return torch.eye(d)
    raise ValueError(f"unknown method {method}")


def fit_module_maps(
    acts_v: dict[str, dict[str, torch.Tensor]],
    acts_v_next: dict[str, dict[str, torch.Tensor]],
    method: str = "ridge",
    lam: float = 1.0,
    center: bool = True,
) -> dict[str, dict[str, torch.Tensor]]:
    """Fit {module: {"P_in", "P_out"}} from cached activations of both bases."""
    maps: dict[str, dict[str, torch.Tensor]] = {}
    for name in acts_v:
        if name not in acts_v_next:
            continue
        p_in = fit_map(acts_v_next[name]["in"], acts_v[name]["in"], method, lam, center)
        p_out = fit_map(acts_v[name]["out"], acts_v_next[name]["out"], method, lam, center)
        maps[name] = {"P_in": p_in, "P_out": p_out}
    return maps


def _match_module(adapter_key: str, maps: dict) -> str | None:
    """LoRA keys look like '...layers.3.self_attn.q_proj.lora_A.weight'.
    Find the activation-module name (e.g. '...self_attn.q_proj') it belongs to."""
    for mod_name in maps:
        if mod_name in adapter_key:
            return mod_name
    return None


def transplant_adapter(
    lora_state: dict[str, torch.Tensor],
    maps: dict[str, dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Apply A' = A P_in, B' = P_out B to a LoRA state_dict.

    Modules without a fitted map are left unchanged (naive on that module).
    Returns a NEW state dict; input is not mutated.
    """
    out = dict(lora_state)
    for key, w in lora_state.items():
        mod = _match_module(key, maps)
        if mod is None:
            continue
        if key.endswith("lora_A.weight"):                # A: (r, d_in)
            out[key] = (w.float() @ maps[mod]["P_in"].to(w.float())).to(w.dtype)
        elif key.endswith("lora_B.weight"):              # B: (d_out, r)
            out[key] = (maps[mod]["P_out"].to(w.float()) @ w.float()).to(w.dtype)
    return out
