"""Cross-LoRA baseline (Xia et al., arXiv 2508.05232): weight-only, data-free
LoRA transfer via rank-truncated SVD subspace alignment.

This is the strongest published training-free competitor and the baseline the
paper must beat. It uses ONLY the two base weight matrices per module -- no
anchors, no activations.

Per adapted module with source base weight W_s (m_s x n_s) and target base
weight W_t (m_t x n_t), the paper (Eqs. 1-7) does:

    W_s ~= U_s S_s V_s^T,  W_t ~= U_t S_t V_t^T          rank-k truncated SVD
    P_U = argmin_P ||P U_s - U_t||_F^2,   U~_s = P_U U_s   ("LoRA-Align")
    P_V = argmin_P ||P V_s - V_t||_F^2,   V~_s = P_V V_s
    dW_t = U~_s (U~_s^T dW_s V~_s) V~_s^T                  ("LoRA-Shift")

The least-squares problems are solved with torch.linalg.lstsq as in the paper.
Note: because U_s has orthonormal columns, the minimum-norm solution is
P_U = U_t U_s^T, so U~_s = U_t exactly (and likewise V~_s = V_t) whenever the
truncation rank k <= min(m, n). In the equal-dimension (same-family) setting
studied here Cross-LoRA therefore reduces to projecting dW_s onto the top-k
left/right singular subspaces of the TARGET base weight:

    dW_t = (U_t U_t^T) dW_s (V_t V_t^T).

The projection is linear and factorizes over the LoRA factors, so the rank-r
structure is preserved:  B' = U~_s U~_s^T B,   A' = A V~_s V~_s^T.
"""
from __future__ import annotations

import torch


def truncated_svd(w: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rank-`rank` truncated SVD of a weight matrix (fp32, CPU-safe).
    Returns U (m x k), S (k,), V (n x k) with k = min(rank, m, n)."""
    w = w.detach().float()
    k = min(rank, *w.shape)
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    return u[:, :k], s[:k], vh[:k].t()


def align_basis(b_s: torch.Tensor, b_t: torch.Tensor) -> torch.Tensor:
    """LoRA-Align: solve P = argmin ||P B_s - B_t||_F and return P B_s.

    b_s: (d_s x k) source singular-vector basis, b_t: (d_t x k) target basis.
    Transposed to standard lstsq form  B_s^T P^T = B_t^T  (k x d_s)(d_s x d_t).
    """
    k = min(b_s.shape[1], b_t.shape[1])
    b_s, b_t = b_s[:, :k].float(), b_t[:, :k].float()
    # gelsd gives the minimum-norm solution for the underdetermined system.
    p_t = torch.linalg.lstsq(b_s.t(), b_t.t(), driver="gelsd").solution   # (d_s x d_t)
    return p_t.t() @ b_s                                                   # (d_t x k)


def fit_crosslora_projectors(
    weights_v: dict[str, torch.Tensor],
    weights_v_next: dict[str, torch.Tensor],
    rank: int = 320,
) -> dict[str, dict[str, torch.Tensor]]:
    """Per module, return the aligned source bases {"U": U~_s (m_t x k),
    "V": V~_s (n_t x k)} from the two base weight dicts {module_name: W}."""
    out: dict[str, dict[str, torch.Tensor]] = {}
    for name, w_s in weights_v.items():
        if name not in weights_v_next:
            continue
        u_s, _, v_s = truncated_svd(w_s, rank)
        u_t, _, v_t = truncated_svd(weights_v_next[name], rank)
        out[name] = {"U": align_basis(u_s, u_t), "V": align_basis(v_s, v_t)}
    return out


def _match_module(adapter_key: str, projectors: dict) -> str | None:
    for mod_name in projectors:
        if mod_name in adapter_key:
            return mod_name
    return None


def crosslora_transplant(
    lora_state: dict[str, torch.Tensor],
    projectors: dict[str, dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """LoRA-Shift on a PEFT state dict:  B' = U~ U~^T B,  A' = A V~ V~^T.
    Equivalent to dW_t = U~ (U~^T dW_s V~) V~^T with dW_s = (alpha/r) B A.
    Modules without a projector are left unchanged. Returns a new dict."""
    out = dict(lora_state)
    for key, w in lora_state.items():
        mod = _match_module(key, projectors)
        if mod is None:
            continue
        u, v = projectors[mod]["U"], projectors[mod]["V"]
        if key.endswith("lora_A.weight"):               # A: (r, n)
            out[key] = ((w.float() @ v) @ v.t()).to(w.dtype)
        elif key.endswith("lora_B.weight"):             # B: (m, r)
            out[key] = (u @ (u.t() @ w.float())).to(w.dtype)
    return out


def collect_target_weights(model, target_substrings: list[str]) -> dict[str, torch.Tensor]:
    """{module_name: W.detach().cpu().float()} for the LoRA-target nn.Linear
    modules of a base model. Keys match the activation-cache module names."""
    from .models import target_linear_modules

    return {n: m.weight.detach().float().cpu()
            for n, m in target_linear_modules(model, target_substrings).items()}
