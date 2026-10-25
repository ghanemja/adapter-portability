"""Linear Centered Kernel Alignment (Kornblith et al., 2019).

drift_l = 1 - CKA(H_v_l, H_{v+1}_l), computed per adapted module over the
anchor activations. We use the linear-CKA closed form that avoids the N x N
Gram matrices:

    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

with X, Y mean-centered over the sample (row) dimension.
"""
from __future__ import annotations

import torch


def _center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """x, y: (N, d) activation matrices over the same N anchor tokens.

    Feature dims may differ; only the sample dim N must match.
    Returns CKA in [0, 1].
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"sample dim mismatch: {x.shape[0]} vs {y.shape[0]}")
    x = _center(x.float())
    y = _center(y.float())
    # ||Y^T X||_F^2 = ||X^T Y||_F^2
    yt_x = y.t() @ x                       # (d_y, d_x)
    hsic = (yt_x ** 2).sum()
    norm_x = (x.t() @ x).norm()
    norm_y = (y.t() @ y).norm()
    denom = norm_x * norm_y
    if denom == 0:
        return 0.0
    return (hsic / denom).item()


def drift(x: torch.Tensor, y: torch.Tensor) -> float:
    """Representational drift = 1 - CKA."""
    return 1.0 - linear_cka(x, y)
