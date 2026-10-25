"""Cache per-module input/output activations over the anchor set.

For each adapted module l we collect:
    H_in[l]  : (N, d_in)  -- the module's input activation
    H_out[l] : (N, d_out) -- the module's output activation
over N real (non-pad) anchor tokens, subsampled to `max_tokens` to bound memory.

These feed cka.py (drift) and alignment.py (fitting P_in, P_out). The SAME
input_ids must be used across both base models so rows of H are token-aligned.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
from tqdm import tqdm

from .models import target_linear_modules


@contextmanager
def _capture(modules: dict[str, torch.nn.Linear]):
    """Forward-hook every target module; yield a dict collecting (in, out) per
    call. Hooks store CPU fp32 tensors to keep GPU memory flat."""
    store: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {n: [] for n in modules}
    handles = []

    def mk(name):
        def hook(_mod, inputs, output):
            x = inputs[0].detach()
            y = output.detach()
            store[name].append((x.float().cpu(), y.float().cpu()))
        return hook

    for name, mod in modules.items():
        handles.append(mod.register_forward_hook(mk(name)))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def extract_activations(
    model,
    loader,
    target_substrings: list[str],
    device: str = "cuda",
    max_tokens: int = 50_000,
    seed: int = 0,
) -> dict[str, dict[str, torch.Tensor]]:
    """Run the anchor loader through `model`, returning
    {module_name: {"in": (N, d_in), "out": (N, d_out)}}.

    Only non-pad tokens are kept; tokens are reservoir-free subsampled by a
    fixed per-batch stride so both models keep the SAME rows when given the same
    loader and seed.
    """
    modules = target_linear_modules(model, target_substrings)
    if not modules:
        raise ValueError(f"no modules matched {target_substrings}")

    gen = torch.Generator().manual_seed(seed)
    flat_in: dict[str, list[torch.Tensor]] = {n: [] for n in modules}
    flat_out: dict[str, list[torch.Tensor]] = {n: [] for n in modules}
    kept = 0

    with _capture(modules) as store:
        for batch in tqdm(loader, desc="extract"):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device).bool()
            for n in store:
                store[n].clear()
            model(input_ids=input_ids, attention_mask=batch["attention_mask"].to(device))

            mask_flat = mask.reshape(-1).cpu()           # (B*T,)
            idx = mask_flat.nonzero(as_tuple=False).squeeze(-1)
            if kept < max_tokens and idx.numel() > 0:
                room = max_tokens - kept
                if idx.numel() > room:
                    perm = torch.randperm(idx.numel(), generator=gen)[:room]
                    idx = idx[perm]
                for n in modules:
                    x, y = store[n][0]                   # one call per forward
                    d_in, d_out = x.shape[-1], y.shape[-1]
                    flat_in[n].append(x.reshape(-1, d_in)[idx])
                    flat_out[n].append(y.reshape(-1, d_out)[idx])
                kept += idx.numel()
            if kept >= max_tokens:
                break

    return {
        n: {"in": torch.cat(flat_in[n], 0), "out": torch.cat(flat_out[n], 0)}
        for n in modules
    }
