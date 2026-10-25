"""Base-model loading and target-module discovery."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_base(model_id: str, device: str = "cuda", dtype: str = "bfloat16"):
    """Load a frozen base causal-LM in eval mode."""
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch_dt = _DTYPES.get(dtype, torch.bfloat16)
    try:                                    # transformers >=5 renamed torch_dtype -> dtype
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dt)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dt)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def target_linear_modules(model, target_substrings: list[str]) -> dict[str, torch.nn.Linear]:
    """Map fully-qualified module name -> nn.Linear, for names containing any
    of the target substrings (the LoRA-adapted / aligned modules)."""
    out: dict[str, torch.nn.Linear] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and any(s in name for s in target_substrings):
            out[name] = mod
    return out
