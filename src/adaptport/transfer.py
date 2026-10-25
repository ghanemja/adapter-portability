"""Orchestrate naive and aligned adapter transfer onto M_{v+1}.

naive   : load A onto M_{v+1} unchanged          (T = identity, lower bound)
aligned : transplant A via fitted maps, then load (T from alignment.py)
"""
from __future__ import annotations

import shutil
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file


def load_lora_state(adapter_path: str) -> dict[str, torch.Tensor]:
    """Read a PEFT adapter's LoRA tensors (safetensors or bin)."""
    p = Path(adapter_path)
    st = p / "adapter_model.safetensors"
    if st.exists():
        return load_file(str(st))
    bin_ = p / "adapter_model.bin"
    if bin_.exists():
        return torch.load(bin_, map_location="cpu")
    raise FileNotFoundError(f"no adapter weights under {adapter_path}")


def write_transplanted_adapter(
    src_adapter_path: str,
    new_state: dict[str, torch.Tensor],
    out_dir: str,
) -> str:
    """Clone the source adapter dir (to keep adapter_config.json) and overwrite
    its weights with the transplanted state. Returns out_dir."""
    src, dst = Path(src_adapter_path), Path(out_dir)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for stale in ("adapter_model.bin",):
        if (dst / stale).exists():
            (dst / stale).unlink()
    save_file(new_state, str(dst / "adapter_model.safetensors"))
    return str(dst)


def attach_adapter(base_model, adapter_dir: str):
    """Return M_{v+1} with the (naive or transplanted) adapter loaded."""
    return PeftModel.from_pretrained(base_model, adapter_dir).eval()


def naive_transfer(base_v_next, adapter_path: str):
    """Drop the original adapter onto M_{v+1} unchanged."""
    return attach_adapter(base_v_next, adapter_path)
