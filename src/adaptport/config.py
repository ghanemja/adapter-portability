"""Experiment configuration: typed dataclasses + YAML loader."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelPair:
    """A same-family version pair, M_v -> M_{v+1}."""
    family: str                      # e.g. "qwen"
    base_v: str                      # HF id of M_v,   e.g. "Qwen/Qwen2-1.5B"
    base_v_next: str                 # HF id of M_{v+1}, e.g. "Qwen/Qwen2.5-1.5B"
    # nn.Linear module name substrings to adapt / align (LoRA targets).
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class AnchorConfig:
    dataset: str = "Salesforce/wikitext"    # HF dataset for unlabeled anchors
    subset: str | None = "wikitext-2-raw-v1"
    split: str = "train"
    n_sequences: int = 2000
    max_length: int = 256
    seed: int = 0


@dataclass
class AlignConfig:
    method: str = "ridge"                    # "ridge" | "procrustes" | "identity"
    ridge_lambda: float = 1.0
    per_layer: bool = True                    # one map per module vs shared
    center: bool = True                       # mean-center activations before fit


@dataclass
class TaskConfig:
    name: str                                 # human-readable task name
    adapter_path: str                         # LoRA adapter trained on M_v
    eval_dataset: str                         # HF dataset id for evaluation
    eval_split: str = "validation"
    metric: str = "perplexity"                # "perplexity" | "accuracy"
    text_field: str = "text"
    max_eval_samples: int = 500


@dataclass
class ExperimentConfig:
    pair: ModelPair
    anchors: AnchorConfig
    align: AlignConfig
    tasks: list[TaskConfig]
    output_dir: str = "runs/default"
    device: str = "cuda"                      # "cuda" | "cpu" | "mps"
    dtype: str = "bfloat16"                    # cached activations downcast to fp32

    def out(self, *parts: str) -> Path:
        p = Path(self.output_dir).joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def _build(cls, data: dict[str, Any]):
    return cls(**data) if data is not None else cls()


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return ExperimentConfig(
        pair=ModelPair(**raw["pair"]),
        anchors=_build(AnchorConfig, raw.get("anchors", {})),
        align=_build(AlignConfig, raw.get("align", {})),
        tasks=[TaskConfig(**t) for t in raw["tasks"]],
        output_dir=raw.get("output_dir", "runs/default"),
        device=raw.get("device", "cuda"),
        dtype=raw.get("dtype", "bfloat16"),
    )
