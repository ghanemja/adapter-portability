"""Build the small unlabeled anchor set used to fit alignment maps.

Anchors are generic text passed through BOTH base checkpoints to obtain
token-aligned activations. No task labels are used.
"""
from __future__ import annotations

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from .config import AnchorConfig


def load_anchor_loader(cfg: AnchorConfig, tokenizer, batch_size: int = 8) -> DataLoader:
    """Returns a DataLoader of tokenized anchor batches.

    The SAME tokenizer must be used for both base models (the paper assumes a
    shared/alignable tokenizer; see Limitations). Pass the M_v tokenizer and
    reuse the produced input_ids for both models so activations are token-aligned.
    """
    ds = load_dataset(cfg.dataset, cfg.subset, split=cfg.split)
    # filter empties, take n_sequences deterministically
    ds = ds.filter(lambda r: len(r.get("text", "").strip()) > 0)
    ds = ds.shuffle(seed=cfg.seed).select(range(min(cfg.n_sequences, len(ds))))

    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_length,
            padding="max_length",
        )

    ds = ds.map(tok, batched=True, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def collate(rows):
        return {
            "input_ids": torch.stack([r["input_ids"] for r in rows]),
            "attention_mask": torch.stack([r["attention_mask"] for r in rows]),
        }

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
