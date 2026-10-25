"""Task evaluation: perplexity (and a hook for accuracy) for a model+adapter.

Used to score the four reference points per task:
    naive | ridge-aligned | procrustes-aligned | oracle (retrained)
and to compute the fraction of the oracle-naive gap recovered.
"""
from __future__ import annotations

import math
import re

import torch
from datasets import load_dataset
from tqdm import tqdm

from .config import TaskConfig

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _last_number(text: str) -> float | None:
    """Final numeric token in a string (commas stripped). None if absent."""
    matches = _NUM.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def _gsm8k_gold(answer: str) -> float | None:
    """GSM8K gold answer is the number after the '####' marker."""
    tail = answer.split("####")[-1] if "####" in answer else answer
    return _last_number(tail)


@torch.no_grad()
def perplexity(model, tokenizer, cfg: TaskConfig, device: str = "cuda",
               max_length: int = 512) -> float:
    ds = load_dataset(cfg.eval_dataset, split=cfg.eval_split)
    ds = ds.filter(lambda r: len(r.get(cfg.text_field, "").strip()) > 0)
    ds = ds.select(range(min(cfg.max_eval_samples, len(ds))))

    total_nll, total_tok = 0.0, 0
    for row in tqdm(ds, desc=f"ppl[{cfg.name}]"):
        enc = tokenizer(row[cfg.text_field], return_tensors="pt",
                        truncation=True, max_length=max_length)
        ids = enc.input_ids.to(device)
        if ids.shape[1] < 2:
            continue
        out = model(ids, labels=ids)
        n = ids.shape[1] - 1
        total_nll += out.loss.item() * n
        total_tok += n
    return math.exp(total_nll / max(total_tok, 1))


@torch.no_grad()
def gsm8k_accuracy(model, tokenizer, cfg: TaskConfig, device: str = "cuda",
                   max_new_tokens: int = 256) -> float:
    """Greedy-decode numeric exact-match accuracy for GSM8K-style tasks.

    Zero-shot: prompt the question, generate, compare the final number in the
    completion against the gold number after '####'.
    """
    # GSM8K ships under the "main" config; fall back to no-config datasets.
    try:
        ds = load_dataset(cfg.eval_dataset, "main", split=cfg.eval_split)
    except (ValueError, FileNotFoundError):
        ds = load_dataset(cfg.eval_dataset, split=cfg.eval_split)
    ds = ds.select(range(min(cfg.max_eval_samples, len(ds))))
    eos = tokenizer.eos_token_id

    correct = 0
    for row in tqdm(ds, desc=f"acc[{cfg.name}]"):
        gold = _gsm8k_gold(str(row.get("answer", "")))
        prompt = f"{row[cfg.text_field]}\nAnswer:"
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=eos)
        gen = tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        pred = _last_number(gen)
        if pred is not None and gold is not None and abs(pred - gold) < 1e-6:
            correct += 1
    return correct / max(len(ds), 1)


def gap_recovered(naive: float, aligned: float, oracle: float,
                  higher_better: bool = False) -> float:
    """Fraction of the naive->oracle improvement that alignment recovers.
    1.0 = matches oracle, 0.0 = no better than naive. Set higher_better=True
    for accuracy-style metrics (vs the default lower-is-better perplexity)."""
    denom = oracle - naive if higher_better else naive - oracle
    if abs(denom) < 1e-9:
        return float("nan")
    num = aligned - naive if higher_better else naive - aligned
    return num / denom


# metrics where a higher value is better (drives gap_recovered direction)
HIGHER_BETTER = {"accuracy"}


def evaluate(model, tokenizer, cfg: TaskConfig, device: str = "cuda") -> dict:
    if cfg.metric == "perplexity":
        return {"metric": "perplexity", "value": perplexity(model, tokenizer, cfg, device)}
    if cfg.metric == "accuracy":
        return {"metric": "accuracy", "value": gsm8k_accuracy(model, tokenizer, cfg, device)}
    raise NotImplementedError(f"metric {cfg.metric} not implemented yet")
