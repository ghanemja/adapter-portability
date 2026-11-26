"""Task evaluation for a model+adapter.

Metrics (TaskConfig.metric):
    perplexity             token-level perplexity over `text_field`
    completion_perplexity  perplexity of `answer_field` tokens only, conditioned
                           on `prompt_template` (loss masked on the prompt)
    accuracy               GSM8K-style numeric exact match (greedy decode)
    exact_match            generic greedy-decode exact match of the normalized
                           completion against `answer_field` (or `label_names`)

Used to score the reference points per task:
    naive | Cross-LoRA | procrustes-aligned | ridge-aligned | oracle (retrained)
and to compute the fraction of the oracle-naive gap recovered.
"""
from __future__ import annotations

import math
import re
import string

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


def _normalize(s: str) -> str:
    s = s.strip().split("\n")[0].lower()
    return s.translate(str.maketrans("", "", string.punctuation)).strip()


def load_task_dataset(cfg: TaskConfig):
    """Load `eval_dataset[eval_split]`, honoring an optional dataset subset
    (`eval_subset`) and the GSM8K 'main' config fallback."""
    if cfg.eval_subset:
        ds = load_dataset(cfg.eval_dataset, cfg.eval_subset, split=cfg.eval_split)
    else:
        try:
            ds = load_dataset(cfg.eval_dataset, split=cfg.eval_split)
        except ValueError:
            ds = load_dataset(cfg.eval_dataset, "main", split=cfg.eval_split)
    return ds.select(range(min(cfg.max_eval_samples, len(ds))))


def render_prompt(cfg: TaskConfig, row) -> str:
    """Prompt for generation / conditioning. `prompt_template` is a Python
    format string over the row's fields; default is '{text_field}\\nAnswer:'."""
    if cfg.prompt_template:
        return cfg.prompt_template.format(**row)
    return f"{row[cfg.text_field]}\nAnswer:"


def gold_answer(cfg: TaskConfig, row) -> str:
    """Target string: `answer_field`, mapped through `label_names` if the
    field is an integer class id."""
    field = cfg.answer_field or "answer"
    val = row[field]
    if cfg.label_names and isinstance(val, int):
        return cfg.label_names[val]
    return str(val)


@torch.no_grad()
def perplexity(model, tokenizer, cfg: TaskConfig, device: str = "cuda",
               max_length: int = 512) -> float:
    ds = load_dataset(cfg.eval_dataset, cfg.eval_subset, split=cfg.eval_split) \
        if cfg.eval_subset else load_dataset(cfg.eval_dataset, split=cfg.eval_split)
    ds = ds.filter(lambda r: len(str(r.get(cfg.text_field, "")).strip()) > 0)
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
def completion_perplexity(model, tokenizer, cfg: TaskConfig, device: str = "cuda",
                          max_length: int = 1024) -> float:
    """Perplexity of the gold completion given the prompt (prompt tokens are
    masked out of the loss). This is the quantity a task adapter actually
    changes, so oracle-naive gaps are large where whole-text perplexity is flat."""
    ds = load_task_dataset(cfg)
    total_nll, total_tok = 0.0, 0
    for row in tqdm(ds, desc=f"cppl[{cfg.name}]"):
        prompt = render_prompt(cfg, row)
        answer = " " + gold_answer(cfg, row).strip()
        p_ids = tokenizer(prompt, add_special_tokens=True).input_ids
        a_ids = tokenizer(answer, add_special_tokens=False).input_ids
        if tokenizer.eos_token_id is not None:
            a_ids = a_ids + [tokenizer.eos_token_id]
        ids = (p_ids + a_ids)[:max_length]
        n_prompt = min(len(p_ids), len(ids))
        if len(ids) - n_prompt < 1:
            continue
        input_ids = torch.tensor([ids], device=device)
        labels = input_ids.clone()
        labels[:, :n_prompt] = -100
        out = model(input_ids, labels=labels)
        n = len(ids) - n_prompt
        total_nll += out.loss.item() * n
        total_tok += n
    return math.exp(total_nll / max(total_tok, 1))


@torch.no_grad()
def _generate_batch(model, tokenizer, prompts: list[str], device: str,
                    max_new_tokens: int) -> list[str]:
    """Greedy decode a batch of prompts with left padding; returns completions."""
    side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    finally:
        tokenizer.padding_side = side
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    gen = out[:, enc.input_ids.shape[1]:]
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


def _batched(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


@torch.no_grad()
def gsm8k_accuracy(model, tokenizer, cfg: TaskConfig, device: str = "cuda",
                   max_new_tokens: int | None = None) -> float:
    """Greedy-decode numeric exact-match accuracy for GSM8K-style tasks.
    Compares the final number in the completion with the gold number after '####'."""
    ds = load_task_dataset(cfg)
    mnt = max_new_tokens or cfg.max_new_tokens or 256
    rows = list(ds)
    correct = 0
    for chunk in tqdm(list(_batched(rows, cfg.eval_batch_size)), desc=f"acc[{cfg.name}]"):
        gens = _generate_batch(model, tokenizer, [render_prompt(cfg, r) for r in chunk],
                               device, mnt)
        for row, gen in zip(chunk, gens):
            gold = _gsm8k_gold(str(row.get(cfg.answer_field or "answer", "")))
            pred = _last_number(gen)
            if pred is not None and gold is not None and abs(pred - gold) < 1e-6:
                correct += 1
    return correct / max(len(rows), 1)


@torch.no_grad()
def exact_match(model, tokenizer, cfg: TaskConfig, device: str = "cuda") -> float:
    """Greedy-decode exact match of the normalized first line of the completion
    against the normalized gold string (label words for classification)."""
    ds = load_task_dataset(cfg)
    mnt = cfg.max_new_tokens or 8
    rows = list(ds)
    correct = 0
    for chunk in tqdm(list(_batched(rows, cfg.eval_batch_size)), desc=f"em[{cfg.name}]"):
        gens = _generate_batch(model, tokenizer, [render_prompt(cfg, r) for r in chunk],
                               device, mnt)
        for row, gen in zip(chunk, gens):
            if _normalize(gen) == _normalize(gold_answer(cfg, row)):
                correct += 1
    return correct / max(len(rows), 1)


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
HIGHER_BETTER = {"accuracy", "exact_match"}


def evaluate(model, tokenizer, cfg: TaskConfig, device: str = "cuda") -> dict:
    fns = {
        "perplexity": perplexity,
        "completion_perplexity": completion_perplexity,
        "accuracy": gsm8k_accuracy,
        "exact_match": exact_match,
    }
    if cfg.metric not in fns:
        raise NotImplementedError(f"metric {cfg.metric} not implemented")
    return {"metric": cfg.metric, "value": fns[cfg.metric](model, tokenizer, cfg, device)}
