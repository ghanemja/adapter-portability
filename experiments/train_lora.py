#!/usr/bin/env python
"""Utility -- train the source (M_v) and oracle (M_{v+1}) LoRA adapters.

These are the inputs/upper-bound for the portability study, NOT the method
itself. Train a task adapter on a base model and save it.

Examples:
    # source adapter on M_v
    python experiments/train_lora.py --base Qwen/Qwen2-1.5B \
        --dataset gsm8k --text-field question --out runs/adapters_v/gsm8k
    # oracle adapter on M_{v+1} (same task, for the upper bound)
    python experiments/train_lora.py --base Qwen/Qwen2.5-1.5B \
        --dataset gsm8k --text-field question --out runs/adapters_oracle/gsm8k
"""
import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def latest_complete_checkpoint(out: Path) -> str | None:
    """Return the newest checkpoint-N dir that finished saving (has
    trainer_state.json + adapter weights). Partial dirs left by a kill
    mid-save are removed so the Trainer never tries to load them."""
    import shutil

    if not out.exists():
        return None
    complete = []
    for d in out.glob("checkpoint-*"):
        if not d.is_dir():
            continue
        ok = (d / "trainer_state.json").exists() and (
            (d / "adapter_model.safetensors").exists() or (d / "adapter_model.bin").exists()
        )
        if ok:
            try:
                complete.append((int(d.name.split("-")[-1]), d))
            except ValueError:
                pass
        else:
            print(f"[train] removing incomplete checkpoint {d}")
            shutil.rmtree(d, ignore_errors=True)
    if not complete:
        return None
    return str(max(complete)[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--subset", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--template", default=None,
                    help="python format string over row fields used as the training text, "
                         "e.g. '{question}\\nAnswer: {answer}'. Overrides --text-field.")
    ap.add_argument("--label-names", nargs="*", default=None,
                    help="map an int 'label' field to words before templating (classification)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-train-samples", type=int, default=None,
                    help="cap dataset size (for fast CPU runs)")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="hard cap on optimizer steps (-1 = use epochs)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default=None, help="force 'cpu' to disable bf16")
    ap.add_argument("--save-steps", type=int, default=25,
                    help="checkpoint every N optimizer steps (resume survives teardown)")
    ap.add_argument("--force", action="store_true",
                    help="retrain even if a finished adapter already exists in --out")
    ap.add_argument("--targets", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj"])
    args = ap.parse_args()
    use_bf16 = args.device != "cpu"

    out = Path(args.out)
    if (out / "adapter_model.safetensors").exists() and not args.force:
        print(f"[train] finished adapter already at {out}; skip (use --force to retrain)")
        return
    resume = latest_complete_checkpoint(out)
    if resume:
        print(f"[train] resuming from {resume}")

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dt = torch.bfloat16 if use_bf16 else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dt)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dt)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, target_modules=args.targets,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    ds = load_dataset(args.dataset, args.subset, split=args.split)
    if args.template:
        tmpl = args.template.replace("\\n", "\n")
        names = args.label_names

        def render(r):
            if names and isinstance(r.get("label"), int):
                r = {**r, "label": names[r["label"]]}
            return {args.text_field: tmpl.format(**r) + (tok.eos_token or "")}

        ds = ds.map(render)
    ds = ds.filter(lambda r: len(str(r.get(args.text_field, "")).strip()) > 0)
    if args.max_train_samples:
        ds = ds.select(range(min(args.max_train_samples, len(ds))))

    def tokenize(batch):
        return tok(batch[args.text_field], truncation=True, max_length=args.max_length)

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=2,
            learning_rate=args.lr, bf16=use_bf16, logging_steps=10,
            # checkpoint so a session teardown mid-train can resume, not restart.
            save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
            report_to=[]),
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    trainer.train(resume_from_checkpoint=resume or None)

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[train] saved adapter -> {args.out}")


if __name__ == "__main__":
    main()
