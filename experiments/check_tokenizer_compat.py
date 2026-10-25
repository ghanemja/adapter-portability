#!/usr/bin/env python
"""Gate: confirm two base versions tokenize plain text IDENTICALLY.

The alignment method needs token-aligned anchor activations, which requires the
SAME input_ids from both models for the same text. A small vocab delta (added
special tokens) is fine as long as ordinary text maps to identical ids. Run this
before trusting any cross-version result -- especially for Gemma 3 -> Gemma 4
(vocab 262208 vs 262144).

Usage:
    python experiments/check_tokenizer_compat.py google/gemma-3-12b-pt google/gemma-4-12b-pt
"""
import sys

from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    if len(sys.argv) != 3:
        print("usage: check_tokenizer_compat.py <base_v> <base_v_next>")
        sys.exit(2)
    a, b = AutoTokenizer.from_pretrained(sys.argv[1]), AutoTokenizer.from_pretrained(sys.argv[2])

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    samples = [r["text"] for r in ds if len(r["text"].strip()) > 0][:1000]

    mismatch = 0
    for text in samples:
        if a(text)["input_ids"] != b(text)["input_ids"]:
            mismatch += 1
    rate = mismatch / len(samples)
    print(f"vocab: {len(a)} vs {len(b)}")
    print(f"tokenization mismatch on {mismatch}/{len(samples)} plain-text samples "
          f"({rate*100:.2f}%)")
    if rate == 0:
        print("PASS -- tokenizers are anchor-compatible; activations will be token-aligned.")
    else:
        print("FAIL -- text tokenizes differently; anchors are NOT token-aligned. "
              "Re-tokenize per model and align on a shared text-span mapping, or "
              "pick a different pair.")
    sys.exit(0 if rate == 0 else 1)


if __name__ == "__main__":
    main()
