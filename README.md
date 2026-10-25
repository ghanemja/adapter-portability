# Cross-Version Adapter Portability via Representational Alignment

Treat a LoRA fine-tuning delta as a **portable object**: transplant an adapter
trained on base model `M_v` onto a same-family successor `M_{v+1}` **without
retraining on task data** — using only the two base checkpoints, the adapter,
and a small *unlabeled* anchor set.

Paper target: ICLR 2027 (abstract ~Sep 19 2026, full ~Sep 24 2026). Backups: TMLR, AAAI 2027.

## Idea in one line

A LoRA update `ΔW = (α/r)·B·A` reads/writes in `M_v`'s activation geometry. The
new base version shifts that geometry, so naive transfer breaks. We fit per-module
linear maps from anchor activations and re-express the adapter:

```
ΔW' = P_out · ΔW · P_in = (α/r)·(P_out·B)·(A·P_in)   # still rank r — no extra inference cost
```

## Layout

```
paper/
  main.tex                       ICLR skeleton (builds standalone or with iclr2027 style)
  sections/problem_formulation.tex   formal problem + alignment derivation
  sections/introduction.tex          intro + 5-bullet contributions
  OUTLINE.md                     section-by-section outline
  references.bib
src/adaptport/
  config.py        typed config + YAML loader
  models.py        base-model loading, target-module discovery
  anchors.py       unlabeled anchor set -> tokenized loader
  extract.py       cache per-module in/out activations over anchors
  cka.py           linear CKA / representational drift
  alignment.py     fit P_in,P_out (ridge / orthogonal Procrustes); transplant adapter
  transfer.py      naive + aligned transfer onto M_{v+1}
  eval.py          perplexity + gap-recovered metric
experiments/
  train_lora.py            train source (M_v) + oracle (M_{v+1}) adapters
  run_cka_drift.py         stage 1: extract activations, measure drift
  run_naive_transfer.py    baseline (T = identity)
  run_linear_alignment.py  method (ridge / procrustes)
  evaluate.py              aggregate naive|ridge|procrustes|oracle table
configs/
  qwen2_to_qwen25_1.5b.yaml
```

## Recommended model pair (fast local iteration)

**Primary: `Qwen/Qwen2-1.5B` → `Qwen/Qwen2.5-1.5B`.** Best controlled setup:
- Identical tokenizer (151k) and hidden size across the bump → isolates *pure
  representational drift*, no tokenizer/dimension confounds (satisfies the paper's
  Section "Assumptions and Scope").
- 1.5B fits a single 16–24 GB GPU in bf16; the 0.5B pair (`Qwen2-0.5B` →
  `Qwen2.5-0.5B`) runs on CPU/MPS for laptop debugging.
- Qwen has 3 clean versions (2 → 2.5 → 3) and many sizes → easy second pair for
  generalization.

**Headline pair (the motivating example): `google/gemma-3-12b-pt` → `google/gemma-4-12b-pt`.**
Verified 2026-06-15 to be **dim-matched**: both `hidden_size=3840`, 48 layers, 16
q-heads / 8 kv-heads, `head_dim=256` → `q/k/v/o_proj` shapes identical, adapter maps
1:1. Vocab differs by 64 (262208 vs 262144 = added special tokens only; embeddings/
lm_head are not adapted, so the transplant is unaffected). This is a *clean* same-dim
pair and is literally the paper's motivating scenario. Costs a real GPU (~24–48 GB,
two 12B models — extract sequentially). Both repos gated: accept license + `hf login`.
Run the pre-flight `experiments/check_tokenizer_compat.py` first.

Avoid for Gemma: the **small** Gemma 4 models (E2B/E4B) use Per-Layer Embeddings — a
new architecture unlike Gemma 3's dense small models — so there is no clean *small*
Gemma 3→4 pair. Also avoid Gemma 2→3 (tokenizer changed 256k→262k, no dim match) and
any GGUF / MLX / abliterated / merged variants (can't hook activations or train LoRA).

## Pipeline

```bash
pip install -r requirements.txt

# 0. train source + oracle adapters (once per task)
python experiments/train_lora.py --base Qwen/Qwen2-1.5B   --dataset gsm8k --text-field question --out runs/adapters_v/gsm8k
python experiments/train_lora.py --base Qwen/Qwen2.5-1.5B --dataset gsm8k --text-field question --out runs/adapters_oracle/gsm8k

# 1. drift + cache anchor activations (both bases)
python experiments/run_cka_drift.py        --config configs/qwen2_to_qwen25_1.5b.yaml
# 2. baseline
python experiments/run_naive_transfer.py   --config configs/qwen2_to_qwen25_1.5b.yaml
# 3. method (reuses cached activations)
python experiments/run_linear_alignment.py --config configs/qwen2_to_qwen25_1.5b.yaml --method ridge
python experiments/run_linear_alignment.py --config configs/qwen2_to_qwen25_1.5b.yaml --method procrustes
# 4. aggregate table (drop oracle eval into runs/.../results_oracle.json for the upper bound)
python experiments/evaluate.py             --config configs/qwen2_to_qwen25_1.5b.yaml
```

## Validated locally (CPU, no GPU)

Two harnesses execute the real `adaptport` code with zero model downloads:

```bash
. .venv/bin/activate
python experiments/validate_mechanism.py   # core method: math + alignment.py + transplant
python experiments/validate_extract.py     # models.py discovery + extract.py hooks on real Llama
python experiments/run_ablations.py --synthetic   # ablation sweep on planted multi-module drift
```

Results on a planted, non-orthogonal drift:

| check | result |
|---|---|
| representational drift visible to CKA | input 0.050, output 0.027 (>0 ✓) |
| naive transfer delta error vs ground truth | **1.36** (broken) |
| ridge-aligned delta error | **0.019** → **98.6%** error reduction |
| procrustes-aligned delta error | 0.257 → 81.4% (lower: drift is non-orthogonal — expected) |
| `transplant_adapter` state-dict vs matrix form | 3e-7 (plumbing correct) |
| real-transformers path: modules discovered / hooks / shapes | 8 modules, (2000,64) acts, shapes preserved ✓ |

Takeaway: method, alignment math, transplant plumbing, hook extraction, and CKA
are all correct and executed. **Ridge > Procrustes when drift is non-orthogonal**
— a concrete ablation finding for the paper.

### Ablation sweep findings (`run_ablations.py --synthetic`, planted per-module drift)

24 configs over method × λ × anchor-budget × map-scope. Patterns (recovery =
fraction of naive error removed; higher better):

- **Per-layer maps ≫ one shared map.** Per-layer recovers 0.79–0.94; a single
  global map collapses (0.1–0.3, and *negative* at low λ — worse than naive),
  because drift is module-specific. Strongest ablation story.
- **Anchor-efficient.** Even 100 anchor tokens gives ~0.93 per-layer recovery
  (small d=64; real hidden dims need more, but the trend — diminishing returns —
  holds). Supports the "small anchor set" claim.
- **Regularization matters under misspecification.** Low λ is best for clean
  per-layer fits but catastrophic for the shared map; λ trades fit vs robustness.
- Ridge ≥ Procrustes in per-layer, consistent with the mechanism test.

These are synthetic (illustrative magnitudes); the same harness runs on real
cached activations via `--config` once adapters are trained on a GPU box.

## Status / next steps

- [x] Core mechanism validated (`validate_mechanism.py`) — naive breaks, aligned recovers.
- [x] Real-transformers extract/discover/transplant path validated (`validate_extract.py`).
- [x] peft LoRA key format matches `alignment._match_module` (substring on `...module.lora_A.weight`).
- [x] GSM8K numeric exact-match accuracy in `eval.py` (greedy decode; direction-aware `gap_recovered`).
- [x] Ablation harness over method × λ × anchor-budget × map-scope (`run_ablations.py`, dual real/synthetic).
- [ ] **Needs GPU box:** train source+oracle adapters (`train_lora.py`), run the 4-stage Qwen2→2.5 pipeline.
- [ ] Optional: nonlinear (MLP/RFF) map ablation — breaks the zero-inference-cost property, so a separate eval path.
- [ ] Lincoln Lab public-release clearance before submission.

> Local box here is CPU-only, 8.5 GB RAM — real LLM training is infeasible (hours +
> RAM pressure), so the Qwen pipeline is left as the GPU step. Everything that does
> not require training a real LLM has been run and passes. A `.venv/` (py3.11) with
> torch+transformers is set up for the validation harnesses.
