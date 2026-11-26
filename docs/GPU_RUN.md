# GPU run: Qwen2-1.5B → Qwen2.5-1.5B (the run that decides the paper)

## What you need to provide

1. **One rented GPU box**: A100 40GB (or any ≥40 GB card: A100 80GB, L40S, H100).
   Host needs **≥48 GB RAM** and **≥60 GB free disk**. Lambda, RunPod and Vast
   all offer this on demand. A box with a recent PyTorch + CUDA image saves time.
2. **No API keys.** All four Qwen checkpoints and all datasets are public and
   ungated. An `HF_TOKEN` only raises download rate limits; it is optional.
3. **One command** after cloning the repo on the box:

```bash
git clone https://github.com/ghanemja/adapter-portability && cd adapter-portability
git checkout claude/wizardly-mendel-uupsel
pip install -r requirements.txt matplotlib
bash experiments/run_gpu_1p5b.sh 2>&1 | tee runs/gpu_1p5b.log
```

4. **Bring back the small result files** (a few hundred KB), then commit them:

```bash
# on the GPU box
tar czf results_1p5b.tgz runs/gpu_1p5b.log runs/qwen2_to_qwen25_1.5b/*.json paper/tables paper/figures
```

`.gitignore` already whitelists `runs/*/*.json`. Do not commit caches or adapters.

Every stage is skip-if-done and adapter training resumes from its last
complete checkpoint, so after a preemption re-run the same command.

## What it runs

| stage | what | outputs |
|---|---|---|
| 0 | download both bases, tokenizer-compat gate | fails fast if tokenizers disagree on plain text |
| 1 | 3 source adapters on Qwen2-1.5B | gsm8k, agnews (label word), samsum (dialogue→summary) |
| 2 | 3 oracle adapters on Qwen2.5-1.5B | same recipe, same step budget |
| 3 | anchor activations (WikiText-2, 2000×256, 16k kept tokens/module) + CKA drift | `drift.json`, `cache/acts_*.pt` |
| 4 | naive transfer | `results_naive.json` |
| 5 | Cross-LoRA k=320 + k ∈ {64,160,640} | `results_crosslora*.json` |
| 6 | Procrustes; ridge λ=1 + λ ∈ {0.1,10,100} | `results_aligned_*.json` |
| 7 | oracle eval | `results_oracle.json` |
| 8 | `summary.json` + paper tables + kill-condition verdict | `paper/tables/*.tex`, `paper/tables/verdict.json` |
| 9 | ablation grid on cached activations | `ablations.json` |

Metrics: GSM8K = greedy numeric exact match, 500 test problems. AG News =
exact match of the generated topic word, 1000 test articles. SAMSum =
perplexity of the gold summary given the dialogue, loss on summary tokens
only, 500 test dialogues. All three should show a large oracle-vs-naive gap.

Defaults: 400 optimizer steps × 16 sequences of ≤512 tokens, LoRA r=16 α=32
on q/k/v/o_proj, lr 2e-4. Override with `STEPS MAXLEN BATCH RANK RIDGE_LAMS
CROSSLORA_RANKS`.

## Cost estimate, A100 40GB

Engineering estimates from FLOP counts and typical HF throughput. Stage 1's
wall-clock will calibrate them.

| stage | work | est. time |
|---|---|---|
| download | 2 × ~3 GB | 2–5 min |
| train 6 adapters | 3.3M tokens each, ~0.5 s/step | 25–35 min |
| anchors + drift | 2 forward passes, ~18 GB cache write per model | 10–15 min |
| 11 evaluated conditions | ~5–6 min each (GSM8K generation dominates) | 55–70 min |
| ablation grid (optional) | 24 refits × 3 task evals | ~2 h |
| **total without stage 9** | | **≈ 1.7–2.2 h** |
| **total with stage 9** | | **≈ 4 h** |

At $1.5–2.0/hr that is **≈ $4 for the decision table, ≈ $8 with the
ablation grid**. Reserve 6 hours to leave room for one retry. To stop after
the decision table, press Ctrl-C once stage 9 starts; everything needed for
the verdict is already written.

If host RAM is under 48 GB, set `anchors.max_tokens: 8192` in the config.
That halves the cache and is still about 5× over-determined for d=1536.

## Reading the result

`paper/tables/verdict.json` applies the kill condition from HANDOFF.md §4:
ridge must beat Cross-LoRA by **≥10 points of mean gap recovered**. The
threshold was fixed before any GPU result existed. Do not move it after.

- `CONTINUE`: write §6 from the generated tables and macros.
- `CONTINUE WITH CARE`: the mean margin clears 10 points but ridge loses on
  some task. The paper must explain that task.
- `KILL CONDITION MET`: activation alignment is not clearly better than
  weight-only Cross-LoRA. Stop, or reposition as a drift-study paper.

The paper picks up `paper/tables/main_results.tex`, `sweeps.tex` and
`results_macros.tex` automatically. For the 1.5B figures run:

```bash
python paper/make_figures.py --run runs/qwen2_to_qwen25_1.5b --tag qwen15b \
  --title 'Qwen2-1.5B $\to$ Qwen2.5-1.5B, input activations'
```

## Not in the script yet

- Per-module naive-degradation vs drift correlation (§5 last paragraph).
- Gemma 3-12B → Gemma 4-12B headline pair: gated weights, 48 GB GPU or
  sequential loading, roughly 8× this cost.
