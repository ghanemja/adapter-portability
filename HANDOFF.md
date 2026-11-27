# HANDOFF — Cross-Version Adapter Portability via Representational Alignment

Last updated: 2026-11-26 (Cross-LoRA baseline, GPU job and paper sections; see §2.5, §6, §7). Repo: `/Users/janelle/Documents/finetune`. Git on `main`, remote: private https://github.com/ghanemja/adapter-portability (pushed 2026-10-25). `.gitignore` excludes `.venv/`, model weights and `runs/**` except small `*.json`/`*.log` result files.
Author: Janelle Ghanem. This file is the single source of truth for project state. Read it before touching anything.

---

## 0. TL;DR

- **Idea:** transplant a LoRA adapter from base model version v to successor v+1 with no task data, by fitting per-module linear maps from a small unlabeled anchor set of activations. Closed form, rank-preserving, zero inference cost.
- **Mechanism validated** on synthetic drift and on real Qwen2-0.5B → Qwen2.5-0.5B weights (ordering oracle < aligned < naive holds). **No publishable headline number yet.** All real runs were tiny CPU runs.
- **Novelty re-checked 2026-10-06.** Original framing ("first training-free cross-version LoRA transfer") is dead: Cross-LoRA (Aug 2025) and LoRA-X (ICLR 2025) exist. Surviving contribution is narrower but real: activation-based (functional) alignment for same-family **version drift**, vs Cross-LoRA's weight-only SVD on cross-family pairs. Details in §4.
- **Deadlines:** ICLR 2027 and AAAI 2027 passed. Target TMLR (rolling) or ICML 2027 (late Jan 2027) or COLM 2027.
- **Blocker:** local machine is CPU-only, 8.5 GB RAM. Everything compelling needs one GPU day.
- **Done:** Cross-LoRA baseline (5th condition), checkpoint-resume, paper §§1–5, 7 (synthetic), 8, 9 plus §6 setup; `paper/make_tables.py` turns results into LaTeX + a kill-condition verdict. GPU job is one command, dry-run verified on CPU. Pushed to `claude/wizardly-mendel-uupsel`.
- **Next step:** rent an A100 40GB (≥48 GB host RAM) and run `bash experiments/run_gpu_1p5b.sh`. Everything in `docs/GPU_RUN.md`. ≈2 h / ≈$4 for the decision table.

---

## 1. Problem and method

**Setting.** A model family ships base checkpoints M_v, M_{v+1}, ... A LoRA adapter ΔW = (α/r)·B·A fine-tuned on M_v stops working (or degrades) when applied naively to M_{v+1}, because internal representations drift between versions. Retraining every adapter per release is expensive and often impossible (task data gone, compliance, cost).

**Goal.** Port the adapter to M_{v+1} with no task data, no gradient steps, same rank r, no extra inference cost.

**Method (per LoRA-targeted module).**
1. Run a small unlabeled anchor set (generic text, ~100–1000 tokens suffice at small d) through both M_v and M_{v+1}; cache module input activations X_v, X_{v+1} and output activations Y_v, Y_{v+1}.
2. Solve two linear maps in closed form:
   - P_in : v+1 input space → v input space (fit X_v ≈ X_{v+1}·P_in)
   - P_out : v output space → v+1 output space (fit Y_{v+1} ≈ Y_v·P_out)
   - Solver: ridge regression (λ-regularized least squares) or orthogonal Procrustes.
3. Transplant: ΔW' = P_out · ΔW · P_in, implemented as A' = A·P_in and B' = P_out·B. Rank r preserved. Adapter loads into M_{v+1} with standard PEFT tooling.

**Reference points.** Naive = apply ΔW unchanged to M_{v+1}. Oracle = LoRA trained directly on M_{v+1} with task data. Metric: gap recovered = (naive − aligned)/(naive − oracle), direction-aware.

**Scope.** Same family, same tokenizer on plain text, matching hidden dims for adapted modules. Cross-family, tokenizer changes, and dim mismatch are out of scope (stated limitation; Cross-LoRA covers dim mismatch).

---

## 2. Results so far

### 2.1 Synthetic planted drift (CPU, 2026-06-15)
Harness: `experiments/validate_mechanism.py`, `experiments/validate_extract.py`, `experiments/run_ablations.py --synthetic`. Output: `runs/ablations_synth.json`.
- Ridge alignment cut transfer error **98.6%** vs naive. Procrustes only **81.4%**. Reason: planted drift is non-orthogonal, Procrustes cannot express it. → **ridge > Procrustes**.
- **Per-layer maps ≫ one shared map.** Shared map collapses, worse than naive at low λ. Drift is module-specific.
- Anchor-efficient: ~100 tokens enough at small d.
- λ trades fit vs robustness, standard ridge behavior.

### 2.2 Real representational drift (CPU, 2026-06-15)
Qwen2-0.5B → Qwen2.5-0.5B, 96 LoRA-target modules (q/k/v/o_proj × 24 layers). Output: `runs/qwen05b_cpu/drift.json`, re-run copy in `runs/qwen05b_agnews/drift.json`.
- Mean CKA drift 0.254, range 0.09–0.69.
- `o_proj` drifts most; peak at layers 16–17 o_proj ≈ 0.69.
- Early layers near floor (layer-0 q/k/v ≈ 0.093). Drift grows with depth.
- This is a real §5 figure. Nobody in the related work has a systematic per-module drift study.

### 2.3 Real transfer, tiny run (CPU, 2026-07-06)
Qwen2-0.5B → Qwen2.5-0.5B, AG News LM task, 20 steps, seqlen 32. Output: `runs/qwen05b_agnews/summary.json`.

| condition | perplexity |
|---|---|
| naive | 36.24 |
| procrustes | 35.75 |
| ridge | 35.72 |
| oracle | 31.52 |

Gap recovered (ridge) = 11.1%. Ordering correct (oracle < aligned < naive). **Proof of plumbing, not a result.** Adapter was badly undertrained so there was little signal to transplant and ridge ≈ Procrustes. Need ≥200 steps and a larger model for a meaningful gap.

### 2.5 Five-condition tiny run with Cross-LoRA (CPU, re-run 2026-11-26)
Config `configs/qwen05b_agnews_tiny5.yaml`, ROOT `runs/qwen05b_agnews_tiny`, 20 steps, seqlen 32, 40 eval docs. Log `runs/tiny5_cpu.log`.

| condition | perplexity | gap recovered |
|---|---|---|
| naive | 36.17 | — |
| Cross-LoRA (k=320, weight-only) | 36.16 | 0.1% |
| procrustes | 35.74 | 9.6% |
| ridge (λ=1) | 35.64 | 11.7% |
| oracle | 31.69 | — |

Plumbing verification only (20-step adapter). Ordering Cross-LoRA ≈ naive < Procrustes < ridge matches the prediction and reproduced across two independent runs (first run, lost with the container: 0.9 / 7.8 / 11.1%).

### 2.6 Drift structure: rotation vs. unconstrained, heads as units (CPU, 2026-11-27)
`experiments/analyze_heads.py`, config `configs/qwen05b_heads.yaml` (q_proj + o_proj only, 12k anchor tokens/module), output `runs/qwen05b_heads/heads.json`, figure `paper/figures/heads_qwen05b.pdf`, text in drift.tex §5.1. Held-out R² of predicting the v activation from the v+1 activation, mean over 24 layers:

| interface | identity (naive) | orthogonal | ridge / full map | per-head, same head | per-head, matched |
|---|---|---|---|---|---|
| residual stream (q/k/v input) | −1.54 | 0.45 | **0.70** | — | — |
| attention output (o_proj input) | −2.44 | — | **0.51** | 0.24 | 0.32 |

- Ridge beats the best rotation in **24/24 layers**: real-model evidence that drift is non-orthogonal (the synthetic ablation's claim, now on real activations).
- **Heads are not preserved across versions.** CKA-matching moves 11.6/14 heads per layer and still explains far less than a full map (24/24 layers). So no head-level modularity; one map per projection is required.
- **Linear ceiling:** ~30% (residual) and ~49% (o_proj) of held-out variance is not linearly recoverable at this anchor budget; layer 16 o_proj worst. Conclusions stable for λ ∈ [1, 100].
- Also verified: fitting the adapter's own features (A' x_{v+1} ≈ A x_v) gives exactly A·P_in for ridge, to float precision. The version-level shim is already adapter-optimal on the input side, so one shim per release serves every adapter (method.tex §4.3).

### 2.4 Attempted 150-step run (CPU, 2026-07-06) — DIED
`runs/s150_cpu.log`, config `configs/qwen05b_agnews_s150.yaml`. Reached step 27/150 at 1–3 min/step, then session teardown killed it. `runs/qwen05b_agnews_s150/` contains only `adapters_v/`. Do not try again on CPU.

---

## 3. Repo map

```
README.md                      project overview (from June; predates novelty re-check — do not trust its claims)
requirements.txt               torch, transformers, peft, datasets, etc.
.venv/                         python 3.11, CPU torch. Activate: source .venv/bin/activate
src/adaptport/                 library: activation caching, CKA, ridge/Procrustes solvers, adapter transplant, eval, crosslora.py (baseline)
                               eval.py metrics: perplexity | completion_perplexity | accuracy (GSM8K EM) | exact_match; batched generation
docs/GPU_RUN.md                what you need (GPU box, no keys), the one command, cost (~$4 / ~$8), how to read the verdict
experiments/
  train_lora.py                stage 1: train LoRA. Resumes from last COMPLETE checkpoint, removes partial ones, skips finished adapters; --template/--label-names; --save-steps; --force
  run_cka_drift.py             stage 2: per-module CKA drift between versions on anchor set
  run_naive_transfer.py        stage 3: apply ΔW unchanged to M_{v+1}
  run_linear_alignment.py      stage 4: fit P_in/P_out per module (ridge | procrustes), transplant. --lam/--tag for sweeps
  run_crosslora_transfer.py    BASELINE: Cross-LoRA weight-only SVD transfer (--rank k, default 320 as in their paper; --tag for sweeps)
  run_gpu_1p5b.sh              THE GPU JOB: 9 skip-if-done stages, resumable, writes paper tables + verdict. DEVICE=cpu dry run passes (runs/gpu_dryrun_cpu.log)
  run_oracle.py                train oracle adapter directly on M_{v+1}
  evaluate.py                  summary.json: naive | crosslora | procrustes | ridge | oracle (+ sweep files), gap_recovered per method
  run_full_cpu.sh              end-to-end runner. Env-parametric: STEPS MAXLEN THREADS CFG ROOT
  run_ablations.py             --synthetic: planted-drift ablation sweep (anchor size, λ, per-layer vs shared)
  validate_mechanism.py        synthetic proof of ΔW' = P_out ΔW P_in
  validate_extract.py          sanity check on activation extraction
  check_tokenizer_compat.py    pre-flight gate: assert plain text tokenizes identically on both versions
  download_models.py           pull base weights from HF
configs/
  qwen05b_agnews_tiny.yaml     20-step smoke test (what produced §2.3)
  qwen05b_agnews_s150.yaml     150-step attempt (died)
  qwen05b_agnews_cpu.yaml      0.5B pair, AG News, CPU defaults
  qwen05b_cpu.yaml             0.5B drift-only
  qwen05b_agnews_tiny5.yaml    fresh-ROOT tiny config for the 5-condition table in §2.5
  qwen2_to_qwen25_1.5b.yaml    PRIMARY GPU pair config (ready, not yet run): gsm8k EM, agnews label EM, samsum completion-ppl
  gemma3_to_gemma4_12b.yaml    HEADLINE GPU pair config (not yet run; weights not downloaded)
runs/                          all outputs; see §2 for which files matter
paper/
  OUTLINE.md                   9-section outline (pre-dates novelty re-check; §2 Related Work must be rewritten)
  main.tex                     skeleton, all \section headers present
  sections/introduction.tex    drafted (~3.5 KB) — contribution list claims "first", must be revised
  sections/problem_formulation.tex  drafted (~7 KB), still valid
  sections/*.tex               all sections written except §6 results paragraphs (need GPU); see §6 table below
  make_tables.py               results_*.json -> tables/{main_results,sweeps,results_macros}.tex + verdict.json (kill condition, 10-pt margin, pre-registered)
  make_figures.py              figures/*.pdf; --run adds 1.5B drift + gap-recovered bars
  references.bib               18 entries incl. all six competitors (authors verified on arXiv)
  NOTE: no LaTeX toolchain available in the cloud container; braces/envs/cites/labels checked by script.
```

Run the pipeline:
```bash
source .venv/bin/activate
STEPS=200 MAXLEN=128 THREADS=6 CFG=configs/qwen05b_agnews_cpu.yaml ROOT=runs/myrun bash experiments/run_full_cpu.sh
```
Notes: torch defaulted to 2 threads on this Apple Silicon box, set `OMP_NUM_THREADS`/`THREADS`. Background jobs die on session teardown — keep runs short or add Trainer checkpoint-resume before any long run. Low-impact recipe: `nice -n 19 env THREADS=4 ...`.

---

## 4. Related work and novelty (re-checked 2026-10-06)

None of these are in `references.bib` yet. All six must be added and discussed.

| Paper | Date / venue | Training-free transfer? | Works on an ordinary pre-existing LoRA? | Alignment signal | Pairs tested |
|---|---|---|---|---|---|
| LoRA-X — arXiv 2501.16559 | ICLR 2025 | yes | **no**: adapter must be trained constrained to base subspace | weight SVD subspace similarity, per-layer gating | Stable Diffusion 1.5 / SDXL only |
| Trans-PEFT — arXiv 2506.06844 | ACL 2025 | yes | **no**: special training (attn-focused, FFN-independent) | architectural heuristic | 7 base models, 12 datasets |
| **Cross-LoRA — arXiv 2508.05232** | Aug 2025 preprint | yes | **yes** | **weight-only**: rank-truncated SVD of base weights + Frobenius projection ΔW_t = (ΔW_s·Ṽ_s)Ṽ_sᵀ | **cross-family only**: Qwen2.5-1.5B/3B, LLaMA-3.2-3B, Gemma-2-2B |
| Fine-tuning Transfer — arXiv 2503.20110 | Mar 2025 | yes | yes | **none**, direct diff-vector addition | Llama 3.0-8B → 3.1-8B |
| ReLoRA — arXiv 2606.02606 | May 2026 | **no**: retrains with Bayesian-optimized init + scheduled regularization | yes | adapter weights as init | unspecified |
| PorTAL — Ramp Labs | Jul 2026 blog | **no**: refits per-base alignment with ~500 labeled examples | no: trains base-agnostic latent up front | learned | Qwen3-1.7B/4B → 8B; → Gemma-3-4B |

**Cross-LoRA is the paper we have to beat.** Full-PDF findings:
- Confirmed 100% weight-only. No activations, no calibration text, no anchors. (PorTAL's blog describes it as "activation alignment" — that is wrong.)
- No same-family version pair anywhere.
- Weak evaluation: their own trained-LoRA oracle averages only +0.98% over base on ARC-c/ARC-e/OpenBookQA/HellaSwag; transferred gets +0.85%. Both ≈ base. "Approaches trained LoRA" means LoRA barely mattered on those tasks. No gap-recovered metric. Only baseline is naive weight interpolation.
- PorTAL reports Cross-LoRA recovers only ~14% of LoRA lift on Qwen3-4B → 8B (dim mismatch, so not our setting, but suggestive).

**What is dead:**
- "First training-free cross-version LoRA transfer."
- "First linear alignment for adapter transfer."

**What survives (the actual contribution now):**
1. **Version drift as the problem**, same family, same dims. Nobody tests alignment here. Fine-tuning Transfer does naive diff-add on Llama 3.0→3.1 and becomes our cited naive baseline.
2. **Activation-grounded (functional) alignment** from unlabeled anchors, vs weight-subspace alignment. Argument: drift lives in what modules compute on real inputs, not in the weight SVD basis. The CKA drift study (§2.2) is the evidence for this.
3. **Ridge vs orthogonal, per-layer vs shared** ablations. Cross-LoRA has neither. Synthetic result (ridge 98.6% vs Procrustes 81.4%) predicts weight/orthogonal methods underperform on real non-orthogonal drift.
4. **Evaluation on tasks where oracle lift is large**, with gap-recovered as the metric, so the comparison means something.
5. Gemma 3 → Gemma 4 as the live motivating example (see §5).

**Repositioned claim:** "Weight-space alignment (Cross-LoRA) is insufficient for real version drift because drift is functional and non-orthogonal. A ~100-token unlabeled anchor set yields per-module ridge maps that recover X% of the oracle gap where weight-only SVD recovers Y%."

**Kill condition:** if activation ridge ≈ Cross-LoRA SVD on the real Qwen 1.5B pair, the paper is incremental and should not be written. Only the GPU run answers this.

**Risk check done:** grep of Cross-LoRA full text for activation / hidden state / calibration / anchor / per-layer / Procrustes / ridge / least squares / orthogonal → zero method-relevant hits (only "SwiGLU activation" architecture mentions). No hidden activation-variant ablation.

Realistic venue tier: TMLR, or ICML 2027 main if the gap-recovered margin over Cross-LoRA is large.

**Kill condition, made operational (2026-11-26):** ridge must beat Cross-LoRA by ≥10 points of mean gap recovered across the three tasks. Fixed before any GPU result; `paper/make_tables.py` computes it into `paper/tables/verdict.json`. Do not move the threshold after seeing results.

**Cross-LoRA implementation note.** Their Eqs. 3–6 align bases by least squares (not Procrustes). With equal dims the min-norm solution gives Ũ_s = U_t exactly, so ΔW_t = U_t U_tᵀ ΔW_s V_t V_tᵀ: projection onto the target weight's top-k subspaces. Verified numerically; written up in method.tex. Their truncation rank is 320 (Table 5); we sweep {64,160,320,640}.

---

## 5. Model pairs

**Primary (cheap, clean):** Qwen2-1.5B → Qwen2.5-1.5B. Same tokenizer (151k), same hidden dim, ungated. Isolates pure drift. Config `configs/qwen2_to_qwen25_1.5b.yaml`. Fits one 24 GB GPU. Iteration pair.

**Headline (the motivating story):** Gemma 3-12B → Gemma 4-12B. Verified 2026-06-15: both hidden_size 3840, 48 layers, 16 q-heads / 8 kv-heads, head_dim 256 → q/k/v/o_proj shapes identical, maps 1:1. Vocab 262208 vs 262144 is +64 special tokens only (embeddings/lm_head not adapted, irrelevant). Needs `google/gemma-3-12b-pt` and `google/gemma-4-12b-pt` (gated, clean base safetensors). Config `configs/gemma3_to_gemma4_12b.yaml`. Pre-flight: `experiments/check_tokenizer_compat.py`. Needs ~48 GB GPU (two 12B in bf16) or sequential loading.

**Rejected:** Gemma 2 → 3 (tokenizer changed, no dim match). Gemma 4 E2B/E4B (Per-Layer Embeddings, new arch, no clean small pair). All Gemma weights currently on disk are MLX/GGUF quant, merged, abliterated or IT — unusable for activation extraction and weight-level LoRA.

---

## 6. Paper state

`paper/OUTLINE.md` has the 9-section plan. Status per section:

| § | Status | Action |
|---|---|---|
| 1 Introduction | **revised** | insert headline numbers after GPU |
| 2 Related Work | **written** | — |
| 3 Problem Formulation | drafted | valid, minor: cite Fine-tuning Transfer as the naive baseline |
| 4 Method | **written** | — |
| 5 Drift | **written (0.5B)** | swap in 1.5B figure; drift-vs-degradation paragraph needs an unscripted per-module experiment |
| 6 Experiments | **setup written; tables auto-generated** | results paragraphs after GPU (structure pre-committed in TODO comments) |
| 7 Ablations | **synthetic half written** | §7.2 uncomment `\input{tables/sweeps}` after GPU |
| 8 Limitations | **written** | — |
| 9 Reproducibility | **written** | clearance note + code URL |

Remaining zero-compute work: abstract numbers, README.md rewrite, per-module drift-vs-degradation script.

---

## 7. Next steps, in order

1. ~~Cross-LoRA baseline~~ DONE (§2.5; also stage 5/8 of `run_full_cpu.sh`).
2. ~~Checkpoint-resume~~ DONE.
3. **Rent GPU and run** `bash experiments/run_gpu_1p5b.sh 2>&1 | tee runs/gpu_1p5b.log`. See `docs/GPU_RUN.md`. Bring back `runs/qwen2_to_qwen25_1.5b/*.json`, `paper/tables/`, `paper/figures/`.
4. **Decide**: if ridge gap-recovered clearly > Cross-LoRA, continue. If not, stop.
5. **Gemma 3-12B → 4-12B** headline run (request gated access first; 48 GB GPU or sequential loading).
6. ~~Write sections 2, 4, 5, 7, 8, 9; bib~~ DONE. Remaining: §6 results paragraphs and §7.2 after GPU.
7. **Submit** TMLR, or ICML 2027 if margin is large.

---

## 8. Known gotchas

- Remote is private `ghanemja/adapter-portability`. Keep it private until README.md is fixed and Lincoln Lab release clearance is done.
- `README.md` predates the novelty re-check and overclaims. A banner now points to HANDOFF.md; rewrite before release.
- **Cloud containers are ephemeral.** On 2026-10-25 two unpushed commits were lost when the container was reclaimed and had to be rebuilt on 2026-11-26. Push after every commit. Cloud containers also lack `.venv`: `python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu && .venv/bin/pip install -r requirements.txt matplotlib`.
- Dry runs must not use ROOT=runs/qwen2_to_qwen25_1.5b: only that ROOT writes into `paper/tables/` (guard added after a dry run leaked test tables there).
- `run_ablations.py` real mode fits shared maps per (d_in, d_out) group (GQA shapes crashed the old global concat).
- **Memory (fixed 2026-11-27):** extraction used to build lists then torch.cat (2x peak) and the drift stage held model v's cache while extracting v+1. That OOM-killed a 0.5B run on 15 GB and would have peaked ~54 GB at 1.5B. `extract.py` now fills preallocated buffers and `run_cka_drift.py` extracts one model at a time; 1.5B peak ≈ 2 caches ≈ 36 GB, so the ≥48 GB host advice in docs/GPU_RUN.md holds.
- Local machine: CPU only, 8.5 GB RAM, torch defaults to 2 threads. 0.5B activation extraction ≈ minutes per 64 sequences. Training 1–3 min/step. Not viable beyond smoke tests.
- Background jobs die on session teardown.
- HF Hub unauthenticated warnings in logs are harmless but set `HF_TOKEN` for gated Gemma.
- `eval.py` perplexity on AG News is a weak task for showing LoRA lift. Prefer tasks where oracle − base is large (GSM8K exact-match is already wired).

---

## 9. Memory / provenance

Project memory lives at `~/.claude/projects/-Users-janelle-Documents-finetune/memory/adapter-portability-paper.md` and mirrors this file. Original idea captured 2026-06-15. Novelty re-check 2026-10-06. This handoff written 2026-10-25.
