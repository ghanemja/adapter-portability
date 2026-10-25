# Paper Outline — Cross-Version Adapter Portability via Representational Alignment

Target: ICLR 2027 (abstract ~Sep 19 2026, full paper ~Sep 24 2026).
Backup: TMLR (rolling), AAAI 2027. ~9 pages + appendix.

## 1. Introduction
One-paragraph problem (new base version ⇒ retrain every adapter), one-paragraph
why prior weight-transfer work doesn't apply (shared-space assumption), framing
(portability = representational alignment), then the 5-bullet contribution list.

## 2. Related Work
Position against task arithmetic/task vectors, TIES, DARE (all assume one shared
space), model stitching (alignment but not weight-level adapter transfer), CKA &
Platonic Representation Hypothesis (motivate alignability), LoRA/PEFT, and
distillation (needs inference + task signal) — none solve cross-version transfer.

## 3. Problem Formulation
Formal setup: family, successive base checkpoints, LoRA delta ΔW = (α/r)BA, the
no-task-data portability objective, the naive/oracle reference points, drift via
CKA, the per-module alignment maps, and assumptions/scope.

## 4. Method: Linear Representational Alignment
Closed-form transplant ΔW′ = Pᵒᵘᵗ ΔW Pⁱⁿ that preserves rank r; algorithm box
(cache anchor activations → solve ridge/Procrustes per module → apply → load);
variants (ridge vs orthogonal, per-layer vs shared, module subset, nonlinear map).

## 5. Representational Drift Across Versions
Empirical study: per-module CKA drift between versions over the anchor set, drift
vs depth, and the correlation between drift magnitude and naive-transfer failure.

## 6. Experiments
Setup (model pair, tasks, anchor set, metrics); main results table comparing
naive / ridge-aligned / Procrustes-aligned / oracle per task, with % of the
oracle–naive gap recovered and relative compute.

## 7. Ablations
Anchor-set size, ridge λ, orthogonal vs unconstrained, which modules to align,
per-layer vs shared map, linear vs nonlinear map, sensitivity to drift magnitude.

## 8. Limitations
Differing tokenizers / hidden dims / cross-family transfer out of scope; single
family pair; caveats on generative-quality evaluation.

## 9. Reproducibility
Released code, anchor sets, fitted maps; Lincoln Lab public-release clearance note.

## Appendix
Full per-module drift tables, derivation of the Procrustes solution, extra tasks,
hyperparameters, qualitative generation samples.
