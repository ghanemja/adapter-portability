#!/usr/bin/env bash
# =============================================================================
# GPU job: Qwen2-1.5B -> Qwen2.5-1.5B, full 5-condition study on 3 tasks.
#
#   download -> tokenizer gate -> train 3 source + 3 oracle adapters (M_v, M_{v+1})
#   -> drift + anchor cache -> naive -> Cross-LoRA (k sweep) -> Procrustes
#   -> ridge (lambda sweep) -> oracle eval -> summary.json -> ablations
#
# One command on an A100 40GB (see docs/GPU_RUN.md for the cost estimate):
#   bash experiments/run_gpu_1p5b.sh 2>&1 | tee runs/gpu_1p5b.log
#
# Every stage is skip-if-done, and train_lora.py resumes from its last complete
# checkpoint, so re-running the same command after a preemption continues.
# Env overrides: STEPS MAXLEN BATCH RANK CFG ROOT RIDGE_LAMS CROSSLORA_RANKS PY DEVICE
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
CFG=${CFG:-configs/qwen2_to_qwen25_1.5b.yaml}
ROOT=${ROOT:-runs/qwen2_to_qwen25_1.5b}
BASE_V=${BASE_V:-Qwen/Qwen2-1.5B}
BASE_V1=${BASE_V1:-Qwen/Qwen2.5-1.5B}
STEPS=${STEPS:-400}            # >= 200 required for a meaningful adapter; 400 ~ 6.4k examples
MAXLEN=${MAXLEN:-512}
BATCH=${BATCH:-8}              # x grad-accum 2 inside train_lora.py = 16 seqs / step
RANK=${RANK:-16}
DEVICE=${DEVICE:-cuda}         # DEVICE=cpu for a dry run (disables bf16 in train_lora.py)
RIDGE_LAMS=${RIDGE_LAMS:-"0.1 10 100"}            # 1.0 is the config default (results_aligned_ridge.json)
CROSSLORA_RANKS=${CROSSLORA_RANKS:-"64 160 640"}  # 320 is the paper default (results_crosslora.json)
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$ROOT"

done_file () { [ -f "$1" ]; }
stamp () { echo "==== [$(date +%H:%M:%S)] $* ===="; }

# Task name | HF dataset | subset | training template (prompt must match CFG prompt_template)
# Label names are only used by agnews (int label -> word).
TASKS=(gsm8k agnews samsum)
declare -A DS=( [gsm8k]="openai/gsm8k" [agnews]="fancyzhx/ag_news" [samsum]="knkarthick/samsum" )
declare -A SUBSET=( [gsm8k]="main" [agnews]="" [samsum]="" )
declare -A TMPL=(
  [gsm8k]='Question: {question}\nAnswer: {answer}'
  [agnews]='Classify the news topic as World, Sports, Business, or Sci/Tech.\nArticle: {text}\nTopic: {label}'
  [samsum]='Dialogue:\n{dialogue}\nSummary: {summary}'
)
declare -A FIELD=( [gsm8k]="question" [agnews]="text" [samsum]="dialogue" )
LABELS=(World Sports Business Sci/Tech)

train_adapter () {   # $1 base model  $2 task  $3 out dir
  local base=$1 task=$2 out=$3
  if done_file "$out/adapter_model.safetensors"; then echo "skip $out (exists)"; return; fi
  local extra=()
  [ -n "${SUBSET[$task]}" ] && extra+=(--subset "${SUBSET[$task]}")
  [ "$task" = agnews ] && extra+=(--label-names "${LABELS[@]}")
  $PY experiments/train_lora.py --base "$base" --device "$DEVICE" \
    --dataset "${DS[$task]}" "${extra[@]}" --split train \
    --text-field "${FIELD[$task]}" --template "${TMPL[$task]}" \
    --out "$out" --max-length "$MAXLEN" --max-train-samples 8000 \
    --max-steps "$STEPS" --batch-size "$BATCH" --rank "$RANK" --save-steps 50
}

stamp "config: CFG=$CFG ROOT=$ROOT STEPS=$STEPS MAXLEN=$MAXLEN BATCH=$BATCH RANK=$RANK"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

stamp "[0/9] download both bases + tokenizer compatibility gate"
$PY experiments/download_models.py --config "$CFG"
$PY experiments/check_tokenizer_compat.py "$BASE_V" "$BASE_V1"

stamp "[1/9] train source adapters on M_v ($BASE_V)"
for t in "${TASKS[@]}"; do train_adapter "$BASE_V" "$t" "$ROOT/adapters_v/$t"; done

stamp "[2/9] train oracle adapters on M_{v+1} ($BASE_V1)"
for t in "${TASKS[@]}"; do train_adapter "$BASE_V1" "$t" "$ROOT/adapters_oracle/$t"; done

stamp "[3/9] drift + cache anchor activations"
if done_file "$ROOT/drift.json"; then echo "skip (exists)"; else
$PY experiments/run_cka_drift.py --config "$CFG"; fi

stamp "[4/9] naive transfer"
if done_file "$ROOT/results_naive.json"; then echo "skip (exists)"; else
$PY experiments/run_naive_transfer.py --config "$CFG"; fi

stamp "[5/9] Cross-LoRA baseline (k=320) + rank sweep ($CROSSLORA_RANKS)"
if done_file "$ROOT/results_crosslora.json"; then echo "skip k=320 (exists)"; else
$PY experiments/run_crosslora_transfer.py --config "$CFG" --rank 320; fi
for k in $CROSSLORA_RANKS; do
  if done_file "$ROOT/results_crosslora_r$k.json"; then echo "skip k=$k (exists)"; else
  $PY experiments/run_crosslora_transfer.py --config "$CFG" --rank "$k" --tag "r$k"; fi
done

stamp "[6/9] activation alignment: procrustes, ridge (lambda=1) + sweep ($RIDGE_LAMS)"
if done_file "$ROOT/results_aligned_procrustes.json"; then echo "skip procrustes (exists)"; else
$PY experiments/run_linear_alignment.py --config "$CFG" --method procrustes; fi
if done_file "$ROOT/results_aligned_ridge.json"; then echo "skip ridge (exists)"; else
$PY experiments/run_linear_alignment.py --config "$CFG" --method ridge; fi
for lam in $RIDGE_LAMS; do
  if done_file "$ROOT/results_aligned_ridge_lam$lam.json"; then echo "skip lam=$lam (exists)"; else
  $PY experiments/run_linear_alignment.py --config "$CFG" --method ridge --lam "$lam" --tag "lam$lam"; fi
done

stamp "[7/9] oracle eval (upper bound)"
if done_file "$ROOT/results_oracle.json"; then echo "skip (exists)"; else
$PY experiments/run_oracle.py --config "$CFG" --oracle-dir "$ROOT/adapters_oracle"; fi

stamp "[8/9] aggregate -> $ROOT/summary.json + paper tables"
$PY experiments/evaluate.py --config "$CFG"
$PY paper/make_tables.py --run "$ROOT" --config "$CFG" || echo "(table build failed; summary.json is intact)"

stamp "[9/9] ablation grid on cached activations (anchor budget x solver x scope)"
if done_file "$ROOT/ablations.json"; then echo "skip (exists)"; else
$PY experiments/run_ablations.py --config "$CFG"; fi

stamp "DONE. Decide: ridge gap-recovered vs crosslora in $ROOT/summary.json (HANDOFF.md kill condition)"
