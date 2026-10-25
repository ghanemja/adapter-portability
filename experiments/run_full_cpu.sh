#!/usr/bin/env bash
# Full adapter-portability pipeline on CPU at 0.5B, end to end.
#   train M_v adapter -> train oracle adapter -> drift -> naive -> aligned
#   (ridge + procrustes) -> oracle eval -> aggregate table.
# Bounded (small steps / eval caps) so it finishes on a laptop CPU.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CFG=${CFG:-configs/qwen05b_agnews_cpu.yaml}
ROOT=${ROOT:-runs/qwen05b_agnews}
export TOKENIZERS_PARALLELISM=false
# Env-overridable so a "tiny + low-impact" run leaves cores free for the laptop.
STEPS=${STEPS:-60}
MAXLEN=${MAXLEN:-64}
THREADS=${THREADS:-6}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS

done_file () { [ -f "$1" ]; }   # stage-skip guard for teardown-resume
echo "==== config: STEPS=$STEPS MAXLEN=$MAXLEN THREADS=$THREADS ROOT=$ROOT ===="

echo "==== [1/7] train M_v adapter (Qwen2-0.5B) ===="
if done_file "$ROOT/adapters_v/agnews/adapter_model.safetensors"; then echo "skip (exists)"; else
$PY experiments/train_lora.py --base Qwen/Qwen2-0.5B \
  --dataset fancyzhx/ag_news --text-field text --split train \
  --out $ROOT/adapters_v/agnews \
  --device cpu --max-length $MAXLEN --max-train-samples 2000 --max-steps $STEPS \
  --batch-size 4 --rank 16
fi

echo "==== [2/7] train ORACLE adapter (Qwen2.5-0.5B) ===="
if done_file "$ROOT/adapters_oracle/agnews/adapter_model.safetensors"; then echo "skip (exists)"; else
$PY experiments/train_lora.py --base Qwen/Qwen2.5-0.5B \
  --dataset fancyzhx/ag_news --text-field text --split train \
  --out $ROOT/adapters_oracle/agnews \
  --device cpu --max-length $MAXLEN --max-train-samples 2000 --max-steps $STEPS \
  --batch-size 4 --rank 16
fi

echo "==== [3/7] drift + cache anchor activations ===="
if done_file "$ROOT/drift.json"; then echo "skip (exists)"; else
$PY experiments/run_cka_drift.py --config $CFG; fi

echo "==== [4/7] naive transfer ===="
if done_file "$ROOT/results_naive.json"; then echo "skip (exists)"; else
$PY experiments/run_naive_transfer.py --config $CFG; fi

echo "==== [5/7] aligned transfer (ridge + procrustes) ===="
if done_file "$ROOT/results_aligned_ridge.json"; then echo "skip ridge (exists)"; else
$PY experiments/run_linear_alignment.py --config $CFG --method ridge; fi
if done_file "$ROOT/results_aligned_procrustes.json"; then echo "skip procr (exists)"; else
$PY experiments/run_linear_alignment.py --config $CFG --method procrustes; fi

echo "==== [6/7] oracle eval (upper bound) ===="
if done_file "$ROOT/results_oracle.json"; then echo "skip (exists)"; else
$PY experiments/run_oracle.py --config $CFG --oracle-dir $ROOT/adapters_oracle; fi

echo "==== [7/7] aggregate ===="
$PY experiments/evaluate.py --config $CFG
echo "==== DONE -> $ROOT/summary.json ===="
