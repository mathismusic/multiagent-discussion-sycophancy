#!/bin/bash
#
# Novel-subject generalization experiment.
#
# Tests whether BSS scores computed on the original 5 subjects generalize
# to 15 completely new MMLU subjects (15 samples each = 225 total).
#
# Runs:
#   1. Data generation (15 new subjects, 15 samples each)
#   2. Knowledge flag pre-computation
#   3. Baseline + BSS debates (batch 1)
#   4. DSS + Accuracy BSS debates (batch 2)
#   5. Evaluation & comparison
#
# All outputs go to separate paths so nothing existing is overwritten.
#
# Usage:
#   bash run_novel_subject_expt.sh

set -e

MODELS="llama3b llama8b qwen3b qwen7b qwen14b qwen32b"
BSS_SCORES="scores/bss_scores_final.json"
ACCURACY_SCORES="scores/accuracy_scores.json"
DBSS_SCORES="scores/bss_scores_post_debate.json"
DATA_CSV="data_for_new_subjects_dss.csv"
KNOWLEDGE_FLAGS="knowledge_flags_new_subjects.json"
LOG_DIR="logs_new_subjects"

# 15 novel subjects (no overlap with the original 5:
#   elementary_mathematics, professional_law, machine_learning,
#   business_ethics, high_school_biology)
NEW_SUBJECTS=(
  abstract_algebra
  anatomy
  astronomy
  clinical_knowledge
  college_chemistry
  college_computer_science
  conceptual_physics
  econometrics
  global_facts
  high_school_geography
  high_school_psychology
  logical_fallacies
  moral_scenarios
  philosophy
  world_religions
)

cd "$(dirname "$0")"

# --- Timing helpers ---
PIPELINE_START=$SECONDS

elapsed() {
  local secs=$1
  printf "%dh %dm %ds" $((secs/3600)) $(((secs%3600)/60)) $((secs%60))
}

step_start() {
  STEP_START=$SECONDS
  echo ""
  echo "────────────────────────────────────────────────"
  echo "[$(date '+%H:%M:%S')] $1"
  echo "────────────────────────────────────────────────"
}

step_done() {
  local dur=$((SECONDS - STEP_START))
  local total=$((SECONDS - PIPELINE_START))
  echo "[$(date '+%H:%M:%S')] $1 finished in $(elapsed $dur)  (total elapsed: $(elapsed $total))"
}

echo "=== Novel-subject generalization experiment ==="
echo "Started at $(date)"
echo ""

mkdir -p "$LOG_DIR"

# --- Step 1: Generate dataset for the new subjects ---
step_start "Step 1: Generate dataset (15 subjects x 15 samples)"
if [ ! -f "$DATA_CSV" ]; then
  python generate_bss_dss_data.py \
    --subjects ${NEW_SUBJECTS[@]} \
    --bss_per_subject 0 --dss_per_subject 15 \
    --bss_out /dev/null --dss_out "$DATA_CSV" \
    --seed 123
  step_done "Step 1"
else
  echo "  $DATA_CSV already exists, skipping."
fi

# --- Step 2: Pre-compute knowledge flags ---
step_start "Step 2: Compute knowledge flags"
if [ ! -f "$KNOWLEDGE_FLAGS" ]; then
  python compute_knowledge_flags.py \
    --data_csv "$DATA_CSV" \
    -m $MODELS \
    -o "$KNOWLEDGE_FLAGS"
  step_done "Step 2"
else
  echo "  $KNOWLEDGE_FLAGS already exists, skipping."
fi

# --- Step 3: Batch 1 — Baseline + BSS ---
step_start "Step 3: Batch 1 — Baseline + BSS debates"

mkdir -p "$LOG_DIR/baseline" "$LOG_DIR/bss"

PIDS=()

if [ ! -f "$LOG_DIR/baseline/log.jsonl" ]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3 python multiagent-debate.py \
    -e baseline \
    -m $MODELS \
    --data_csv "$DATA_CSV" \
    --knowledge_flags_path "$KNOWLEDGE_FLAGS" \
    --log_dir "$LOG_DIR" \
    > "$LOG_DIR/baseline/stdout.log" 2>&1 &
  PIDS+=($!)
  echo "  (baseline) PID=${PIDS[-1]}  [GPUs 0,1,2,3]"
else
  echo "  (baseline) already complete, skipping."
fi

if [ ! -f "$LOG_DIR/bss/log.jsonl" ]; then
  CUDA_VISIBLE_DEVICES=4,5,6,7 python multiagent-debate.py \
    -e bss \
    -m $MODELS \
    --data_csv "$DATA_CSV" \
    --knowledge_flags_path "$KNOWLEDGE_FLAGS" \
    --log_dir "$LOG_DIR" \
    --use_bss_scores --bss_scores_path "$BSS_SCORES" \
    > "$LOG_DIR/bss/stdout.log" 2>&1 &
  PIDS+=($!)
  echo "  (bss) PID=${PIDS[-1]}  [GPUs 4,5,6,7]"
else
  echo "  (bss) already complete, skipping."
fi

[ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"
step_done "Step 3"

# --- Step 4: Batch 2 — DSS + Accuracy BSS ---
step_start "Step 4: Batch 2 — DSS + Accuracy BSS debates"

mkdir -p "$LOG_DIR/dss" "$LOG_DIR/accuracy_bss"

PIDS=()

if [ ! -f "$LOG_DIR/dss/log.jsonl" ]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3 python multiagent-debate.py \
    -e dss \
    -m $MODELS \
    --data_csv "$DATA_CSV" \
    --knowledge_flags_path "$KNOWLEDGE_FLAGS" \
    --log_dir "$LOG_DIR" \
    --use_dss_scores --bss_scores_path "$BSS_SCORES" \
    --alpha 0.2 --beta 0.0 \
    > "$LOG_DIR/dss/stdout.log" 2>&1 &
  PIDS+=($!)
  echo "  (dss) PID=${PIDS[-1]}  [GPUs 0,1,2,3]"
else
  echo "  (dss) already complete, skipping."
fi

if [ ! -f "$LOG_DIR/accuracy_bss/log.jsonl" ]; then
  CUDA_VISIBLE_DEVICES=4,5,6,7 python multiagent-debate.py \
    -e accuracy_bss \
    -m $MODELS \
    --data_csv "$DATA_CSV" \
    --knowledge_flags_path "$KNOWLEDGE_FLAGS" \
    --log_dir "$LOG_DIR" \
    --use_bss_scores --bss_scores_path "$ACCURACY_SCORES" \
    > "$LOG_DIR/accuracy_bss/stdout.log" 2>&1 &
  PIDS+=($!)
  echo "  (accuracy_bss) PID=${PIDS[-1]}  [GPUs 4,5,6,7]"
else
  echo "  (accuracy_bss) already complete, skipping."
fi

[ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"
step_done "Step 4"

# --- Step 5: Batch 3 — Binary BSS + DBSS ---
step_start "Step 5: Batch 3 — Binary BSS + DBSS debates"

mkdir -p "$LOG_DIR/binary_bss" "$LOG_DIR/dbss"

PIDS=()

if [ ! -f "$LOG_DIR/binary_bss/log.jsonl" ]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3 python multiagent-debate.py \
    -e binary_bss \
    -m $MODELS \
    --data_csv "$DATA_CSV" \
    --knowledge_flags_path "$KNOWLEDGE_FLAGS" \
    --log_dir "$LOG_DIR" \
    --use_binary_syco_flags --bss_scores_path "$BSS_SCORES" \
    > "$LOG_DIR/binary_bss/stdout.log" 2>&1 &
  PIDS+=($!)
  echo "  (binary_bss) PID=${PIDS[-1]}  [GPUs 0,1,2,3]"
else
  echo "  (binary_bss) already complete, skipping."
fi

if [ ! -f "$LOG_DIR/dbss/log.jsonl" ]; then
  CUDA_VISIBLE_DEVICES=4,5,6,7 python multiagent-debate.py \
    -e dbss \
    -m $MODELS \
    --data_csv "$DATA_CSV" \
    --knowledge_flags_path "$KNOWLEDGE_FLAGS" \
    --log_dir "$LOG_DIR" \
    --use_bss_scores --bss_scores_path "$DBSS_SCORES" \
    > "$LOG_DIR/dbss/stdout.log" 2>&1 &
  PIDS+=($!)
  echo "  (dbss) PID=${PIDS[-1]}  [GPUs 4,5,6,7]"
else
  echo "  (dbss) already complete, skipping."
fi

[ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"
step_done "Step 5"

# --- Step 6: Evaluate ---
step_start "Step 6: Evaluate all experiments"
python evaluate.py --all --log_dir "$LOG_DIR"
step_done "Step 6"

# --- Final summary ---
TOTAL=$((SECONDS - PIPELINE_START))
echo ""
echo "════════════════════════════════════════════════"
echo "  Novel-subject experiment complete at $(date)"
echo "  Total wall time: $(elapsed $TOTAL)"
echo "════════════════════════════════════════════════"
echo ""
echo "Logs:        $LOG_DIR/<experiment>/log.jsonl"
echo "Evaluations: $LOG_DIR/<experiment>/eval/"
echo "Comparison:  $LOG_DIR/comparison.csv"
