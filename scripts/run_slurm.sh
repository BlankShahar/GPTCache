#!/bin/bash
# ==============================================================================
# SLURM Runner — EchoLLM Semantic Cache Experiment
# Runs experiment.py across a grid of embedding models, thresholds, and seeds.
# ==============================================================================

set -euo pipefail

# --- Paths & Env ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONDA_ENV="your_conda_env_name"  # <-- IMPORTANT: Change this

# --- Experiment Grid (Edit as needed) ---
EMBEDDING_MODELS=("nomic-embed-text" "all-minilm" "mxbai-embed-large")
SIMILARITY_THRESHOLDS=(0.8 0.85 0.9 0.95)
SEEDS=(42 100 2024)

# --- SLURM Resources ---
PARTITION="gpu_partition"      # <-- Change to your cluster's partition
GPUS="1"
CPUS=4
MEM="32G"
TIME="0-04:00:00"

# --- Main Script and Output ---
PYTHON_SCRIPT="${PROJECT_ROOT}/scripts/experiment.py"
OUTPUT_DIR="${PROJECT_ROOT}/slurm_results"

# --- Main Loop ---
echo "🔥 Submitting SLURM job grid for EchoLLM experiments"
echo "Models:     ${EMBEDDING_MODELS[*]}"
echo "Thresholds: ${SIMILARITY_THRESHOLDS[*]}"
echo "Seeds:      ${SEEDS[*]}"
echo "----------------------------------------"

for model in "${EMBEDDING_MODELS[@]}"; do
  for threshold in "${SIMILARITY_THRESHOLDS[@]}"; do
    for seed in "${SEEDS[@]}"; do

      JOB_NAME="cache_${model}_t${threshold}_s${seed}"
      RUN_OUTPUT_DIR="${OUTPUT_DIR}/${model}/${threshold}/seed_${seed}"
      LOG_DIR="${RUN_OUTPUT_DIR}/_logs"
      mkdir -p "$LOG_DIR"
      LOG_FILE="${LOG_DIR}/slurm_%j.out"

      if [[ -f "${RUN_OUTPUT_DIR}/summary.csv" ]]; then
        echo "✅ SKIP: Results already exist for ${JOB_NAME}"
        continue
      fi

      CMD="python ${PYTHON_SCRIPT}"
      CMD+=" --embedding-model ${model}"
      CMD+=" --similarity-threshold ${threshold}"
      CMD+=" --seed ${seed}"
      CMD+=" --output-dir ${OUTPUT_DIR}"

      echo "🚀 SUBMIT: ${JOB_NAME}"
      sbatch \
        --partition="$PARTITION" \
        --job-name="$JOB_NAME" \
        --output="$LOG_FILE" \
        --time="$TIME" \
        --ntasks=1 \
        --gpus="$GPUS" \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --wrap="
          echo '========================================'
          echo '🔬 SLURM JOB: EchoLLM Experiment'
          echo '========================================'
          echo 'Job ID       : $SLURM_JOB_ID'
          echo 'Host         : $(hostname)'
          echo 'Start Time   : $(date)'
          echo '----------------------------------------'
          echo 'Config:'
          echo '  Model      : ${model}'
          echo '  Threshold  : ${threshold}'
          echo '  Seed       : ${seed}'
          echo '----------------------------------------'
          module load anaconda || true
          source activate ${CONDA_ENV} || conda activate ${CONDA_ENV}
          export PYTHONPATH='${PROJECT_ROOT}:$PYTHONPATH'
          cd '${PROJECT_ROOT}'
          echo 'CMD: ${CMD}'
          ${CMD}
          echo '----------------------------------------'
          echo 'End Time     : $(date)'
          echo '========================================'
        "
    done
  done
done

echo "🎉 All jobs submitted."
