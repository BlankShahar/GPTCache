#!/bin/bash
# ==============================================================================
# SLURM Runner — EchoLLM Semantic Cache Experiment (GPU-ready)
# ==============================================================================

set -euo pipefail

# --- Paths & Env you control ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONDA_ENV="gptcache-env"        # << change
PYTHON_SCRIPT="${PROJECT_ROOT}/scripts/experiment.py"
OUTPUT_DIR="${PROJECT_ROOT}/slurm_results"



# Shared prewarmed models directory (must match scripts/prewarm_ollama.sh)
SHARED_OLLAMA_MODELS="${HOME}/.ollama_shared_models"

# --- Grid ---
EMBEDDING_MODELS=("nomic-embed-text")
SIMILARITY_THRESHOLDS=(0.99)
SEEDS=(42)

# --- Slurm resources (adjust to your cluster) ---
PARTITION="gpu_partition"  # << change
# Prefer --gres on modern clusters; if your cluster uses --gpus, set GRES=""
GRES="gpu:1"
GPUS="1"
CPUS=4
MEM="32G"
TIME="0-08:00:00"

# All models you will touch (embed + chat/judge). Pre-pulled per job.
ALL_MODELS=("nomic-embed-text" "all-minilm" "mxbai-embed-large" "llama3.1:8b")

ALGO_NAME="search_distance"
VECTOR_METRIC="ip"
ALGO_DIR="${ALGO_NAME}-${VECTOR_METRIC}"

echo "🔥 Submitting SLURM job grid"
for model in "${EMBEDDING_MODELS[@]}"; do
  for threshold in "${SIMILARITY_THRESHOLDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      JOB_NAME="cache_${model}_t${threshold}_s${seed}"
      RUN_OUTPUT_DIR="${OUTPUT_DIR}/${model}/${ALGO_DIR}/${threshold}/seed_${seed}"
      LOG_DIR="${RUN_OUTPUT_DIR}/_logs"
      mkdir -p "$LOG_DIR"
      LOG_FILE="${LOG_DIR}/slurm_%j.out"

      # Skip if already done
      if [[ -f "${RUN_OUTPUT_DIR}/summary.csv" ]]; then
        echo "✅ SKIP: ${JOB_NAME}"
        continue
      fi

      # Compose experiment command
      CMD="python ${PYTHON_SCRIPT}"
      CMD+=" --embedding-model ${model}"
      CMD+=" --similarity-threshold ${threshold}"
      CMD+=" --similarity-algo ${ALGO_NAME}"
      CMD+=" --vector-metric ${VECTOR_METRIC}"
      CMD+=" --seed ${seed}"
      CMD+=" --output-dir ${OUTPUT_DIR}"

      sbatch \
        --partition="$PARTITION" \
        --job-name="$JOB_NAME" \
        --output="$LOG_FILE" \
        --time="$TIME" \
        --ntasks=1 \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        ${GRES:+--gres="$GRES"} \
        ${GRES:+'--gpus' '""'} << EOF
#!/bin/bash
set -euo pipefail

echo '========================================'
echo '🔬 SLURM JOB: EchoLLM Experiment'
echo 'Host         : ' \$(hostname)
echo 'Job ID       : ' \$SLURM_JOB_ID
echo 'SLURM TmpDir : ' \${SLURM_TMPDIR:-'(Not set)'}
echo 'Start Time   : ' \$(date)
echo 'CUDA devices : ' \${CUDA_VISIBLE_DEVICES:-'(unset)'}
echo '----------------------------------------'

# Ensure log and run output dirs also exist on this node
mkdir -p "${LOG_DIR}"
mkdir -p "${RUN_OUTPUT_DIR}"

# Ensure Ollama is available inside the job
export PATH="\$HOME/bin:\$PATH"
export LD_LIBRARY_PATH="\$HOME/opt/ollama/lib:\${LD_LIBRARY_PATH:-}"

module load anaconda || true
module load cuda/12.1 || true
source activate ${CONDA_ENV} 2>/dev/null || conda activate ${CONDA_ENV}
export PYTHONPATH='${PROJECT_ROOT}':\${PYTHONPATH:-}
cd '${PROJECT_ROOT}'

# --- Node-local dirs ----------------------------------------------------------
JOB_TMPDIR="\${SLURM_TMPDIR:-/tmp/\$USER/\$SLURM_JOB_ID}"
mkdir -p "\$JOB_TMPDIR"
echo "Job temp directory: \$JOB_TMPDIR"
export OLLAMA_MODELS="\$JOB_TMPDIR/ollama_models"
export OLLAMA_TMPDIR="\$JOB_TMPDIR/ollama_temp"
mkdir -p "\$OLLAMA_MODELS" "\$OLLAMA_TMPDIR"

# --- Stage models from the shared pre-warm cache ------------------------------
# Use flock so concurrent jobs on the SAME NODE don't step on each other
LOCK_FILE="/tmp/\${USER}_ollama_models.lock"
exec 9>"\$LOCK_FILE"
if command -v flock >/dev/null 2>&1; then
  flock 9
fi

echo "\U0001F4E6 Staging models from ${SHARED_OLLAMA_MODELS} → \${OLLAMA_MODELS}"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete "${SHARED_OLLAMA_MODELS}/" "\${OLLAMA_MODELS}/"
else
  # Fallback if rsync is unavailable
  tar -C "${SHARED_OLLAMA_MODELS}" -cf - . | tar -C "\${OLLAMA_MODELS}" -xf -
fi
echo "✅ Models staged."

# --- Version consistency check ------------------------------------------------
if [[ -f "${SHARED_OLLAMA_MODELS}/PREWARM_METADATA.txt" ]]; then
  PREWARM_VER=\$(grep -E '^ollama_version:' "${SHARED_OLLAMA_MODELS}/PREWARM_METADATA.txt" | sed 's/.*: *//')
  LOCAL_VER=\$(ollama --version || true)
  if [[ -n "\$PREWARM_VER" && -n "\$LOCAL_VER" && "\$PREWARM_VER" != "\$LOCAL_VER" ]]; then
    echo "❌ Ollama version mismatch. Prewarm: \$PREWARM_VER, Compute: \$LOCAL_VER"
    echo "   Ensure the same ollama build is used on login and compute nodes."
    exit 1
  fi
fi

# --- Per-job Ollama server (bind via env, not flags) ---
PORT=\$(( 12000 + (\$SLURM_JOB_ID % 1000) ))
export OLLAMA_HOST="127.0.0.1:\${PORT}"
export OPENAI_API_BASE="http://\${OLLAMA_HOST}/v1"
export OPENAI_API_KEY="ollama"

# Start server
echo "Starting Ollama server..."
( ollama serve > "${RUN_OUTPUT_DIR}/ollama_\$SLURM_JOB_ID.log" 2>&1 ) & O_PID=\$!
echo "Ollama server PID: \$O_PID"
# Ensure server is cleaned up on exit
trap 'echo "Cleaning up Ollama server..."; kill -TERM \$O_PID 2>/dev/null || true' EXIT INT TERM

# Wait until ready
echo "Waiting for Ollama server to be ready..."
for i in {1..60}; do
  if curl -sf "http://\${OLLAMA_HOST}/api/tags" >/dev/null; then
    echo "✅ Ollama server is up!"
    break
  fi
  if ! kill -0 "\$O_PID" 2>/dev/null; then
    echo '❌ Ollama crashed on startup. See log: ${LOG_DIR}/ollama_'"\$SLURM_JOB_ID"'.log'
    tail -n 100 "${LOG_DIR}/ollama_\$SLURM_JOB_ID.log" || true
    exit 1
  fi
  sleep 1
done

# Final readiness check
if ! curl -sf "http://\${OLLAMA_HOST}/api/tags" >/dev/null; then
  echo "❌ Ollama server failed to start in 60 seconds."
  exit 1
fi

echo '✅ Ollama ready on' "\$OLLAMA_HOST"

# --- Validate expected models are actually visible ----------------------------
MISSING=()
for m in ${ALL_MODELS[@]}; do
  if ! ollama list | cut -d' ' -f1 | grep -qx "\$m"; then
    MISSING+=("\$m")
  fi
done
if (( \${#MISSING[@]} > 0 )); then
  echo "❌ Missing expected models after staging: \${MISSING[*]}"
  echo "   Check that the pre-warm cache contains these tags (and tag names match)."
  exit 1
fi

echo 'CMD: ${CMD}'
${CMD}

echo '----------------------------------------'
echo 'End Time     : ' \$(date)
echo '========================================'
EOF
    done
  done
done

echo '🎉 All jobs submitted.'
