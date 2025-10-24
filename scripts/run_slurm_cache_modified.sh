#!/bin/bash
# ==============================================================================
# SLURM Runner — EchoLLM Semantic Cache Experiment (Ensemble-style launcher)
# - Runs scripts/experiment.py across embedding_models / thresholds / seeds
# - Skips configs whose summary.csv already exists
# - Stages pre-warmed Ollama models onto node-local storage (no WAN pulls)
# - Starts a per-job Ollama server, sets OPENAI_API_BASE, then runs the job
# - NEW: Supports custom workload path (OASST dataset or mock)
# ==============================================================================

set -euo pipefail

# --------------------
# Defaults (override via CLI below)
# --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONDA_ENV="gptcache-env"

EMBEDDING_MODELS=("mxbai-embed-large")
SEEDS=(42)

# Algorithm-specific configurations (algo:metric:thr1,thr2,...)
declare -a ALGO_CONFIGS=(
  "search_distance:ip:0.9,0.99,0.95"
  "search_distance:l2:0.15,0.2,0.25,0.3"
  "sbert_crossencoder:none:0.95"
  "sequence_match:none:0.7,0.8,0.9"
  "exact_match:none:1.0"
)

# Add eviction policies (can be overridden via CLI)
EVICTION_POLICIES=("LFU")

# NEW: Workload configuration
# Empty string = use mock workload (default for backward compatibility)
# Set to path = use OASST workload with ground truth
WORKLOAD_PATH="workload_oasst.json"  # Use OASST workload

# SLURM Resources (tune for your cluster)
PARTITION="gpu_partition"
# If your site uses --gpus (typed), prefer GPUS and leave GRES empty
GRES=""
GPUS="rtx_4090:1"
CPUS=4
MEM="32G"
TIME="0-04:00:00"

# Paths
PYTHON_SCRIPT="${PROJECT_ROOT}/scripts/experiment.py"
OUTPUT_DIR="${PROJECT_ROOT}/slurm_results"

# Pre-warmed models location (must match your prewarm_ollama.sh)
SHARED_OLLAMA_MODELS="${HOME}/.ollama_shared_models"

# All models required on compute nodes (MUST exist in SHARED_OLLAMA_MODELS)
ALL_MODELS=("nomic-embed-text:latest" "mxbai-embed-large:latest" "llama3.1:8b")

# ----------------------------------------------------------------------
# CLI overrides (mimic ensemble-style flags)
# ----------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --embedding-models)       IFS=' ' read -r -a EMBEDDING_MODELS <<< "$2"; shift 2;;
    --thresholds)             IFS=' ' read -r -a THRESHOLDS <<< "$2"; shift 2;;
    --seeds)                  IFS=' ' read -r -a SEEDS <<< "$2"; shift 2;;
    --algo-configs)           IFS=' ' read -r -a ALGO_CONFIGS <<< "$2"; shift 2;;
    --conda-env)              CONDA_ENV="$2"; shift 2;;
    --output-dir)             OUTPUT_DIR="$2"; shift 2;;
    --shared-models)          SHARED_OLLAMA_MODELS="$2"; shift 2;;
    --all-models)             IFS=' ' read -r -a ALL_MODELS <<< "$2"; shift 2;;
    --eviction-policies)      IFS=' ' read -r -a EVICTION_POLICIES <<< "$2"; shift 2;;
    
    # NEW: Workload path option
    --workload-path)          WORKLOAD_PATH="$2"; shift 2;;
    --use-mock)               WORKLOAD_PATH=""; shift 1;;  # Explicit mock mode

    # SLURM resources
    --partition)              PARTITION="$2"; shift 2;;
    --gres)                   GRES="$2"; shift 2;;
    --gpus)                   GPUS="$2"; shift 2;;
    --cpus)                   CPUS="$2"; shift 2;;
    --mem)                    MEM="$2"; shift 2;;
    --time)                   TIME="$2"; shift 2;;

    --help|-h)
      echo "Usage: $0 [all|status] [options]"
      echo "Options:"
      echo "  --embedding-models 'm1 m2'        (default: ${EMBEDDING_MODELS[*]})"
      echo "  --thresholds 't1 t2'              (default: ${THRESHOLDS[*]:-})"
      echo "  --seeds 's1 s2'                   (default: ${SEEDS[*]})"
      echo "  --algo-configs 'a:m:t1,t2 ...'    (default: predefined)"
      echo "                                     ex: 'search_distance:ip:0.9,0.95 exact_match:none:1.0'"
      echo "  --eviction-policies 'p1 p2'       (default: ${EVICTION_POLICIES[*]})"
      echo "  --workload-path PATH              OASST workload JSON (default: mock)"
      echo "  --use-mock                        Explicitly use mock workload"
      echo "  --conda-env NAME                  (default: $CONDA_ENV)"
      echo "  --output-dir PATH                 (default: $OUTPUT_DIR)"
      echo "  --shared-models PATH              prewarmed models dir (default: $SHARED_OLLAMA_MODELS)"
      echo "  --all-models 'n1 n2'              expected model tags (default: ${ALL_MODELS[*]})"
      echo "  --partition P                     (default: $PARTITION)"
      echo "  --gres 'gpu:1' OR --gpus '1'      set ONE of these; others auto-handled"
      echo "  --cpus N                          (default: $CPUS)"
      echo "  --mem SIZE                        (default: $MEM)"
      echo "  --time D-HH:MM:SS                 (default: $TIME)"
      echo ""
      echo "Examples:"
      echo "  # Use OASST workload with ground truth:"
      echo "  $0 all --workload-path workload_oasst.json"
      echo ""
      echo "  # Use mock workload (default):"
      echo "  $0 all"
      echo "  $0 all --use-mock"
      echo ""
      echo "  # Test single config with OASST:"
      echo "  $0 all --embedding-models 'nomic-embed-text' \\
         --algo-configs 'search_distance:ip:0.95' \\
         --eviction-policies 'AP LRU' \\
         --seeds '42' \\
         --workload-path workload_oasst.json"
      exit 0;;
    *) break;;
  esac
done

# Build algorithm path naming per (algo, metric)
get_algo_dir() {
  local algo="$1" metric="$2"
  if [[ "${algo}" == "search_distance" ]]; then
    echo "${algo}-${metric}"
  else
    echo "${algo}"
  fi
}

# Determine workload name for path structure
get_workload_name() {
  if [[ -z "${WORKLOAD_PATH}" ]]; then
    echo "mock"
  else
    # Extract filename without extension (e.g., workload_oasst.json -> oasst)
    local basename=$(basename "${WORKLOAD_PATH}" .json)
    echo "${basename#workload_}"  # Remove 'workload_' prefix if present
  fi
}

# GPU flags array (safe under 'set -u') — prefer --gpus if set
GPU_FLAGS=()
if [[ -n "${GPUS:-}" ]]; then
  GPU_FLAGS+=(--gpus="${GPUS}")
elif [[ -n "${GRES:-}" ]]; then
  GPU_FLAGS+=(--gres="${GRES}")
fi

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

check_output_exists() {
  local embedding_model="$1" algo="$2" metric="$3" threshold="$4" eviction="$5" seed="$6"
  local algo_dir=$(get_algo_dir "$algo" "$metric")
  local workload_name=$(get_workload_name)
  local run_dir="${OUTPUT_DIR}/${workload_name}/${embedding_model}/${algo_dir}/${threshold}/evict_${eviction}/seed_${seed}"
  [[ -s "${run_dir}/summary.csv" ]]
}

submit_single_job() {
  local embedding_model="$1" algo="$2" metric="$3" threshold="$4" eviction="$5" seed="$6"

  if check_output_exists "${embedding_model}" "${algo}" "${metric}" "${threshold}" "${eviction}" "${seed}"; then
    echo "✅ SKIP: ${embedding_model} | ${algo}(${metric}) | thr=${threshold} | evict=${eviction} | seed=${seed}"
    return 2
  fi

  local algo_dir=$(get_algo_dir "$algo" "$metric")
  local workload_name=$(get_workload_name)
  local run_dir="${OUTPUT_DIR}/${workload_name}/${embedding_model}/${algo_dir}/${threshold}/evict_${eviction}/seed_${seed}"
  # Central log directory (per-workload)
  local log_dir="${OUTPUT_DIR}/${workload_name}/logs"
  mkdir -p "${log_dir}"
  # More descriptive log filename with job details
  local log_file="${log_dir}/${embedding_model}_${algo}_${metric}_thr${threshold}_${eviction}_s${seed}_%j.out"

  local job_name="cache_${embedding_model}_${algo}_${eviction}_t${threshold}_s${seed}"

  echo "🚀 SUBMIT: ${job_name}"
  # Flatten ALL_MODELS array to a space-separated string for export
  local all_models_str="${ALL_MODELS[*]}"
  sbatch \
    --partition="${PARTITION}" \
    --job-name="${job_name}" \
    --output="${log_file}" \
    --time="${TIME}" \
    --ntasks=1 \
    --cpus-per-task="${CPUS}" \
    --mem="${MEM}" \
    --export=ALL,LOG_DIR="${log_dir}",CONDA_ENV="${CONDA_ENV}",PROJECT_ROOT="${PROJECT_ROOT}",PYTHON_SCRIPT="${PYTHON_SCRIPT}",EMBEDDING_MODEL="${embedding_model}",THRESHOLD="${threshold}",SIMILARITY_ALGO="${algo}",VECTOR_METRIC="${metric}",EVICTION_POLICY="${eviction}",SEED="${seed}",OUTPUT_DIR="${OUTPUT_DIR}",WORKLOAD_NAME="${workload_name}",SHARED_OLLAMA_MODELS="${SHARED_OLLAMA_MODELS}",ALL_MODELS_STR="${all_models_str}",WORKLOAD_PATH="${WORKLOAD_PATH}" \
    "${GPU_FLAGS[@]}" << 'EOF'
#!/bin/bash
set -euo pipefail

echo '========================================'
echo '🔬 SLURM JOB: EchoLLM Cache Experiment'
echo 'Job ID       : ' ${SLURM_JOB_ID}
echo 'Host         : ' $(hostname)
echo 'Start Time   : ' $(date)
echo 'CUDA devices : ' ${CUDA_VISIBLE_DEVICES:-'(unset)'}
echo '----------------------------------------'

mkdir -p "$LOG_DIR"

export PATH="$HOME/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/ollama/lib:${LD_LIBRARY_PATH:-}"
unset OLLAMA_NUM_GPU

module load anaconda || true
module load cuda/12.1 || true
source activate ${CONDA_ENV} 2>/dev/null || conda activate ${CONDA_ENV}
export PYTHONPATH="${PROJECT_ROOT}":${PYTHONPATH:-}
cd "${PROJECT_ROOT}"

JOB_TMPDIR="$HOME/slurm_tmp/$SLURM_JOB_ID"
mkdir -p "$JOB_TMPDIR"
export OLLAMA_MODELS="$JOB_TMPDIR/ollama_models"
export OLLAMA_TMPDIR="$JOB_TMPDIR/ollama_tmp"
mkdir -p "$OLLAMA_MODELS" "$OLLAMA_TMPDIR"

LOCK_FILE="/tmp/${USER}_ollama_models.lock"
exec 9>"$LOCK_FILE"
if command -v flock >/dev/null 2>&1; then flock 9; fi

echo "📦 Staging models from '${SHARED_OLLAMA_MODELS}' → '$OLLAMA_MODELS'"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete "${SHARED_OLLAMA_MODELS}/" "$OLLAMA_MODELS/"
else
  tar -C "${SHARED_OLLAMA_MODELS}" -cf - . | tar -C "$OLLAMA_MODELS" -xf -
fi
echo "✅ Models staged."

# ---- Optional client version check (non-fatal) -------------------------------
if [[ -f "${SHARED_OLLAMA_MODELS}/PREWARM_METADATA.txt" ]]; then
  PREWARM_CLIENT_VER="$(awk -F': ' '/^ollama_client_version:/ {print $2}' "${SHARED_OLLAMA_MODELS}/PREWARM_METADATA.txt" | tail -n1)"
  LOCAL_CLIENT_VER="$(ollama --version 2>&1 | sed -n 's/.*client version is \([0-9.]\+\).*/\1/p')"

  if [[ -n "${PREWARM_CLIENT_VER}" && -n "${LOCAL_CLIENT_VER}" && "${PREWARM_CLIENT_VER}" != "${LOCAL_CLIENT_VER}" ]]; then
    echo "⚠️  Ollama client version differs: prewarm=${PREWARM_CLIENT_VER}, compute=${LOCAL_CLIENT_VER}. Proceeding anyway."
  else
    echo "✅ Ollama client version OK: ${LOCAL_CLIENT_VER:-unknown}"
  fi
fi

PORT=$(( 12000 + ($SLURM_JOB_ID % 1000) ))
export OLLAMA_HOST="127.0.0.1:${PORT}"
export OPENAI_API_BASE="http://${OLLAMA_HOST}/v1"
export OPENAI_API_KEY="ollama"

echo "Starting Ollama server on ${OLLAMA_HOST} …"
( ollama serve > "$LOG_DIR/ollama_$SLURM_JOB_ID.log" 2>&1 ) & O_PID=$!
trap 'echo "Cleaning up Ollama server…"; kill -TERM $O_PID 2>/dev/null || true' EXIT INT TERM

for i in {1..60}; do
  if curl -sf "http://${OLLAMA_HOST}/api/tags" >/dev/null; then
    echo "✅ Ollama ready."
    break
  fi
  if ! kill -0 "$O_PID" 2>/dev/null; then
    echo '❌ Ollama crashed on startup. Tail:'
    tail -n 150 "$LOG_DIR/ollama_$SLURM_JOB_ID.log" || true
    exit 1
  fi
  sleep 1
done

MISSING=()
MAP=$(ollama list | awk '{print $1}')
for m in ${ALL_MODELS_STR}; do
  echo "$MAP" | grep -qx "$m" || MISSING+=("$m")
done
if (( ${#MISSING[@]} > 0 )); then
  echo "❌ Missing expected models after staging: ${MISSING[*]}"
  echo "   Make sure they were pre-warmed into ${SHARED_OLLAMA_MODELS}."
  exit 1
fi

nvidia-smi || true

echo "Executing experiment with:"
echo "  Model: ${EMBEDDING_MODEL}"
echo "  Threshold: ${THRESHOLD}"
echo "  Algorithm: ${SIMILARITY_ALGO}"
echo "  Metric: ${VECTOR_METRIC}"
echo "  Eviction: ${EVICTION_POLICY}"
echo "  Seed: ${SEED}"

# NEW: Check workload type and add argument if specified
if [[ -n "${WORKLOAD_PATH}" ]]; then
  echo "  Workload: ${WORKLOAD_PATH} (OASST dataset)"
  
  # Verify workload file exists
  if [[ ! -f "${PROJECT_ROOT}/${WORKLOAD_PATH}" ]]; then
    echo "❌ ERROR: Workload file not found: ${PROJECT_ROOT}/${WORKLOAD_PATH}"
    echo "   Generate it first with: python scripts/prepare_oasst_workload.py"
    exit 1
  fi
  
  WORKLOAD_ARG="--workload-path ${WORKLOAD_PATH}"
else
  echo "  Workload: mock (default)"
  WORKLOAD_ARG=""
fi

python "${PYTHON_SCRIPT}" \
  --embedding-model "${EMBEDDING_MODEL}" \
  --similarity-threshold "${THRESHOLD}" \
  --similarity-algo "${SIMILARITY_ALGO}" \
  --vector-metric "${VECTOR_METRIC}" \
  --eviction-policy "${EVICTION_POLICY}" \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}/${WORKLOAD_NAME}" \
  ${WORKLOAD_ARG}

echo '----------------------------------------'
echo 'End Time     : ' $(date)
echo '========================================'
EOF

  return 0
}

run_all() {
  echo "🔥 Submitting EchoLLM Cache grid"
  echo "Embedding models  : ${EMBEDDING_MODELS[*]}"
  echo "Algorithm configs : ${ALGO_CONFIGS[*]}"
  echo "Eviction policies : ${EVICTION_POLICIES[*]}"
  echo "Seeds             : ${SEEDS[*]}"
  
  # NEW: Show workload configuration
  if [[ -n "${WORKLOAD_PATH}" ]]; then
    echo "Workload          : ${WORKLOAD_PATH} (OASST with ground truth)"
    
    # Verify workload exists before submitting all jobs
    if [[ ! -f "${PROJECT_ROOT}/${WORKLOAD_PATH}" ]]; then
      echo ""
      echo "❌ ERROR: Workload file not found: ${PROJECT_ROOT}/${WORKLOAD_PATH}"
      echo ""
      echo "Generate it first with:"
      echo "  python scripts/prepare_oasst_workload.py \\\n+    --embedding-model nomic-embed-text \\\n+    --num-prompts 800 \\\n+    --num-queries 2000 \\\n+    --output ${WORKLOAD_PATH}"
      echo ""
      exit 1
    fi
  else
    echo "Workload          : mock (default, no ground truth)"
  fi
  
  echo "Output dir        : ${OUTPUT_DIR}"
  echo "Shared models     : ${SHARED_OLLAMA_MODELS}"
  echo "Partition         : ${PARTITION}"
  echo "GPU flags         : ${GPU_FLAGS[*]:-<none>}"
  echo "----------------------------------------"

  local total=0 sub=0 skip=0
  for em in "${EMBEDDING_MODELS[@]}"; do
    for config in "${ALGO_CONFIGS[@]}"; do
      IFS=':' read -r algo metric thresholds_str <<< "$config"
      IFS=',' read -r -a thresholds <<< "$thresholds_str"
      for thr in "${thresholds[@]}"; do
        for evict in "${EVICTION_POLICIES[@]}"; do
          for sd in "${SEEDS[@]}"; do
            total=$((total+1))
            if submit_single_job "${em}" "${algo}" "${metric}" "${thr}" "${evict}" "${sd}"; then
              rc=$?
            else
              rc=$?
            fi
            case ${rc} in
              0) sub=$((sub+1));;
              2) skip=$((skip+1));;
              *) ;;
            esac
            sleep 0.2
            if (( total % 10 == 0 )); then
              echo "Pausing for 5 seconds to avoid overwhelming scheduler..."
              sleep 5
            fi
          done
        done
      done
    done
  done

  echo "========================================"
  echo "📊 Submission Summary"
  echo "Total     : ${total}"
  echo "Submitted : ${sub}"
  echo "Skipped   : ${skip}"
  echo "========================================"
  echo "Use: squeue -u \\${USER} --name='cache_*'"
}

status() {
  echo "📊 Status (completed configs)"
  local total=0 done=0
  for em in "${EMBEDDING_MODELS[@]}"; do
    for config in "${ALGO_CONFIGS[@]}"; do
      IFS=':' read -r algo metric thresholds_str <<< "$config"
      IFS=',' read -r -a thresholds <<< "$thresholds_str"
      for thr in "${thresholds[@]}"; do
        for evict in "${EVICTION_POLICIES[@]}"; do
          for sd in "${SEEDS[@]}"; do
            total=$((total+1))
            check_output_exists "${em}" "${algo}" "${metric}" "${thr}" "${evict}" "${sd}" && done=$((done+1))
          done
        done
      done
    done
  done
  local pct=0
  if [[ ${total} -gt 0 ]]; then pct=$((100*done/total)); fi
  echo "Done: ${done} / ${total} (${pct}%)"
  echo "Queued jobs:"
  squeue -u "${USER}" --name="cache_*" --format="%.18j %.8T %.10M" 2>/dev/null || true
}

case "${1:-all}" in
  all)    run_all ;;
  status) status ;;
  *) echo "Unknown command '$1' (use all|status)"; exit 1 ;;
esac

