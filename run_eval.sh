#!/bin/bash
# K9Bench end-to-end evaluation launcher.
#
# Single SLURM job (download -> inference -> scoring):
#   sbatch run_eval.sh                                  # 10-sample smoke test
#   MAX_SAMPLES=-1 sbatch run_eval.sh                   # full 4744 samples
#
# Sharded across the cluster (run on the login node; submits one download job,
# NUM_SHARDS inference jobs, then a dependent merge+score job):
#   NUM_SHARDS=24 bash run_eval.sh
#
# Protocol and model family (either mode):
#   PROMPT_STYLE=openended   # default: options hidden, cosine-similarity scoring
#   PROMPT_STYLE=mcq         # options in prompt, "ANSWER: X", exact-match scoring
#   MODEL_FAMILY=qwen        # default
#   MODEL_FAMILY=internvl
#
# Examples:
#   MODEL_NAME=Qwen/Qwen3-VL-32B-Instruct GPUS=2 NUM_SHARDS=24 bash run_eval.sh
#   MODEL_FAMILY=internvl MODEL_NAME=OpenGVLab/InternVL3_5-14B PROMPT_STYLE=mcq \
#       NUM_SHARDS=24 bash run_eval.sh
#   MODEL_NAME=Qwen/Qwen3-VL-8B-Thinking THINKING_MODE=1 NUM_SHARDS=24 bash run_eval.sh

#SBATCH --job-name=k9bench_eval
#SBATCH --output=slurm_logs/k9bench_eval-%j.out
#SBATCH --error=slurm_logs/k9bench_eval-%j.err
#SBATCH --gpus=a40:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --requeue
#SBATCH --signal=USR1@100

set -euo pipefail

# Resolve the repository root. Under sbatch, SLURM executes a *copy* of this
# script from its spool directory, so BASH_SOURCE points there rather than at
# the repo; SLURM_SUBMIT_DIR is the directory sbatch was called from. K9BENCH_ROOT
# is exported by the launcher below so shard jobs never have to guess.
if [ -n "${K9BENCH_ROOT:-}" ]; then
    SCRIPT_DIR="${K9BENCH_ROOT}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

if [ ! -f "${SCRIPT_DIR}/run_eval.sh" ]; then
    echo "Cannot locate the repository root (tried ${SCRIPT_DIR})." >&2
    echo "Submit from the repo root, or set K9BENCH_ROOT=/path/to/K9-Bench-Anon" >&2
    exit 1
fi
cd "${SCRIPT_DIR}"
export K9BENCH_ROOT="${SCRIPT_DIR}"

# ---- config (override via env) ----
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-4B-Instruct}"
MAX_FRAMES="${MAX_FRAMES:-32}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"   # -1 for full dataset; ignored when NUM_SHARDS > 1
VIDEO_DIR="${VIDEO_DIR:-k9bench_videos}"
OUT_DIR="${OUT_DIR:-eval_results}"
CONDA_ENV="${CONDA_ENV:-k9bench}"
PROMPT_STYLE="${PROMPT_STYLE:-openended}"
MODEL_FAMILY="${MODEL_FAMILY:-qwen}"
THINKING_MODE="${THINKING_MODE:-0}"

# ---- sharding (override via env) ----
NUM_SHARDS="${NUM_SHARDS:-1}"
DATASET_SIZE="${DATASET_SIZE:-4744}"
GPUS="${GPUS:-1}"                  # A40s per inference job; 32B models need 2
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"

case "${MODEL_FAMILY}:${PROMPT_STYLE}" in
    qwen:openended)     EVAL_SCRIPT="local_evals/eval_video_llm.py" ;;
    qwen:mcq)           EVAL_SCRIPT="local_evals/eval_video_llm_mcq.py" ;;
    internvl:openended) EVAL_SCRIPT="local_evals/eval_video_llm_internvl.py" ;;
    internvl:mcq)       EVAL_SCRIPT="local_evals/eval_video_llm_internvl_mcq.py" ;;
    *) echo "Unsupported MODEL_FAMILY=${MODEL_FAMILY} PROMPT_STYLE=${PROMPT_STYLE}" >&2
       echo "MODEL_FAMILY must be qwen|internvl, PROMPT_STYLE must be openended|mcq" >&2
       exit 1 ;;
esac

THINKING_FLAG=""
if [ "${THINKING_MODE}" = "1" ]; then
    THINKING_FLAG="--thinking_mode"
fi

MODEL_NAME_SHORT=$(echo "$MODEL_NAME" | rev | cut -d "/" -f 1 | rev)
STYLE_SUFFIX=""
if [ "${PROMPT_STYLE}" = "mcq" ]; then
    STYLE_SUFFIX="_mcq"
fi
EXP_TAG="${EXP_TAG:-${MODEL_NAME_SHORT}_mf${MAX_FRAMES}${STYLE_SUFFIX}}"
EXP_DIR="${OUT_DIR}/${EXP_TAG}"
SHARD_DIR="${EXP_DIR}/shards"
SLURM_LOG_DIR="${EXP_DIR}/slurm_logs"

activate_env() {
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda deactivate || true
    conda activate "${CONDA_ENV}"
    export DECORD_EOF_RETRY_MAX=40960
}

score() {
    # $1 = raw json, $2 = output prefix
    if [ "${PROMPT_STYLE}" = "mcq" ]; then
        python evaluate_results_mcq.py --input_file "$1" --output_file "${2}_eval.json"
    else
        python evaluate_results.py --mode calculate --input_file "$1" --output_file "${2}_sim.json"
        python evaluate_results.py --mode evaluate --input_file "${2}_sim.json" \
            --output_file "${2}_eval.json" --threshold 0.5
    fi
    echo "Final eval JSON: ${2}_eval.json"
}

# ---------------------------------------------------------------- download job
if [ -n "${K9_DOWNLOAD:-}" ]; then
    activate_env
    python download_videos.py --output_dir "${VIDEO_DIR}"
    exit 0
fi

# ------------------------------------------------------------------ shard job
if [ -n "${K9_SHARD_START:-}" ]; then
    activate_env
    srun -u python "${EVAL_SCRIPT}" \
        --model_name_or_path "${MODEL_NAME}" \
        --bf16 --torch_dtype bfloat16 \
        --trust_remote_code ${THINKING_FLAG} \
        --max_frames "${MAX_FRAMES}" \
        --video_dir "${VIDEO_DIR}" \
        --start_idx "${K9_SHARD_START}" \
        --end_idx "${K9_SHARD_END}" \
        --output_file "${SHARD_DIR}/raw_${MODEL_NAME_SHORT}_mf${MAX_FRAMES}_idx_${K9_SHARD_START}_${K9_SHARD_END}.json"
    exit 0
fi

# ------------------------------------------------------------------ merge job
if [ -n "${K9_MERGE:-}" ]; then
    activate_env
    MERGED="${EXP_DIR}/raw_${MODEL_NAME_SHORT}_mf${MAX_FRAMES}_full.json"
    python combine_shards.py \
        --out_dir "${SHARD_DIR}" \
        --model "${MODEL_NAME_SHORT}" \
        --max_frames "${MAX_FRAMES}" \
        --chunk_size "${K9_CHUNK_SIZE:-200}" \
        --output "${MERGED}"
    score "${MERGED}" "${EXP_DIR}/${MODEL_NAME_SHORT}_mf${MAX_FRAMES}"
    exit 0
fi

# ------------------------------------------------------------- launcher mode
if [ "${NUM_SHARDS}" -gt 1 ]; then
    CHUNK_SIZE=$(( (DATASET_SIZE + NUM_SHARDS - 1) / NUM_SHARDS ))
    mkdir -p "${VIDEO_DIR}" "${SHARD_DIR}" "${SLURM_LOG_DIR}"

    echo "=========================================="
    echo "K9Bench sharded evaluation"
    echo "  exp tag      : ${EXP_TAG}"
    echo "  model        : ${MODEL_NAME}  (family=${MODEL_FAMILY})"
    echo "  prompt style : ${PROMPT_STYLE}  -> ${EVAL_SCRIPT}"
    echo "  max_frames   : ${MAX_FRAMES}"
    echo "  dataset      : ${DATASET_SIZE} samples"
    echo "  shards       : ${NUM_SHARDS} x ${CHUNK_SIZE}"
    echo "  gpus per job : ${GPUS}"
    echo "  out dir      : ${EXP_DIR}"
    echo "=========================================="

    DL_JOB=$(sbatch --parsable \
        --job-name=k9b_dl --gpus=0 --cpus-per-task=4 --time=08:00:00 \
        --output="${SLURM_LOG_DIR}/download-%j.out" \
        --error="${SLURM_LOG_DIR}/download-%j.err" \
        --export=ALL,K9BENCH_ROOT="${SCRIPT_DIR}",K9_DOWNLOAD=1 \
        run_eval.sh)
    echo "Submitted download job ${DL_JOB}"

    DEPS=""
    for ((i = 0; i < NUM_SHARDS; i++)); do
        START_IDX=$((i * CHUNK_SIZE))
        END_IDX=$(((i + 1) * CHUNK_SIZE))
        if [ ${END_IDX} -gt ${DATASET_SIZE} ]; then
            END_IDX=${DATASET_SIZE}
        fi
        [ ${START_IDX} -ge ${DATASET_SIZE} ] && break

        JID=$(sbatch --parsable \
            --job-name="k9b_${MODEL_NAME_SHORT}" \
            --gpus="a40:${GPUS}" --cpus-per-task=10 --time="${TIME_LIMIT}" \
            --output="${SLURM_LOG_DIR}/shard_${START_IDX}_${END_IDX}-%j.out" \
            --error="${SLURM_LOG_DIR}/shard_${START_IDX}_${END_IDX}-%j.err" \
            --dependency=afterok:"${DL_JOB}" \
            --export=ALL,K9BENCH_ROOT="${SCRIPT_DIR}",K9_SHARD_START="${START_IDX}",K9_SHARD_END="${END_IDX}" \
            run_eval.sh)
        DEPS="${DEPS}:${JID}"
        echo "  shard $((i + 1))/${NUM_SHARDS}: indices ${START_IDX}-$((END_IDX - 1))  job ${JID}"
    done

    MERGE_GPUS=1
    if [ "${PROMPT_STYLE}" = "mcq" ]; then
        MERGE_GPUS=0   # exact-match scoring needs no embedding model
    fi
    MERGE_JOB=$(sbatch --parsable \
        --job-name=k9b_merge --gpus="${MERGE_GPUS}" --cpus-per-task=10 --time=04:00:00 \
        --output="${SLURM_LOG_DIR}/merge-%j.out" \
        --error="${SLURM_LOG_DIR}/merge-%j.err" \
        --dependency=afterok"${DEPS}" \
        --export=ALL,K9BENCH_ROOT="${SCRIPT_DIR}",K9_MERGE=1,K9_CHUNK_SIZE="${CHUNK_SIZE}" \
        run_eval.sh)

    echo ""
    echo "Submitted merge+score job ${MERGE_JOB} (runs after all shards succeed)."
    echo "Monitor with: squeue -u \$USER"
    exit 0
fi

# ----------------------------------------------------------- single-job mode
mkdir -p "${VIDEO_DIR}" "${EXP_DIR}" slurm_logs
activate_env

PREFIX="${EXP_DIR}/${MODEL_NAME_SHORT}_mf${MAX_FRAMES}_n${MAX_SAMPLES}"
RAW_OUTPUT="${PREFIX}_raw.json"

echo "============================================"
echo "K9Bench evaluation"
echo "  model        : ${MODEL_NAME}  (family=${MODEL_FAMILY})"
echo "  prompt style : ${PROMPT_STYLE}  -> ${EVAL_SCRIPT}"
echo "  max_frames   : ${MAX_FRAMES}"
echo "  max_samples  : ${MAX_SAMPLES}"
echo "  video_dir    : ${VIDEO_DIR}"
echo "  out dir      : ${EXP_DIR}"
echo "  started at   : $(date)"
echo "============================================"

DL_ARGS=(--output_dir "${VIDEO_DIR}")
if [ "${MAX_SAMPLES}" -gt 0 ]; then
    DL_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

echo "[1/3] Downloading videos..."
python download_videos.py "${DL_ARGS[@]}"

EVAL_ARGS=(
    --model_name_or_path "${MODEL_NAME}"
    --bf16 --torch_dtype bfloat16
    --trust_remote_code
    --max_frames "${MAX_FRAMES}"
    --video_dir "${VIDEO_DIR}"
    --auto_download
    --output_file "${RAW_OUTPUT}"
)
if [ -n "${THINKING_FLAG}" ]; then
    EVAL_ARGS+=("${THINKING_FLAG}")
fi
if [ "${MAX_SAMPLES}" -gt 0 ]; then
    EVAL_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

echo "[2/3] Running inference..."
srun -u python "${EVAL_SCRIPT}" "${EVAL_ARGS[@]}"

echo "[3/3] Scoring..."
score "${RAW_OUTPUT}" "${PREFIX}"

echo "Done at $(date)."
