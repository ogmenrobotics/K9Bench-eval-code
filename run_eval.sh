#!/bin/bash
# K9Bench end-to-end evaluation launcher (single SLURM job, 1x A40).
#
# Pipeline:
#   1. Download YouTube videos for the dataset into ${VIDEO_DIR}.
#   2. Run model inference, writing raw outputs to ${RAW_OUTPUT}.
#   3. Compute cosine similarities -> ${SIM_OUTPUT}.
#   4. Score with cosine-similarity thresholds -> ${EVAL_OUTPUT}.
#
# Set MAX_SAMPLES > 0 to run on only the first N dataset rows (used for
# the 10-sample reproducibility smoke test; set to -1 for the full run).

#SBATCH --job-name=k9bench_eval
#SBATCH --output=slurm_logs/k9bench_eval-%j.out
#SBATCH --error=slurm_logs/k9bench_eval-%j.err
#SBATCH --gpus=a40:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --requeue
#SBATCH --signal=USR1@100

set -euo pipefail

# ---- config (override via env) ----
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-4B-Instruct}"
MAX_FRAMES="${MAX_FRAMES:-32}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"   # -1 for full dataset
VIDEO_DIR="${VIDEO_DIR:-pawbench/k9bench_release/k9bench_videos}"
OUT_DIR="${OUT_DIR:-pawbench/k9bench_release/eval_results}"

MODEL_NAME_SHORT=$(echo "$MODEL_NAME" | rev | cut -d "/" -f 1 | rev)
RAW_OUTPUT="${OUT_DIR}/raw_${MODEL_NAME_SHORT}_mf${MAX_FRAMES}_n${MAX_SAMPLES}.json"
SIM_OUTPUT="${OUT_DIR}/sim_${MODEL_NAME_SHORT}_mf${MAX_FRAMES}_n${MAX_SAMPLES}.json"
EVAL_OUTPUT="${OUT_DIR}/eval_${MODEL_NAME_SHORT}_mf${MAX_FRAMES}_n${MAX_SAMPLES}.json"

mkdir -p "${VIDEO_DIR}" "${OUT_DIR}" "pawbench/k9bench_release/slurm_logs"

export TRANSFORMERS_CACHE=/coc/testnvme/yali30/code/hf_cache/hf_models
export HF_HOME=/coc/testnvme/yali30/code/hf_cache/hf_datasets
export HF_DATASETS_CACHE=/coc/testnvme/yali30/code/hf_cache/hf_datasets

source /coc/testnvme/yali30/miniforge3/etc/profile.d/conda.sh
conda deactivate || true
conda activate vsibench

echo "============================================"
echo "K9Bench evaluation"
echo "  model        : ${MODEL_NAME}"
echo "  max_frames   : ${MAX_FRAMES}"
echo "  max_samples  : ${MAX_SAMPLES}"
echo "  video_dir    : ${VIDEO_DIR}"
echo "  raw output   : ${RAW_OUTPUT}"
echo "  eval output  : ${EVAL_OUTPUT}"
echo "  started at   : $(date)"
echo "============================================"

DL_ARGS=(--output_dir "${VIDEO_DIR}")
if [ "${MAX_SAMPLES}" -gt 0 ]; then
    DL_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

echo "[1/3] Downloading videos..."
python pawbench/k9bench_release/download_videos.py "${DL_ARGS[@]}"

EVAL_ARGS=(
    --model_name_or_path "${MODEL_NAME}"
    --bf16 --torch_dtype bfloat16
    --trust_remote_code
    --use_system_message True
    --max_frames "${MAX_FRAMES}"
    --video_dir "${VIDEO_DIR}"
    --auto_download
    --output_file "${RAW_OUTPUT}"
)
if [ "${MAX_SAMPLES}" -gt 0 ]; then
    EVAL_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

echo "[2/3] Running inference..."
srun -u python pawbench/k9bench_release/eval_video_llm.py "${EVAL_ARGS[@]}"

echo "[3/3] Scoring (cosine similarity + thresholds)..."
python pawbench/k9bench_release/evaluate_results.py \
    --mode calculate \
    --input_file "${RAW_OUTPUT}" \
    --output_file "${SIM_OUTPUT}"

python pawbench/k9bench_release/evaluate_results.py \
    --mode evaluate \
    --input_file "${SIM_OUTPUT}" \
    --output_file "${EVAL_OUTPUT}" \
    --threshold 0.5

echo "Done at $(date). Final eval JSON: ${EVAL_OUTPUT}"
