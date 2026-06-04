# K9Bench evaluation pipeline

End-to-end scripts to evaluate a video LLM on the
[`ogmen/K9Bench`](https://huggingface.co/datasets/ogmen/K9Bench) dataset:

1. Download YouTube videos referenced by the dataset (`video_url` →
   `{scene_name}.mp4`).
2. Run the model and write raw model outputs.
3. Score outputs with cosine-similarity matching against the MCQ options.

## Prerequisites

- A conda env with `transformers`, `qwen-vl-utils`, `sentence-transformers`,
  `flash-attn`, `datasets`, `torch`, `tqdm`, `peft`, `trl`.
- `yt-dlp` and an `ffmpeg` binary. If `ffmpeg` is not on `$PATH`, the scripts
  fall back to the binary bundled with `imageio-ffmpeg`
  (`pip install imageio-ffmpeg`).
- A GPU that supports `bf16` + `flash_attention_2` (an A40 is fine).

## Quick start (single SLURM job, 1× A40)

From the repo root:

```bash
sbatch run_eval.sh                    # default: 10 samples
MAX_SAMPLES=100 sbatch run_eval.sh    # 100-sample smoke test
MAX_SAMPLES=-1  sbatch run_eval.sh    # full 4744 samples
```

[run_eval.sh](run_eval.sh) does all three steps in one job:

| step | what it does | output |
|---|---|---|
| 1 | `download_videos.py` (yt-dlp + ffmpeg passthrough remux) | `${VIDEO_DIR}/{scene}.mp4` |
| 2 | `eval_video_llm.py` (model inference) | `raw_<model>_mf<frames>_n<N>.json` |
| 3 | `evaluate_results.py --mode calculate` | `sim_<model>_mf<frames>_n<N>.json` |
| 4 | `evaluate_results.py --mode evaluate`  | `eval_<model>_mf<frames>_n<N>.json` |

Artifacts land under `${OUT_DIR}` (default `eval_results/`).

### Environment-variable overrides for `run_eval.sh`

| var | default | meaning |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen3-VL-4B-Instruct` | HF model id |
| `MAX_FRAMES` | `32` | frames sampled per video |
| `MAX_SAMPLES` | `10` | run on first N rows only; `-1` = full dataset |
| `VIDEO_DIR` | `k9bench_videos` | where `{scene}.mp4` lives |
| `OUT_DIR` | `eval_results` | where JSONs go |

Examples:

```bash
# Reuse an existing video cache (no downloads needed)
VIDEO_DIR=/path/to/cached_videos OUT_DIR=results_oldcache MAX_SAMPLES=100 \
    sbatch run_eval.sh

# Try a different model
MODEL_NAME=Qwen/Qwen3-VL-32B-Instruct MAX_SAMPLES=-1 \
    sbatch run_eval.sh
```

## Running the scripts directly (no SLURM)

```bash
# 1. Download (or skip if videos already exist)
python download_videos.py \
    --output_dir k9bench_videos \
    --max_samples 10

# 2. Inference
python eval_video_llm.py \
    --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
    --bf16 --torch_dtype bfloat16 --trust_remote_code \
    --use_system_message True --max_frames 32 \
    --video_dir k9bench_videos \
    --max_samples 10 --auto_download \
    --output_file results/raw.json

# 3. Cosine similarities
python evaluate_results.py \
    --mode calculate --input_file results/raw.json --output_file results/sim.json

# 4. Score
python evaluate_results.py \
    --mode evaluate --input_file results/sim.json --output_file results/eval.json \
    --threshold 0.5
```

## Major CLI args

### [`download_videos.py`](download_videos.py)

| arg | default | what it does |
|---|---|---|
| `--dataset_name` | `ogmen/K9Bench` | HF dataset id |
| `--split` | `test` | dataset split |
| `--output_dir` | `./k9bench_videos` | where each `{scene_name}.mp4` is saved (`scene_name` = the 11-char YouTube id parsed from the dataset's `video_url` column) |
| `--max_samples` | `-1` | only download videos for the first N dataset rows; `-1` = all |

Each download is `yt-dlp` (preferring an mp4 ≤720p, video stream only) followed
by an `ffmpeg -c copy -movflags +faststart` passthrough remux. The remux is
**not** a re-encode: the H.264 stream is byte-identical, the container is just
rewritten to be seekable, which avoids hangs in the video reader.

### [`eval_video_llm.py`](eval_video_llm.py)

| arg | default | what it does |
|---|---|---|
| `--model_name_or_path` | `Qwen/Qwen3-VL-4B-Instruct` | HF model id |
| `--dataset_name` / `--split` | `ogmen/K9Bench` / `test` | dataset to evaluate |
| `--video_dir` | `./k9bench_videos` | local directory of `{scene}.mp4` files |
| `--auto_download` | off | if a video file is missing, fetch + remux on the fly via `download_videos.download_video` |
| `--bf16`, `--torch_dtype` | off / `None` | bf16 mixed precision; pass `--torch_dtype bfloat16` to also load weights in bf16 |
| `--trust_remote_code` | off | passed to `from_pretrained` |
| `--use_system_message` | `True` | prepend the JSON-format reasoning system prompt |
| `--max_frames` | `32` | upper bound on frames sampled per video |
| `--start_idx` / `--end_idx` | `None` / `None` | optional slice of the dataset (used for parallelizing across SLURM jobs) |
| `--max_samples` | `-1` | additional cap *after* slicing — useful for smoke tests |
| `--thinking_mode` | off | bumps `max_new_tokens` from 2048 → 3072 (used with thinking-mode models) |
| `--output_file` | `evaluation_results.json` | path for the raw output JSON |

Output JSON shape: `{"individual_results": [...], "total_count": N}`. Each
record carries `idx`, `question`, `options`, `ground_truth`, `response`,
`answer`, `scene_name`, `question_category`, `prompt`, `model_name`,
`max_frames`, `timestamp`.

### [`evaluate_results.py`](evaluate_results.py)

Two stages, picked via `--mode`:

| arg | default | what it does |
|---|---|---|
| `--mode` | required, `calculate` or `evaluate` | stage 1 (embed + cosine) or stage 2 (threshold check) |
| `--input_file` | required | input JSON (raw model outputs for `calculate`; sim JSON for `evaluate`) |
| `--output_file` | required | where to write the result JSON |
| `--threshold` | `0.5` | cosine threshold used in `evaluate` mode |

Scoring rule (fine mode, the only one we ship):
```
is_correct = sim(answer, gt_option) > threshold
             AND sim(answer, gt_option) is the strict argmax over all options
```
Embedding model is fixed to `Qwen/Qwen3-Embedding-8B`.

The final eval JSON contains `accuracy`, `correct_count`, `total_count`,
`category_wise_accuracy`, and the per-sample list under `individual_results`
with added `is_correct` / `evaluation_explanation` fields.


## Subjective Evaluation (LLM-as-Judge)

In addition to cosine-similarity scoring, open-ended answers can be evaluated
with GPT-4o as a judge via `subjective_evals.py`. The judge scores each model
output against the ground-truth answer on five axes: **Logic**, **Factuality**,
**Accuracy**, **Conciseness**, and **Overall** (all 1–10).

### Prerequisites

```bash
pip install openai datasets
export OPENAI_API_KEY=sk-...
```

### Usage

```bash
python subjective_evals.py \
    --evaluations eval_results/eval_Qwen3-VL-4B-Instruct_mf32_n100.json \
    --output      eval_results/llm_judge_Qwen3-VL-4B-Instruct.json \
    --skipped     eval_results/llm_judge_Qwen3-VL-4B-Instruct_skipped.json
```

### How it works

1. Loads ground-truth questions and options directly from the
   `ogmen/K9Bench` HuggingFace dataset (no local file needed).
2. Matches each entry to the model answer by `idx`.
3. Sends a structured prompt to `gpt-4o` and parses the scored JSON response.
4. Saves results incrementally after every entry — safe to interrupt and resume.

### Output format

Each entry in the output JSON contains:

```json
{
  "idx": 0,
  "video_url": "https://www.youtube.com/watch?v=rbrbj6Olzg4",
  "scene_name": "rbrbj6Olzg4",
  "question_category": "action sequence",
  "question": "...",
  "correct_answer": "...",
  "model_output": "...",
  "cot": "...",
  "scores": {
    "Logic": 8,
    "Factuality": 7,
    "Accuracy": 8,
    "Conciseness": 9,
    "Overall": 8
  },
  "judge_model": "gpt-4o",
  "timestamp": "2026-05-05 08:30:00"
}
```

Entries that fail (API error, parse error) are written to the `--skipped` file
and can be retried by re-running the same command — already-processed `idx`
values are skipped automatically.

### [`subjective_evals.py`](subjective_evals.py)

| arg | what it does |
|---|---|
| `--evaluations` | path to the pipeline output JSON (`eval_*.json`) |
| `--output` | path to save judge results JSON |
| `--skipped` | path to save skipped/failed entries JSON |

## Files in this folder

- [`download_videos.py`](download_videos.py) — yt-dlp + ffmpeg remux
- [`eval_video_llm.py`](eval_video_llm.py) — model inference
- [`evaluate_results.py`](evaluate_results.py) — cosine scoring (fine mode)
- [`run_eval.sh`](run_eval.sh) — SLURM launcher (1× A40) running the full pipeline
- [`subjective_evals.py`](subjective_evals.py) — subjective evaluation script
