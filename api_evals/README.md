# Closed-source API evaluations

Evaluation of API-served models on K9Bench. Local open-source models are
evaluated from the repository root instead (see the top-level `README.md`).

All commands are run **from the repository root**, so that results land in the
same `eval_results/` tree as the local-model runs.

## Install

```bash
pip install -r api_evals/requirements-api.txt
```

Then put your keys in the relevant config under `api_evals/configs/`, or export
`DASHSCOPE_API_KEY` for the Qwen runners.

## Runners

| Script | Model | Paper |
|---|---|---|
| `eval_gemini.py` | Gemini 2.5 Pro (Files API, full video) | Table 1, and Table 2 with `modality: audio_video` |
| `eval_dashscope_qwen.py` | Qwen3-VL-235B-A22B (DashScope, 1 FPS) | Table 1 |
| `eval_qwen_omni.py` | Qwen3-Omni-Flash (audio+video, <150s subset) | Table 2 |

GPT-4o is not yet included.

## Two-step flow

Each runner writes JSONL as answers arrive, so an interrupted run resumes by
re-invoking the same command. The output records only what the provider
returned; `convert_api_results.py` then joins in the options, gold letter and
question category from the dataset to produce the scorer's input.

```bash
# 1. Inference
python api_evals/eval_gemini.py --config api_evals/configs/gemini.yaml

# 2. Convert to the scorer schema
python api_evals/convert_api_results.py \
    --input_file eval_results/gemini-2.5-pro/raw.jsonl \
    --output_file eval_results/gemini-2.5-pro/raw_full.json \
    --model_name gemini-2.5-pro --modality video

# 3. Score (identical to the local-model path)
python evaluate_results.py --mode calculate \
    --input_file eval_results/gemini-2.5-pro/raw_full.json \
    --output_file eval_results/gemini-2.5-pro/sim.json
python evaluate_results.py --mode evaluate \
    --input_file eval_results/gemini-2.5-pro/sim.json \
    --output_file eval_results/gemini-2.5-pro/eval.json --threshold 0.5
```

## Notes

**Video vs audio+video.** `modality` in the Gemini config selects the input.
`video` strips the audio track into `silent_videos_dir` before upload;
`audio_video` uploads the original. The chosen modality is recorded in every
result row, so the two conditions are distinguishable from the artifacts alone.

**The 150s limit.** Qwen3-Omni only accepts videos shorter than 150 seconds,
which is why it is reported on a 2693-QA subset. `eval_qwen_omni.py` measures
durations with ffprobe, caches them, and prints how many QAs were excluded.

**Join key.** `convert_api_results.py` matches on `(scene_name, question)` by
default. Use `--join_on idx` for files you know were produced against the
released dataset; result files predating publication carry row numbers from a
larger pre-release pool whose indices do not line up with `ogmen/K9Bench`.
