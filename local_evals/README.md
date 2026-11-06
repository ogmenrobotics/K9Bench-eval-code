# Local open-source model evaluations

Inference runners for models loaded locally with `transformers`. API-served
models live in `../api_evals/`; scoring, video download and the launcher are
shared and live at the repository root.

All commands are run **from the repository root**.

## Runners

| Script | Family | Protocol |
|---|---|---|
| `eval_video_llm.py` | Qwen3-VL | open-ended (options hidden, cosine scoring) |
| `eval_video_llm_mcq.py` | Qwen3-VL | MCQ (options in prompt, `ANSWER: X`) |
| `eval_video_llm_internvl.py` | InternVL | open-ended |
| `eval_video_llm_internvl_mcq.py` | InternVL | MCQ |

`eval_video_llm.py` is the reference implementation: the other three import its
dataset adaptation (`normalize_example`), video handling (`ensure_video`) and
answer extraction, so every model sees identical question text.

The InternVL runners do not use `qwen_vl_utils`. They sample frames with decord,
tile them to 448px with ImageNet normalization, and generate through
`model.chat()`; `eval_video_llm_internvl.py` owns that machinery and the MCQ
variant imports `process_video` from it.

## Running

Prefer the launcher at the repository root, which selects the runner, handles
sharding, and scores the result:

```bash
sbatch run_eval.sh                                   # 10-sample smoke test
NUM_SHARDS=24 bash run_eval.sh                       # full sharded run
PROMPT_STYLE=mcq MODEL_FAMILY=internvl \
    MODEL_NAME=OpenGVLab/InternVL3_5-14B NUM_SHARDS=24 bash run_eval.sh
```

To invoke a runner directly:

```bash
python local_evals/eval_video_llm.py \
    --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
    --bf16 --torch_dtype bfloat16 --trust_remote_code \
    --max_frames 32 --video_dir k9bench_videos --auto_download \
    --max_samples 10 --output_file eval_results/raw.json
```

Then score with `evaluate_results.py` (open-ended) or `evaluate_results_mcq.py`
(MCQ) from the repository root.

## Notes

**Resume.** The MCQ runners checkpoint atomically after every sample and skip
already-completed `idx` values on restart, so a preempted shard loses at most
one sample. The open-ended runners write once at the end.

**Thinking models.** Pass `--thinking_mode` (or `THINKING_MODE=1` to the
launcher). For Qwen3-VL this raises the generation budget; for InternVL it sets
the R1 reasoning-protocol system message and lowers the temperature.
