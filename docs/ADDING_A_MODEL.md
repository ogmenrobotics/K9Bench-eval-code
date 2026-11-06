# Adding a model

Two cases. If your model is already supported by one of the four runners
(Qwen-VL-style or InternVL-style locally, Gemini / DashScope / OpenAI-compatible
remotely), adding it is **one registry entry**. If it needs different loading or
a different API, you also write one runner.

## Case 1: an existing runner already fits

Add an entry to `k9bench/models/local_models.py` or `api_models.py`:

```python
register(ModelSpec(
    name="my-vlm-7b",                  # what you type on the command line
    model_id="my-org/MyVLM-7B",        # HF id, or the provider's model name
    family="qwen",                     # selects the runner
    gpus=1,
))
```

That is the whole change:

```bash
k9bench models | grep my-vlm
k9bench eval --model my-vlm-7b --max_samples 10
```

`family="qwen"` means the model is loaded with `AutoModelForVision2Seq` and
consumes the Qwen chat-template message format. `family="internvl"` means
`AutoModel` plus `model.chat()` with 448px tiles. Pick whichever matches how
your model expects video.

Useful fields: `protocols=("openended",)` if the model cannot do the MCQ
prompt, `thinking=True` to pass `--thinking_mode`, `gpus=2` for larger models,
and `extra_args=[...]` for flags the runner already accepts.

## Case 2: the model needs its own runner

Write a runner that produces the standard record schema, then register it.

A runner reads the dataset, generates one free-form answer per question, and
writes `{"individual_results": [...], "total_count": N}`. Each record must
carry at least:

| field | meaning |
|---|---|
| `idx` | dataset row id |
| `question`, `options`, `ground_truth` | copied from the dataset |
| `question_category` | for the per-category table |
| `response` | the raw model output |
| `answer` | the extracted answer, scored against the options |
| `scene_name`, `model_name`, `max_frames` | provenance |

The fastest route is to copy `local_evals/eval_video_llm.py` and replace the
model-loading and generation block. Import `normalize_example`, `ensure_video`
and `extract_answer_from_json` from it so your model sees byte-identical
question text — that is what keeps the comparison fair.

For an API model, copy `api_evals/eval_dashscope_qwen.py` instead and reuse
`api_common.build_prompt` and `api_common.extract_reasoning_answer`. API runners
write JSONL of provider responses; `api_evals/convert_api_results.py` then joins
in the options and gold letter from the dataset.

Then declare the runner in `k9bench/registry.py`:

```python
RUNNERS = {
    ...
    ("myfamily", "openended"): "local_evals/eval_my_model.py",
}
```

and register the model with `family="myfamily"`.

## Verifying

```bash
k9bench eval --model my-vlm-7b --dry-run          # check the command
k9bench eval --model my-vlm-7b --max_samples 5 --auto_download
k9bench score --input_file eval_results/.../raw.json \
              --output_file eval_results/.../eval.json
```

Five samples is enough to confirm the records are well-formed and the scorer
accepts them. Scale up with the launcher, which shards across the cluster:

```bash
NUM_SHARDS=24 MODEL_NAME=my-org/MyVLM-7B MODEL_FAMILY=qwen bash run_eval.sh
```

## Note on comparability

Anything that changes the prompt, the frame count, or the sampling rate makes
your number incomparable to the published table. If you must change one, say so
alongside the result — the two protocols in this repo (open-ended with cosine
scoring, and MCQ with exact match) are not interchangeable either, and the
paper's tables use the open-ended one.
