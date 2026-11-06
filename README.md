<p align="center">
  <a href="https://arxiv.org/abs/2607.02680">
    <img src="https://img.shields.io/badge/arXiv-2607.02680-B31B1B?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://ogmenrobotics.github.io/K9Bench/">
    <img src="https://img.shields.io/badge/Website-K9--Bench-blue?style=flat-square" alt="Website">
  </a>
  <img src="https://img.shields.io/badge/NeurIPS_E%26D-2026-2E8B57?style=flat-square" alt="Accepted at NeurIPS E&D 2026">
  <a href="https://huggingface.co/datasets/ogmen/K9Bench">
    <img src="https://img.shields.io/badge/Dataset-Hugging_Face-yellow?style=flat-square&logo=huggingface&logoColor=white" alt="Dataset">
  </a>
</p>

<h1 align="center">
  K9-Bench: Evaluating Multimodal LLMs on Canine-Centric Videos
</h1>

<p align="center">
  <a href="#">Khush Attarde*</a>,
  <a href="#">Yusuf Ali*</a>,
  <a href="#">Megha Thukral<sup>†</sup></a>,
  <a href="#">Divye Bhutani</a>,
  <a href="#">Thomas Ploetz</a>,
  <a href="#">Zsolt Kira</a>
</p>

<p align="center">
  <em>
    * Equal contribution &nbsp; · &nbsp; † Core contributor
  </em>
</p>

---


## ⚠️ License & Opt-Out

> **Please read the [LICENSE.md](./LICENSE.md) for the Terms of Use and licensing conditions before accessing or using K9-Bench.**
>
> Content creators and rights holders may request the removal of their videos from K9-Bench at any time by submitting the [Takedown and Opt-Out Form](https://forms.gle/9YtQAwaoqnKhWriU6), or by emailing [khushattarde2003@gmail.com](mailto:khushattarde2003@gmail.com) or [yusufali.dtu.98@gmail.com](mailto:yusufali.dtu.98@gmail.com).

---



# K9Bench evaluation suite

Evaluate multimodal LLMs on
`[ogmen/K9Bench](https://huggingface.co/datasets/ogmen/K9Bench)` — 4744
five-way multiple-choice questions over 907 real-world videos of domestic dogs,
covering posture, action sequences, interaction, context and cause-effect
reasoning.

The suite covers local open-source models loaded with `transformers` and
closed-source models served over an API, under two evaluation protocols, with
sharded multi-GPU execution and both scoring methods used in the paper.

---



## 1. Install

```bash
bash install.sh          # creates the `k9bench` conda env and installs everything
conda activate k9bench
pip install -e .         # exposes the `k9bench` command
```

`install.sh` pins every dependency to the exact version used for the reported
results (see `requirements.txt`) and installs `flash-attn` last, since it needs
torch present at build time.

For closed-source API models, additionally:

```bash
pip install -r api_evals/requirements-api.txt
```

Requirements: a GPU supporting `bf16` and `flash_attention_2` (an A40 is
sufficient for models up to 14B; 32B needs two), plus `yt-dlp` and `ffmpeg`.

---



## 2. Quick start

Ten samples, run right here. No cluster required:

```bash
k9bench eval --model qwen3-vl-4b --max_samples 10 --auto_download
k9bench score --input_file eval_results/qwen3-vl-4b_mf32/raw.json \
              --output_file eval_results/qwen3-vl-4b_mf32/eval.json
```

This is the command to run first: it confirms the install, the dataset and the
video download all work before committing GPU hours.

On a SLURM cluster, add `--shards` to submit instead of running locally:

```bash
k9bench eval --model qwen3-vl-4b --shards 1     # one job: download, infer, score
k9bench eval --model qwen3-vl-8b --shards 24    # 24 parallel shards, merged and scored
```

`--shards` hands off to `run_eval.sh`, deriving the model's family, GPU count
and thinking flag from the registry, and scoring the merged result for you.

---



## 3. The `k9bench` command

```bash
k9bench models                                   # list every registered model
k9bench models --kind local

k9bench eval --model qwen3-vl-8b --max_samples 10 --auto_download
k9bench score --input_file eval_results/qwen3-vl-8b_mf32/raw.json \
              --output_file eval_results/qwen3-vl-8b_mf32/eval.json
```

Add `--dry-run` to any command to print what would run without running it.

`k9bench eval` selects the right runner from the model name, so you do not need
to know whether a model is Qwen-style or InternVL-style, or which script handles
which prompt protocol. Anything after the recognised flags is passed straight
through to the underlying runner.


| Command                                        | Purpose                                                                                                        |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `k9bench models [--kind local|api]`            | list registered models, their family and supported protocols                                                   |
| `k9bench eval --model NAME`                    | run inference here (`--protocol`, `--max_frames`, `--max_samples`, `--video_dir`, `--output_file`, `--config`) |
| `k9bench eval --model NAME --shards N`         | submit to SLURM instead: `1` = one job doing everything, `N>1` = N parallel shards, merged and scored          |
| `k9bench score --input_file … --output_file …` | score a raw result file (`--protocol`, `--threshold`)                                                          |


The CLI only builds command lines: without `--shards` it invokes a runner in
`local_evals/` or `api_evals/`, and with `--shards` it invokes `run_eval.sh`.
Both remain directly callable if you prefer to drive them yourself.

### Registered models

Local (evaluated at 32 frames): `qwen3-vl-4b`, `qwen3-vl-8b`, `qwen3-vl-32b`,
their `-thinking` variants, `internvl3.5-8b`, `internvl3.5-14b`.

API: `gemini-2.5-pro`, `gemini-2.5-pro-av` (audio+video), `qwen3-vl-235b`,
`qwen3-omni-flash` (audio+video, <150s subset), `gpt-4o` (32 frames).

---



## 4. Two evaluation protocols

These are **not interchangeable**, and the distinction matters when comparing
numbers.

**Open-ended + cosine similarity** (default; this is what the paper's tables
use). The model never sees the options. It answers in free form, the answer is
embedded with `Qwen3-Embedding-8B`, and a response counts as correct only if
similarity to the gold option exceeds 0.5 *and* beats every distractor.

```bash
k9bench eval --model qwen3-vl-8b --protocol openended
k9bench score --input_file … --output_file … --protocol openended
```

**MCQ prompting + exact match.** All five options go in the prompt, the model
ends with `ANSWER: X`, and the letter is compared directly. Needs no GPU to
score.

```bash
k9bench eval --model qwen3-vl-8b --protocol mcq
k9bench score --input_file … --output_file … --protocol mcq
```

---



## 5. Reproducing the paper



### Table 1 and Table 3 — local open-source models

```bash
k9bench eval --model qwen3-vl-8b          --shards 24
k9bench eval --model internvl3.5-14b      --shards 24
k9bench eval --model qwen3-vl-32b-thinking --shards 24
```

Each writes `eval_results/<tag>/<model>_mf32_eval.json`, containing `accuracy`
and `category_wise_accuracy` — the Table 1 row and its per-category columns.
GPU count and the thinking flag come from the registry, so the 32B models
request two A40s and the `-thinking` variants get `--thinking_mode` without you
having to remember either.

### Table 1 and Table 2 — API models

Three steps: inference, convert, score. See `api_evals/README.md`.

```bash
python api_evals/eval_gemini.py --config api_evals/configs/gemini.yaml
python api_evals/convert_api_results.py \
    --input_file eval_results/gemini-2.5-pro/raw.jsonl \
    --output_file eval_results/gemini-2.5-pro/raw_full.json \
    --model_name gemini-2.5-pro --modality video
k9bench score --input_file eval_results/gemini-2.5-pro/raw_full.json \
              --output_file eval_results/gemini-2.5-pro/eval.json
```

For Table 2, set `modality: audio_video` in the Gemini config, or use
`api_evals/configs/qwen3_omni.yaml` for Qwen3-Omni. The Omni runner applies the
150-second limit that produces the 2693-QA subset and reports how many questions
it excluded.

### Table 3 — LLM-as-judge

```bash
export OPENAI_API_KEY=sk-...
python subjective_evals.py \
    --evaluations eval_results/<tag>/<model>_mf32_eval.json \
    --output      eval_results/<tag>/llm_judge.json \
    --skipped     eval_results/<tag>/llm_judge_skipped.json
```

GPT-4o scores each answer 1–10 on logic, factuality, accuracy, conciseness and
overall quality. Saves after every entry, so it is safe to interrupt and resume.

---



## 6. Repository layout

```
run_eval.sh              launcher: single job or sharded, both protocols
install.sh               creates the conda env
local_evals/             inference for locally-loaded models
api_evals/               inference for API-served models, plus the converter
k9bench/                 CLI and model registry
docs/ADDING_A_MODEL.md   how to plug in a new model
download_videos.py       yt-dlp fetch + ffmpeg remux
evaluate_results.py      cosine-similarity scorer (open-ended)
evaluate_results_mcq.py  exact-match scorer (MCQ, no GPU)
subjective_evals.py      GPT-4o judge
combine_shards.py        merge sharded runs, report missing shards
```

Inference lives in the two `*_evals/` folders because it is the part that varies
per model family. Everything downstream — download, scoring, judging, merging —
is shared and sits at the root.

---



## 7. Running at scale

`run_eval.sh` is the SLURM backend behind `k9bench eval --shards N`, and the
only shell script. `NUM_SHARDS=1` runs everything in one job; `NUM_SHARDS>1`
submits a dependency chain of download → shards → merge and score.

You normally do not call it directly — `--shards` sets these variables from the
registry. Drive it yourself when you need a knob the registry does not carry,
such as a custom time limit or an unregistered checkpoint:

```bash
NUM_SHARDS=24 MODEL_NAME=my-org/MyVLM-7B MODEL_FAMILY=qwen TIME_LIMIT=08:00:00 \
    bash run_eval.sh
```


| Variable                | Default                           | Meaning                                         |
| ----------------------- | --------------------------------- | ----------------------------------------------- |
| `MODEL_NAME`            | `Qwen/Qwen3-VL-4B-Instruct`       | HuggingFace model id                            |
| `MODEL_FAMILY`          | `qwen`                            | `qwen` or `internvl`                            |
| `PROMPT_STYLE`          | `openended`                       | `openended` or `mcq`                            |
| `NUM_SHARDS`            | `1`                               | >1 shards across that many SLURM jobs           |
| `MAX_FRAMES`            | `32`                              | frames sampled per video                        |
| `MAX_SAMPLES`           | `10`                              | single-job mode only; `-1` for the full dataset |
| `THINKING_MODE`         | `0`                               | `1` for `-Thinking` model variants              |
| `GPUS`                  | `1`                               | A40s per shard; 32B models need 2               |
| `VIDEO_DIR` / `OUT_DIR` | `k9bench_videos` / `eval_results` | paths                                           |
| `CONDA_ENV`             | `k9bench`                         | environment to activate inside the job          |


If a shard fails, rerun the launcher: the MCQ runners resume from their
checkpoint, and `combine_shards.py` reports which index ranges are missing.

---



## 8. Adding a model

For a model that an existing runner already fits, this is a single registry
entry in `k9bench/models/`:

```python
register(ModelSpec(name="my-vlm-7b", model_id="my-org/MyVLM-7B",
                   family="qwen", gpus=1))
```

```bash
k9bench eval --model my-vlm-7b --max_samples 5 --auto_download
```

See [docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md) for models needing their
own loading or a different API.

---



## 9. Notes

**Videos are not redistributed.** Only YouTube ids ship with the dataset;
`download_videos.py` fetches each one and remuxes it with
`ffmpeg -c copy -movflags +faststart`. That remux is not a re-encode — it makes
the container seekable, which otherwise causes the video reader to hang.

---



## Citation

```bash
@article{attarde2026k9bench,
  title={K9-Bench: Evaluating Multimodal LLMs on Canine-Centric Videos},
  author={Attarde, Khush and Ali, Yusuf and Thukral, Megha and Bhutani, Divye and Ploetz, Thomas and Kira, Zsolt},
  journal={arXiv preprint arXiv:2607.02680},
  year={2026},
  url={https://arxiv.org/abs/2607.02680}
}
```

