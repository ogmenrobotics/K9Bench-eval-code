#!/usr/bin/env python3
# Copyright 2025 The K9Bench Authors.
# SPDX-License-Identifier: Apache-2.0

"""K9Bench evaluation with Qwen3-VL via Alibaba DashScope.

Videos are passed as local ``file://`` URIs and sampled server-side at ``fps``
(1 by default, matching the reported runs). Results are appended as JSONL so an
interrupted run resumes. Convert with ``convert_api_results.py`` and score with
``evaluate_results.py``.
"""

import argparse
import os
import sys
import time

import dashscope
import yaml
from dashscope import MultiModalConversation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_common import (  # noqa: E402
    TokenBucket,
    append_jsonl,
    build_prompt,
    extract_reasoning_answer,
    load_jsonl,
    load_k9bench_qas,
)


def main():
    ap = argparse.ArgumentParser(description="DashScope Qwen3-VL evaluator for K9Bench.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--start_idx", type=int, default=None)
    ap.add_argument("--end_idx", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    api_key = cfg.get("api_key") or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key or api_key.startswith("YOUR_"):
        raise SystemExit("Set api_key in the config or DASHSCOPE_API_KEY in the environment.")

    if cfg.get("base_url"):
        dashscope.base_http_api_url = cfg["base_url"]

    model = cfg.get("model", "qwen3-vl-235b-a22b-instruct")
    fps = cfg.get("fps", 1)
    videos_dir = cfg.get("videos_dir", "k9bench_videos")
    results_file = cfg.get("results_file", "eval_results/qwen3vl-235b/raw.jsonl")
    failed_log = cfg.get("failed_log", "eval_results/qwen3vl-235b/failed.log")
    modality = cfg.get("modality", "video")
    bucket = TokenBucket(int(cfg.get("rate_limit_rpm", 60)),
                         int(cfg.get("rate_limit_tpm", 100_000)))

    data = load_k9bench_qas(cfg.get("hf_dataset", "ogmen/K9Bench"),
                            cfg.get("hf_split", "test"))
    if args.start_idx is not None:
        data = [q for q in data if q["idx"] >= args.start_idx]
    if args.end_idx is not None:
        data = [q for q in data if q["idx"] <= args.end_idx]

    done = {r["idx"] for r in load_jsonl(results_file) if "idx" in r}
    todo = [q for q in data if q["idx"] not in done]
    print(f"model={model} fps={fps}  already done={len(done)}  to process={len(todo)}")

    os.makedirs(os.path.dirname(failed_log) or ".", exist_ok=True)

    for n, qa in enumerate(todo, 1):
        video_path = os.path.join(videos_dir, f"{qa['scene_name']}.mp4")
        if not os.path.exists(video_path):
            with open(failed_log, "a") as f:
                f.write(f"{qa['scene_name']} (idx:{qa['idx']}) missing video\n")
            continue

        messages = [{"role": "user", "content": [
            {"video": f"file://{os.path.abspath(video_path)}", "fps": fps},
            {"text": build_prompt(qa["question"])},
        ]}]

        try:
            bucket.wait(token_estimate=400 + fps * 1000 + 500)
            response = MultiModalConversation.call(
                api_key=api_key, model=model, messages=messages)
            if response.status_code != 200:
                raise RuntimeError(f"status {response.status_code}: {response.message}")

            content = response.output.choices[0].message.content
            text = content[0]["text"] if isinstance(content, list) else str(content)
            reasoning, answer = extract_reasoning_answer(text)

            usage = getattr(response, "usage", None)
            tokens = (getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "output_tokens", 0) or 0)
            bucket.record_tokens(tokens)

            append_jsonl(results_file, {
                "idx": qa["idx"],
                "scene_name": qa["scene_name"],
                "question": qa["question"],
                "question_category": qa["question_category"],
                "response": {"reasoning": reasoning, "answer": answer},
                "model": model,
                "modality": modality,
                "total_tokens": tokens,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            print(f"[{n}/{len(todo)}] idx {qa['idx']}: {answer[:70]}")
        except Exception as e:
            print(f"[{n}/{len(todo)}] idx {qa['idx']} FAILED: {e}")
            with open(failed_log, "a") as f:
                f.write(f"{qa['scene_name']} (idx:{qa['idx']}) {e}\n")

    print(f"\nDone. Raw results: {results_file}")
    print(f"Next: python api_evals/convert_api_results.py --input_file {results_file} "
          f"--output_file <raw.json> --modality {modality}")


if __name__ == "__main__":
    main()
