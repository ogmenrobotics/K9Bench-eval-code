#!/usr/bin/env python3
# Copyright 2025 The K9Bench Authors.
# SPDX-License-Identifier: Apache-2.0

"""K9Bench audio+video evaluation with Qwen3-Omni via the DashScope
OpenAI-compatible endpoint.

Qwen3-Omni only accepts videos shorter than ``max_duration_sec`` (150s), which
is why the paper reports this model on the 2693-QA subset rather than all 4744.
That filter is applied here: durations are measured with ffprobe, cached to
``duration_cache``, and longer videos are skipped and listed explicitly.

Video is inlined as a base64 data URI, so files above ``max_bytes`` (256 MB) are
skipped rather than silently failing. Convert with ``convert_api_results.py``
and score with ``evaluate_results.py``.
"""

import argparse
import base64
import json
import os
import sys
import time

import yaml
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_common import (  # noqa: E402
    TokenBucket,
    append_jsonl,
    build_prompt,
    extract_reasoning_answer,
    load_jsonl,
    load_k9bench_qas,
    video_duration_seconds,
)


def build_duration_index(scenes, videos_dir, cache_path):
    """scene -> duration in seconds, measured once and cached to disk."""
    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except json.JSONDecodeError:
            cache = {}

    missing = [s for s in scenes if s not in cache]
    if missing:
        print(f"Measuring duration for {len(missing)} videos with ffprobe...")
        for i, scene in enumerate(missing, 1):
            path = os.path.join(videos_dir, f"{scene}.mp4")
            cache[scene] = video_duration_seconds(path) if os.path.exists(path) else None
            if i % 100 == 0:
                print(f"  {i}/{len(missing)}")
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(cache, f, indent=2)
            print(f"Cached durations -> {cache_path}")
    return cache


def main():
    ap = argparse.ArgumentParser(description="Qwen3-Omni audio+video evaluator for K9Bench.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--start_idx", type=int, default=None)
    ap.add_argument("--end_idx", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    api_key = cfg.get("api_key") or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key or api_key.startswith("YOUR_"):
        raise SystemExit("Set api_key in the config or DASHSCOPE_API_KEY in the environment.")

    model = cfg.get("model", "qwen3-omni-flash")
    base_url = cfg.get("base_url", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    videos_dir = cfg.get("videos_dir", "k9bench_videos")
    results_file = cfg.get("results_file", "eval_results/qwen3-omni/raw.jsonl")
    failed_log = cfg.get("failed_log", "eval_results/qwen3-omni/failed.log")
    duration_cache = cfg.get("duration_cache", "eval_results/video_durations.json")
    max_duration = float(cfg.get("max_duration_sec", 150))
    max_bytes = int(cfg.get("max_bytes", 256 * 1024 * 1024))
    modality = cfg.get("modality", "audio_video")
    bucket = TokenBucket(int(cfg.get("rate_limit_rpm", 60)),
                         int(cfg.get("rate_limit_tpm", 100_000)))

    client = OpenAI(api_key=api_key, base_url=base_url)

    data = load_k9bench_qas(cfg.get("hf_dataset", "ogmen/K9Bench"),
                            cfg.get("hf_split", "test"))
    if args.start_idx is not None:
        data = [q for q in data if q["idx"] >= args.start_idx]
    if args.end_idx is not None:
        data = [q for q in data if q["idx"] <= args.end_idx]

    durations = build_duration_index(sorted({q["scene_name"] for q in data}),
                                     videos_dir, duration_cache)

    eligible, too_long, unknown = [], [], []
    for q in data:
        d = durations.get(q["scene_name"])
        if d is None:
            unknown.append(q)
        elif d < max_duration:
            eligible.append(q)
        else:
            too_long.append(q)

    print(f"\nDuration filter (< {max_duration:.0f}s): {len(eligible)} eligible QAs, "
          f"{len(too_long)} too long, {len(unknown)} with unknown duration (skipped)")
    print(f"The paper reports Qwen3-Omni on this subset rather than all {len(data)} QAs.")

    done = {r["idx"] for r in load_jsonl(results_file) if "idx" in r}
    todo = [q for q in eligible if q["idx"] not in done]
    print(f"already done={len(done)}  to process={len(todo)}")

    os.makedirs(os.path.dirname(failed_log) or ".", exist_ok=True)

    for n, qa in enumerate(todo, 1):
        video_path = os.path.join(videos_dir, f"{qa['scene_name']}.mp4")
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"missing video {video_path}")
            size = os.path.getsize(video_path)
            if size > max_bytes:
                raise ValueError(f"video is {size / 1e6:.0f} MB, above the "
                                 f"{max_bytes / 1e6:.0f} MB inline limit")

            with open(video_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")

            messages = [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": f"data:;base64,{b64}"}},
                {"type": "text", "text": build_prompt(qa["question"], include_audio=True)},
            ]}]

            bucket.wait(token_estimate=15_000 + 800)
            completion = client.chat.completions.create(
                model=model, messages=messages, modalities=["text"],
                stream=True, stream_options={"include_usage": True})

            full_text, usage = "", None
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_text += chunk.choices[0].delta.content
                if chunk.usage:
                    usage = chunk.usage

            reasoning, answer = extract_reasoning_answer(full_text.strip())
            tokens = ((usage.prompt_tokens if usage else 0) +
                      (usage.completion_tokens if usage else 0))
            bucket.record_tokens(tokens)

            append_jsonl(results_file, {
                "idx": qa["idx"],
                "scene_name": qa["scene_name"],
                "question": qa["question"],
                "question_category": qa["question_category"],
                "response": {"reasoning": reasoning, "answer": answer},
                "model": model,
                "modality": modality,
                "duration_sec": durations.get(qa["scene_name"]),
                "total_tokens": tokens,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            print(f"[{n}/{len(todo)}] idx {qa['idx']}: {answer[:70]}")
        except Exception as e:
            print(f"[{n}/{len(todo)}] idx {qa['idx']} FAILED: {e}")
            with open(failed_log, "a") as f:
                f.write(f"{qa['scene_name']} (idx:{qa['idx']}) {e}\n")

    print(f"\nDone. Raw results: {results_file}")
    print(f"Excluded by the {max_duration:.0f}s limit: {len(too_long)} QAs")
    print(f"Next: python api_evals/convert_api_results.py --input_file {results_file} "
          f"--output_file <raw.json> --modality {modality}")


if __name__ == "__main__":
    main()
