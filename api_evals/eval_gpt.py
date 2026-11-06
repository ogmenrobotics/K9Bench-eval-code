#!/usr/bin/env python3
# Copyright 2025 The K9Bench Authors.
# SPDX-License-Identifier: Apache-2.0

"""K9Bench evaluation with GPT-4o via uniformly-sampled frames.

Unlike the Gemini/DashScope runners (which pass the full video to the model),
GPT-4o here only accepts images, so each scene's video is decoded once with
OpenCV and ``max_frames`` frames are sampled uniformly across its length, then
base64-encoded and reused across all of that scene's questions (same reuse
pattern as the Gemini runner's single upload-per-scene).

Output schema matches the other API runners exactly: prompt, parsing, and the
``{"reasoning", "answer"}`` JSON come from ``api_common.py``.

Results are written as JSONL (one record per line, appended as they complete)
so an interrupted run resumes where it stopped. Convert with
``convert_api_results.py`` and score with ``evaluate_results.py``.
"""

import argparse
import base64
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
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
)


def sample_frames_b64(path: str, max_frames: int = 32):
    """Uniformly sample up to ``max_frames`` frames from a video, base64-jpeg encoded.

    Reads sequentially (no random seeks) for codec-friendly decoding, picking
    off frames at the target indices as they're hit.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"could not read frame count: {path}")

    wanted = sorted(set(np.linspace(0, total - 1, min(max_frames, total)).astype(int).tolist()))

    frames_b64 = []
    wi = 0
    current = 0
    while wi < len(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if current == wanted[wi]:
            ok2, buf = cv2.imencode(".jpg", frame)
            if ok2:
                frames_b64.append(base64.b64encode(buf).decode("utf-8"))
            wi += 1
        current += 1
    cap.release()

    if not frames_b64:
        raise RuntimeError(f"no frames extracted: {path}")
    return frames_b64


class GPTAgent:
    def __init__(self, api_key: str, model_name: str, image_detail: str = "low"):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.image_detail = image_detail

    def ask(self, frames_b64, question: str, retries: int, bucket):
        """Returns (parsed_dict_or_None, tokens, quota_hit)."""
        prompt = build_prompt(question)
        content = [{"type": "text", "text": prompt}]
        for img in frames_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": self.image_detail},
            })

        for attempt in range(retries):
            try:
                bucket.wait(token_estimate=100_000)
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.7,
                    max_tokens=700,
                )
                txt = resp.choices[0].message.content
                reasoning, answer = extract_reasoning_answer(txt)
                if not answer:
                    raise ValueError("no answer field in model output")

                usage = getattr(resp, "usage", None)
                tokens = getattr(usage, "total_tokens", 0) if usage else 0
                bucket.record_tokens(tokens)
                return {"reasoning": reasoning, "answer": answer}, tokens, False
            except Exception as e:
                err = str(e)
                print(f"  generate attempt {attempt+1}/{retries} failed: {e}")
                if "429" in err or "quota" in err.lower() or "rate_limit" in err.lower():
                    return None, 0, True
                if attempt < retries - 1:
                    time.sleep(12)
        return None, 0, False


def main():
    ap = argparse.ArgumentParser(description="GPT-4o frame-based evaluator for K9Bench.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--start_idx", type=int, default=None)
    ap.add_argument("--end_idx", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    api_keys = [k for k in cfg.get("api_keys", []) if k and not k.startswith("OPENAI_API_KEY")]
    if not api_keys:
        env_key = os.environ.get("OPENAI_API_KEY")
        if env_key:
            api_keys = [env_key]
    if not api_keys:
        raise SystemExit("No usable API keys in config (placeholders are ignored) or OPENAI_API_KEY env var.")

    model_name = cfg.get("model_name", "gpt-4o")
    videos_dir = cfg.get("videos_dir", "k9bench_videos")
    results_file = cfg.get("results_file", "eval_results/gpt4o/raw.jsonl")
    failed_log = cfg.get("failed_log", "eval_results/gpt4o/failed.log")
    modality = cfg.get("modality", "video_frames")
    max_frames = int(cfg.get("max_frames", 32))
    image_detail = cfg.get("image_detail", "low")
    max_retries = int(cfg.get("max_retries", 3))
    bucket = TokenBucket(int(cfg.get("rate_limit_rpm", 8)),
                         int(cfg.get("rate_limit_tpm", 1_000_000)))

    data = load_k9bench_qas(cfg.get("hf_dataset", "ogmen/K9Bench"),
                            cfg.get("hf_split", "test"))
    if args.start_idx is not None:
        data = [q for q in data if q["idx"] >= args.start_idx]
    if args.end_idx is not None:
        data = [q for q in data if q["idx"] <= args.end_idx]

    done = {r["idx"] for r in load_jsonl(results_file) if "idx" in r}
    todo = [q for q in data if q["idx"] not in done]
    print(f"model={model_name} max_frames={max_frames}  already done={len(done)}  to process={len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    by_scene = defaultdict(list)
    for q in todo:
        by_scene[q["scene_name"]].append(q)
    print(f"Grouped into {len(by_scene)} unique videos")

    os.makedirs(os.path.dirname(failed_log) or ".", exist_ok=True)
    key_idx = 0

    for n, (scene, qa_list) in enumerate(by_scene.items(), 1):
        video_path = os.path.join(videos_dir, f"{scene}.mp4")
        if not os.path.exists(video_path):
            with open(failed_log, "a") as f:
                f.write(f"{scene} (missing video)\n")
            continue

        print(f"\n[{n}/{len(by_scene)}] {scene}: {len(qa_list)} questions")
        try:
            frames_b64 = sample_frames_b64(video_path, max_frames=max_frames)
        except Exception as e:
            print(f"  frame extraction failed: {e}")
            with open(failed_log, "a") as f:
                f.write(f"{scene} (frame extraction failed: {e})\n")
            continue

        agent = GPTAgent(api_keys[key_idx % len(api_keys)], model_name, image_detail)

        for qa in qa_list:
            resp, tokens, quota = agent.ask(frames_b64, qa["question"], max_retries, bucket)
            if resp is None and quota:
                # Rotate key and retry once under the new one.
                key_idx = (key_idx + 1) % len(api_keys)
                agent = GPTAgent(api_keys[key_idx], model_name, image_detail)
                resp, tokens, _ = agent.ask(frames_b64, qa["question"], max_retries, bucket)
            if resp is None:
                with open(failed_log, "a") as f:
                    f.write(f"{scene} (idx:{qa['idx']})\n")
                continue

            append_jsonl(results_file, {
                "idx": qa["idx"],
                "scene_name": scene,
                "question": qa["question"],
                "question_category": qa["question_category"],
                "response": resp,
                "model": model_name,
                "modality": modality,
                "max_frames": max_frames,
                "total_tokens": tokens,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            print(f"    idx {qa['idx']}: {resp['answer'][:70]}")

    print(f"\nDone. Raw results: {results_file}")
    print(f"Next: python api_evals/convert_api_results.py --input_file {results_file} "
          f"--output_file <raw.json> --modality {modality}")


if __name__ == "__main__":
    main()