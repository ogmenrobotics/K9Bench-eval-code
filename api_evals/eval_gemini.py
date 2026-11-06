#!/usr/bin/env python3
# Copyright 2025 The K9Bench Authors.
# SPDX-License-Identifier: Apache-2.0

"""K9Bench evaluation with Gemini via the Files API.

Videos are grouped by scene so each file is uploaded once, reused across all of
its questions, then deleted. API keys rotate on quota errors, re-uploading the
video under the new key when needed.

Set ``modality: audio_video`` in the config to evaluate with the audio track
(Table 2); ``video`` strips audio into a cache dir first (Table 1). The chosen
modality is recorded in every result row.

Results are written as JSONL (one record per line, appended as they complete) so
an interrupted run resumes where it stopped. Convert with
``convert_api_results.py`` and score with ``evaluate_results.py``.
"""

import argparse
import os
import subprocess
import sys
import time
from collections import defaultdict

import yaml
from google import genai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_common import (  # noqa: E402
    TokenBucket,
    append_jsonl,
    build_prompt,
    extract_reasoning_answer,
    load_jsonl,
    load_k9bench_qas,
)


def strip_audio(src: str, dst: str) -> str:
    """Re-mux without the audio stream so a video-only run cannot see audio."""
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                    "-c", "copy", "-an", dst], check=True)
    return dst


class GeminiAgent:
    def __init__(self, api_key: str, model_name: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def upload_video(self, path: str, retries: int, timeout: int):
        """Returns (file_obj, quota_hit)."""
        if not os.path.exists(path):
            print(f"  video missing: {path}")
            return None, False
        size_mb = os.path.getsize(path) / (1024 * 1024)
        for attempt in range(retries):
            try:
                print(f"  uploading {os.path.basename(path)} ({size_mb:.1f} MB), attempt {attempt+1}")
                file = self.client.files.upload(file=path)
                start = time.time()
                while file.state.name == "PROCESSING":
                    if time.time() - start > timeout:
                        raise TimeoutError("upload processing timeout")
                    time.sleep(2)
                    file = self.client.files.get(name=file.name)
                if file.state.name == "ACTIVE":
                    return file, False
                raise RuntimeError(f"upload state: {file.state.name}")
            except Exception as e:
                if "429" in str(e) or "Quota" in str(e):
                    return None, True
                print(f"  upload attempt {attempt+1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(12)
        return None, False

    def ask(self, file_obj, question: str, include_audio: bool, retries: int, bucket):
        """Returns (parsed_dict_or_None, tokens, quota_hit)."""
        prompt = build_prompt(question, include_audio=include_audio)
        for attempt in range(retries):
            try:
                bucket.wait(token_estimate=100_000)
                resp = self.client.models.generate_content(
                    model=self.model_name, contents=[file_obj, prompt])
                txt = resp.text if hasattr(resp, "text") else str(resp)
                reasoning, answer = extract_reasoning_answer(txt)
                if not answer:
                    raise ValueError("no answer field in model output")
                tokens = getattr(getattr(resp, "usage_metadata", None), "total_token_count", 0)
                bucket.record_tokens(tokens)
                return {"reasoning": reasoning, "answer": answer}, tokens, False
            except Exception as e:
                err = str(e)
                print(f"  generate attempt {attempt+1}/{retries} failed: {e}")
                if "429" in err or "Quota" in err:
                    return None, 0, True
                if attempt < retries - 1:
                    time.sleep(12)
        return None, 0, False


def main():
    ap = argparse.ArgumentParser(description="Gemini video QA evaluator for K9Bench.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--start_idx", type=int, default=None)
    ap.add_argument("--end_idx", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    api_keys = [k for k in cfg.get("api_keys", []) if k and not k.startswith("GEMINI_API_KEY")]
    if not api_keys:
        raise SystemExit("No usable API keys in config (placeholders are ignored).")

    model_name = cfg.get("model_name", "gemini-2.5-pro")
    videos_dir = cfg.get("videos_dir", "k9bench_videos")
    results_file = cfg.get("results_file", "eval_results/gemini/raw.jsonl")
    failed_log = cfg.get("failed_log", "eval_results/gemini/failed.log")
    modality = cfg.get("modality", "video")
    if modality not in ("video", "audio_video"):
        raise SystemExit("modality must be 'video' or 'audio_video'")
    include_audio = modality == "audio_video"
    silent_dir = cfg.get("silent_videos_dir", "k9bench_videos_silent")
    max_retries = int(cfg.get("max_retries", 3))
    upload_timeout = int(cfg.get("upload_timeout", 600))
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
    print(f"modality={modality}  already done={len(done)}  to process={len(todo)}")
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
        src = os.path.join(videos_dir, f"{scene}.mp4")
        if not os.path.exists(src):
            with open(failed_log, "a") as f:
                f.write(f"{scene} (missing video)\n")
            continue
        path = src if include_audio else strip_audio(src, os.path.join(silent_dir, f"{scene}.mp4"))

        print(f"\n[{n}/{len(by_scene)}] {scene}: {len(qa_list)} questions")
        agent = GeminiAgent(api_keys[key_idx % len(api_keys)], model_name)

        file_obj, switches = None, 0
        while file_obj is None and switches < len(api_keys):
            file_obj, quota = agent.upload_video(path, max_retries, upload_timeout)
            if file_obj:
                break
            if not quota:
                break
            key_idx = (key_idx + 1) % len(api_keys)
            agent = GeminiAgent(api_keys[key_idx], model_name)
            switches += 1
        if file_obj is None:
            with open(failed_log, "a") as f:
                f.write(f"{scene} (upload failed)\n")
            continue

        for qa in qa_list:
            resp, tokens, quota = agent.ask(file_obj, qa["question"], include_audio,
                                            max_retries, bucket)
            if resp is None and quota:
                # Rotate key and re-upload under the new one, then retry once.
                try:
                    agent.client.files.delete(name=file_obj.name)
                except Exception:
                    pass
                key_idx = (key_idx + 1) % len(api_keys)
                agent = GeminiAgent(api_keys[key_idx], model_name)
                file_obj, _ = agent.upload_video(path, max_retries, upload_timeout)
                if file_obj is None:
                    with open(failed_log, "a") as f:
                        f.write(f"{scene} (re-upload failed)\n")
                    break
                resp, tokens, _ = agent.ask(file_obj, qa["question"], include_audio,
                                            max_retries, bucket)
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
                "total_tokens": tokens,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            print(f"    idx {qa['idx']}: {resp['answer'][:70]}")

        try:
            agent.client.files.delete(name=file_obj.name)
        except Exception:
            pass

    print(f"\nDone. Raw results: {results_file}")
    print(f"Next: python api_evals/convert_api_results.py --input_file {results_file} "
          f"--output_file <raw.json> --modality {modality}")


if __name__ == "__main__":
    main()
