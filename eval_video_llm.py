#!/usr/bin/env python3
# Copyright 2025 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""K9Bench evaluation script.

Loads ``K9Bench/K9Bench`` from the Hugging Face Hub, attempts to download any
missing videos via yt-dlp, runs the video LLM, and writes a JSON of raw model
outputs ready to be scored by ``evaluate_results.py``.

The model interaction (prompt, vision processing, generation kwargs, JSON
extraction) is identical to the original ``pawbench/eval_video_llm.py`` so that
results are reproducible against earlier reference runs.
"""

# Force UTF-8 everywhere
import sys, io, os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import json
import re
from datetime import datetime
from typing import Any

import torch
import tqdm
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForVision2Seq, AutoProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "true"


# ---------------------------------------------------------------------------
# Dataset adaptation: K9Bench/K9Bench -> the schema expected by the original
# eval pipeline (scene_name, ground_truth, options-as-list, question_category).
# ---------------------------------------------------------------------------

YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|/embed/|/v/|/shorts/)([\w-]{11})")

# Mapping from any legacy category labels to the labels used by the released
# K9Bench dataset / the published reference results. Applied defensively: rows
# whose category is already in the released form pass through unchanged.
CATEGORY_MAP = {
    "causal_inference": "cause-effect analysis",
    "contextual_interpretation": "context analysis",
    "interaction_loop_analysis": "action sequence",
    "posture_analysis": "posture analysis",
    "social_interaction_analysis": "interaction analysis",
    "steps_of_actions": "action sequence",
}


def extract_scene_name(video_url: str) -> str:
    m = YT_ID_RE.search(video_url)
    if not m:
        raise ValueError(f"Could not extract YouTube id from URL: {video_url}")
    return m.group(1)


def options_dict_to_list(options: Any) -> list[str]:
    """Convert the dataset's ``{"A": "...", "B": "..."}`` options dict into the
    list-of-strings form ``["A. ...", "B. ...", ...]`` used by the legacy
    pipeline (and by the cosine-similarity scoring step).
    """
    if isinstance(options, list):
        return options
    if not isinstance(options, dict):
        raise TypeError(f"Unexpected options type: {type(options)}")
    return [f"{k}. {options[k]}" for k in sorted(options.keys())]


def normalize_example(ex: dict) -> dict:
    """Adapt a K9Bench row to the legacy schema."""
    cat = ex["question_category"]
    cat = CATEGORY_MAP.get(cat, cat)
    return {
        "idx": ex["idx"],
        "scene_name": extract_scene_name(ex["video_url"]),
        "question": ex["question"],
        "question_category": cat,
        "options": options_dict_to_list(ex["options"]),
        "ground_truth": ex["correct_answer"],
        "video_url": ex["video_url"],
    }


# ---------------------------------------------------------------------------
# Optional on-the-fly video download (kept here so eval is self-contained).
# ---------------------------------------------------------------------------

def ensure_video(scene_name: str, video_url: str, video_dir: str) -> str:
    """Return absolute path to ``{scene_name}.mp4``, downloading via yt-dlp if
    missing. Reuses :mod:`download_videos` to keep the remux step in sync."""
    os.makedirs(video_dir, exist_ok=True)
    out_path = os.path.join(video_dir, f"{scene_name}.mp4")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    sys.path.insert(0, os.path.dirname(__file__))
    from download_videos import download_video, get_ffmpeg_exe  # type: ignore
    ok = download_video(video_url, out_path, get_ffmpeg_exe())
    if not ok or not os.path.exists(out_path):
        raise FileNotFoundError(f"Failed to download/remux {video_url}")
    return out_path


# ---------------------------------------------------------------------------
# Identical to the legacy script below this line (prompt / extraction logic).
# ---------------------------------------------------------------------------

def get_system_message(use_system_message: bool) -> str:
    if use_system_message:
        return (
            "You are an expert in video understanding and reasoning. "
            "Carefully watch and analyze the entire video before answering.\n\n"
            "### Instructions:\n"
            "1. Watch the entire video to identify subtle behaviors, postures, or interactions relevant to the question.\n"
            "2. Reason step-by-step, explaining how each observation leads to your conclusion.\n\n"
            "Respond **strictly** in the following JSON format:\n"
            "{\n"
            '  "reasoning": "<your step-by-step explanation>",\n'
            '  "answer": "<a concise sentence with the final answer>"\n'
            "}"
        )
    return ""


def extract_assistant_response(text):
    if "assistant\n" in text:
        return text.split("assistant\n", 1)[1].strip()
    return text


def extract_answer_from_json(output_text):
    try:
        json_obj = json.loads(output_text)
        return json_obj.get("answer", output_text)
    except json.JSONDecodeError:
        json_match = re.search(r"\{[\s\S]*\}", output_text)
        if json_match:
            try:
                json_obj = json.loads(json_match.group(0))
                return json_obj.get("answer", output_text)
            except json.JSONDecodeError:
                pass
        answer_match = re.search(r'"answer"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', output_text)
        if answer_match:
            answer = answer_match.group(1)
            answer = answer.replace(r'\"', '"').replace(r'\\', '\\').replace(r'\n', '\n')
            return answer
    return output_text


def build_messages(example: dict, use_system_message: bool, video_path: str, max_frames: int):
    system_message = get_system_message(use_system_message)
    return [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "max_pixels": 360 * 420,
                 "fps": 1.0, "max_frames": max_frames},
                {"type": "text", "text": example["question"]},
            ],
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a video LLM on K9Bench.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="K9Bench/K9Bench")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--video_dir", type=str, default="./k9bench_videos",
                        help="Local directory holding {scene_name}.mp4 files.")
    parser.add_argument("--auto_download", action="store_true",
                        help="If set, missing videos are fetched via yt-dlp at eval time.")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--torch_dtype", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_system_message", type=lambda s: s.lower() == "true",
                        default=True)
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="If > 0, only run on the first N rows (after subset).")
    parser.add_argument("--thinking_mode", action="store_true")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    if args.torch_dtype:
        torch_dtype = getattr(torch, args.torch_dtype)
    print(f"Using device: {device}, dtype: {torch_dtype}")

    print(f"Loading base model from {args.model_name_or_path}...")
    model_kwargs = dict(
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model = AutoModelForVision2Seq.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_name_or_path,
                                              trust_remote_code=args.trust_remote_code)

    print(f"Loading dataset: {args.dataset_name} (split={args.split})...")
    dataset = load_dataset(args.dataset_name, split=args.split)
    total_examples = len(dataset)
    print(f"Found total samples: {total_examples}")

    if args.start_idx is not None and args.end_idx is not None:
        s = max(0, min(args.start_idx, total_examples))
        e = max(s, min(args.end_idx, total_examples))
        dataset = dataset.select(range(s, e))
        print(f"Selected {len(dataset)} samples (indices {s}..{e - 1})")

    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        print(f"Capped to first {len(dataset)} samples (--max_samples)")

    normalized = [normalize_example(ex) for ex in dataset]

    print("Starting inference...")
    results = []
    for i, ex in enumerate(tqdm.tqdm(normalized)):
        scene_name = ex["scene_name"]
        print(f"####### {ex['idx']} {scene_name} #######")
        try:
            video_path = os.path.join(args.video_dir, f"{scene_name}.mp4")
            if not os.path.exists(video_path):
                if args.auto_download:
                    video_path = ensure_video(scene_name, ex["video_url"], args.video_dir)
                else:
                    raise FileNotFoundError(
                        f"Missing video {video_path}. Run download_videos.py first "
                        "or pass --auto_download.")

            messages = build_messages(ex, args.use_system_message, video_path, args.max_frames)
            print(f"Video has {process_vision_info(messages)[1][0].shape[0]} frames")

            with torch.no_grad():
                inputs = processor(
                    text=processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                    videos=process_vision_info(messages)[1][0],
                    return_tensors="pt",
                ).to(device)

                if args.thinking_mode:
                    print("Using thinking mode")
                    max_new_tokens = 3072
                else:
                    max_new_tokens = 2048
                print(f"Using max new tokens: {max_new_tokens}")

                output_obj = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    output_logits=True,
                    return_dict_in_generate=True,
                    use_cache=True,
                )
                output_ids = output_obj.sequences
                output_text = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                output_text = extract_assistant_response(output_text)
                answer_text = extract_answer_from_json(output_text)

                results.append({
                    "idx": ex["idx"],
                    "question_category": ex["question_category"],
                    "video": os.path.basename(video_path),
                    "ground_truth": ex["ground_truth"],
                    "response": output_text,
                    "answer": answer_text,
                    "scene_name": scene_name,
                    "question": ex["question"],
                    "options": ex["options"],
                    "prompt": processor.apply_chat_template(messages, tokenize=False),
                    "model_name": args.model_name_or_path.split("/")[-1],
                    "max_frames": args.max_frames,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            import traceback; traceback.print_exc()
            results.append({"example_id": i, "error": str(e)})

    final_results = {"individual_results": results, "total_count": len(results)}
    out_dir = os.path.dirname(args.output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, "w") as f:
        json.dump(final_results, f, indent=2)
    print("Inference complete! Run evaluate_results.py to score.")


if __name__ == "__main__":
    main()
