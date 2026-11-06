#!/usr/bin/env python3
# Copyright 2025 The K9Bench Authors.
# SPDX-License-Identifier: Apache-2.0

"""MCQ (options-in-prompt) K9Bench evaluation for InternVL models.

Combines the InternVL inference machinery from ``eval_video_llm_internvl.py``
(decord frame sampling, 448px ImageNet-normalized tiles, ``model.chat()``) with
the MCQ prompt and answer extraction from ``eval_video_llm_mcq.py``.

Records carry the same audit fields as ``eval_video_llm_mcq.py``: exact
``prompt`` string, full ``response``, ``extracted_option``, ``parse_failed``,
``prompt_style``. Score with ``evaluate_results_mcq.py``.
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
from datetime import datetime

import torch
import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_video_llm import ensure_video, normalize_example          # noqa: E402
from eval_video_llm_mcq import (                                    # noqa: E402
    PROMPT_STYLE,
    extract_mcq_answer,
    get_mcq_system_message,
    load_resume_state,
    save_results_atomic,
)
from eval_video_llm_internvl import process_video                   # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def build_mcq_prompt(example: dict, num_frames: int) -> str:
    """InternVL-style single-string prompt: frame placeholders, then the MCQ
    system message, then question + options."""
    video_prefix = "".join(f"Frame{i+1}: <image>\n" for i in range(num_frames))
    user_text = example["question"] + "\n\nOptions:\n" + "\n".join(example["options"])
    return video_prefix + get_mcq_system_message() + "\n" + user_text


def main():
    parser = argparse.ArgumentParser(description="Evaluate an InternVL video LLM on K9Bench (MCQ prompting).")
    parser.add_argument("--model_name_or_path", type=str, default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--dataset_name", type=str, default="ogmen/K9Bench")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--video_dir", type=str, default="./k9bench_videos",
                        help="Local directory holding {scene_name}.mp4 files.")
    parser.add_argument("--auto_download", action="store_true",
                        help="If set, missing videos are fetched via yt-dlp at eval time.")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--torch_dtype", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="If > 0, only run on the first N rows (after subset).")
    parser.add_argument("--thinking_mode", action="store_true",
                        help="Accepted for launcher compatibility; not used for InternVL MCQ runs.")
    parser.add_argument("--output_file", type=str, default="evaluation_results_internvl_mcq.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    if args.torch_dtype:
        torch_dtype = getattr(torch, args.torch_dtype)
    print(f"Using device: {device}, dtype: {torch_dtype}")
    print(f"Prompt style: {PROMPT_STYLE}")

    print(f"Loading base model from {args.model_name_or_path}...")
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    print("Model loaded successfully")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              trust_remote_code=True, use_fast=False)
    # Same generation config as the InternVL open-ended runs.
    generation_config = dict(max_new_tokens=1024, do_sample=True, temperature=0.7)

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

    results, done_idxs = load_resume_state(args.output_file)
    if done_idxs:
        print(f"Resuming: {len(done_idxs)} samples already saved in {args.output_file}")

    print("Starting inference...")
    for i, ex in enumerate(tqdm.tqdm(normalized)):
        if ex["idx"] in done_idxs:
            continue
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

            pixel_values, num_patches_list = process_video(video_path, args.max_frames)
            print(f"Video has {len(num_patches_list)} frames")

            full_prompt = build_mcq_prompt(ex, len(num_patches_list))

            with torch.no_grad():
                output_text, _history = model.chat(
                    tokenizer, pixel_values, full_prompt, generation_config,
                    history=None, return_history=True)

            extracted_option = extract_mcq_answer(output_text)
            parse_failed = extracted_option is None
            if parse_failed:
                print(f"[PARSE FAIL] idx={ex['idx']}: no 'ANSWER: X' line found")

            results.append({
                "idx": ex["idx"],
                "question_category": ex["question_category"],
                "video": os.path.basename(video_path),
                "ground_truth": ex["ground_truth"],
                "response": output_text,
                "answer": extracted_option if not parse_failed else output_text,
                "extracted_option": extracted_option,
                "parse_failed": parse_failed,
                "prompt_style": PROMPT_STYLE,
                "scene_name": scene_name,
                "question": ex["question"],
                "options": ex["options"],
                "prompt": full_prompt,
                "model_name": args.model_name_or_path.split("/")[-1],
                "max_frames": args.max_frames,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            import traceback; traceback.print_exc()
            results.append({"example_id": i, "error": str(e)})

        # Incremental checkpoint after every sample so timeouts/preemptions
        # lose at most the in-flight sample; resumed runs pick up from here.
        save_results_atomic(results, args.output_file)

    save_results_atomic(results, args.output_file)
    print(f"Saved {len(results)} results to {args.output_file}")
    print("Inference complete! Run evaluate_results_mcq.py to score.")


if __name__ == "__main__":
    main()
