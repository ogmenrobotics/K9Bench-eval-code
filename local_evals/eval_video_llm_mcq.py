#!/usr/bin/env python3
# Copyright 2025 The K9Bench Authors.
# SPDX-License-Identifier: Apache-2.0

"""MCQ (options-in-prompt) variant of the K9Bench evaluation.

Unlike ``eval_video_llm.py`` (which is option-blind and scored post-hoc via
cosine similarity), this script supplies all answer options inside the prompt
and instructs the model to end its response with a final line ``ANSWER: X``.
The selected letter is regex-extracted and stored per record together with the
exact prompt string, the full raw response, and an explicit parse-failure flag,
so every result is fully auditable post-hoc.

Score the output with ``evaluate_results_mcq.py`` (exact match, no GPU).

This file deliberately does NOT modify ``eval_video_llm.py`` — it imports the
shared pieces from it so the open-ended pipeline stays byte-identical.
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

import torch
import tqdm
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForVision2Seq, AutoProcessor

# Reuse the frozen release pipeline (dataset adaptation, video handling,
# response splitting) without modifying it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_video_llm import (  # noqa: E402
    ensure_video,
    extract_assistant_response,
    normalize_example,
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

PROMPT_STYLE = "mcq_answer_letter"

ANSWER_RE = re.compile(r"ANSWER\s*[:\-]?\s*\(?([A-E])\)?", re.IGNORECASE)


def get_mcq_system_message() -> str:
    return (
        "You are an expert in video understanding and reasoning. "
        "Carefully watch and analyze the entire video before answering.\n\n"
        "### Instructions:\n"
        "1. Watch the entire video to identify subtle behaviors, postures, or interactions relevant to the question.\n"
        "2. Reason step-by-step, explaining how each observation leads to your conclusion.\n"
        "3. Then select exactly one of the given options.\n\n"
        "End your response with a single final line in exactly this format:\n"
        "ANSWER: <option letter>"
    )


def extract_mcq_answer(output_text: str):
    """Extract the selected option letter from the model output.

    Takes the LAST ``ANSWER: X`` style match so that thinking preambles or a
    restatement of the instruction earlier in the response do not win over the
    model's final selection. Returns an uppercase letter or ``None``.
    """
    matches = ANSWER_RE.findall(output_text)
    if matches:
        return matches[-1].upper()
    return None


def save_results_atomic(results, output_file):
    """Write results to output_file atomically (tmp + rename)."""
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp = output_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"individual_results": results, "total_count": len(results)}, f, indent=2)
    os.replace(tmp, output_file)


def load_resume_state(output_file):
    """Load previously saved results so an interrupted shard can resume.

    Only successful records (those carrying an ``idx``) count as done —
    error records are dropped and their samples re-attempted.
    """
    if not os.path.exists(output_file):
        return [], set()
    try:
        with open(output_file) as f:
            prev = json.load(f)
        results = [r for r in prev.get("individual_results", []) if "idx" in r]
        return results, {r["idx"] for r in results}
    except Exception as e:
        print(f"Could not resume from {output_file} ({e}); starting fresh")
        return [], set()


def build_mcq_messages(example: dict, video_path: str, max_frames: int):
    """Same video block as the open-ended eval; user text carries the options."""
    user_text = example["question"] + "\n\nOptions:\n" + "\n".join(example["options"])
    return [
        {"role": "system", "content": [{"type": "text", "text": get_mcq_system_message()}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "max_pixels": 360 * 420,
                 "fps": 1.0, "max_frames": max_frames},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a video LLM on K9Bench (MCQ prompting).")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
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
    parser.add_argument("--thinking_mode", action="store_true")
    parser.add_argument("--output_file", type=str, default="evaluation_results_mcq.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    if args.torch_dtype:
        torch_dtype = getattr(torch, args.torch_dtype)
    print(f"Using device: {device}, dtype: {torch_dtype}")
    print(f"Prompt style: {PROMPT_STYLE}")

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

            messages = build_mcq_messages(ex, video_path, args.max_frames)
            video_inputs = process_vision_info(messages)[1][0]
            print(f"Video has {video_inputs.shape[0]} frames")

            with torch.no_grad():
                inputs = processor(
                    text=processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                    videos=video_inputs,
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
                    return_dict_in_generate=True,
                    use_cache=True,
                )
                output_ids = output_obj.sequences
                output_text = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                output_text = extract_assistant_response(output_text)

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
                    "prompt": processor.apply_chat_template(messages, tokenize=False),
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
