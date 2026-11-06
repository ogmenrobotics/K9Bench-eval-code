#!/usr/bin/env python3
# Copyright 2025 The K9Bench Authors.
# SPDX-License-Identifier: Apache-2.0

"""K9Bench evaluation for InternVL models (open-ended prompting).

InternVL does not go through the Qwen ``qwen_vl_utils`` path: frames are sampled
with decord, split into 448px ImageNet-normalized tiles, and passed to
``model.chat()`` together with a single flat prompt string carrying one
``Frame{i}: <image>`` placeholder per frame.

Dataset adaptation, video download and answer extraction are imported from
``eval_video_llm.py`` so both model families see the identical question text and
identical scoring path. Score the output with ``evaluate_results.py``.
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
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms as T
import tqdm
from datasets import load_dataset
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_video_llm import (  # noqa: E402
    ensure_video,
    extract_answer_from_json,
    get_system_message,
    normalize_example,
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Applied as model.system_message when --thinking_mode is set.
R1_SYSTEM_PROMPT = """
    You are an AI assistant that rigorously follows this response protocol:

    1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

    2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

    Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
    """.strip()


# ---------------------------------------------------------------------------
# InternVL frame preparation (dynamic 448px tiling, decord sampling).
# ---------------------------------------------------------------------------

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def process_video(video_path, max_frames):
    pixel_values, num_patches_list = load_video(
        video_path, bound=None, input_size=448, max_num=1, num_segments=max_frames)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    return pixel_values, num_patches_list


def build_prompt(example: dict, use_system_message: bool, num_frames: int) -> str:
    """InternVL-style single-string prompt: frame placeholders, then the system
    message, then the question."""
    video_prefix = "".join(f"Frame{i+1}: <image>\n" for i in range(num_frames))
    return video_prefix + get_system_message(use_system_message) + "\n" + example["question"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate an InternVL video LLM on K9Bench.")
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
    parser.add_argument("--use_system_message", type=lambda s: s.lower() == "true",
                        default=True)
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="If > 0, only run on the first N rows (after subset).")
    parser.add_argument("--thinking_mode", action="store_true")
    parser.add_argument("--output_file", type=str, default="evaluation_results_internvl.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    if args.torch_dtype:
        torch_dtype = getattr(torch, args.torch_dtype)
    print(f"Using device: {device}, dtype: {torch_dtype}")

    print(f"Loading base model from {args.model_name_or_path}...")
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    if args.thinking_mode:
        model.system_message = R1_SYSTEM_PROMPT
    model.eval()
    print("Model loaded successfully")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1024, do_sample=True,
                             temperature=0.6 if args.thinking_mode else 0.7)

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

            pixel_values, num_patches_list = process_video(video_path, args.max_frames)
            print(f"Video has {len(num_patches_list)} frames")

            full_prompt = build_prompt(ex, args.use_system_message, len(num_patches_list))

            with torch.no_grad():
                output_text, _history = model.chat(
                    tokenizer, pixel_values, full_prompt, generation_config,
                    history=None, return_history=True)

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
                "prompt": full_prompt,
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
