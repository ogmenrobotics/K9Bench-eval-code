#!/usr/bin/env python3
# Copyright 2025 The K9Bench Authors.
# SPDX-License-Identifier: Apache-2.0

"""K9Bench scoring pipeline.

Two stages:

  1. ``--mode calculate``: embed each model answer + each MCQ option with
     ``Qwen/Qwen3-Embedding-8B`` and store cosine similarities under
     ``cosine_similarities`` in the result JSON.

  2. ``--mode evaluate``: an answer is correct iff the similarity to the
     ground-truth option is (a) above ``--threshold`` AND (b) higher than the
     similarity to every distractor. Reports overall and per-category accuracy.
"""

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
from collections import defaultdict
from datetime import datetime

import torch
from sentence_transformers import SentenceTransformer

SIMILARITY_MODEL = "Qwen/Qwen3-Embedding-8B"


def is_failed_record(r: dict) -> bool:
    """True for records that inference could not produce an answer for (e.g. the
    video was unavailable on YouTube). These are excluded from the accuracy
    denominator rather than counted as wrong."""
    return bool(r.get("error")) or "answer" not in r or "options" not in r


def report_failed(failed: list, total: int, what: str) -> None:
    if not failed:
        return
    print(f"\n!! {len(failed)}/{total} records {what}:")
    for r in failed:
        ident = r.get("idx", r.get("example_id", "?"))
        scene = r.get("scene_name", r.get("video", "?"))
        print(f"     idx={ident} scene={scene} error={r.get('error', 'no answer field')}")
    print()


def cosine_similarities(answer, options, model):
    a = model.encode(answer, prompt_name="query",
                     convert_to_tensor=True, normalize_embeddings=True)
    o = model.encode(options, convert_to_tensor=True, normalize_embeddings=True)
    return model.similarity(a, o)[0].cpu().tolist()


def step_calculate(input_file: str, output_file: str):
    print(f"Loading {input_file}...")
    data = json.load(open(input_file))
    results = data["individual_results"]
    print(f"Found {len(results)} examples")

    print(f"Loading similarity model: {SIMILARITY_MODEL}...")
    model = SentenceTransformer(
        SIMILARITY_MODEL,
        model_kwargs={"attn_implementation": "flash_attention_2",
                      "device_map": "auto", "torch_dtype": torch.bfloat16},
        tokenizer_kwargs={"padding_side": "left"},
    )

    out = []
    failed = []
    for i, r in enumerate(results):
        if is_failed_record(r):
            failed.append(r)
            out.append(r)
        else:
            sims = cosine_similarities(r["answer"], r["options"], model)
            sim_dict = {chr(ord('A') + j): float(s) for j, s in enumerate(sims)}
            out.append({**r, "cosine_similarities": sim_dict})
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  {i + 1}/{len(results)}")

    report_failed(failed, len(results), "skipped during similarity calculation")

    payload = {**data, "individual_results": out,
               "similarity_model": SIMILARITY_MODEL,
               "similarity_calculated_at": datetime.now().isoformat(),
               "inference_error_count": len(failed)}
    print(f"Saving -> {output_file}")
    json.dump(payload, open(output_file, "w"), indent=2)


def is_correct_fine(gt_label: str, sims: list, threshold: float) -> tuple[bool, str]:
    """Fine-mode correctness: gt similarity must exceed threshold AND be the
    strict argmax among all options."""
    gt_idx = ord(gt_label.upper()) - ord('A')
    if not (0 <= gt_idx < len(sims)):
        return False, f"Invalid ground-truth label: {gt_label}"

    gt_sim = sims[gt_idx]
    if gt_sim <= threshold:
        return False, f"gt sim {gt_sim:.4f} <= threshold {threshold}"

    for j, s in enumerate(sims):
        if j != gt_idx and s > gt_sim:
            return False, (f"distractor {chr(ord('A') + j)} sim {s:.4f} > "
                           f"gt sim {gt_sim:.4f}")
    return True, f"gt sim {gt_sim:.4f} > threshold and is argmax"


def step_evaluate(input_file: str, output_file: str, threshold: float):
    print(f"Loading {input_file}...")
    data = json.load(open(input_file))
    results = data["individual_results"]
    print(f"Found {len(results)} examples; threshold={threshold}; mode=fine")

    correct = 0
    scored = 0
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    evaluated = []
    failed = []

    for i, r in enumerate(results):
        if is_failed_record(r) or "cosine_similarities" not in r:
            failed.append(r)
            evaluated.append({**r, "is_correct": None,
                              "evaluation_explanation": "not scored: inference failed",
                              "eval_mode": "fine"})
            continue

        sims_dict = r["cosine_similarities"]
        sims = [sims_dict[chr(ord('A') + j)] for j in range(len(sims_dict))]
        ok, why = is_correct_fine(r["ground_truth"], sims, threshold)

        cat = r["question_category"]
        cat_stats[cat]["total"] += 1
        scored += 1
        if ok:
            correct += 1
            cat_stats[cat]["correct"] += 1

        evaluated.append({**r, "is_correct": ok, "evaluation_explanation": why,
                          "eval_mode": "fine"})

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  {i + 1}/{len(results)}  acc={correct / scored:.4f}")

    report_failed(failed, len(results), "excluded from scoring (inference failed)")

    total = scored
    accuracy = correct / total if total else 0.0
    print(f"\nAccuracy: {accuracy:.4f}  ({correct}/{total})")
    if failed:
        print(f"Excluded from denominator: {len(failed)} failed record(s) "
              f"out of {len(results)} total.")
    print(f"\n{'category':<30} {'accuracy':<10} {'correct/total':<15}")
    print("-" * 60)
    cat_acc = {}
    for cat in sorted(cat_stats):
        s = cat_stats[cat]
        a = s["correct"] / s["total"] if s["total"] else 0.0
        cat_acc[cat] = {"accuracy": a, "correct": s["correct"], "total": s["total"]}
        print(f"{cat:<30} {a:<10.4f} {s['correct']}/{s['total']}")

    payload = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "eval_mode": "fine",
        "threshold": threshold,
        "accuracy": accuracy,
        "correct_count": correct,
        "total_count": total,
        "inference_error_count": len(failed),
        "inference_errors": [
            {"idx": r.get("idx", r.get("example_id")),
             "scene_name": r.get("scene_name"),
             "error": r.get("error", "no answer field")}
            for r in failed
        ],
        "category_wise_accuracy": cat_acc,
        "individual_results": evaluated,
        "metadata": {
            "similarity_model": data.get("similarity_model"),
            "similarity_calculated_at": data.get("similarity_calculated_at"),
        },
    }
    print(f"\nSaving -> {output_file}")
    json.dump(payload, open(output_file, "w"), indent=2)


def main():
    parser = argparse.ArgumentParser(description="K9Bench scoring (fine mode).")
    parser.add_argument("--mode", required=True, choices=["calculate", "evaluate"])
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Cosine similarity threshold (used in --mode evaluate).")
    args = parser.parse_args()

    if args.mode == "calculate":
        step_calculate(args.input_file, args.output_file)
    else:
        step_evaluate(args.input_file, args.output_file, args.threshold)


if __name__ == "__main__":
    main()
