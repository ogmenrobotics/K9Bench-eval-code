#!/usr/bin/env python3
# Copyright 2025 The K9Bench Authors.
# SPDX-License-Identifier: Apache-2.0

"""Exact-match scorer for MCQ (options-in-prompt) K9Bench runs.

Reads a raw JSON produced by ``eval_video_llm_mcq.py`` (single shard or the
merged full file from ``combine_shards.py``) and scores each record by
comparing the regex-extracted option letter against the ground-truth letter.

No GPU / torch / sentence-transformers needed — runs anywhere.

Output JSON mirrors the cosine scorer's schema (``accuracy``,
``correct_count``, ``total_count``, ``category_wise_accuracy``,
``individual_results`` with ``is_correct``/``evaluation_explanation``) plus
``eval_mode: "mcq_exact_match"`` and parse-failure counts, so downstream
comparison/viz tooling works unchanged.
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
import re
from collections import defaultdict
from datetime import datetime

# Keep in sync with eval_video_llm_mcq.ANSWER_RE (duplicated here so this
# scorer stays importable on nodes without torch/transformers installed).
ANSWER_RE = re.compile(r"ANSWER\s*[:\-]?\s*\(?([A-E])\)?", re.IGNORECASE)


def extract_mcq_answer(output_text: str):
    matches = ANSWER_RE.findall(output_text)
    return matches[-1].upper() if matches else None


def score_record(r: dict):
    """Return (is_correct, extracted_option, parse_failed, explanation)."""
    extracted = r.get("extracted_option")
    if extracted is None and not r.get("parse_failed", False):
        # Older/foreign record without stored extraction — re-run the regex.
        extracted = extract_mcq_answer(r.get("response", ""))

    if extracted is None:
        return False, None, True, "[PARSE FAIL] no 'ANSWER: X' found in response"

    gt = str(r["ground_truth"]).strip().upper()
    extracted = str(extracted).strip().upper()
    if extracted == gt:
        return True, extracted, False, f"extracted {extracted} == ground truth {gt}"
    return False, extracted, False, f"extracted {extracted} != ground truth {gt}"


def main():
    parser = argparse.ArgumentParser(description="K9Bench MCQ exact-match scoring.")
    parser.add_argument("--input_file", required=True,
                        help="Raw JSON from eval_video_llm_mcq.py (shard or merged).")
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    print(f"Loading {args.input_file}...")
    data = json.load(open(args.input_file))
    results = data["individual_results"]
    print(f"Found {len(results)} examples; mode=mcq_exact_match")

    correct = 0
    parse_fails = 0
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0, "parse_failed": 0})
    evaluated = []
    skipped_errors = 0

    for i, r in enumerate(results):
        if "error" in r and "ground_truth" not in r:
            # Inference-time failure record; keep it visible but unscored.
            skipped_errors += 1
            evaluated.append({**r, "is_correct": False,
                              "evaluation_explanation": "[ERROR RECORD] inference failed",
                              "eval_mode": "mcq_exact_match"})
            continue

        ok, extracted, parse_failed, why = score_record(r)

        cat = r["question_category"]
        cat_stats[cat]["total"] += 1
        if ok:
            correct += 1
            cat_stats[cat]["correct"] += 1
        if parse_failed:
            parse_fails += 1
            cat_stats[cat]["parse_failed"] += 1

        evaluated.append({**r, "extracted_option": extracted, "parse_failed": parse_failed,
                          "is_correct": ok, "evaluation_explanation": why,
                          "eval_mode": "mcq_exact_match"})

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  {i + 1}/{len(results)}  acc={correct / (i + 1):.4f}")

    total = len(results) - skipped_errors
    accuracy = correct / total if total else 0.0
    print(f"\nAccuracy: {accuracy:.4f}  ({correct}/{total})")
    print(f"Parse failures (counted incorrect): {parse_fails}/{total}")
    if skipped_errors:
        print(f"Inference-error records excluded from denominator: {skipped_errors}")

    print(f"\n{'category':<30} {'accuracy':<10} {'correct/total':<15} {'parse_fails':<10}")
    print("-" * 70)
    cat_acc = {}
    for cat in sorted(cat_stats):
        s = cat_stats[cat]
        a = s["correct"] / s["total"] if s["total"] else 0.0
        cat_acc[cat] = {"accuracy": a, "correct": s["correct"], "total": s["total"],
                        "parse_failed": s["parse_failed"]}
        print(f"{cat:<30} {a:<10.4f} {s['correct']}/{s['total']:<15} {s['parse_failed']}")

    payload = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "eval_mode": "mcq_exact_match",
        "accuracy": accuracy,
        "correct_count": correct,
        "total_count": total,
        "parse_fail_count": parse_fails,
        "inference_error_count": skipped_errors,
        "category_wise_accuracy": cat_acc,
        "individual_results": evaluated,
        "metadata": {
            "prompt_style": next((r.get("prompt_style") for r in results
                                  if r.get("prompt_style")), None),
            "input_file": args.input_file,
        },
    }
    out_dir = os.path.dirname(args.output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    print(f"\nSaving -> {args.output_file}")
    json.dump(payload, open(args.output_file, "w"), indent=2)


if __name__ == "__main__":
    main()
