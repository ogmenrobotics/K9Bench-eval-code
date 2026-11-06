#!/usr/bin/env python3
# Copyright 2025 The K9Bench Authors.
# SPDX-License-Identifier: Apache-2.0

"""Convert closed-source API eval output into the local scorer's schema.

The API runners record only what the provider returned (the question and a
``{"reasoning", "answer"}`` payload). ``evaluate_results.py`` additionally needs
``options`` as a list, the ground-truth letter and the question category. This
script supplies them from the released dataset.

The join key is ``(scene_name, question)``, NOT ``idx``. Result files produced
before the dataset was published carry row numbers from a larger pre-release
pool, and those numbers do not correspond to ``idx`` in ``ogmen/K9Bench`` —
joining on them silently pairs answers with the wrong question. Pass
``--join_on idx`` only for files you know were produced against the released
dataset.

Accepts either a JSON list, a ``{"individual_results": [...]}`` object, or JSONL
(one record per line), so it handles the output of all three API runners.

Example:
    python api_evals/convert_api_results.py \
        --input_file gemini_results.jsonl \
        --output_file eval_results/gemini-2.5-pro/raw_full.json \
        --model_name gemini-2.5-pro
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_common import load_k9bench_qas, options_dict_to_list  # noqa: E402


def load_any(path: str) -> list:
    """Load a JSON list, a dict with individual_results, or JSONL."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("individual_results", "results"):
                if key in data:
                    return data[key]
            return [data]
    except json.JSONDecodeError:
        pass

    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def extract_answer(rec: dict) -> str:
    """Pull the free-form answer out of whichever shape the runner used."""
    for key in ("response", "processed_response"):
        payload = rec.get(key)
        if isinstance(payload, dict) and "answer" in payload:
            return str(payload["answer"] or "").strip()
    if isinstance(rec.get("answer"), str):
        return rec["answer"].strip()
    return ""


def extract_reasoning(rec: dict) -> str:
    for key in ("response", "processed_response"):
        payload = rec.get(key)
        if isinstance(payload, dict) and "reasoning" in payload:
            return str(payload["reasoning"] or "")
    return str(rec.get("reasoning", "") or "")


def main():
    ap = argparse.ArgumentParser(description="Convert API eval output to the scorer schema.")
    ap.add_argument("--input_file", required=True, help="Raw output from an API runner.")
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--model_name", default=None,
                    help="Override the model name recorded per result.")
    ap.add_argument("--dataset_name", default="ogmen/K9Bench")
    ap.add_argument("--split", default="test")
    ap.add_argument("--modality", default=None, choices=["video", "audio_video"],
                    help="Stamp the input modality into each record (Table 1 vs Table 2).")
    ap.add_argument("--join_on", default="question", choices=["question", "idx"],
                    help="Join key. 'question' matches on (scene_name, question) and is "
                         "correct for every result file; 'idx' is only safe for files "
                         "produced against the released dataset.")
    args = ap.parse_args()

    records = load_any(args.input_file)
    print(f"Loaded {len(records)} records from {args.input_file}")
    if not records:
        raise SystemExit("No records found in input file.")

    dataset = load_k9bench_qas(args.dataset_name, args.split)
    by_idx = {q["idx"]: q for q in dataset}
    by_scene_q, by_q = {}, {}
    for q in dataset:
        by_scene_q.setdefault((q["scene_name"], q["question"].strip()), []).append(q)
        by_q.setdefault(q["question"].strip(), []).append(q)

    def disambiguate(candidates, rec, used):
        """A handful of questions repeat within one video with different option
        sets. Prefer the candidate whose options match the record's own, then
        any candidate not already consumed."""
        if len(candidates) == 1:
            return candidates[0]
        rec_opts = rec.get("options")
        if rec_opts:
            rec_list = options_dict_to_list(rec_opts) if isinstance(rec_opts, dict) else list(rec_opts)
            rec_set = {str(o).strip() for o in rec_list}
            for cand in candidates:
                cand_set = {str(o).strip() for o in options_dict_to_list(cand["options"])}
                if cand_set == rec_set:
                    return cand
        for cand in candidates:
            if cand["idx"] not in used:
                return cand
        return candidates[0]

    def lookup(rec, used):
        if args.join_on == "idx":
            idx = rec.get("idx")
            return by_idx.get(int(idx)) if idx is not None else None
        question = str(rec.get("question", "")).strip()
        scene = rec.get("scene_name") or rec.get("video_id")
        candidates = by_scene_q.get((scene, question)) or by_q.get(question)
        if not candidates:
            return None
        return disambiguate(candidates, rec, used)

    converted, unmatched, empty = [], 0, 0
    seen = set()
    for rec in records:
        qa = lookup(rec, seen)
        if qa is None:
            unmatched += 1
            continue
        if qa["idx"] in seen:
            continue
        seen.add(qa["idx"])

        answer = extract_answer(rec)
        if not answer:
            empty += 1

        out = {
            "idx": qa["idx"],
            "question_category": qa["question_category"],
            "scene_name": qa["scene_name"],
            "question": qa["question"],
            "options": options_dict_to_list(qa["options"]),
            "ground_truth": qa["correct_answer"],
            "response": extract_reasoning(rec),
            "answer": answer,
            "model_name": args.model_name or rec.get("model") or rec.get("model_name"),
            "timestamp": rec.get("timestamp"),
        }
        if args.modality:
            out["modality"] = args.modality
        converted.append(out)

    converted.sort(key=lambda r: r["idx"])

    print(f"\nJoined on: {args.join_on}")
    print(f"  matched   : {len(converted)}")
    print(f"  unmatched : {unmatched} (not part of the released 4744-QA test split)")
    if empty:
        print(f"  !! {empty} record(s) had an empty answer field (kept; scored as-is)")
    if converted and unmatched > len(converted):
        print("  !! More records failed to match than matched. If this file was produced "
              "against the released dataset, try --join_on idx.")

    payload = {"individual_results": converted, "total_count": len(converted)}
    out_dir = os.path.dirname(args.output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    # ensure_ascii (the json default) keeps the file pure ASCII, matching what
    # eval_video_llm.py writes. Readers that open() under a non-UTF-8 locale
    # would otherwise mojibake em dashes and smart quotes.
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nConverted {len(converted)} records -> {args.output_file}")
    print("Next: python evaluate_results.py --mode calculate --input_file "
          f"{args.output_file} --output_file <sim.json>")


if __name__ == "__main__":
    main()
