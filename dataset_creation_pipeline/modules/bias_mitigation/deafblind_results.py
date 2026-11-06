#!/usr/bin/env python3
"""Majority vote over deaf-blind LLM runs.

Takes the per-model outputs of ``deafblind_eval.py`` and keeps the question IDs
that at least ``--min_votes`` models answered correctly without seeing the
video. Those questions are solvable from text alone, so they are removed from
the benchmark.

Each ``--model`` argument is one model's results: either a glob over the worker
JSON files, or a plain text file with one correct-answer ID per line (the format
used before the runner was added).

Example:
    python deafblind_results.py \\
        --model "../../data/intermediary_files/bias_mitigation/deafblind/outputs/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/*.json" \\
        --model "../../data/intermediary_files/bias_mitigation/deafblind/outputs/Qwen/Qwen3-32B/*.json" \\
        --model "../../data/intermediary_files/bias_mitigation/deafblind/outputs/mistralai/Mistral-Small-3.2-24B-Instruct-2506/*.json" \\
        --output example_ids_majority_voted.txt
"""

import argparse
import glob
import json
import os
from collections import Counter


def ids_from_json(paths) -> set:
    """IDs this model answered correctly (exact_match == 1.0)."""
    correct, seen = set(), 0
    for path in paths:
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"    x skipping unreadable {os.path.basename(path)}: {e}")
            continue
        for r in data.get("individual_results", []):
            if "example_id" not in r:
                continue
            seen += 1
            if r.get("exact_match", 0) == 1.0:
                correct.add(str(r["example_id"]))
    return correct, seen


def ids_from_txt(path: str) -> set:
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def load_model_ids(spec: str):
    """Resolve one --model argument to (correct_ids, n_records)."""
    paths = sorted(glob.glob(spec))
    if not paths:
        raise SystemExit(f"No files matched: {spec}")
    if len(paths) == 1 and paths[0].endswith(".txt"):
        ids = ids_from_txt(paths[0])
        return ids, len(ids)
    return ids_from_json(paths)


def main():
    ap = argparse.ArgumentParser(description="Majority vote over deaf-blind LLM runs.")
    ap.add_argument("--model", action="append", required=True, dest="models",
                    metavar="GLOB",
                    help="One model's results: a glob over worker JSONs, or a "
                         ".txt of correct-answer IDs. Repeat per model.")
    ap.add_argument("--output", default="example_ids_majority_voted.txt",
                    help="Where to write the IDs to remove, one per line.")
    ap.add_argument("--min_votes", type=int, default=2,
                    help="Models that must answer correctly for a question to be "
                         "removed (default 2, i.e. majority of 3).")
    args = ap.parse_args()

    counter = Counter()
    all_ids = set()
    for spec in args.models:
        print(f"Reading {spec}")
        correct, seen = load_model_ids(spec)
        print(f"    {len(correct)} answered correctly out of {seen} records")
        counter.update(correct)      # set, so a model votes at most once per id
        all_ids |= correct

    majority = [eid for eid, votes in counter.items() if votes >= args.min_votes]
    try:
        majority.sort(key=int)
    except ValueError:
        majority.sort()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        for eid in majority:
            f.write(f"{eid}\n")

    print()
    print(f"Models compared              : {len(args.models)}")
    print(f"IDs answered by >= 1 model   : {len(all_ids)}")
    print(f"IDs answered by >= {args.min_votes} models : {len(majority)}  (removed)")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
