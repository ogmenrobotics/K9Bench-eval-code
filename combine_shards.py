#!/usr/bin/env python3
"""Merge sharded raw_*_idx_<start>_<end>.json files from run_eval.sh into a
single raw JSON ready for evaluate_results.py / evaluate_results_mcq.py.

Dedupes by `idx`, sorts, and reports missing index chunks so failed shards
are easy to spot and relaunch.

Example:
    python combine_shards.py \
        --out_dir eval_results/Qwen3-VL-4B-Instruct_mf32/shards \
        --model Qwen3-VL-4B-Instruct --max_frames 32 --chunk_size 200 \
        --output eval_results/Qwen3-VL-4B-Instruct_mf32/raw_full.json
"""

import argparse
import glob
import json
import os
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Directory containing shard raw_*.json files")
    ap.add_argument("--model", required=True, help="Model short name, e.g. Qwen3-VL-4B-Instruct")
    ap.add_argument("--max_frames", type=int, default=32)
    ap.add_argument("--chunk_size", type=int, default=150, help="Expected shard size (for missing-chunk report)")
    ap.add_argument("--output", default=None, help="Merged output path (default: <out_dir>/../raw_<model>_mf<frames>_full.json)")
    args = ap.parse_args()

    pattern = os.path.join(args.out_dir, f"raw_{args.model}_mf{args.max_frames}_idx_*.json")
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} shard files matching {pattern}")
    if not files:
        raise SystemExit("No shard files found — check --out_dir/--model/--max_frames.")

    rng = re.compile(r"_idx_(\d+)_(\d+)\.json$")
    starts = []
    all_results, seen = [], set()
    for f in files:
        m = rng.search(f)
        if m:
            starts.append(int(m.group(1)))
        try:
            with open(f) as fh:
                data = json.load(fh)
        except json.JSONDecodeError as e:
            print(f"  x Skipping unreadable {f}: {e}")
            continue
        n_new = 0
        for r in data.get("individual_results", []):
            if "idx" in r and r["idx"] not in seen:
                seen.add(r["idx"])
                all_results.append(r)
                n_new += 1
        print(f"  + {os.path.basename(f)}: {n_new} new results")

    if starts:
        starts.sort()
        expected = set(range(starts[0], starts[-1] + 1, args.chunk_size))
        missing = sorted(expected - set(starts))
        if missing:
            print(f"\n!! Missing shard start indices (chunk_size={args.chunk_size}): {missing}")
        else:
            print("\nNo missing shards detected")

    all_results.sort(key=lambda r: r["idx"])
    merged = {"individual_results": all_results, "total_count": len(all_results)}

    output = args.output or os.path.join(
        os.path.dirname(args.out_dir.rstrip("/")),
        f"raw_{args.model}_mf{args.max_frames}_full.json",
    )
    out_dir = os.path.dirname(output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(output, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nMerged {len(all_results)} unique results -> {output}")


if __name__ == "__main__":
    main()
