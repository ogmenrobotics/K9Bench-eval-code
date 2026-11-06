#!/usr/bin/env python3
"""Deaf-blind LLM filtering: answer the MCQs with no video, text only.

A question that a text-only LLM can already answer does not require watching the
video, so it is a language-only shortcut and is removed from the benchmark. This
script runs one such model over the QA pool; ``deafblind_results.py`` then takes
the majority vote across models and writes the IDs to drop.

The prompt, sampling parameters, JSON extraction and exact-match scoring are
unchanged from the runs reported in the paper. The only difference is that
generation happens locally through ``transformers`` instead of against a vLLM
server, so no separate inference service is needed.

Shard across GPUs with ``--num_workers``/``--worker_id``; each worker writes its
own results file, and ``deafblind_results.py`` globs them back together.

Example:
    python deafblind_eval.py --config ../../configs/deafblind_eval.yaml \\
        --model_name Qwen/Qwen3-32B --num_workers 8 --worker_id 0
"""

import argparse
import json
import os
import re

import torch
import tqdm
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Sampling parameters, per model, exactly as used for the reported runs.
MISTRAL_MODEL = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
SAMPLING = {
    MISTRAL_MODEL: dict(max_new_tokens=4096, temperature=0.15, top_p=0.8,
                        top_k=20, repetition_penalty=1.05),
    "default": dict(max_new_tokens=4096, temperature=0.7, top_p=0.8,
                    top_k=20, repetition_penalty=1.05),
}


def load_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(example: dict, prompt_template: str) -> str:
    """Fill the question and options into the prompt template.

    ``options`` is stringified exactly as before (``str(example['options'])``),
    so the model sees the same text it saw in the original runs.
    """
    prompt = prompt_template.replace("{question}", example["question"])
    prompt = prompt.replace("{options}", str(example["options"]))
    return prompt


def extract_reasoning_trace_prediction_from_response(model_response: str):
    """Extract reasoning_trace and selected_option from the JSON part of the
    model response. Unchanged from the original implementation."""
    try:
        json_match = re.search(r'```json\n(\{.*?\})\n```', model_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            json_data = json.loads(json_str)
            reasoning_trace = json_data.get("reasoning_trace", "")
            predicted_option = json_data.get("selected_option", "")
            return reasoning_trace, predicted_option
    except (json.JSONDecodeError, AttributeError):
        pass
    return "", ""


def calculate_exact_match(pred_text, ground_truth):
    """1.0 if the predicted option matches the ground truth exactly."""
    return 1.0 if pred_text.strip() == ground_truth.strip() else 0.0


def shard(dataset, num_workers: int, worker_id: int):
    """Contiguous split; the last worker absorbs the remainder."""
    if num_workers <= 1:
        print(f"Single worker: using all {len(dataset)} samples.")
        return dataset
    total = len(dataset)
    chunk = total // num_workers
    start = worker_id * chunk
    end = start + chunk
    if worker_id == num_workers - 1:
        end += total % num_workers
    print(f"Worker {worker_id}/{num_workers}: samples {start} to {end - 1} "
          f"({end - start} of {total})")
    return dataset.select(range(start, min(end, total)))


def main():
    ap = argparse.ArgumentParser(description="Deaf-blind (text-only) MCQ evaluation.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model_name", default=None,
                    help="Overrides model_name in the config.")
    ap.add_argument("--output_file", default=None,
                    help="Overrides the config-derived output path.")
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--worker_id", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=-1)
    args = ap.parse_args()

    if not 0 <= args.worker_id < max(args.num_workers, 1):
        raise SystemExit("worker_id must satisfy 0 <= worker_id < num_workers")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = args.model_name or cfg["model_name"]
    dataset_name = cfg["dataset_name"]
    split = cfg.get("split", "test")
    prompt_template = load_file(cfg["prompt_file"])

    system_prompt = ""
    if model_name == MISTRAL_MODEL and cfg.get("mistral_system_prompt_file"):
        system_prompt = load_file(cfg["mistral_system_prompt_file"])

    output_file = args.output_file
    if output_file is None:
        out_dir = os.path.join(cfg.get("output_dir", "outputs"), model_name)
        output_file = os.path.join(out_dir, f"worker_{args.worker_id}.json")
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    print(f"Loading dataset: {dataset_name} (split={split})...")
    dataset = load_dataset(dataset_name, split=split)
    print(f"Found total samples: {len(dataset)}")

    if args.max_samples != -1:
        step = max(1, len(dataset) // args.max_samples)
        dataset = dataset.select(list(range(0, len(dataset), step))[:args.max_samples])
        print(f"After max_samples filtering: {len(dataset)} samples (step {step})")

    dataset = shard(dataset, args.num_workers, args.worker_id)

    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    gen_kwargs = dict(SAMPLING.get(model_name, SAMPLING["default"]))
    gen_kwargs["do_sample"] = True
    print(f"Sampling: {gen_kwargs}")

    results = []
    exact_match_scores = []

    for i, example in enumerate(tqdm.tqdm(dataset)):
        example_id = example["idx"]
        ground_truth = example["ground_truth"]
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_prompt(example, prompt_template)},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id,
                                     **gen_kwargs)
            output_text = tokenizer.decode(
                out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

            reasoning, predicted_option = \
                extract_reasoning_trace_prediction_from_response(output_text)
            exact_match = calculate_exact_match(predicted_option, ground_truth)
            exact_match_scores.append(exact_match)

            results.append({
                "example_id": example_id,
                "question_category": example["question_category"],
                "scene_name": example["scene_name"],
                "ground_truth": ground_truth,
                "exact_match": exact_match,
                "model_output": output_text,
                "reasoning": reasoning,
                "predicted_option": predicted_option,
            })
            print(f"[{i}] id={example_id} gt={ground_truth} "
                  f"pred={predicted_option} match={exact_match:.1f}")

        except Exception as e:
            print(f"Error processing example {i}: {e}")
            results.append({
                "example_id": example_id,
                "question_category": example.get("question_category"),
                "scene_name": example.get("scene_name"),
                "ground_truth": ground_truth,
                "model_output": "",
                "error": str(e),
            })

        # Checkpoint so a preempted worker does not lose everything.
        if (i + 1) % 25 == 0:
            save(results, exact_match_scores, output_file)

    avg = sum(exact_match_scores) / len(exact_match_scores) if exact_match_scores else 0.0
    print("\n" + "=" * 50)
    print(f"Overall Exact Match Score: {avg:.4f}")
    print("=" * 50)
    save(results, exact_match_scores, output_file)
    print(f"Saved results to {output_file}")


def save(results, scores, output_file):
    avg = sum(scores) / len(scores) if scores else 0.0
    tmp = output_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"individual_results": results, "average_exact_match": avg}, f, indent=2)
    os.replace(tmp, output_file)


if __name__ == "__main__":
    main()
