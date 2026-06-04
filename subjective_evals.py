"""LLM-as-judge evaluation for K9Bench, aligned with the release pipeline.

Reads any JSON produced by the release pipeline (``raw_*.json``, ``sim_*.json``,
or ``eval_*.json`` from ``evaluate_results.py``) — they all expose
``individual_results`` at the top level — and asks GPT-4o to score each
``answer`` against the ground-truth option text.

Example::

    export OPENAI_API_KEY=sk-...

    python pawbench/k9bench_release/subjective_evals.py \\
        --evaluations pawbench/k9bench_release/eval_results/eval_Qwen3-VL-4B-Instruct_mf32_n100.json \\
        --output     pawbench/k9bench_release/eval_results/llm_judge_Qwen3-VL-4B-Instruct.json \\
        --skipped    pawbench/k9bench_release/eval_results/llm_judge_Qwen3-VL-4B-Instruct_skipped.json
"""

import os
import re
import json
import time
import argparse
import traceback
import logging
from urllib.parse import urlparse, parse_qs

from openai import OpenAI
from datasets import load_dataset

# ─────────────────────────── Logging ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


# ─────────────────────────── Config ─────────────────────────────
MODEL_NAME    = "gpt-4o"
SLEEP_SECONDS = 1


# ─────────────────────────── Argument Parsing ───────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="LLM-as-Judge Evaluation for K9Bench")
    parser.add_argument("--evaluations", required=True, help="Path to model evaluations JSON")
    parser.add_argument("--output",      required=True, help="Path to save judge results JSON")
    parser.add_argument("--skipped",     required=True, help="Path to save skipped entries JSON")
    return parser.parse_args()


# ─────────────────────────── Scene Name Extraction ──────────────
def extract_scene_name(video_url: str) -> str:
    try:
        parsed = urlparse(video_url)
        if parsed.hostname in ("www.youtube.com", "youtube.com"):
            qs = parse_qs(parsed.query)
            if "v" in qs:
                return qs["v"][0]
        if parsed.hostname == "youtu.be":
            return parsed.path.lstrip("/")
    except Exception:
        pass
    match = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", video_url)
    if match:
        return match.group(1)
    return video_url


# ─────────────────────────── Data Loading ───────────────────────
def load_k9bench() -> dict:
    """Load K9Bench from HuggingFace and index by idx."""
    log.info("Loading K9Bench dataset from HuggingFace...")
    ds = load_dataset("ogmen/K9Bench", split="test")
    log.info(f"Loaded {len(ds)} samples")

    index = {}
    for sample in ds:
        idx = int(sample["idx"])

        # Use dataset's scene_name if present, otherwise extract from video_url
        scene_name = (sample.get("scene_name") or "").strip()
        if not scene_name:
            scene_name = extract_scene_name(sample["video_url"])

        index[idx] = {
            "idx":               idx,
            "video_url":         sample["video_url"],
            "scene_name":        scene_name,
            "question_category": sample["question_category"],
            "question":          sample["question"],
            "options":           dict(sample["options"]),
            "correct_answer":    sample["correct_answer"],
        }

    return index


def load_model_answers(evaluations_path: str) -> dict:
    """Load {idx: answer} from either pipeline JSON format:
    - New flat format:   top-level 'individual_results'
    - Legacy format:     'threshold_results' -> 'threshold_0.5' -> 'individual_results'
    """
    with open(evaluations_path, "r") as f:
        data = json.load(f)

    # New flat format
    if "individual_results" in data:
        items = data["individual_results"]
        log.info(f"Detected new flat format")

    # Legacy threshold format
    elif "threshold_results" in data:
        threshold_key = next(iter(data["threshold_results"]))  # takes first threshold
        items = data["threshold_results"][threshold_key]["individual_results"]
        log.info(f"Detected legacy format, using threshold key: '{threshold_key}'")

    else:
        raise ValueError(
            f"{evaluations_path} does not contain 'individual_results' or "
            "'threshold_results' at the top level."
        )

    answer_map = {int(item["idx"]): item["answer"] for item in items}
    log.info(f"Loaded {len(answer_map)} model answers from {evaluations_path}")
    return answer_map


def load_json(path: str):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []


def save_json(path: str, data):
    """Atomic write: write to .tmp then rename to avoid corruption on interrupt."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ─────────────────────────── Prompt Builder ─────────────────────
def build_prompt(question, correct_answer_text, model_output):
    return f"""You are a scoring assistant for evaluating text quality.

Evaluate the Model Output based strictly on the given Question and Correct Answer.

Question:
{question}

Correct Answer:
{correct_answer_text}

Model Output:
{model_output}

Evaluation Instructions:
Score each of the following aspects on a scale from 1 to 10 (integers only):

1. Logic:
Evaluate how well the reasoning and structure of the response align with the question, and whether the conclusions follow coherently.
- 1–2: Entirely illogical
- 3–4: Inconsistent or poorly structured
- 5–6: Partially logical with minor gaps
- 7–8: Mostly logical with rare issues
- 9–10: Fully logical and coherent

2. Factuality:
Assess the correctness of the information and the absence of factual errors.
- 1–2: Mostly incorrect or misleading
- 3–4: Significant factual inaccuracies
- 5–6: Some minor inaccuracies
- 7–8: Highly factual with rare errors
- 9–10: Entirely factual

3. Accuracy:
Consider how precisely the response addresses the question.
- 1–2: Irrelevant or off-topic
- 3–4: Partially inaccurate
- 5–6: Moderately accurate
- 7–8: Accurate with minimal flaws
- 9–10: Perfectly accurate

4. Conciseness:
Evaluate how effectively the response conveys its message without unnecessary verbosity.
- 1–2: Excessively wordy or incomplete
- 3–4: Moderately verbose or unfocused
- 5–6: Somewhat concise but improvable
- 7–8: Mostly concise with rare verbosity
- 9–10: Perfectly concise and to the point

5. Overall:
Provide an integrated score reflecting the holistic quality of the response.

Output Requirements:
1. First, provide a brief Chain-of-Thought (CoT) explaining the reasoning behind the scores.
   Format exactly as:
   CoT: {{your concise reasoning here}}

2. Then, output ONLY a JSON dictionary in the exact format below:
{{'Logic': X, 'Factuality': X, 'Accuracy': X, 'Conciseness': X, 'Overall': X}}

Do not include any text outside the CoT and the JSON dictionary.
"""


# ─────────────────────────── Response Parsing ───────────────────
def parse_response(text: str) -> tuple:
    cot_match = re.search(r"CoT:\s*(.*?)(?:\n\n|\Z)", text, re.DOTALL)
    cot = cot_match.group(1).strip() if cot_match else ""

    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()

    # Find last {...} block
    json_matches = list(re.finditer(r"\{[^{}]+\}", cleaned, re.DOTALL))
    if not json_matches:
        raise ValueError(f"No score dictionary found in response:\n{text}")

    score_text = json_matches[-1].group(0)
    score_text = score_text.replace("'", '"')
    scores = json.loads(score_text)

    required_keys = {"Logic", "Factuality", "Accuracy", "Conciseness", "Overall"}
    missing = required_keys - scores.keys()
    if missing:
        raise ValueError(f"Score dict missing keys: {missing}")

    scores = {k: int(v) for k, v in scores.items()}
    return cot, scores


# ─────────────────────────── Main ───────────────────────────────
def main():
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable not set.\n"
            "Run:  export OPENAI_API_KEY=sk-..."
        )
    client = OpenAI(api_key=api_key)

    k9bench       = load_k9bench()
    model_answers = load_model_answers(args.evaluations)

    results: list = load_json(args.output)
    skipped: list = load_json(args.skipped)
    processed_ids = {r["idx"] for r in results}

    log.info(f"Loaded {len(k9bench)} questions")
    log.info(f"Already processed: {len(processed_ids)}")

    for idx in sorted(k9bench.keys()):

        if idx in processed_ids:
            continue

        if idx not in model_answers:
            log.warning(f"Missing model answer for idx={idx}")
            skipped.append({
                "idx":       idx,
                "error":     "missing_model_answer",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            save_json(args.skipped, skipped)
            continue

        entry = k9bench[idx]

        try:
            log.info(f"Processing idx {idx}")

            question            = entry["question"]
            options             = entry["options"]
            correct_answer_text = options[entry["correct_answer"]]
            model_output        = model_answers[idx]

            prompt = build_prompt(question, correct_answer_text, model_output)

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.choices[0].message.content.strip()
            cot, scores = parse_response(content)

            result = {
                "idx":               idx,
                "video_url":         entry["video_url"],
                "scene_name":        entry["scene_name"],
                "question_category": entry["question_category"],
                "question":          question,
                "correct_answer":    correct_answer_text,
                "model_output":      model_output,
                "cot":               cot,
                "scores":            scores,
                "judge_model":       MODEL_NAME,
                "timestamp":         time.strftime("%Y-%m-%d %H:%M:%S")
            }

            results.append(result)
            save_json(args.output, results)

            log.info(f"Saved idx {idx}")
            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            log.error(f"Failed idx {idx}: {e}")
            traceback.print_exc()

            skipped.append({
                "idx":       idx,
                "error":     str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            save_json(args.skipped, skipped)

    log.info("DONE")
    log.info(f"Saved results : {len(results)}")
    log.info(f"Skipped       : {len(skipped)}")


if __name__ == "__main__":
    main()

