#!/usr/bin/env python3
# Copyright 2025 The K9Bench Authors.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the closed-source API evaluation runners.

The three API runners (Gemini, DashScope Qwen3-VL, Qwen3-Omni) previously each
carried their own copy of the dataset loader, prompt, JSON extraction and rate
limiter, which had drifted apart. They are collected here so every model is
asked the identical question and the outputs share one schema.
"""

import json
import os
import re
import subprocess
import time
from collections import deque

from datasets import load_dataset

# The released dataset stores options as a dict A..E and the gold letter under
# `correct_answer`; `scene_name` is the YouTube id parsed from `video_url`.
YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|/embed/|/v/|/shorts/)([\w-]{11})")


def extract_scene_name(video_url: str) -> str:
    m = YT_ID_RE.search(video_url)
    if not m:
        raise ValueError(f"Could not extract YouTube id from URL: {video_url}")
    return m.group(1)


def options_dict_to_list(options) -> list:
    """``{"A": "..."} -> ["A. ...", ...]`` (the form the scorer embeds)."""
    if isinstance(options, list):
        return options
    return [f"{k}. {options[k]}" for k in sorted(options.keys())]


def load_k9bench_qas(dataset_name: str = "ogmen/K9Bench", split: str = "test") -> list:
    """Load the released dataset into one canonical per-QA record."""
    ds = load_dataset(dataset_name, split=split)
    data = []
    for i, row in enumerate(ds):
        video_url = row["video_url"]
        data.append({
            "idx": row.get("idx", i),
            "scene_name": extract_scene_name(video_url),
            "video_url": video_url,
            "question": row["question"],
            "question_category": row["question_category"],
            "options": dict(row["options"]),
            "correct_answer": row["correct_answer"],
        })
    print(f"Loaded {len(data)} QA entries from {dataset_name} ({split})")
    return data


def sanitize_text(text: str) -> str:
    return (text.replace("’", "'").replace("‘", "'")
                .replace("“", '"').replace("”", '"'))


def sanitize_json_string(txt: str) -> str:
    txt = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", txt)
    txt = txt.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")
    return " ".join(txt.split())


def build_prompt(question: str, include_audio: bool = False) -> str:
    """The evaluation prompt (paper Figure 17).

    ``include_audio`` adds the audio wording used for the audio+video runs; the
    video-only wording is otherwise identical across all models.
    """
    if include_audio:
        watch = "Carefully watch and analyze the entire video (including any audio) before answering the question."
        step1 = "1. Watch the entire video to identify subtle behaviors, postures, interactions, or audio cues relevant to the question."
    else:
        watch = "Carefully watch and analyze the entire video before answering the question."
        step1 = "1. Watch the entire video to identify subtle behaviors, postures, or interactions relevant to the question."

    return f"""You are an expert in video understanding and reasoning. {watch}

### Instructions:
{step1}
2. Reason step-by-step, explaining how each observation leads to your conclusion.

Return the response strictly in JSON format with two keys:
- "reasoning": A detailed freeform explanation of your logical steps leading to the correct answer.
- "answer": A concise sentence giving the proper answer.

### Question
{sanitize_text(question)}"""


def extract_reasoning_answer(text: str):
    """Parse the model's ``{"reasoning", "answer"}`` JSON, tolerating fences.

    Returns ``(reasoning, answer)``; on failure the raw text becomes the
    reasoning and the answer is empty.
    """
    if not text:
        return "", ""
    cleaned = text.strip()
    m = re.search(r"```json\s*(.*?)\s*```", cleaned, re.DOTALL)
    if m:
        cleaned = m.group(1).strip()
    elif cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    for candidate in (cleaned, sanitize_json_string(cleaned)):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return str(parsed.get("reasoning", "")).strip(), str(parsed.get("answer", "")).strip()
        except json.JSONDecodeError:
            pass

    m = re.search(r'"answer"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', cleaned)
    if m:
        return cleaned, m.group(1).replace(r'\"', '"').replace(r"\n", "\n")
    return text, ""


def video_duration_seconds(path: str):
    """Duration in seconds via ffprobe, or None if it cannot be determined."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, check=True).stdout.strip()
        return float(out)
    except Exception:
        return None


class TokenBucket:
    """Sliding-window requests-per-minute + tokens-per-minute limiter."""

    def __init__(self, rpm: int, tpm: int):
        self.rpm = rpm
        self.tpm = tpm
        self.requests = deque()
        self.tokens = deque()

    def _trim(self, now):
        while self.requests and now - self.requests[0] > 60:
            self.requests.popleft()
        while self.tokens and now - self.tokens[0][0] > 60:
            self.tokens.popleft()

    def wait(self, token_estimate: int = 0):
        now = time.time()
        self._trim(now)

        sleep_for = 0.0
        if self.rpm and len(self.requests) >= self.rpm:
            sleep_for = max(sleep_for, 60 - (now - self.requests[0]))
        if self.tpm and token_estimate:
            used = sum(t for _, t in self.tokens)
            if used + token_estimate > self.tpm and self.tokens:
                sleep_for = max(sleep_for, 60 - (now - self.tokens[0][0]))

        if sleep_for > 0:
            print(f"Rate limit: sleeping {sleep_for:.1f}s")
            time.sleep(sleep_for)
            now = time.time()
            self._trim(now)

        self.requests.append(now)
        if token_estimate:
            self.tokens.append((now, token_estimate))

    def record_tokens(self, tokens: int):
        if tokens:
            self.tokens.append((time.time(), tokens))


def append_jsonl(path: str, record: dict) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


def load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out
