"""Closed-source API-served models (Tables 1 and 2)."""

from ..registry import ModelSpec, register

register(ModelSpec(
    name="gemini-2.5-pro", model_id="gemini-2.5-pro", family="gemini",
    kind="api", protocols=("openended",), config="api_evals/configs/gemini.yaml",
    modality="video"))

register(ModelSpec(
    name="gemini-2.5-pro-av", model_id="gemini-2.5-pro", family="gemini",
    kind="api", protocols=("openended",), config="api_evals/configs/gemini.yaml",
    modality="audio_video",
    note="audio+video; set modality: audio_video in the config"))

register(ModelSpec(
    name="qwen3-vl-235b", model_id="qwen3-vl-235b-a22b-instruct", family="dashscope",
    kind="api", protocols=("openended",), config="api_evals/configs/qwen3vl_235b.yaml",
    modality="video", note="DashScope, 1 FPS"))

register(ModelSpec(
    name="qwen3-omni-flash", model_id="qwen3-omni-flash", family="omni",
    kind="api", protocols=("openended",), config="api_evals/configs/qwen3_omni.yaml",
    modality="audio_video",
    note="audio+video; videos under 150s only (2693 QAs)"))

register(ModelSpec(
    name="gpt-4o", model_id="gpt-4o", family="openai",
    kind="api", protocols=("openended",), config="api_evals/configs/gpt.yaml",
    modality="video", note="32 uniformly sampled frames"))
