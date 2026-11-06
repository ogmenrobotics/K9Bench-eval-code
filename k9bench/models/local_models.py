"""Locally-loaded open-source models (Table 1)."""

from ..registry import ModelSpec, register

# --- Qwen3-VL, instruct -----------------------------------------------------
register(ModelSpec(name="qwen3-vl-4b", model_id="Qwen/Qwen3-VL-4B-Instruct",
                   family="qwen", gpus=1))
register(ModelSpec(name="qwen3-vl-8b", model_id="Qwen/Qwen3-VL-8B-Instruct",
                   family="qwen", gpus=1))
register(ModelSpec(name="qwen3-vl-32b", model_id="Qwen/Qwen3-VL-32B-Instruct",
                   family="qwen", gpus=2, note="needs 2 A40s"))

# --- Qwen3-VL, thinking -----------------------------------------------------
register(ModelSpec(name="qwen3-vl-4b-thinking", model_id="Qwen/Qwen3-VL-4B-Thinking",
                   family="qwen", thinking=True, gpus=1))
register(ModelSpec(name="qwen3-vl-8b-thinking", model_id="Qwen/Qwen3-VL-8B-Thinking",
                   family="qwen", thinking=True, gpus=1))
register(ModelSpec(name="qwen3-vl-32b-thinking", model_id="Qwen/Qwen3-VL-32B-Thinking",
                   family="qwen", thinking=True, gpus=2, note="needs 2 A40s"))

# --- InternVL ---------------------------------------------------------------
register(ModelSpec(name="internvl3.5-8b", model_id="OpenGVLab/InternVL3_5-8B",
                   family="internvl", gpus=1))
register(ModelSpec(name="internvl3.5-14b", model_id="OpenGVLab/InternVL3_5-14B",
                   family="internvl", gpus=2, note="needs 2 A40s"))
