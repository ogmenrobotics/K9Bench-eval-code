"""Model registry for K9Bench.

Every evaluable model is one :class:`ModelSpec` passed to :func:`register`.
The spec says which runner executes it and what arguments that runner needs;
the runners themselves are unchanged.

Adding a model is one entry (see ``docs/ADDING_A_MODEL.md``).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Runner scripts, relative to the repository root.
RUNNERS = {
    ("qwen", "openended"): "local_evals/eval_video_llm.py",
    ("qwen", "mcq"): "local_evals/eval_video_llm_mcq.py",
    ("internvl", "openended"): "local_evals/eval_video_llm_internvl.py",
    ("internvl", "mcq"): "local_evals/eval_video_llm_internvl_mcq.py",
    ("gemini", "openended"): "api_evals/eval_gemini.py",
    ("dashscope", "openended"): "api_evals/eval_dashscope_qwen.py",
    ("omni", "openended"): "api_evals/eval_qwen_omni.py",
    ("openai", "openended"): "api_evals/eval_gpt.py",
}


@dataclass(frozen=True)
class ModelSpec:
    """How to evaluate one model.

    name        short identifier used on the command line
    model_id    HuggingFace id (local) or provider model name (API)
    family      selects the runner: qwen | internvl | gemini | dashscope | omni
    kind        "local" (loaded with transformers) or "api" (served remotely)
    protocols   prompt styles this model supports
    thinking    pass --thinking_mode to the runner
    gpus        A40s needed per shard (local models only)
    config      default config file for API models
    modality    "video" or "audio_video" (API models)
    available   False for models the suite cannot yet run
    note        shown by `k9bench models`
    """

    name: str
    model_id: str
    family: str
    kind: str = "local"
    protocols: tuple = ("openended", "mcq")
    thinking: bool = False
    gpus: int = 1
    config: Optional[str] = None
    modality: str = "video"
    available: bool = True
    note: str = ""
    extra_args: List[str] = field(default_factory=list)

    def runner(self, protocol: str) -> str:
        key = (self.family, protocol)
        if key not in RUNNERS:
            raise KeyError(
                f"{self.name} does not support protocol '{protocol}' "
                f"(supported: {', '.join(self.protocols)})")
        return RUNNERS[key]


MODELS: Dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    if spec.name in MODELS:
        raise ValueError(f"Duplicate model name: {spec.name}")
    MODELS[spec.name] = spec
    return spec


def get(name: str) -> ModelSpec:
    if name not in MODELS:
        raise SystemExit(
            f"Unknown model '{name}'.\nRun `k9bench models` to list the "
            f"{len(MODELS)} registered models.")
    return MODELS[name]


def all_specs() -> List[ModelSpec]:
    return sorted(MODELS.values(), key=lambda s: (s.kind, s.family, s.name))
