"""K9Bench evaluation suite."""

from . import models  # noqa: F401  (registers the built-in models)
from .registry import MODELS, ModelSpec, all_specs, get, register

__version__ = "1.0.0"
__all__ = ["MODELS", "ModelSpec", "all_specs", "get", "register"]
