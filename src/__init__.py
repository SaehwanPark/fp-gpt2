"""
JAX/Flax GPT-2 implementation.

Exports core building blocks (config, model, utilities) plus example
training and inference scripts.
"""

from .config import ModelConfig, load_hf_gpt2_config
from .model import GPT2LMHeadModel
from .utils import load_dataset, prepare_tokenizer, map_hf_params

__all__ = [
  "ModelConfig",
  "load_hf_gpt2_config",
  "GPT2LMHeadModel",
  "load_dataset",
  "prepare_tokenizer",
  "map_hf_params",
]
