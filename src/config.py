"""Model configuration for the GPT-2 Flax modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelConfig:
  """GPT-2 hyperparameters used to instantiate the Flax model."""

  vocab_size: int
  n_positions: int
  n_embd: int
  n_layer: int
  n_head: int
  resid_pdrop: float
  embd_pdrop: float
  attn_pdrop: float
  n_inner: Optional[int] = None
  layer_norm_epsilon: float = 1e-5

  @property
  def head_dim(self) -> int:
    """Per-head hidden size."""
    return self.n_embd // self.n_head


def load_hf_gpt2_config(hf_config) -> ModelConfig:
  """Build ``ModelConfig`` from a Hugging Face ``GPT2Config``."""
  # Import inside the function to avoid a hard dependency when the
  # transformers package is not installed.  Importing at module
  # import time would raise an ImportError for users who only work
  # with the bare model implementation.
  from transformers import GPT2Config as _GPT2Config  # type: ignore

  if not isinstance(hf_config, _GPT2Config):
    raise TypeError(f"hf_config must be a GPT2Config, got {type(hf_config)!r}")

  return ModelConfig(
    vocab_size=hf_config.vocab_size,
    n_positions=hf_config.n_positions,
    n_embd=hf_config.n_embd,
    n_layer=hf_config.n_layer,
    n_head=hf_config.n_head,
    resid_pdrop=hf_config.resid_pdrop,
    embd_pdrop=hf_config.embd_pdrop,
    attn_pdrop=hf_config.attn_pdrop,
    n_inner=getattr(hf_config, "n_inner", None),
    layer_norm_epsilon=getattr(hf_config, "layer_norm_epsilon", 1e-5),
  )
