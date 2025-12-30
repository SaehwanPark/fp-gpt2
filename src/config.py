"""Model configuration utilities.

This module defines a lightweight configuration data class
(`ModelConfig`) that captures the key hyperparameters of the GPT‑2
architecture.  The intent is to provide an immutable record of
model parameters that can be passed into Flax modules.  A helper
function is provided to create a `ModelConfig` instance from a
Hugging Face ``GPT2Config`` object, which makes it easy to
instantiate the custom Flax model with the same settings as a
pre‑trained model.

Examples
--------

>>> from transformers import GPT2Config
>>> from src.config import load_hf_gpt2_config
>>> hf_config = GPT2Config.from_pretrained('gpt2')
>>> model_cfg = load_hf_gpt2_config(hf_config)
>>> model_cfg.n_layer
12

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelConfig:
  """Container for the key hyperparameters of a GPT‑2 model.

  Attributes
  ----------
  vocab_size:
      Number of distinct tokens that can be embedded.
  n_positions:
      Maximum sequence length in tokens.  This value defines the
      number of distinct positional embeddings.
  n_embd:
      Dimensionality of the token and positional embeddings.
  n_layer:
      Number of transformer blocks.
  n_head:
      Number of attention heads per block.
  resid_pdrop:
      Dropout probability applied to residual connections.
  embd_pdrop:
      Dropout probability applied immediately after adding
      positional embeddings.
  attn_pdrop:
      Dropout probability applied to attention weights.
  n_inner:
      Dimensionality of the inner feed‑forward layer.  If ``None``
      (the default), a value of ``4 * n_embd`` will be used.
  layer_norm_epsilon:
      Small constant added to the variance in layer normalization to
      prevent division by zero.
  """

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
    """Compute the dimensionality of a single attention head.

    Returns
    -------
    int
        The number of features per attention head.
    """
    return self.n_embd // self.n_head


def load_hf_gpt2_config(hf_config) -> ModelConfig:
  """Convert a Hugging Face ``GPT2Config`` into a ``ModelConfig``.

  Parameters
  ----------
  hf_config:
      A Hugging Face GPT‑2 configuration instance from which to
      extract hyperparameters.

  Returns
  -------
  ModelConfig
      A new configuration containing only the parameters required
      for the custom Flax implementation.

  Notes
  -----
  Not all attributes of the Hugging Face configuration are used by
  this implementation.  Only those pertaining to the architecture
  layout and dropout rates are transferred.  If you modify the
  Hugging Face configuration, ensure that the corresponding fields
  exist on ``hf_config`` or provide fallback values.
  """
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
