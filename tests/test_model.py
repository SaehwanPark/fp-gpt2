"""Unit tests for the GPT‑2 Flax implementation.

These tests verify that the shapes of the tensors flowing through
individual components match expectations and that helper functions
behave sensibly.  They can be run with ``pytest``.
"""

import jax
import jax.numpy as jnp

from src.config import ModelConfig
from src.model import GPT2Block, GPT2LMHeadModel
from src.utils import map_hf_params


def test_block_output_shape() -> None:
  """A transformer block should preserve the input shape."""
  cfg = ModelConfig(
    vocab_size=100,
    n_positions=16,
    n_embd=32,
    n_layer=2,
    n_head=4,
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
  )
  block = GPT2Block(config=cfg)
  rng = jax.random.PRNGKey(0)
  x = jax.random.normal(rng, (2, 5, cfg.n_embd))
  mask = jnp.tril(jnp.ones((5, 5), dtype=jnp.bool_))[None, None, :, :]
  variables = block.init(rng, x, mask, deterministic=True)
  y = block.apply(variables, x, mask, deterministic=True)
  assert y.shape == x.shape


def test_model_forward_shape() -> None:
  """The language model should produce logits of shape (batch, seq, vocab)."""
  cfg = ModelConfig(
    vocab_size=50,
    n_positions=16,
    n_embd=32,
    n_layer=1,
    n_head=4,
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
  )
  model = GPT2LMHeadModel(config=cfg)
  rng = jax.random.PRNGKey(0)
  input_ids = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
  variables = model.init(rng, input_ids, deterministic=True)
  logits = model.apply(variables, input_ids, deterministic=True)
  assert logits.shape == (1, 4, cfg.vocab_size)


def test_map_hf_params_structure() -> None:
  """The parameter mapping should contain expected top‑level keys."""
  # Fake Hugging Face parameter structure for testing
  hf_params = {
    "transformer": {
      "wte": {"embedding": jnp.zeros((10, 8))},
      "wpe": {"embedding": jnp.zeros((4, 8))},
      "h": [
        {
          "ln_1": {"scale": jnp.ones(8), "bias": jnp.zeros(8)},
          "ln_2": {"scale": jnp.ones(8), "bias": jnp.zeros(8)},
          "attn": {
            "c_attn": {"kernel": jnp.zeros((8, 24)), "bias": jnp.zeros(24)},
            "c_proj": {"kernel": jnp.zeros((8, 8)), "bias": jnp.zeros(8)},
          },
          "mlp": {
            "c_fc": {"kernel": jnp.zeros((8, 32)), "bias": jnp.zeros(32)},
            "c_proj": {"kernel": jnp.zeros((32, 8)), "bias": jnp.zeros(8)},
          },
        }
      ],
      "ln_f": {"scale": jnp.ones(8), "bias": jnp.zeros(8)},
    }
  }
  cfg = ModelConfig(
    vocab_size=10,
    n_positions=4,
    n_embd=8,
    n_layer=1,
    n_head=2,
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
  )
  params_out = map_hf_params(hf_params, cfg)
  assert "wte" in params_out and "wpe" in params_out
  assert "blocks" in params_out
  assert "block_0" in params_out["blocks"]
  assert "ln_f" in params_out
