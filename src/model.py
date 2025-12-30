"""GPT-2 architecture in Flax (attention, blocks, LM head)."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

from .config import ModelConfig


def gelu(x: jnp.ndarray) -> jnp.ndarray:
  """GELU activation (GPT-2 approximation)."""
  return (
    0.5
    * x
    * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3))))
  )


class CausalSelfAttention(nn.Module):
  """Multi-head causal self-attention."""

  config: ModelConfig
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
    self, x: jnp.ndarray, mask: Optional[jnp.ndarray], deterministic: bool = True
  ) -> jnp.ndarray:
    """Apply attention.

    Args:
      x: (batch, seq, n_embd)
      mask: (1, 1, seq, seq) where True allows attention
      deterministic: disables dropout when True
    """
    cfg = self.config
    bsz, seq_len, _ = x.shape

    # Compute queries, keys and values in a single projection to
    # preserve the weight layout used by OpenAI.  The projection
    # outputs a tensor of shape (batch, seq_len, 3 * n_embd).
    proj = nn.Dense(
      features=3 * cfg.n_embd, use_bias=True, dtype=self.dtype, name="c_attn"
    )
    qkv = proj(x)
    # Split along the last dimension into q, k, v.  Each has shape
    # (batch, seq_len, n_embd).
    q, k, v = jnp.split(qkv, 3, axis=-1)

    # Reshape for multi‑head attention: (batch, heads, seq_len, head_dim).
    head_dim = cfg.head_dim
    q = q.reshape(bsz, seq_len, cfg.n_head, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(bsz, seq_len, cfg.n_head, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(bsz, seq_len, cfg.n_head, head_dim).transpose(0, 2, 1, 3)

    # Scale queries to prevent large logits.
    scale = 1.0 / jnp.sqrt(head_dim)
    q = q * scale

    # Compute dot‑products between queries and keys.  The resulting
    # tensor has shape (batch, heads, seq_len, seq_len).
    attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k)

    # Apply the causal mask by setting disallowed positions to a
    # large negative value.  This relies on the mask being a boolean
    # array where ``True`` indicates that the attention weight is
    # permitted.
    if mask is not None:
      # Broadcast mask to match the attention weight shape.  The
      # incoming mask is assumed to have dimensions
      # (1, 1, seq_len, seq_len).
      attn_weights = jnp.where(mask, attn_weights, -1e10)

    # Normalize the attention weights.
    attn_probs = jax.nn.softmax(attn_weights, axis=-1)
    attn_probs = nn.Dropout(rate=cfg.attn_pdrop)(
      attn_probs, deterministic=deterministic
    )

    # Compute the weighted values.
    attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_probs, v)
    # Reshape back to (batch, seq_len, n_embd).
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, seq_len, cfg.n_embd)

    # Final linear projection and dropout.
    out_proj = nn.Dense(
      features=cfg.n_embd, use_bias=True, dtype=self.dtype, name="c_proj"
    )
    output = out_proj(attn_output)
    output = nn.Dropout(rate=cfg.resid_pdrop)(output, deterministic=deterministic)

    return output


class MLP(nn.Module):
  """Two-layer feed-forward network used in each block."""

  config: ModelConfig
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
    cfg = self.config
    inner_dim = cfg.n_inner if cfg.n_inner is not None else 4 * cfg.n_embd
    fc = nn.Dense(features=inner_dim, dtype=self.dtype, name="c_fc")
    proj = nn.Dense(features=cfg.n_embd, dtype=self.dtype, name="c_proj")

    x = fc(x)
    x = gelu(x)
    x = proj(x)
    x = nn.Dropout(rate=cfg.resid_pdrop)(x, deterministic=deterministic)
    return x


class GPT2Block(nn.Module):
  """Transformer block: LN → attn → residual → LN → MLP → residual."""

  config: ModelConfig
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
    self, x: jnp.ndarray, mask: Optional[jnp.ndarray], deterministic: bool = True
  ) -> jnp.ndarray:
    cfg = self.config
    ln1 = nn.LayerNorm(epsilon=cfg.layer_norm_epsilon, dtype=self.dtype, name="ln_1")
    ln2 = nn.LayerNorm(epsilon=cfg.layer_norm_epsilon, dtype=self.dtype, name="ln_2")

    # Attention block
    attn_out = CausalSelfAttention(cfg, dtype=self.dtype, name="attn")(
      ln1(x), mask, deterministic
    )
    x = x + attn_out

    # Feed‑forward block
    mlp_out = MLP(cfg, dtype=self.dtype, name="mlp")(ln2(x), deterministic)
    x = x + mlp_out

    return x


class GPT2LMHeadModel(nn.Module):
  """GPT-2 language model with tied input/output embeddings."""

  config: ModelConfig
  dtype: jnp.dtype = jnp.float32

  def setup(self) -> None:
    # Create a learnable token embedding table.  The same table
    # will be used for the output projection to tie the input and
    # output representations.  The variable is registered under
    # ``params`` so that it will be saved/loaded with checkpoints.
    self.wte = self.param(
      "wte",
      jax.nn.initializers.normal(stddev=0.02),
      (self.config.vocab_size, self.config.n_embd),
    )
    # Positional embeddings are learned as well.  A single
    # parameter stores all positions; indexing is handled in the
    # forward pass.
    self.wpe = self.param(
      "wpe",
      jax.nn.initializers.normal(stddev=0.02),
      (self.config.n_positions, self.config.n_embd),
    )
    # Instantiate the sequence of blocks.  Each block has its own
    # scope/name so that parameters do not collide.
    self.blocks = [
      GPT2Block(config=self.config, dtype=self.dtype, name=f"block_{i}")
      for i in range(self.config.n_layer)
    ]
    # Final layer normalisation.
    self.ln_f = nn.LayerNorm(
      epsilon=self.config.layer_norm_epsilon, dtype=self.dtype, name="ln_f"
    )

    # Define dropout modules.  Dropout layers are submodules and must be
    # created in ``setup`` rather than in the ``__call__`` method.  This
    # avoids the ``AssignSubModuleError`` raised by Flax when submodules
    # are instantiated outside of ``setup`` or a ``@compact`` function.
    # ``embd_pdrop`` controls dropout applied immediately after adding
    # positional embeddings.  Additional dropout modules in attention
    # and MLP layers are created within those modules.
    self.embd_dropout = nn.Dropout(rate=self.config.embd_pdrop)

  def __call__(self, input_ids: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
    """Compute logits over the vocabulary for each input position."""
    batch_size, seq_len = input_ids.shape
    # Lookup token and position embeddings.  Positional indices are
    # computed on the fly.
    token_embeds = jnp.take(self.wte, input_ids, axis=0)
    positions = jnp.arange(seq_len)[None, :]
    pos_embeds = jnp.take(self.wpe, positions, axis=0)
    x = token_embeds + pos_embeds
    # Dropout on embeddings.  Use the dropout module defined in setup.
    x = self.embd_dropout(x, deterministic=deterministic)
    # Prepare the causal mask once per sequence length.  The mask
    # shape is (1, 1, seq_len, seq_len) so that broadcasting over
    # batch and heads works correctly.
    # True values indicate positions that are allowed to attend.
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    causal_mask = causal_mask[None, None, :, :]

    # Iterate over blocks.  We pass the same mask to all blocks.
    for block in self.blocks:
      x = block(x, mask=causal_mask, deterministic=deterministic)

    x = self.ln_f(x)
    # Compute logits via weight tying.  We reshape wte to (n_embd,
    # vocab_size) for the matrix multiplication.
    logits = jnp.einsum("bld,vd->blv", x, self.wte)
    return logits
