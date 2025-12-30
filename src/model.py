"""gpt-2 architecture in flax (attention, blocks, lm head)."""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from flax import linen as nn

from .config import ModelConfig


def gelu(x: jnp.ndarray) -> jnp.ndarray:
  """gelu activation (gpt-2 approximation).

  uses the tanh approximation from the original gelu paper:
  https://arxiv.org/abs/1606.08415

  gelu(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])
  """
  return (
    0.5
    * x
    * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3))))
  )


def create_causal_mask(seq_len: int) -> jnp.ndarray:
  """precompute causal attention mask: (1, 1, seq, seq).

  lower triangular matrix where mask[i, j] = (i >= j), ensuring
  position i can only attend to positions 0..i (not future tokens).

  shape is (1, 1, seq, seq) for broadcasting over (batch, heads, seq, seq).
  """
  return jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))[None, None, :, :]


class CausalSelfAttention(nn.Module):
  """multi-head causal self-attention.

  implements scaled dot-product attention with causal masking:
  attention(q, k, v) = softmax(qk^t / √d_k) v

  preserves openai's weight layout: single projection for q,k,v.
  """

  config: ModelConfig
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
    self, x: jnp.ndarray, mask: Optional[jnp.ndarray], deterministic: bool = True
  ) -> jnp.ndarray:
    """apply causal self-attention.

    args:
      x: (batch, seq, n_embd) input embeddings
      mask: (1, 1, seq, seq) where true allows attention
      deterministic: disables dropout when true

    returns:
      (batch, seq, n_embd) contextualized representations
    """
    cfg = self.config
    bsz, seq_len, _ = x.shape

    # compute q, k, v in a single projection to preserve openai's layout.
    # outputs (batch, seq, 3 * n_embd) which splits into [q, k, v].
    # each of q, k, v has shape (batch, seq, n_embd).
    proj = nn.Dense(
      features=3 * cfg.n_embd, use_bias=True, dtype=self.dtype, name="c_attn"
    )
    qkv = proj(x)
    q, k, v = jnp.split(qkv, 3, axis=-1)

    # reshape for multi-head attention: (batch, n_head, seq, head_dim).
    # this enables parallel computation across attention heads.
    head_dim = cfg.head_dim
    q = q.reshape(bsz, seq_len, cfg.n_head, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(bsz, seq_len, cfg.n_head, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(bsz, seq_len, cfg.n_head, head_dim).transpose(0, 2, 1, 3)

    # scaled dot-product attention: (batch, heads, seq, seq)
    # scaling by 1/√d_k prevents dot products from growing too large
    attn_scores = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)

    # apply causal mask: set future positions to -inf before softmax
    if mask is not None:
      # broadcast mask from (1, 1, seq, seq) to (batch, heads, seq, seq)
      attn_scores = jnp.where(mask, attn_scores, -1e10)

    # normalize attention weights and apply dropout
    attn_weights = nn.softmax(attn_scores, axis=-1)
    attn_weights = nn.Dropout(rate=cfg.attn_pdrop, deterministic=deterministic)(
      attn_weights
    )

    # weighted sum of values: (batch, heads, seq, head_dim)
    attn_output = attn_weights @ v

    # merge heads: (batch, seq, n_embd)
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, seq_len, cfg.n_embd)

    # output projection
    out_proj = nn.Dense(
      features=cfg.n_embd, use_bias=True, dtype=self.dtype, name="c_proj"
    )
    output = out_proj(attn_output)
    output = nn.Dropout(rate=cfg.resid_pdrop, deterministic=deterministic)(output)

    return output


class MLP(nn.Module):
  """position-wise feedforward network.

  two-layer network with gelu activation: ffn(x) = gelu(xw1)w2
  expands to n_inner (default 4 * n_embd), then contracts back.
  """

  config: ModelConfig
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
    """apply position-wise feedforward.

    args:
      x: (batch, seq, n_embd)
      deterministic: disables dropout when true

    returns:
      (batch, seq, n_embd)
    """
    cfg = self.config
    # expand: (batch, seq, n_embd) -> (batch, seq, n_inner)
    expand = nn.Dense(
      features=cfg.ffn_dim, use_bias=True, dtype=self.dtype, name="c_fc"
    )
    hidden = gelu(expand(x))

    # contract: (batch, seq, n_inner) -> (batch, seq, n_embd)
    contract = nn.Dense(
      features=cfg.n_embd, use_bias=True, dtype=self.dtype, name="c_proj"
    )
    output = contract(hidden)
    output = nn.Dropout(rate=cfg.resid_pdrop, deterministic=deterministic)(output)

    return output


class GPT2Block(nn.Module):
  """single transformer block: layernorm -> attention -> layernorm -> mlp.

  uses pre-normalization (layernorm before attention/mlp) as in gpt-2.
  residual connections allow gradient flow through deep networks.
  """

  config: ModelConfig
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
    self, x: jnp.ndarray, mask: Optional[jnp.ndarray], deterministic: bool = True
  ) -> jnp.ndarray:
    """apply transformer block.

    args:
      x: (batch, seq, n_embd)
      mask: (1, 1, seq, seq) causal attention mask
      deterministic: disables dropout when true

    returns:
      (batch, seq, n_embd)
    """
    cfg = self.config

    # attention sub-layer with residual connection
    ln1 = nn.LayerNorm(epsilon=cfg.layer_norm_epsilon, dtype=self.dtype, name="ln_1")
    attn = CausalSelfAttention(config=cfg, dtype=self.dtype, name="attn")
    x = x + attn(ln1(x), mask, deterministic=deterministic)

    # mlp sub-layer with residual connection
    ln2 = nn.LayerNorm(epsilon=cfg.layer_norm_epsilon, dtype=self.dtype, name="ln_2")
    mlp = MLP(config=cfg, dtype=self.dtype, name="mlp")
    x = x + mlp(ln2(x), deterministic=deterministic)

    return x


class GPT2LMHeadModel(nn.Module):
  """gpt-2 language model: embeddings + blocks + lm head.

  architecture:
    1. token + position embeddings
    2. n_layer transformer blocks
    3. final layer norm
    4. project to vocabulary (tied with token embeddings)
  """

  config: ModelConfig
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, input_ids: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
    """forward pass producing next-token logits.

    args:
      input_ids: (batch, seq) token indices
      deterministic: disables dropout when true

    returns:
      (batch, seq, vocab_size) logits for next token prediction
    """
    cfg = self.config
    bsz, seq_len = input_ids.shape

    # embeddings: token + position
    wte = nn.Embed(
      num_embeddings=cfg.vocab_size,
      features=cfg.n_embd,
      dtype=self.dtype,
      name="wte",
    )
    wpe = nn.Embed(
      num_embeddings=cfg.n_positions,
      features=cfg.n_embd,
      dtype=self.dtype,
      name="wpe",
    )

    token_emb = wte(input_ids)
    pos_ids = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    pos_emb = wpe(pos_ids)

    x = token_emb + pos_emb
    x = nn.Dropout(rate=cfg.embd_pdrop, deterministic=deterministic)(x)

    # precompute causal mask once for all blocks
    mask = create_causal_mask(seq_len)

    # apply transformer blocks sequentially
    for i in range(cfg.n_layer):
      block = GPT2Block(config=cfg, dtype=self.dtype, name=f"block_{i}")
      x = block(x, mask, deterministic=deterministic)

    # final layer norm
    ln_f = nn.LayerNorm(epsilon=cfg.layer_norm_epsilon, dtype=self.dtype, name="ln_f")
    x = ln_f(x)

    # project to vocabulary (weight tying with token embeddings)
    logits = x @ wte.embedding.T

    return logits
