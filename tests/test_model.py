"""shape, utility, and functionality tests for gpt-2 flax implementation."""

import jax
import jax.numpy as jnp
import numpy as np

from src.config import ModelConfig
from src.model import GPT2Block, GPT2LMHeadModel, create_causal_mask
from src.utils import map_hf_params
from src.eval import compute_perplexity, compute_accuracy


def test_block_output_shape() -> None:
  """a transformer block should preserve the input shape."""
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
  mask = create_causal_mask(5)
  variables = block.init(rng, x, mask, deterministic=True)
  y = block.apply(variables, x, mask, deterministic=True)
  assert y.shape == x.shape


def test_model_forward_shape() -> None:
  """the language model should produce logits of shape (batch, seq, vocab)."""
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


def test_causal_mask_correctness() -> None:
  """causal mask should allow attention only to past positions."""
  seq_len = 4
  mask = create_causal_mask(seq_len)

  # shape should be (1, 1, seq, seq) for broadcasting
  assert mask.shape == (1, 1, seq_len, seq_len)

  # mask should be lower triangular
  mask_2d = mask[0, 0]
  expected = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
  assert jnp.array_equal(mask_2d, expected)

  # verify specific positions
  # position 0 can only attend to position 0
  assert mask_2d[0, 0]
  assert jnp.all(mask_2d[0, 1:] == False)

  # position 2 can attend to positions 0, 1, 2 but not 3
  assert jnp.all(mask_2d[2, :3])
  assert not mask_2d[2, 3]


def test_gradient_flow() -> None:
  """gradients should flow through the model (no disconnected params)."""
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
  input_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
  variables = model.init(rng, input_ids, deterministic=True)

  def loss_fn(params):
    logits = model.apply({"params": params}, input_ids, deterministic=True)
    # simple loss: sum of all logits
    return jnp.sum(logits)

  grads = jax.grad(loss_fn)(variables["params"])

  # check that gradients exist for key parameters
  assert "wte" in grads
  assert "wpe" in grads
  assert "block_0" in grads  # blocks are at top level, not nested
  assert "ln_f" in grads

  # check that gradients are non-zero (model is actually used)
  # note: nn.Embed creates {"embedding": array}, not flat array
  assert jnp.sum(jnp.abs(grads["wte"]["embedding"])) > 0
  assert jnp.sum(jnp.abs(grads["wpe"]["embedding"])) > 0


def test_deterministic_mode() -> None:
  """deterministic=True should produce identical outputs."""
  cfg = ModelConfig(
    vocab_size=50,
    n_positions=16,
    n_embd=32,
    n_layer=1,
    n_head=4,
    resid_pdrop=0.1,  # non-zero dropout
    embd_pdrop=0.1,
    attn_pdrop=0.1,
  )
  model = GPT2LMHeadModel(config=cfg)
  rng = jax.random.PRNGKey(0)
  input_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
  variables = model.init(rng, input_ids, deterministic=True)

  # two forward passes with deterministic=True should be identical
  logits1 = model.apply(variables, input_ids, deterministic=True)
  logits2 = model.apply(variables, input_ids, deterministic=True)
  assert jnp.allclose(logits1, logits2)


def test_stochastic_mode() -> None:
  """deterministic=False with dropout should produce different outputs."""
  cfg = ModelConfig(
    vocab_size=50,
    n_positions=16,
    n_embd=32,
    n_layer=2,
    n_head=4,
    resid_pdrop=0.2,  # significant dropout
    embd_pdrop=0.2,
    attn_pdrop=0.2,
  )
  model = GPT2LMHeadModel(config=cfg)
  rng = jax.random.PRNGKey(0)
  input_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)

  # initialize with different rngs
  rng1, rng2 = jax.random.split(rng)
  variables1 = model.init(rng1, input_ids, deterministic=False)
  variables2 = model.init(rng2, input_ids, deterministic=False)

  # forward passes with deterministic=False should differ
  logits1 = model.apply(
    variables1, input_ids, deterministic=False, rngs={"dropout": rng1}
  )
  logits2 = model.apply(
    variables2, input_ids, deterministic=False, rngs={"dropout": rng2}
  )

  # outputs should be different due to dropout
  assert not jnp.allclose(logits1, logits2)


def test_map_hf_params_structure() -> None:
  """the parameter mapping should contain expected top-level keys."""
  # fake hugging face parameter structure for testing
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

  # check structure - blocks are at top level
  assert "wte" in params_out and "wpe" in params_out
  assert "block_0" in params_out  # not nested under "blocks"
  assert "ln_f" in params_out

  # check embedding structure (nn.Embed creates {"embedding": array})
  assert "embedding" in params_out["wte"]
  assert "embedding" in params_out["wpe"]

  # check that shapes are preserved
  assert params_out["wte"]["embedding"].shape == (10, 8)
  assert params_out["wpe"]["embedding"].shape == (4, 8)


def test_perplexity_computation() -> None:
  """perplexity should be finite and positive."""
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

  # create dummy data
  inputs = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
  targets = jnp.array([[2, 3, 4, 5]], dtype=jnp.int32)

  variables = model.init(rng, inputs, deterministic=True)
  params = variables["params"]

  ppl = compute_perplexity(model, params, inputs, targets)

  # perplexity should be positive and finite
  assert ppl > 0
  assert np.isfinite(ppl)

  # for random model, perplexity should be roughly vocab_size
  # (uniform distribution over vocab)
  assert 10 < ppl < 100  # reasonable range for untrained model


def test_accuracy_computation() -> None:
  """accuracy should be between 0 and 1."""
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

  inputs = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
  targets = jnp.array([[2, 3, 4, 5]], dtype=jnp.int32)

  variables = model.init(rng, inputs, deterministic=True)
  params = variables["params"]

  acc = compute_accuracy(model, params, inputs, targets)

  # accuracy should be in [0, 1]
  assert 0.0 <= acc <= 1.0

  # for random model with vocab=50, expect ~2% accuracy
  assert 0.0 <= acc <= 0.5


def test_parameter_count() -> None:
  """verify parameter count matches expected gpt-2 architecture."""
  # gpt2-small config
  cfg = ModelConfig(
    vocab_size=50257,
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
  )
  model = GPT2LMHeadModel(config=cfg)
  rng = jax.random.PRNGKey(0)
  dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
  variables = model.init(rng, dummy_input, deterministic=True)

  # count parameters
  params = variables["params"]
  total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))

  # gpt2-small has approximately 124m parameters
  # allow some tolerance for implementation differences
  expected_params = 124_000_000
  tolerance = 5_000_000

  assert abs(total_params - expected_params) < tolerance, (
    f"expected ~{expected_params:,} params, got {total_params:,}"
  )
