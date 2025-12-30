"""run text generation with the custom gpt-2 flax model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import dotenv
import jax
import jax.numpy as jnp
import numpy as np

from .config import load_hf_gpt2_config
from .model import GPT2LMHeadModel
from .utils import map_hf_params, prepare_tokenizer
from transformers import FlaxGPT2LMHeadModel

logger = logging.getLogger(__name__)


def load_pretrained_parameters(model_name: str, config) -> dict[str, Any]:
  """load parameters from a hugging face gpt-2 model.

  retrieves pre-trained weights and maps them to the custom flax
  implementation's parameter structure.

  parameters
  ----------
  model_name:
      name of the pre-trained model, e.g. ``'gpt2'``.
  config:
      the custom :class:`~src.config.ModelConfig` describing the
      expected architecture.

  returns
  -------
  dict
      parameter dictionary ready for ``GPT2LMHeadModel``.
  """
  logger.info("loading hugging face model %s", model_name)
  hf_model = FlaxGPT2LMHeadModel.from_pretrained(model_name)
  hf_params = hf_model.params
  mapped = map_hf_params(hf_params, config, hf_key_prefix="transformer")
  return mapped


def sample_token(
  logits: jnp.ndarray,
  rng: jax.random.PRNGKey,
  temperature: float = 1.0,
  top_k: int | None = None,
  top_p: float | None = None,
) -> int:
  """sample next token from logits with various strategies.

  implements temperature scaling, top-k filtering, and nucleus (top-p) sampling.

  parameters
  ----------
  logits:
      (vocab_size,) unnormalized log probabilities
  rng:
      jax random key for sampling
  temperature:
      sampling temperature (>1 more random, <1 more deterministic)
  top_k:
      if set, only sample from top k tokens
  top_p:
      if set, sample from smallest set with cumulative prob >= p

  returns
  -------
  int
      sampled token id
  """
  # apply temperature
  if temperature != 1.0:
    logits = logits / temperature

  # top-k filtering
  if top_k is not None:
    top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
    logits = jnp.full_like(logits, -1e10)
    logits = logits.at[top_k_indices].set(top_k_logits)

  # nucleus (top-p) filtering
  if top_p is not None:
    sorted_indices = jnp.argsort(logits, descending=True)
    sorted_logits = logits[sorted_indices]
    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits))

    # find cutoff: first position where cumsum exceeds p
    cutoff_idx = jnp.searchsorted(cumulative_probs, top_p, side="right")

    # keep only tokens up to cutoff
    mask = jnp.arange(len(logits)) <= cutoff_idx
    nucleus_indices = jnp.where(mask, sorted_indices, -1)

    # zero out logits not in nucleus
    logits = jnp.where(
      jnp.isin(jnp.arange(len(logits)), nucleus_indices), logits, -1e10
    )

  # sample from distribution
  return int(jax.random.categorical(rng, logits))


def generate_text_greedy(
  model: GPT2LMHeadModel,
  params: dict[str, Any],
  tokenizer,
  prompt: str,
  max_length: int,
) -> str:
  """greedy decode up to ``max_length`` tokens (including prompt).

  uses python loop for simplicity since sequence length varies.
  forward pass is jit-compiled for efficiency.
  """
  input_ids = list(tokenizer.encode(prompt))

  # jit-compile the forward pass once
  @jax.jit
  def get_next_token_logits(ids_array: jnp.ndarray) -> jnp.ndarray:
    """compute logits for next token given current sequence."""
    logits = model.apply({"params": params}, ids_array, deterministic=True)
    return logits[0, -1]  # last position logits

  # generate tokens one at a time
  while len(input_ids) < max_length:
    # prepare input: (1, seq_len)
    ids_array = jnp.array([input_ids], dtype=jnp.int32)
    next_token_logits = get_next_token_logits(ids_array)
    # convert to int outside jit boundary
    next_token = int(jnp.argmax(next_token_logits))
    input_ids.append(next_token)

    # stop if eos token
    if next_token == tokenizer.eos_token_id:
      break

  return tokenizer.decode(input_ids)


def generate_text_sampling(
  model: GPT2LMHeadModel,
  params: dict[str, Any],
  tokenizer,
  prompt: str,
  max_length: int,
  rng: jax.random.PRNGKey,
  temperature: float = 1.0,
  top_k: int | None = None,
  top_p: float | None = None,
) -> str:
  """sample tokens with temperature, top-k, or nucleus sampling.

  uses python loop with jit-compiled sampling function for efficiency.
  """
  input_ids = list(tokenizer.encode(prompt))

  # jit-compile the sampling logic
  @jax.jit
  def get_next_token_logits(ids_array: jnp.ndarray) -> jnp.ndarray:
    """get logits for next token given current sequence."""
    logits = model.apply({"params": params}, ids_array, deterministic=True)
    return logits[0, -1]  # last position logits

  # generate tokens with rng threading
  for step in range(max_length - len(input_ids)):
    # split rng for this step
    rng, step_rng = jax.random.split(rng)

    # prepare input
    ids_array = jnp.array([input_ids], dtype=jnp.int32)
    next_token_logits = get_next_token_logits(ids_array)
    # sample outside jit boundary
    next_token = sample_token(next_token_logits, step_rng, temperature, top_k, top_p)
    input_ids.append(next_token)

    # stop if eos token
    if next_token == tokenizer.eos_token_id:
      break

  return tokenizer.decode(input_ids)


def main() -> None:
  import argparse

  dotenv.load_dotenv()
  parser = argparse.ArgumentParser(description="run inference with a gpt-2 flax model")
  parser.add_argument("--prompt", type=str, required=True, help="initial text prompt")
  parser.add_argument(
    "--max-length", type=int, default=20, help="maximum length including prompt"
  )
  parser.add_argument(
    "--model-name", type=str, default="gpt2", help="hugging face model name"
  )
  parser.add_argument(
    "--checkpoint",
    type=Path,
    default=None,
    help="path to .npz file with fine-tuned parameters",
  )
  parser.add_argument(
    "--sampling",
    type=str,
    choices=["greedy", "sample"],
    default="greedy",
    help="decoding strategy",
  )
  parser.add_argument(
    "--temperature", type=float, default=1.0, help="sampling temperature"
  )
  parser.add_argument(
    "--top-k", type=int, default=None, help="top-k filtering (optional)"
  )
  parser.add_argument(
    "--top-p", type=float, default=None, help="nucleus sampling threshold (optional)"
  )
  parser.add_argument("--seed", type=int, default=0, help="random seed for sampling")
  args = parser.parse_args()

  # load tokenizer and config
  tokenizer = prepare_tokenizer(args.model_name)
  from transformers import GPT2Config  # type: ignore

  hf_cfg = GPT2Config.from_pretrained(args.model_name)
  cfg = load_hf_gpt2_config(hf_cfg)

  # instantiate model
  model = GPT2LMHeadModel(config=cfg)
  rng = jax.random.PRNGKey(args.seed)
  dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
  variables = model.init(rng, dummy_input, deterministic=True)
  params = variables["params"]

  # load parameters
  if args.checkpoint is not None:
    logger.info("loading parameters from checkpoint %s", args.checkpoint)
    npz = np.load(args.checkpoint, allow_pickle=True)
    params = jax.tree_util.tree_unflatten(
      jax.tree_util.tree_structure(params), [npz[key] for key in sorted(npz.files)]
    )
  else:
    params = load_pretrained_parameters(args.model_name, cfg)

  # generate text
  if args.sampling == "greedy":
    generated = generate_text_greedy(
      model, params, tokenizer, args.prompt, args.max_length
    )
  else:
    rng = jax.random.PRNGKey(args.seed)
    generated = generate_text_sampling(
      model,
      params,
      tokenizer,
      args.prompt,
      args.max_length,
      rng,
      args.temperature,
      args.top_k,
      args.top_p,
    )

  print(generated)


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
