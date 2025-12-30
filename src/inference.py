"""Run text generation with the custom GPT-2 Flax model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import dotenv
import jax
import jax.numpy as jnp
import numpy as np

from .config import load_hf_gpt2_config
from .model import GPT2LMHeadModel
from .utils import map_hf_params, prepare_tokenizer
from transformers import FlaxGPT2LMHeadModel

logger = logging.getLogger(__name__)


def load_pretrained_parameters(model_name: str, config) -> Dict[str, Any]:
  """Load parameters from a Hugging Face GPT‑2 model.

  This function instantiates a Flax GPT‑2 model from the
  transformers library, retrieves its parameters and maps them to the
  structure expected by the custom Flax implementation.

  Parameters
  ----------
  model_name:
      Name of the pre‑trained model to load, e.g. ``'gpt2'``.
  config:
      The custom :class:`~src.config.ModelConfig` describing the
      expected architecture.  The Hugging Face configuration is
      derived from the same model name to ensure compatibility.

  Returns
  -------
  dict
      A parameter dictionary ready to be passed into
      ``GPT2LMHeadModel``.  The returned dictionary matches the
      structure of the ``params`` collection used in the custom
      model.
  """

  logger.info("Loading Hugging Face model %s", model_name)
  hf_model = FlaxGPT2LMHeadModel.from_pretrained(model_name)
  hf_params = hf_model.params
  mapped = map_hf_params(hf_params, config, hf_key_prefix="transformer")
  return mapped


def generate_text(
  model: GPT2LMHeadModel,
  params: Dict[str, Any],
  tokenizer,
  prompt: str,
  max_length: int,
) -> str:
  """Greedy decode up to ``max_length`` tokens (including prompt)."""
  # Encode the prompt into token ids.
  input_ids = tokenizer.encode(prompt)
  input_ids = list(input_ids)
  for _ in range(max_length - len(input_ids)):
    # Create array and run model.
    arr = jnp.array([input_ids], dtype=jnp.int32)
    logits = model.apply({"params": params}, arr, deterministic=True)
    next_token_logits = logits[0, len(input_ids) - 1]
    next_token_id = int(jnp.argmax(next_token_logits))
    input_ids.append(next_token_id)
    # Stop if EOS token encountered.
    if next_token_id == tokenizer.eos_token_id:
      break
  return tokenizer.decode(input_ids)


def main() -> None:
  import argparse

  dotenv.load_dotenv()
  parser = argparse.ArgumentParser(description="Run inference with a GPT‑2 Flax model")
  parser.add_argument("--prompt", type=str, required=True, help="Initial text prompt")
  parser.add_argument(
    "--max-length", type=int, default=20, help="Maximum length including prompt"
  )
  parser.add_argument(
    "--model-name", type=str, default="gpt2", help="Hugging Face model name to load"
  )
  parser.add_argument(
    "--checkpoint",
    type=Path,
    default=None,
    help="Path to a .npz file with fine‑tuned parameters",
  )
  args = parser.parse_args()

  tokenizer = prepare_tokenizer(args.model_name)
  from transformers import GPT2Config  # type: ignore

  hf_cfg = GPT2Config.from_pretrained(args.model_name)
  cfg = load_hf_gpt2_config(hf_cfg)
  model = GPT2LMHeadModel(config=cfg)
  # Initialise params to get the correct tree structure
  rng = jax.random.PRNGKey(0)
  dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
  variables = model.init(rng, dummy_input, deterministic=True)
  params = variables["params"]

  if args.checkpoint is not None:
    logger.info("Loading parameters from checkpoint %s", args.checkpoint)
    npz = np.load(args.checkpoint, allow_pickle=True)
    # Flatten param dict for assignment; rely on same structure
    params = jax.tree_util.tree_unflatten(
      jax.tree_util.tree_structure(params), [npz[key] for key in sorted(npz.files)]
    )
  else:
    params = load_pretrained_parameters(args.model_name, cfg)

  generated = generate_text(model, params, tokenizer, args.prompt, args.max_length)
  print(generated)


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
