"""Minimal fine-tuning loop for the custom GPT-2 Flax model."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable, Tuple

import dotenv
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from .config import load_hf_gpt2_config
from .model import GPT2LMHeadModel
from .utils import load_dataset, prepare_tokenizer


logger = logging.getLogger(__name__)


def cross_entropy_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
  """Mean cross-entropy over batch and sequence."""
  vocab_size = logits.shape[-1]
  one_hot_targets = jax.nn.one_hot(targets, num_classes=vocab_size)
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  loss = -jnp.sum(one_hot_targets * log_probs, axis=-1)
  return jnp.mean(loss)


def create_learning_rate_fn(learning_rate: float) -> optax.Schedule:
  """Constant learning-rate schedule."""
  return optax.constant_schedule(learning_rate)


def train_step(
  state: train_state.TrainState, batch_inputs: jnp.ndarray, batch_targets: jnp.ndarray
) -> Tuple[train_state.TrainState, jnp.ndarray]:
  """One optimization step (forward → loss → grad → update)."""

  def loss_fn(params: Any) -> jnp.ndarray:
    logits = state.apply_fn({"params": params}, batch_inputs, deterministic=False)
    return cross_entropy_loss(logits, batch_targets)

  grad_fn = jax.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  new_state = state.apply_gradients(grads=grads)
  return new_state, loss


def iterate_batches(
  token_ids: np.ndarray, batch_size: int, seq_length: int
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
  """Yield infinite (inputs, targets) batches from a token-id stream."""
  data_len = len(token_ids)
  # Starting offset for reading sequences; this will advance
  # sequentially through the corpus.
  offset = 0
  while True:
    inputs = np.zeros((batch_size, seq_length), dtype=np.int32)
    targets = np.zeros((batch_size, seq_length), dtype=np.int32)
    for i in range(batch_size):
      # Ensure there is enough room to take a sequence of length + 1
      if offset + seq_length + 1 >= data_len:
        offset = 0
      seq = token_ids[offset : offset + seq_length + 1]
      inputs[i] = seq[:-1]
      targets[i] = seq[1:]
      offset += seq_length
    yield inputs, targets


def main() -> None:
  import argparse

  dotenv.load_dotenv()
  parser = argparse.ArgumentParser(description="Fine‑tune a GPT‑2 model using JAX/Flax")
  parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Name of the text file in DATA_DIR to use for training",
  )
  parser.add_argument(
    "--model-name",
    type=str,
    default="gpt2",
    help="Pre‑trained model name to load configuration and tokenizer from",
  )
  parser.add_argument(
    "--batch-size", type=int, default=2, help="Number of sequences per batch"
  )
  parser.add_argument(
    "--seq-length", type=int, default=64, help="Sequence length in tokens"
  )
  parser.add_argument(
    "--steps", type=int, default=10, help="Number of optimisation steps"
  )
  parser.add_argument(
    "--learning-rate", type=float, default=3e-4, help="Initial learning rate"
  )
  parser.add_argument(
    "--log-every", type=int, default=1, help="How often to log the loss"
  )
  args = parser.parse_args()

  # Resolve the data directory from the environment.
  data_dir_str = os.environ.get("DATA_DIR")
  if not data_dir_str:
    raise RuntimeError("DATA_DIR environment variable must be set")
  data_dir = Path(data_dir_str)

  # Prepare tokenizer and load dataset.
  tokenizer = prepare_tokenizer(args.model_name)
  lines = load_dataset(data_dir, args.dataset)
  # Concatenate all lines together with the EOS token in between.
  text = "\n".join(lines)
  encoded = tokenizer.encode(text)
  token_ids = np.array(encoded, dtype=np.int32)
  logger.info("Encoded dataset into %d tokens", len(token_ids))

  # Load configuration from Hugging Face and create custom model config.
  from transformers import GPT2Config  # type: ignore

  hf_cfg = GPT2Config.from_pretrained(args.model_name)
  model_cfg = load_hf_gpt2_config(hf_cfg)

  # Instantiate model and obtain initial parameters.
  model = GPT2LMHeadModel(config=model_cfg)
  rng = jax.random.PRNGKey(0)
  dummy_input = jnp.ones((args.batch_size, args.seq_length), dtype=jnp.int32)
  variables = model.init(rng, dummy_input, deterministic=False)
  params = variables["params"]

  # Setup optimiser and training state.
  tx = optax.adamw(learning_rate=create_learning_rate_fn(args.learning_rate))
  state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

  # Prepare the batch iterator.
  batch_iter = iterate_batches(token_ids, args.batch_size, args.seq_length)

  # jit compile the train step
  jit_train_step = jax.jit(train_step)

  for step in range(1, args.steps + 1):
    batch_inputs, batch_targets = next(batch_iter)
    state, loss = jit_train_step(
      state, jnp.array(batch_inputs), jnp.array(batch_targets)
    )
    if step % args.log_every == 0:
      logger.info("Step %d, loss %.4f", step, float(loss))

  # Save final parameters
  save_path = Path("gpt2_finetuned.npz")
  np.savez(save_path, **jax.tree_util.tree_map(lambda x: np.array(x), state.params))
  logger.info("Finished training; model parameters saved to %s", save_path)


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
