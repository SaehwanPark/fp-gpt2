"""Example script for fine‑tuning the GPT‑2 model with JAX/Flax.

This script demonstrates how to perform a basic fine‑tuning of the
custom GPT‑2 model on a user‑provided text corpus.  The goal is
educational rather than to achieve state‑of‑the‑art performance.  It
illustrates data loading, tokenisation, loss computation, gradient
updates with Optax and model parameter saving.  The training loop
adheres to functional programming principles: immutable state is
passed between steps and side effects are confined to logging and
checkpoint writing.

Prior to running this script you must set a ``DATA_DIR`` environment
variable pointing at the directory containing your training text file.
You can specify the file name via the ``--dataset`` argument.  The
Hugging Face tokenizer for the model specified in ``--model-name``
will be used to encode the text.

Examples
--------

The following command fine‑tunes the model on a file called
``train.txt`` located in ``DATA_DIR`` for 10 steps with a batch size
of 2 and a sequence length of 64 tokens:

.. code:: bash

    DATA_DIR=/path/to/data python -m src.train --dataset train.txt \
      --model-name gpt2 --batch-size 2 --seq-length 64 --steps 10

This example prints the loss at each step and writes a final
checkpoint to the working directory.
"""

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
  """Compute the cross‑entropy loss between logits and targets.

  Parameters
  ----------
  logits:
      Logits output by the model of shape ``(batch, seq_len, vocab_size)``.
  targets:
      Target token ids of shape ``(batch, seq_len)``.

  Returns
  -------
  jnp.ndarray
      The mean cross‑entropy loss over the batch and sequence.
  """
  vocab_size = logits.shape[-1]
  one_hot_targets = jax.nn.one_hot(targets, num_classes=vocab_size)
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  loss = -jnp.sum(one_hot_targets * log_probs, axis=-1)
  return jnp.mean(loss)


def create_learning_rate_fn(learning_rate: float) -> optax.Schedule:
  """Create a constant learning rate schedule.

  In a real project you might use warm‑up and decay.  Here we use a
  constant schedule for simplicity.
  """
  return optax.constant_schedule(learning_rate)


def train_step(
  state: train_state.TrainState, batch_inputs: jnp.ndarray, batch_targets: jnp.ndarray
) -> Tuple[train_state.TrainState, jnp.ndarray]:
  """Perform a single optimisation step.

  A JAX transformation will jit‑compile this function so that it runs
  efficiently on accelerators.  It returns the updated training state
  and the computed loss for logging.

  Parameters
  ----------
  state:
      Current training state containing parameters and optimiser.
  batch_inputs:
      Input token ids of shape ``(batch, seq_len)``.
  batch_targets:
      Target token ids.  For language modelling this is typically the
      same as ``batch_inputs`` shifted by one position.

  Returns
  -------
  Tuple[train_state.TrainState, jnp.ndarray]
      Updated state and the loss for the batch.
  """

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
  """Generate batches of inputs and targets from a stream of token ids.

  The generator yields tuples of arrays with shapes
  ``(batch_size, seq_length)``.  Each input sequence is accompanied by
  a target sequence which is the input shifted one token to the
  left.  When the end of the token stream is reached the iterator
  wraps around.

  Parameters
  ----------
  token_ids:
      A one‑dimensional array of encoded token ids.
  batch_size:
      Number of sequences per batch.
  seq_length:
      Length of each sequence in tokens.

  Yields
  ------
  tuple of ndarray
      A pair ``(inputs, targets)`` representing a batch.
  """
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
