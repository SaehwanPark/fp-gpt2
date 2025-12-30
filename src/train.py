"""example fine-tuning script with functional training loop."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterator

import dotenv
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from .config import load_hf_gpt2_config
from .model import GPT2LMHeadModel
from .utils import load_dataset, map_hf_params, prepare_tokenizer
from transformers import FlaxGPT2LMHeadModel

logger = logging.getLogger(__name__)


def create_learning_rate_fn(base_lr: float, warmup_steps: int = 100) -> Any:
  """create learning rate schedule with linear warmup.

  pure function returning another function (closure over hyperparameters).
  warmup helps stabilize training in early steps.
  """

  def schedule_fn(step):
    warmup_factor = jnp.minimum(step / warmup_steps, 1.0)
    return base_lr * warmup_factor

  return schedule_fn


def compute_loss(
  params: dict, apply_fn, inputs: jnp.ndarray, targets: jnp.ndarray
) -> jnp.ndarray:
  """compute cross-entropy loss for next-token prediction.

  pure function: given params and data, returns scalar loss.
  """
  logits = apply_fn({"params": params}, inputs, deterministic=False)
  # flatten: (batch, seq, vocab) -> (batch * seq, vocab)
  logits_flat = logits.reshape(-1, logits.shape[-1])
  targets_flat = targets.reshape(-1)

  # cross-entropy: -log p(target)
  loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, targets_flat)
  return jnp.mean(loss)


def train_step(
  state: train_state.TrainState, inputs: jnp.ndarray, targets: jnp.ndarray
) -> tuple[train_state.TrainState, jnp.ndarray]:
  """single training step: compute loss, gradients, update params.

  pure function (modulo rng state in dropout).
  returns new state and loss for logging.
  """
  loss, grads = jax.value_and_grad(compute_loss)(
    state.params, state.apply_fn, inputs, targets
  )
  state = state.apply_gradients(grads=grads)
  return state, loss


def iterate_batches(
  token_ids: np.ndarray, batch_size: int, seq_length: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
  """infinite generator yielding (inputs, targets) batches.

  targets are inputs shifted by one position (next-token prediction).
  wraps around when reaching end of dataset.
  """
  n_tokens = len(token_ids)
  idx = 0

  while True:
    batch_inputs = []
    batch_targets = []

    for _ in range(batch_size):
      # ensure we have seq_length + 1 tokens for input + target
      if idx + seq_length + 1 > n_tokens:
        idx = 0  # wrap around

      chunk = token_ids[idx : idx + seq_length + 1]
      batch_inputs.append(chunk[:-1])
      batch_targets.append(chunk[1:])
      idx += seq_length

    yield np.stack(batch_inputs), np.stack(batch_targets)


def train_with_scan(
  state: train_state.TrainState,
  batches: list[tuple[jnp.ndarray, jnp.ndarray]],
  log_every: int = 10,
) -> tuple[train_state.TrainState, list[float]]:
  """functional training loop using jax.lax.scan.

  scan is a functional fold: accumulates state while mapping over sequence.
  more efficient than python loop as it compiles the entire sequence.

  parameters
  ----------
  state:
      initial training state
  batches:
      list of (inputs, targets) pairs
  log_every:
      log loss every n steps

  returns
  -------
  tuple[TrainState, list[float]]
      final state and list of losses
  """

  @jax.jit
  def step_fn(carry_state, batch_and_step):
    """single step function for scan."""
    batch, step = batch_and_step
    inputs, targets = batch
    new_state, loss = train_step(carry_state, inputs, targets)

    # log conditionally (note: in jit this doesn't print, but we track loss)
    should_log = (step % log_every) == 0
    return new_state, (loss, should_log, step)

  # prepare scan inputs: (batch, step_number) pairs
  scan_inputs = [(batch, i + 1) for i, batch in enumerate(batches)]

  # run scan: functional fold over all batches
  final_state, outputs = jax.lax.scan(step_fn, state, scan_inputs)

  # extract losses and log info
  losses, should_logs, steps = zip(*outputs)

  # log outside of jit (post-hoc)
  for loss, should_log, step in zip(losses, should_logs, steps):
    if should_log:
      logger.info("step %d, loss %.4f", int(step), float(loss))

  return final_state, list(losses)


def main() -> None:
  import argparse

  dotenv.load_dotenv()
  parser = argparse.ArgumentParser(description="fine-tune gpt-2 on custom text data")
  parser.add_argument("--dataset", type=str, required=True, help="filename in data_dir")
  parser.add_argument(
    "--model-name", type=str, default="gpt2", help="base model to fine-tune"
  )
  parser.add_argument("--batch-size", type=int, default=2, help="batch size")
  parser.add_argument("--seq-length", type=int, default=64, help="sequence length")
  parser.add_argument("--steps", type=int, default=10, help="training steps")
  parser.add_argument("--learning-rate", type=float, default=5e-5, help="learning rate")
  parser.add_argument("--log-every", type=int, default=10, help="log interval")
  parser.add_argument(
    "--use-scan",
    action="store_true",
    help="use jax.lax.scan for functional training (faster but less flexible logging)",
  )
  args = parser.parse_args()

  # load dataset
  data_dir = Path(os.getenv("DATA_DIR", "."))
  tokenizer = prepare_tokenizer(args.model_name)
  lines = load_dataset(data_dir, args.dataset)

  # encode dataset into token ids
  text = "\n".join(lines)
  encoded = tokenizer.encode(text)
  token_ids = np.array(encoded, dtype=np.int32)
  logger.info("encoded dataset into %d tokens", len(token_ids))

  # load config and create model
  from transformers import GPT2Config  # type: ignore

  hf_cfg = GPT2Config.from_pretrained(args.model_name)
  model_cfg = load_hf_gpt2_config(hf_cfg)
  model = GPT2LMHeadModel(config=model_cfg)

  # load pretrained parameters
  logger.info("loading pretrained parameters from %s", args.model_name)
  hf_model = FlaxGPT2LMHeadModel.from_pretrained(args.model_name)
  params = map_hf_params(hf_model.params, model_cfg, hf_key_prefix="transformer")

  # setup optimizer and training state
  tx = optax.adamw(learning_rate=create_learning_rate_fn(args.learning_rate))
  state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

  # prepare batches
  batch_iter = iterate_batches(token_ids, args.batch_size, args.seq_length)

  if args.use_scan:
    # functional training with scan (collect all batches first)
    logger.info("training with jax.lax.scan (functional fold)")
    batches = [next(batch_iter) for _ in range(args.steps)]
    batches_jax = [(jnp.array(inp), jnp.array(tgt)) for inp, tgt in batches]
    state, losses = train_with_scan(state, batches_jax, args.log_every)
  else:
    # traditional training loop (more flexible for logging)
    logger.info("training with traditional loop")
    jit_train_step = jax.jit(train_step)

    for step in range(1, args.steps + 1):
      batch_inputs, batch_targets = next(batch_iter)
      state, loss = jit_train_step(
        state, jnp.array(batch_inputs), jnp.array(batch_targets)
      )
      if step % args.log_every == 0:
        logger.info("step %d, loss %.4f", step, float(loss))

  # save final parameters
  save_path = Path("gpt2_finetuned.npz")
  np.savez(save_path, **jax.tree_util.tree_map(lambda x: np.array(x), state.params))
  logger.info("finished training; parameters saved to %s", save_path)


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
