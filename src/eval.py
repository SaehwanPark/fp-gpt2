"""evaluation utilities for language modeling."""

from __future__ import annotations

import jax.numpy as jnp
import optax


def compute_perplexity(
  model,
  params: dict,
  inputs: jnp.ndarray,
  targets: jnp.ndarray,
) -> float:
  """compute perplexity on a batch of sequences.

  perplexity = exp(cross_entropy_loss)
  lower perplexity indicates better model fit to the data.

  parameters
  ----------
  model:
      the gpt2 model
  params:
      model parameters
  inputs:
      (batch, seq) input token ids
  targets:
      (batch, seq) target token ids (inputs shifted by 1)

  returns
  -------
  float
      perplexity score
  """
  logits = model.apply({"params": params}, inputs, deterministic=True)

  # flatten: (batch, seq, vocab) -> (batch * seq, vocab)
  logits_flat = logits.reshape(-1, logits.shape[-1])
  targets_flat = targets.reshape(-1)

  # cross-entropy loss
  loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, targets_flat)
  avg_loss = jnp.mean(loss)

  # perplexity = exp(loss)
  return float(jnp.exp(avg_loss))


def compute_accuracy(
  model,
  params: dict,
  inputs: jnp.ndarray,
  targets: jnp.ndarray,
) -> float:
  """compute next-token prediction accuracy.

  useful sanity check: good models should have >50% accuracy
  on in-domain data for common tokens.

  parameters
  ----------
  model:
      the gpt2 model
  params:
      model parameters
  inputs:
      (batch, seq) input token ids
  targets:
      (batch, seq) target token ids

  returns
  -------
  float
      fraction of correct next-token predictions
  """
  logits = model.apply({"params": params}, inputs, deterministic=True)
  predictions = jnp.argmax(logits, axis=-1)

  correct = jnp.sum(predictions == targets)
  total = targets.size

  return float(correct / total)
