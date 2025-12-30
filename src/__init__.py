"""Top-level package for the Jax/Flax GPT‑2 implementation.

Modules
=======

model
    Defines the neural network layers making up the GPT‑2 architecture,
    including the causal self‑attention block and the transformer block.

config
    Provides a simple data class describing the essential
    hyperparameters required to instantiate the model.  A helper is
    included to load the configuration from a pre‑trained Hugging Face
    `GPT2Config` object.

utils
    Contains helper functions for loading datasets from the file
    system and performing tokenization using the Hugging Face
    tokenizers.  It also provides convenience routines for generating
    causal masks and for mapping pre‑trained parameters into the
    custom Flax model parameter tree.

train
    Demonstrates how to fine‑tune the model on a text corpus using
    JAX and Optax.  The training loop is deliberately simple but
    illustrates how to handle data loading, loss computation and
    parameter updates.

inference
    Provides an end‑to‑end example for loading a pre‑trained model,
    running it on an input prompt and decoding the generated text.

This project assumes that the path to any training data is provided
via the `DATA_DIR` environment variable.
"""

from .config import ModelConfig, load_hf_gpt2_config
from .model import GPT2LMHeadModel
from .utils import load_dataset, prepare_tokenizer, map_hf_params

__all__ = [
  "ModelConfig",
  "load_hf_gpt2_config",
  "GPT2LMHeadModel",
  "load_dataset",
  "prepare_tokenizer",
  "map_hf_params",
]
