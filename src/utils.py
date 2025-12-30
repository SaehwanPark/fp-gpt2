"""Utility functions for dataset handling, tokenization and parameter mapping.

This module collects helper routines used throughout the project.
Functions defined here avoid external side effects and are intended to
be easily testable.  The helpers cover three broad areas: loading
text datasets from disk, preparing Hugging Face tokenizers, and
converting pre‑trained Hugging Face parameters into the layout
expected by the custom Flax model defined in :mod:`src.model`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def load_dataset(data_dir: Path, filename: str) -> List[str]:
  """Load a text dataset from the given directory.

  Each line of the file will be returned as a separate string.  Empty
  lines are skipped.  The caller is responsible for ensuring that
  ``data_dir`` exists and that the filename is valid.

  Parameters
  ----------
  data_dir:
      Base directory from which to load the file.  Typically this
      comes from the ``DATA_DIR`` environment variable.
  filename:
      Name of the file within ``data_dir`` to open.

  Returns
  -------
  list of str
      A list of non‑empty lines from the file.
  """
  file_path = data_dir / filename
  if not file_path.is_file():
    raise FileNotFoundError(f"Dataset file not found: {file_path}")
  lines: List[str] = []
  with file_path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        lines.append(line)
  logger.info("Loaded %d non‑empty lines from %s", len(lines), file_path)
  return lines


def prepare_tokenizer(model_name: str = "gpt2"):
  """Load a Hugging Face tokenizer for the given model name.

  The returned tokenizer is configured to return tensors as Python
  ``int`` arrays.  Tokenizers require an active internet connection
  for the first call; subsequent calls will use the cached files.

  Parameters
  ----------
  model_name:
      Name of the pre‑trained model whose tokenizer should be
      instantiated.  The default is ``'gpt2'``.

  Returns
  -------
  transformers.PreTrainedTokenizer
      A tokenizer instance ready for encoding and decoding text.
  """
  from transformers import AutoTokenizer  # type: ignore

  logger.info("Loading tokenizer for model %s", model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  # Ensure the tokenizer does not add special tokens by default.
  tokenizer.pad_token = tokenizer.eos_token
  return tokenizer


def map_hf_params(
  hf_params: dict, config, *, hf_key_prefix: str = "transformer"
) -> dict:
  """Map Hugging Face GPT‑2 parameters to the custom Flax model layout.

  Hugging Face uses a nested dictionary of parameters for its Flax
  implementation of GPT‑2.  This function extracts the relevant
  weights and biases and places them into a structure matching the
  parameter tree of :class:`~src.model.GPT2LMHeadModel`.  Only
  parameters that exist in both models are copied; other entries are
  ignored.

  Parameters
  ----------
  hf_params:
      The parameter dictionary from a Hugging Face
      ``FlaxGPT2LMHeadModel``.  It must have the same structure as
      returned by ``model.params``.
  config:
      Configuration of the custom model used to determine the number
      of layers and shapes.
  hf_key_prefix:
      Root key under which transformer parameters live in ``hf_params``.
      The default is ``'transformer'``, which is how Hugging Face
      namespaces the GPT‑2 transformer stack.

  Returns
  -------
  dict
      A parameter dictionary structured like the ``params`` portion
      of the custom Flax model.  Keys that do not correspond to a
      parameter in the custom model are omitted.

  Notes
  -----
  The mapping here assumes that the Hugging Face parameter names
  conform to the conventions used in version 4.x of the
  transformers library.  Future versions may require updates.  The
  mapping is implemented explicitly to avoid dependence on private
  APIs.
  """
  params_out: dict = {}

  # Word token embeddings and positional embeddings
  try:
    wte = hf_params[hf_key_prefix]["wte"]["embedding"]
    wpe = hf_params[hf_key_prefix]["wpe"]["embedding"]
  except KeyError as exc:
    raise KeyError(
      f"Expected wte/wpe in Hugging Face parameters under '{hf_key_prefix}'"
    ) from exc
  params_out["wte"] = wte
  params_out["wpe"] = wpe

  # Blocks
  hf_blocks = hf_params[hf_key_prefix]["h"]
  blocks_out = {}
  for idx in range(config.n_layer):
    hf_block = hf_blocks[str(idx)] if isinstance(hf_blocks, dict) else hf_blocks[idx]
    block_out: dict = {}
    # Layer norms
    block_out["ln_1"] = {
      "scale": hf_block["ln_1"]["scale"],
      "bias": hf_block["ln_1"]["bias"],
    }
    block_out["ln_2"] = {
      "scale": hf_block["ln_2"]["scale"],
      "bias": hf_block["ln_2"]["bias"],
    }
    # Attention projections
    attn = hf_block["attn"]
    block_out["attn"] = {
      "c_attn": {
        "kernel": attn["c_attn"]["kernel"],
        "bias": attn["c_attn"]["bias"],
      },
      "c_proj": {
        "kernel": attn["c_proj"]["kernel"],
        "bias": attn["c_proj"]["bias"],
      },
    }
    # MLP projections
    mlp = hf_block["mlp"]
    block_out["mlp"] = {
      "c_fc": {
        "kernel": mlp["c_fc"]["kernel"],
        "bias": mlp["c_fc"]["bias"],
      },
      "c_proj": {
        "kernel": mlp["c_proj"]["kernel"],
        "bias": mlp["c_proj"]["bias"],
      },
    }
    blocks_out[f"block_{idx}"] = block_out

  params_out["blocks"] = blocks_out
  # Final layer norm
  ln_f = hf_params[hf_key_prefix]["ln_f"]
  params_out["ln_f"] = {
    "scale": ln_f["scale"],
    "bias": ln_f["bias"],
  }

  return params_out
