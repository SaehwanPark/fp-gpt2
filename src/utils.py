"""Utilities for datasets, tokenization, and HF→Flax param mapping."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def load_dataset(data_dir: Path, filename: str) -> List[str]:
  """Load non-empty lines from a text file in ``data_dir``."""
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
  """Load a Hugging Face tokenizer and set pad_token to eos_token."""
  from transformers import AutoTokenizer  # type: ignore

  logger.info("Loading tokenizer for model %s", model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  # Ensure the tokenizer does not add special tokens by default.
  tokenizer.pad_token = tokenizer.eos_token
  return tokenizer


def map_hf_params(
  hf_params: dict, config, *, hf_key_prefix: str = "transformer"
) -> dict:
  """Convert HF Flax GPT-2 param dict into this model’s param tree."""
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
