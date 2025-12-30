"""utilities for datasets, tokenization, and hf→flax param mapping."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

import jax.numpy as jnp

logger = logging.getLogger(__name__)


# precise types for parameter structure
class LayerNormParams(TypedDict):
  scale: jnp.ndarray
  bias: jnp.ndarray


class DenseParams(TypedDict):
  kernel: jnp.ndarray
  bias: jnp.ndarray


class AttentionParams(TypedDict):
  c_attn: DenseParams  # qkv projection
  c_proj: DenseParams  # output projection


class MLPParams(TypedDict):
  c_fc: DenseParams  # feedforward expand
  c_proj: DenseParams  # feedforward contract


class BlockParams(TypedDict):
  ln_1: LayerNormParams
  ln_2: LayerNormParams
  attn: AttentionParams
  mlp: MLPParams


# note: full model params structure is dynamic based on n_layer
# top level keys: wte, wpe, block_0, block_1, ..., block_{n-1}, ln_f
# each block_i contains BlockParams structure


def load_dataset(data_dir: Path, filename: str) -> list[str]:
  """load non-empty lines from a text file in ``data_dir``.

  pure function: given same inputs, returns same output.
  file i/o is the only side effect.
  """
  file_path = data_dir / filename
  if not file_path.is_file():
    raise FileNotFoundError(f"dataset file not found: {file_path}")

  with file_path.open("r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

  logger.info("loaded %d non-empty lines from %s", len(lines), file_path)
  return lines


def prepare_tokenizer(model_name: str = "gpt2"):
  """load a hugging face tokenizer and set pad_token to eos_token.

  side effect: downloads tokenizer if not cached.
  """
  from transformers import AutoTokenizer  # type: ignore

  logger.info("loading tokenizer for model %s", model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  tokenizer.pad_token = tokenizer.eos_token
  return tokenizer


# pure extraction functions for parameter mapping
def _extract_embeddings(hf_params: dict, prefix: str) -> dict[str, dict]:
  """extract word and position embeddings.

  note: nn.Embed creates {"embedding": array} structure, not flat arrays.
  we preserve this structure for compatibility with flax.
  """
  return {
    "wte": {"embedding": hf_params[prefix]["wte"]["embedding"]},
    "wpe": {"embedding": hf_params[prefix]["wpe"]["embedding"]},
  }


def _extract_layer_norm(hf_ln: dict) -> LayerNormParams:
  """extract layer norm parameters."""
  return LayerNormParams(scale=hf_ln["scale"], bias=hf_ln["bias"])


def _extract_dense(hf_dense: dict) -> DenseParams:
  """extract dense layer parameters.

  important: hugging face stores kernels as (out_features, in_features)
  but flax expects (in_features, out_features), so we transpose.
  """
  return DenseParams(
    kernel=hf_dense["kernel"].T,  # transpose for flax convention
    bias=hf_dense["bias"],
  )


def _extract_block(hf_block: dict) -> BlockParams:
  """extract single transformer block parameters.

  preserves openai's weight layout: c_attn contains concatenated qkv.
  """
  return BlockParams(
    ln_1=_extract_layer_norm(hf_block["ln_1"]),
    ln_2=_extract_layer_norm(hf_block["ln_2"]),
    attn=AttentionParams(
      c_attn=_extract_dense(hf_block["attn"]["c_attn"]),
      c_proj=_extract_dense(hf_block["attn"]["c_proj"]),
    ),
    mlp=MLPParams(
      c_fc=_extract_dense(hf_block["mlp"]["c_fc"]),
      c_proj=_extract_dense(hf_block["mlp"]["c_proj"]),
    ),
  )


def map_hf_params(
  hf_params: dict, config, *, hf_key_prefix: str = "transformer"
) -> dict:
  """convert hf flax gpt-2 param dict into this model's param tree.

  functional pipeline: compose pure extraction functions to build
  the target parameter structure. each extraction function handles
  one level of the hierarchy.

  note: flax creates block_0, block_1, etc. as top-level keys, not
  nested under a "blocks" container. this matches the model structure.

  important: hugging face stores dense layer kernels as (out, in) while
  flax expects (in, out). we transpose during mapping to convert between
  conventions. this is done once at load time, not during forward passes.

  raises:
    KeyError: if expected keys are missing from hf_params
  """
  try:
    hf_blocks = hf_params[hf_key_prefix]["h"]

    # extract blocks as top-level keys (not nested in container)
    blocks_dict = {
      f"block_{idx}": _extract_block(
        hf_blocks[str(idx)] if isinstance(hf_blocks, dict) else hf_blocks[idx]
      )
      for idx in range(config.n_layer)
    }

    # compose into final structure
    return {
      **_extract_embeddings(hf_params, hf_key_prefix),
      **blocks_dict,  # blocks are at top level, not nested
      "ln_f": _extract_layer_norm(hf_params[hf_key_prefix]["ln_f"]),
    }
  except KeyError as exc:
    raise KeyError(
      f"expected transformer structure in hf parameters under '{hf_key_prefix}'"
    ) from exc
