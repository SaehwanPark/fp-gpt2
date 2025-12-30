# FP-styled GPT-2 in JAX/Flax

A minimal implementation of **GPT-2** using [JAX](https://docs.jax.dev/en/latest/) and [Flax](https://flax.readthedocs.io/en/stable/), with explicit parameter mapping from Hugging Face and minimal training and inference pipelines.

## Key motivations

* **Readable first**: prioritizes clarity and modularity over micro-optimizations
* **Faithful architecture**: mirrors GPT-2 layer structure and weight layout
* **Framework-native**: pure JAX/Flax, no PyTorch dependencies
* **Interoperable**: loads Hugging Face GPT-2 configs, tokenizers, and weights

---

## Features

* GPT-2 transformer blocks with causal self-attention
* Immutable model configuration via `ModelConfig`
* Hugging Face → Flax parameter mapping
* Simple fine-tuning loop with Optax
* Greedy text generation for inference
* Lightweight test suite with `pytest`

---

## Project structure

```
src/
  config.py     # ModelConfig and HF config adapter
  model.py      # GPT-2 architecture (attention, MLP, blocks, LM head)
  utils.py      # Dataset loading, tokenizer prep, parameter mapping
  train.py      # Example fine-tuning script
  inference.py  # Text generation script
tests/
  test_model.py
```

---

## Setup

### Requirements

* Python 3.9+ (tested on 3.13)
* uv

Install dependencies:

```bash
uv sync
```

---

## Training

Set the dataset directory and run fine-tuning:

```bash
export DATA_DIR=/path/to/data
uv run -m src.train \
  --dataset train.txt \
  --model-name gpt2 \
  --batch-size 2 \
  --seq-length 64 \
  --steps 10
```

The final checkpoint is saved as:

```
gpt2_finetuned.npz
```

---

## Inference

Generate text from a prompt using pretrained or fine-tuned weights:

```bash
uv run -m src.inference \
  --prompt "Hello, I'm Einstgein, " \
  --max-length 20
```

To load a fine-tuned checkpoint:

```bash
uv run -m src.inference \
  --prompt "Once upon a time" \
  --checkpoint gpt2_finetuned.npz
```

---

## Notes

* Decoding uses **greedy generation** (no KV cache; full forward pass per step)
* Dropout is controlled via a `deterministic` flag
* Parameter mapping assumes Hugging Face `transformers` v4.x layout

---

## Testing

Run the test suite from the project root:

```bash
uv run pytest
```
