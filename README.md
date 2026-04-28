# smollm3.go

Minimal Go implementation for local inference with
[HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B).

This is a small, readable FP32 runtime inspired by `llama2.c` and adapted from
the SmolLM2 Go implementation. The current target is a minimal-correct SmolLM3
path: float32 export, tokenizer export, GQA, the SmolLM3 RoPE/NoPE layer
pattern, batched prompt prefill, and chat rendering.

## Layout

```text
cmd/smollm3/          CLI entry point
internal/model/       SML3 loader, weights, KV cache, forward pass
internal/tokenizer/   TOK3 loader and byte-level BPE tokenizer
internal/sampler/     greedy, multinomial, and top-p sampling
tools/                Hugging Face model/tokenizer export scripts
docs/CHECKPOINT.md    SML3/TOK3 binary format notes
```

## Prepare Python Environment

The export scripts need `torch`, `transformers`, `sentencepiece`, `safetensors`,
`accelerate`, and `numpy`.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip torch transformers sentencepiece safetensors accelerate numpy
```

SmolLM3 requires recent Transformers support. Use `transformers>=4.53`.

## Export Model And Tokenizer

```sh
mkdir -p models

.venv/bin/python tools/export.py models/smollm3-3b-f32.bin \
  --hf HuggingFaceTB/SmolLM3-3B

.venv/bin/python tools/export_tokenizer.py models/smollm3-tokenizer.bin \
  --hf HuggingFaceTB/SmolLM3-3B
```

## Build

```sh
mkdir -p bin
go build -o bin/smollm3 ./cmd/smollm3
```

## Run

```sh
bin/smollm3 \
  -model models/smollm3-3b-f32.bin \
  -tokenizer models/smollm3-tokenizer.bin \
  -mode chat \
  -prompt "Give me a brief explanation of gravity in simple terms." \
  -temp 0.6 \
  -top-p 0.95
```

Disable extended thinking in chat rendering:

```sh
bin/smollm3 \
  -model models/smollm3-3b-f32.bin \
  -tokenizer models/smollm3-tokenizer.bin \
  -mode chat \
  -think=false \
  -prompt "What is 2+2?" \
  -temp 0
```
