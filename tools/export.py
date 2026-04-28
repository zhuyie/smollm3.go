"""Export Hugging Face SmolLM3 weights to SML3 fp32 format."""

import argparse
import json
import os
import struct

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM


CHECKPOINT_MAGIC = 0x334C4D53  # "SML3" in little-endian bytes
CHECKPOINT_VERSION_V1 = 1
CHECKPOINT_HEADER_SIZE = 256
MAX_ROPE_HEADER_LAYERS = 48


def serialize_fp32(file, tensor):
    data = tensor.detach().cpu().reshape(-1).to(torch.float32).numpy()
    file.write(data.tobytes())


def reverse_rope_permute(weight, n_heads, out_dim, in_dim):
    return (
        weight.view(n_heads, 2, out_dim // n_heads // 2, in_dim)
        .transpose(1, 2)
        .reshape(out_dim, in_dim)
    )


def config_value(config, raw_config, name, default):
    value = getattr(config, name, None)
    if value is None and hasattr(config, "to_dict"):
        value = config.to_dict().get(name)
    if value is None and raw_config is not None:
        value = raw_config.get(name, default)
    return default if value is None else value


def load_raw_config(model_name):
    config_path = None
    if os.path.isdir(model_name):
        candidate = os.path.join(model_name, "config.json")
        if os.path.exists(candidate):
            config_path = candidate
    else:
        try:
            config_path = hf_hub_download(model_name, "config.json")
        except Exception:
            config_path = None
    if config_path is None:
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def export_hf(model_name, output_path):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    state = model.state_dict()
    config = model.config
    raw_config = load_raw_config(model_name)

    dim = config.hidden_size
    hidden_dim = config.intermediate_size
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv_heads = config_value(config, raw_config, "num_key_value_heads", n_heads)
    vocab_size = config.vocab_size
    seq_len = config.max_position_embeddings
    rope_theta = float(config_value(config, raw_config, "rope_theta", 10000.0))
    rms_norm_eps = float(config_value(config, raw_config, "rms_norm_eps", 1e-6))
    shared_classifier = bool(config_value(config, raw_config, "tie_word_embeddings", True))
    bos_id = config_value(config, raw_config, "bos_token_id", -1)
    eos_id = config_value(config, raw_config, "eos_token_id", -1)
    pad_id = config_value(config, raw_config, "pad_token_id", -1)
    head_dim = dim // n_heads
    kv_dim = n_kv_heads * head_dim
    rope_layers = config_value(config, raw_config, "no_rope_layers", None)
    if rope_layers is None:
        no_rope_layer_interval = int(config_value(config, raw_config, "no_rope_layer_interval", 4))
        rope_layers = [0 if (layer + 1) % no_rope_layer_interval == 0 else 1 for layer in range(n_layers)]
    rope_layers = [int(v) for v in rope_layers]
    if len(rope_layers) != n_layers:
        raise ValueError(f"expected {n_layers} rope layer flags, got {len(rope_layers)}")
    if n_layers > MAX_ROPE_HEADER_LAYERS:
        raise ValueError(f"n_layers={n_layers} exceeds header maximum {MAX_ROPE_HEADER_LAYERS}")

    with open(output_path, "wb") as out:
        header = struct.pack(
            "<I" + "i" * 12 + "ff",
            CHECKPOINT_MAGIC,
            CHECKPOINT_VERSION_V1,
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            seq_len,
            int(shared_classifier),
            int(bos_id) if bos_id is not None else -1,
            int(eos_id) if eos_id is not None else -1,
            int(pad_id) if pad_id is not None else -1,
            rope_theta,
            rms_norm_eps,
        )
        header += struct.pack("<" + "i" * MAX_ROPE_HEADER_LAYERS, *(rope_layers + [0] * (MAX_ROPE_HEADER_LAYERS - n_layers)))
        out.write(header)
        out.write(b"\0" * (CHECKPOINT_HEADER_SIZE - len(header)))

        serialize_fp32(out, state["model.embed_tokens.weight"])
        for layer in range(n_layers):
            prefix = f"model.layers.{layer}"
            serialize_fp32(out, state[f"{prefix}.input_layernorm.weight"])
            serialize_fp32(
                out,
                reverse_rope_permute(
                    state[f"{prefix}.self_attn.q_proj.weight"],
                    n_heads,
                    dim,
                    dim,
                ),
            )
            serialize_fp32(
                out,
                reverse_rope_permute(
                    state[f"{prefix}.self_attn.k_proj.weight"],
                    n_kv_heads,
                    kv_dim,
                    dim,
                ),
            )
            serialize_fp32(out, state[f"{prefix}.self_attn.v_proj.weight"])
            serialize_fp32(out, state[f"{prefix}.self_attn.o_proj.weight"])
            serialize_fp32(out, state[f"{prefix}.post_attention_layernorm.weight"])
            serialize_fp32(out, state[f"{prefix}.mlp.gate_proj.weight"])
            serialize_fp32(out, state[f"{prefix}.mlp.down_proj.weight"])
            serialize_fp32(out, state[f"{prefix}.mlp.up_proj.weight"])

        serialize_fp32(out, state["model.norm.weight"])
        if not shared_classifier:
            serialize_fp32(out, state["lm_head.weight"])

    print(f"wrote {output_path}")
    print(
        "config: "
        f"dim={dim} hidden_dim={hidden_dim} layers={n_layers} "
        f"heads={n_heads} kv_heads={n_kv_heads} vocab={vocab_size} "
        f"seq_len={seq_len} rope_theta={rope_theta} rms_norm_eps={rms_norm_eps} "
        f"shared_classifier={int(shared_classifier)} rope_layers={''.join(str(v) for v in rope_layers)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="output SML3 checkpoint path")
    parser.add_argument("--hf", default="HuggingFaceTB/SmolLM3-3B")
    args = parser.parse_args()
    export_hf(args.hf, args.output)


if __name__ == "__main__":
    main()
