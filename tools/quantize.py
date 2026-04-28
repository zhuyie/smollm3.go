"""Convert an SML3 fp32 checkpoint to SML3 int8 weight format."""

import argparse
import struct

import numpy as np


CHECKPOINT_MAGIC = 0x334C4D53
CHECKPOINT_VERSION = 1
CHECKPOINT_HEADER_SIZE = 256
MAX_ROPE_HEADER_LAYERS = 48
WEIGHT_TYPE_FP32 = 0
WEIGHT_TYPE_INT8 = 1


def read_fp32(file, count):
    data = np.fromfile(file, dtype="<f4", count=count)
    if data.size != count:
        raise EOFError(f"expected {count} float32 values, got {data.size}")
    return data


def write_fp32(file, data):
    file.write(np.asarray(data, dtype="<f4").reshape(-1).tobytes())


def write_int8_matrix(file, data, inputs, rows):
    matrix = np.asarray(data, dtype=np.float32).reshape(rows, inputs)
    max_abs = np.max(np.abs(matrix), axis=1)
    scale = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype("<f4")
    quantized = np.rint(matrix / scale[:, None])
    quantized = np.clip(quantized, -127, 127).astype(np.int8)
    file.write(quantized.reshape(-1).tobytes())
    file.write(scale.tobytes())


def read_header(file):
    header = file.read(CHECKPOINT_HEADER_SIZE)
    if len(header) != CHECKPOINT_HEADER_SIZE:
        raise EOFError("checkpoint header is truncated")
    magic, version, dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, shared, bos_id, eos_id, pad_id, rope_theta, rms_norm_eps = struct.unpack_from(
        "<I" + "i" * 12 + "ff", header
    )
    if magic != CHECKPOINT_MAGIC or version != CHECKPOINT_VERSION:
        raise ValueError(f"expected SML3 v1 checkpoint, got magic={magic:#x} version={version}")
    rope_offset = struct.calcsize("<I" + "i" * 12 + "ff")
    rope_layers = list(struct.unpack_from("<" + "i" * MAX_ROPE_HEADER_LAYERS, header, rope_offset))
    weight_type_offset = rope_offset + struct.calcsize("<" + "i" * MAX_ROPE_HEADER_LAYERS)
    (weight_type,) = struct.unpack_from("<i", header, weight_type_offset)
    if weight_type != WEIGHT_TYPE_FP32:
        raise ValueError(f"expected FP32 checkpoint weight_type=0, got {weight_type}")
    if n_layers > MAX_ROPE_HEADER_LAYERS:
        raise ValueError(f"n_layers={n_layers} exceeds header maximum {MAX_ROPE_HEADER_LAYERS}")
    return {
        "dim": dim,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "shared": shared,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "pad_id": pad_id,
        "rope_theta": rope_theta,
        "rms_norm_eps": rms_norm_eps,
        "rope_layers": rope_layers[:n_layers],
    }


def write_header(file, cfg):
    header = struct.pack(
        "<I" + "i" * 12 + "ff",
        CHECKPOINT_MAGIC,
        CHECKPOINT_VERSION,
        cfg["dim"],
        cfg["hidden_dim"],
        cfg["n_layers"],
        cfg["n_heads"],
        cfg["n_kv_heads"],
        cfg["vocab_size"],
        cfg["seq_len"],
        cfg["shared"],
        cfg["bos_id"],
        cfg["eos_id"],
        cfg["pad_id"],
        cfg["rope_theta"],
        cfg["rms_norm_eps"],
    )
    header += struct.pack(
        "<" + "i" * MAX_ROPE_HEADER_LAYERS,
        *(cfg["rope_layers"] + [0] * (MAX_ROPE_HEADER_LAYERS - cfg["n_layers"])),
    )
    header += struct.pack("<i", WEIGHT_TYPE_INT8)
    file.write(header)
    file.write(b"\0" * (CHECKPOINT_HEADER_SIZE - len(header)))


def convert(input_path, output_path):
    with open(input_path, "rb") as src, open(output_path, "wb") as dst:
        cfg = read_header(src)
        write_header(dst, cfg)

        dim = cfg["dim"]
        hidden_dim = cfg["hidden_dim"]
        n_layers = cfg["n_layers"]
        n_heads = cfg["n_heads"]
        n_kv_heads = cfg["n_kv_heads"]
        vocab_size = cfg["vocab_size"]
        kv_dim = dim * n_kv_heads // n_heads

        token_embedding = read_fp32(src, vocab_size * dim)
        write_fp32(dst, token_embedding)

        for _ in range(n_layers):
            write_fp32(dst, read_fp32(src, dim))
            write_int8_matrix(dst, read_fp32(src, dim * dim), dim, dim)
            write_int8_matrix(dst, read_fp32(src, dim * kv_dim), dim, kv_dim)
            write_int8_matrix(dst, read_fp32(src, dim * kv_dim), dim, kv_dim)
            write_int8_matrix(dst, read_fp32(src, dim * dim), dim, dim)
            write_fp32(dst, read_fp32(src, dim))
            write_int8_matrix(dst, read_fp32(src, dim * hidden_dim), dim, hidden_dim)
            write_int8_matrix(dst, read_fp32(src, hidden_dim * dim), hidden_dim, dim)
            write_int8_matrix(dst, read_fp32(src, dim * hidden_dim), dim, hidden_dim)

        write_fp32(dst, read_fp32(src, dim))
        if cfg["shared"]:
            write_int8_matrix(dst, token_embedding, dim, vocab_size)
        else:
            write_int8_matrix(dst, read_fp32(src, vocab_size * dim), dim, vocab_size)

        trailing = src.read(1)
        if trailing:
            raise ValueError("input checkpoint has trailing bytes")

    print(f"wrote {output_path}")
    print(
        "config: "
        f"dim={cfg['dim']} hidden_dim={cfg['hidden_dim']} layers={cfg['n_layers']} "
        f"heads={cfg['n_heads']} kv_heads={cfg['n_kv_heads']} vocab={cfg['vocab_size']} "
        f"seq_len={cfg['seq_len']} rope_theta={cfg['rope_theta']} rms_norm_eps={cfg['rms_norm_eps']} "
        f"shared_classifier={cfg['shared']} rope_layers={''.join(str(v) for v in cfg['rope_layers'])} "
        "weight_type=int8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input SML3 fp32 checkpoint path")
    parser.add_argument("output", help="output SML3 int8 checkpoint path")
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
