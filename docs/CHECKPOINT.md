# Binary Formats

This project uses compact local binary formats for SmolLM3 inference. All
numeric fields are little-endian. The formats are intentionally simple and are
produced by the scripts in `tools/`.

## SML3 Checkpoint

The first 256 bytes are the fixed checkpoint header.

| Field | Type | Notes |
| --- | --- | --- |
| `magic` | `uint32` | `0x334C4D53`, bytes spell `SML3` |
| `version` | `int32` | currently `1` |
| `dim` | `int32` | hidden size |
| `hidden_dim` | `int32` | MLP hidden size |
| `n_layers` | `int32` | Transformer block count |
| `n_heads` | `int32` | query attention heads |
| `n_kv_heads` | `int32` | key/value heads |
| `vocab_size` | `int32` | tokenizer vocabulary size |
| `seq_len` | `int32` | maximum KV cache length |
| `shared_classifier` | `int32` | nonzero means `wcls` aliases token embeddings |
| `bos_id` | `int32` | model BOS id, or `-1` |
| `eos_id` | `int32` | model EOS id |
| `pad_id` | `int32` | model PAD id, or `-1` |
| `rope_theta` | `float32` | RoPE base frequency |
| `rms_norm_eps` | `float32` | RMSNorm epsilon |
| `rope_layers` | `int32[48]` | first `n_layers` entries; nonzero means apply RoPE |

The SmolLM3-3B instruct checkpoint uses 36 layers and a 3:1 RoPE/NoPE pattern:
`1,1,1,0` repeated.

## Weights

All tensors are written as contiguous float32 values in row-major order.

1. `token_embedding_table`: `(vocab_size, dim)`
2. For each layer:
   - `rms_att_weight`: `(dim,)`
   - `wq`: `(dim, dim)`
   - `wk`: `(n_kv_heads * head_size, dim)`
   - `wv`: `(n_kv_heads * head_size, dim)`
   - `wo`: `(dim, dim)`
   - `rms_ffn_weight`: `(dim,)`
   - `w1`: `(hidden_dim, dim)`
   - `w2`: `(dim, hidden_dim)`
   - `w3`: `(hidden_dim, dim)`
3. `rms_final_weight`: `(dim,)`
4. If `shared_classifier == 0`, `wcls`: `(vocab_size, dim)`

## TOK3 Tokenizer

The tokenizer file is a compact binary form of SmolLM3's byte-level BPE
tokenizer.

The first 256 bytes are the fixed header.

| Field | Type | Notes |
| --- | --- | --- |
| `magic` | `uint32` | `0x334B4F54`, bytes spell `TOK3` |
| `version` | `int32` | currently `1` |
| `vocab_size` | `int32` | number of token strings |
| `merge_count` | `int32` | number of BPE merge rules |
| `max_token_length` | `int32` | maximum UTF-8 byte length of a token string |
| `bos_id` | `int32` | BOS id, or `-1` |
| `eos_id` | `int32` | EOS id |
| `pad_id` | `int32` | PAD id, or `-1` |
| `unk_id` | `int32` | UNK id, or `-1` |
| `special_count` | `int32` | number of special token ids stored in the body |

The body stores:

1. `vocab_size` token strings, each as `uint32 byte_length` followed by UTF-8 bytes.
2. `merge_count` merge rules, each as three `int32` values: left id, right id, output id.
3. `special_count` token ids as `int32`.
