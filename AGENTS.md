# AGENTS.md

## Project Goal

`smollm3.go` is a small, readable Go runtime for local SmolLM3-3B inference. It includes tokenizer loading, byte-level BPE tokenization, model loading, KV cache, sampling, int8 weight-only quantization, and ARM64/amd64 SIMD kernels.

Keep the project easy to understand and hack on. Prefer clear Go and local conventions over clever abstractions.

## Working Style

- Keep changes narrowly scoped to the requested behavior.
- Read the existing implementation before editing; follow the style already present in nearby files.
- Do not rewrite unrelated code, generated artifacts, binary model files, or benchmark data unless explicitly asked.
- Preserve public CLI flags, checkpoint formats, tokenizer behavior, and model outputs unless the task is specifically about changing them.
- Be especially careful around `internal/model` numeric code and assembly kernels. Small changes can affect correctness or performance.
- Prefer explicit, readable code paths over reflection, global state, or unnecessary abstraction.
- Add comments only where they explain non-obvious model, tokenizer, binary format, or SIMD details.

## Repository Layout

- `cmd/smollm3/`: CLI entry point and command behavior.
- `internal/model/`: SML3 loader, weights, KV cache, matmul, forward pass, and platform kernels.
- `internal/tokenizer/`: TOK3 loader and byte-level BPE tokenizer.
- `internal/sampler/`: greedy, multinomial, and top-p sampling.
- `tools/`: Python export and quantization scripts for Hugging Face checkpoints.
- `docs/CHECKPOINT.md`: SML3/TOK3 binary format notes.
- `models/`: local model/tokenizer artifacts. Treat these as large local assets, not source files to casually modify.

## Build And Test

Use these commands from the repository root:

```sh
go test ./...
go build -o bin/smollm3 ./cmd/smollm3
```

After code changes, always run the build command so `bin/smollm3` is refreshed and matches the latest source.

For model benchmark work:

```sh
go test ./internal/model -bench='Benchmark(Prefill|Decode)' -benchtime=1x -run '^$'
```

Do not run multiple benchmarks at the same time. Concurrent benchmark runs can interfere with each other and make the numbers unreliable.

When touching tokenizer behavior, run tokenizer tests at minimum:

```sh
go test ./internal/tokenizer
```

When touching CLI behavior, run CLI tests at minimum:

```sh
go test ./cmd/smollm3
```

Do not add tests mechanically for every change. Let test coverage follow risk and value: correctness-sensitive parsing, tokenizer behavior, model behavior, checkpoint formats, and CLI contract changes should be tested, but small output formatting tweaks or updates to previously hard-coded strings often do not need dedicated unit tests when the added test would mostly restate the implementation.

## Commit Log Style

Use concise conventional-style commit messages:

```text
fix(scope): short imperative summary
feat(scope): short imperative summary
test(scope): short imperative summary
docs(scope): short imperative summary
refactor(scope): short imperative summary
perf(scope): short imperative summary
```

Guidelines:

- Keep the subject under roughly 72 characters when practical.
- Use lowercase after the prefix unless a proper noun or acronym requires otherwise.
- Prefer the smallest accurate scope, such as `tokenizer`, `model`, `cli`, `sampler`, `tools`, `docs`, or `kernel`.
- Use an imperative summary: `fix(tokenizer): handle empty merges`, not `fixed...` or `fixes...`.
- Add a body only when the reason or tradeoff is not obvious from the diff.

Examples:

```text
fix(tokenizer): align official tokenization
fix(toolcall): render tool responses with tags
perf(model): speed up int8 decode on arm64
test(cli): cover disabled thinking mode
docs(checkpoint): clarify TOK3 token records
```

## Pull Requests

PR descriptions should be short and practical:

- Summarize the user-facing or developer-facing change.
- Mention correctness, compatibility, or performance implications when relevant.
- List the tests or benchmarks run.
- Call out intentionally skipped tests, large model requirements, or platform-specific coverage gaps.

## Model And Data Files

- Do not commit regenerated model binaries, Hugging Face checkpoint shards, or tokenizer binaries unless explicitly requested.
- Avoid running export or quantization scripts unless the task requires it; they may need large downloads and local Python dependencies.
- If model files are needed for verification, prefer existing files under `models/`.

## Performance Notes

- For benchmark-sensitive changes, compare before and after numbers on the same machine when possible.
- Run benchmarks one at a time so CPU, thermal, and memory effects do not contaminate results.
- Do not trade correctness or checkpoint compatibility for speed without making that tradeoff explicit.
- Keep scalar and architecture-specific kernel behavior aligned.
