// Package smollm3 exposes a small integration API for local SmolLM3 inference.
//
// A Client owns a loaded model checkpoint and tokenizer. It is intended for
// straightforward embedding in Go programs; methods are not safe for concurrent
// use because generation mutates the transformer's KV cache and scratch state.
package smollm3

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/zhuyie/smollm3.go/internal/model"
	"github.com/zhuyie/smollm3.go/internal/sampler"
	"github.com/zhuyie/smollm3.go/internal/tokenizer"
)

const (
	defaultMaxNewTokens   = 1024
	defaultTemperature    = 0.6
	defaultTopP           = 0.95
	defaultSystemPrompt   = "You are a helpful AI assistant named SmolLM, trained by Hugging Face."
	defaultThinkingPrompt = defaultSystemPrompt + " Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. " +
		"This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracking, and iteration to develop well-considered thinking process. " +
		"Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. " +
		"In the Thought section, detail your reasoning process in steps. In the Solution section, systematically present the final solution that you deem correct."
)

// Config selects the model and tokenizer files to load.
type Config struct {
	ModelPath     string
	TokenizerPath string
}

// Client owns a loaded SmolLM3 model and tokenizer.
type Client struct {
	transformer *model.Transformer
	tokenizer   *tokenizer.Tokenizer
}

// GenerateOptions controls text generation.
type GenerateOptions struct {
	// MaxNewTokens limits how many tokens are sampled. Values <= 0 use the
	// package default.
	MaxNewTokens int
	// Temperature controls sampling entropy. Temperature 0 selects greedy
	// decoding. Invalid negative, NaN, or infinite values use the package default.
	Temperature float64
	// TopP enables nucleus sampling when 0 < TopP < 1. Invalid values use the
	// package default.
	TopP float64
	// Seed initializes the sampler. Zero is a valid deterministic seed.
	Seed int64
	// TokenCallback is called with each decoded token piece as it is generated.
	// Returning an error stops generation and returns the partial output.
	TokenCallback func(string) error
}

// ChatOptions controls chat prompt rendering and answer generation.
type ChatOptions struct {
	GenerateOptions
	SystemPrompt string
	Thinking     bool
}

// Message is one chat-template message.
type Message struct {
	Role    string
	Content string
}

// GenerationStats describes one generation call.
type GenerationStats struct {
	PromptTokens    int
	GeneratedTokens int
	Duration        time.Duration
}

// TokensPerSecond returns the generation throughput, excluding prompt prefill.
func (s GenerationStats) TokensPerSecond() float64 {
	if s.GeneratedTokens == 0 || s.Duration <= 0 {
		return 0
	}
	return float64(s.GeneratedTokens) / s.Duration.Seconds()
}

// Load opens a model checkpoint and matching tokenizer.
func Load(cfg Config) (*Client, error) {
	if strings.TrimSpace(cfg.ModelPath) == "" {
		return nil, errors.New("model path is required")
	}
	if strings.TrimSpace(cfg.TokenizerPath) == "" {
		return nil, errors.New("tokenizer path is required")
	}
	t, err := model.Load(cfg.ModelPath)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.Load(cfg.TokenizerPath, t.Config.VocabSize)
	if err != nil {
		return nil, err
	}
	return &Client{transformer: t, tokenizer: tok}, nil
}

// Encode tokenizes text with the loaded tokenizer.
func (c *Client) Encode(text string, bos bool, eos bool) []int {
	return c.tokenizer.Encode(text, bos, eos)
}

// Decode decodes one token id.
func (c *Client) Decode(id int) string {
	return c.tokenizer.Decode(id)
}

// EOS returns the tokenizer's end-of-sequence token id.
func (c *Client) EOS() int {
	return c.tokenizer.EOS()
}

// Generate returns a plain text continuation for prompt.
func (c *Client) Generate(ctx context.Context, prompt string, opts GenerateOptions) (string, GenerationStats, error) {
	if c == nil || c.transformer == nil || c.tokenizer == nil {
		return "", GenerationStats{}, errors.New("nil smollm3 client")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	opts = normalizeGenerateOptions(opts)
	ids := c.tokenizer.Encode(prompt, false, false)
	if len(ids) == 0 {
		ids = append(ids, c.tokenizer.EOS())
	}
	if err := validateContextWindow(len(ids), 0, opts.MaxNewTokens, c.transformer.Config.SeqLen); err != nil {
		return "", GenerationStats{PromptTokens: len(ids)}, err
	}
	logits, pos := c.forwardTokens(ids, 0)
	out, pos, generated, duration, err := c.generateFromLogits(ctx, sampler.New(float32(opts.Temperature), float32(opts.TopP), opts.Seed), logits, pos, opts.MaxNewTokens, opts.TokenCallback, false)
	_ = pos
	return out, GenerationStats{PromptTokens: len(ids), GeneratedTokens: generated, Duration: duration}, err
}

// Chat renders messages with the SmolLM3 chat template and generates one reply.
func (c *Client) Chat(ctx context.Context, messages []Message, opts ChatOptions) (string, GenerationStats, error) {
	if c == nil || c.transformer == nil || c.tokenizer == nil {
		return "", GenerationStats{}, errors.New("nil smollm3 client")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	opts.GenerateOptions = normalizeGenerateOptions(opts.GenerateOptions)
	rendered := RenderChatPrompt(messages, opts.SystemPrompt, opts.Thinking)
	ids := c.tokenizer.Encode(rendered, false, false)
	if err := validateContextWindow(len(ids), 0, opts.MaxNewTokens, c.transformer.Config.SeqLen); err != nil {
		return "", GenerationStats{PromptTokens: len(ids)}, err
	}
	logits, pos := c.forwardTokens(ids, 0)
	out, pos, generated, duration, err := c.generateFromLogits(ctx, sampler.New(float32(opts.Temperature), float32(opts.TopP), opts.Seed), logits, pos, opts.MaxNewTokens, opts.TokenCallback, true)
	_ = pos
	return out, GenerationStats{PromptTokens: len(ids), GeneratedTokens: generated, Duration: duration}, err
}

// NewChatSession starts a stateful chat session. The session reuses the
// client's KV cache, so do not interleave it with other generation calls on the
// same Client.
func (c *Client) NewChatSession(opts ChatOptions) *ChatSession {
	opts.GenerateOptions = normalizeGenerateOptions(opts.GenerateOptions)
	return &ChatSession{
		client:  c,
		opts:    opts,
		sampler: sampler.New(float32(opts.Temperature), float32(opts.TopP), opts.Seed),
	}
}

// ChatSession keeps a running chat context in the model KV cache.
type ChatSession struct {
	client  *Client
	opts    ChatOptions
	sampler *sampler.Sampler
	pos     int
	ready   bool
}

// Reply appends one user turn to the session and returns the assistant reply.
func (s *ChatSession) Reply(ctx context.Context, userPrompt string) (string, GenerationStats, error) {
	if s == nil || s.client == nil {
		return "", GenerationStats{}, errors.New("nil smollm3 chat session")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	c := s.client
	prompt := renderUserTurn(userPrompt, s.opts.Thinking)
	if !s.ready {
		prompt = renderSystemPrompt(s.opts.SystemPrompt, s.opts.Thinking) + prompt
	}
	ids := c.tokenizer.Encode(prompt, false, false)
	if err := validateContextWindow(len(ids), s.pos, s.opts.MaxNewTokens, c.transformer.Config.SeqLen); err != nil {
		return "", GenerationStats{PromptTokens: len(ids)}, err
	}
	s.ready = true
	logits, pos := c.forwardTokens(ids, s.pos)
	out, pos, generated, duration, err := c.generateFromLogits(ctx, s.sampler, logits, pos, s.opts.MaxNewTokens, s.opts.TokenCallback, true)
	s.pos = pos
	return out, GenerationStats{PromptTokens: len(ids), GeneratedTokens: generated, Duration: duration}, err
}

// RenderChatPrompt renders messages with the SmolLM3 chat template.
func RenderChatPrompt(messages []Message, systemPrompt string, thinking bool) string {
	var b strings.Builder
	b.WriteString(renderSystemPrompt(systemPrompt, thinking))
	for _, msg := range messages {
		b.WriteString("<|im_start|>")
		b.WriteString(msg.Role)
		b.WriteByte('\n')
		if msg.Role == "assistant" && !thinking {
			b.WriteString("<think>\n\n</think>\n")
		}
		b.WriteString(msg.Content)
		b.WriteString("<|im_end|>\n")
	}
	b.WriteString("<|im_start|>assistant\n")
	if !thinking {
		b.WriteString("<think>\n\n</think>\n")
	}
	return b.String()
}

func normalizeGenerateOptions(opts GenerateOptions) GenerateOptions {
	if opts.MaxNewTokens <= 0 {
		opts.MaxNewTokens = defaultMaxNewTokens
	}
	if opts.Temperature < 0 || math.IsNaN(opts.Temperature) || math.IsInf(opts.Temperature, 0) {
		opts.Temperature = defaultTemperature
	}
	if opts.TopP < 0 || opts.TopP > 1 || math.IsNaN(opts.TopP) || math.IsInf(opts.TopP, 0) {
		opts.TopP = defaultTopP
	}
	return opts
}

func validateContextWindow(promptTokens int, existingTokens int, maxNewTokens int, seqLen int) error {
	if existingTokens < 0 {
		existingTokens = 0
	}
	if promptTokens > seqLen-existingTokens {
		return fmt.Errorf("prompt uses %d tokens with %d existing context tokens, exceeding context length %d", promptTokens, existingTokens, seqLen)
	}
	remaining := seqLen - existingTokens - promptTokens
	if maxNewTokens > remaining {
		return fmt.Errorf("max new tokens %d exceeds remaining context %d tokens", maxNewTokens, remaining)
	}
	return nil
}

func (c *Client) forwardTokens(ids []int, pos int) ([]float32, int) {
	if len(ids) == 0 || pos >= c.transformer.Config.SeqLen {
		return nil, pos
	}
	end := min(len(ids), c.transformer.Config.SeqLen-pos)
	return c.transformer.Prefill(ids[:end], pos), pos + end
}

func (c *Client) generateFromLogits(ctx context.Context, samp *sampler.Sampler, logits []float32, pos int, maxNew int, tokenCallback func(string) error, closeTurn bool) (string, int, int, time.Duration, error) {
	if len(logits) == 0 {
		return "", pos, 0, 0, nil
	}
	var out strings.Builder
	start := time.Now()
	generated := 0
	for generated < maxNew && pos < c.transformer.Config.SeqLen {
		if err := ctx.Err(); err != nil {
			return out.String(), pos, generated, elapsed(start, generated), err
		}
		next := samp.Sample(logits)
		if next == c.tokenizer.EOS() {
			if closeTurn {
				pos = c.closeAssistantTurn(pos)
			}
			break
		}
		piece := c.tokenizer.Decode(next)
		if tokenCallback != nil {
			if err := tokenCallback(piece); err != nil {
				return out.String(), pos, generated, elapsed(start, generated), fmt.Errorf("token callback: %w", err)
			}
		}
		out.WriteString(piece)
		logits = c.transformer.Forward(next, pos)
		pos++
		generated++
	}
	if generated == maxNew && closeTurn && pos < c.transformer.Config.SeqLen {
		pos = c.closeAssistantTurn(pos)
	}
	return out.String(), pos, generated, elapsed(start, generated), nil
}

func elapsed(start time.Time, generated int) time.Duration {
	if generated == 0 {
		return 0
	}
	return time.Since(start)
}

func (c *Client) closeAssistantTurn(pos int) int {
	if pos < c.transformer.Config.SeqLen {
		c.transformer.Forward(c.tokenizer.EOS(), pos)
		pos++
	}
	ids := c.tokenizer.Encode("\n", false, false)
	for i := 0; i < len(ids) && pos < c.transformer.Config.SeqLen; i++ {
		c.transformer.Forward(ids[i], pos)
		pos++
	}
	return pos
}

func renderSystemPrompt(systemPrompt string, thinking bool) string {
	if strings.Contains(systemPrompt, "/no_think") {
		thinking = false
	} else if strings.Contains(systemPrompt, "/think") {
		thinking = true
	}
	systemPrompt = strings.ReplaceAll(systemPrompt, "/no_think", "")
	systemPrompt = strings.ReplaceAll(systemPrompt, "/think", "")
	systemPrompt = strings.TrimSpace(systemPrompt)
	if systemPrompt == "" {
		systemPrompt = defaultSystemPrompt
		if thinking {
			systemPrompt = defaultThinkingPrompt
		}
	}
	reasoningMode := "/no_think"
	if thinking {
		reasoningMode = "/think"
	}
	return "<|im_start|>system\n## Metadata\n\nReasoning Mode: " + reasoningMode + "\n\n## Custom Instructions\n\n" + systemPrompt + "\n<|im_end|>\n"
}

func renderUserTurn(userPrompt string, thinking bool) string {
	prompt := "<|im_start|>user\n" + userPrompt + "<|im_end|>\n<|im_start|>assistant\n"
	if !thinking {
		prompt += "<think>\n\n</think>\n"
	}
	return prompt
}
