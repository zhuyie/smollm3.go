package smollm3

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestLoadRequiresPaths(t *testing.T) {
	if _, err := Load(Config{}); err == nil {
		t.Fatal("Load(Config{}) returned nil error")
	}
	if _, err := Load(Config{ModelPath: "model.bin"}); err == nil {
		t.Fatal("Load without tokenizer returned nil error")
	}
}

func TestRenderChatPromptIncludesHistory(t *testing.T) {
	messages := []Message{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi"},
		{Role: "user", Content: "again"},
	}
	got := RenderChatPrompt(messages, "system", true)
	want := "<|im_start|>system\n## Metadata\n\nReasoning Mode: /think\n\n## Custom Instructions\n\nsystem\n<|im_end|>\n" +
		"<|im_start|>user\nhello<|im_end|>\n" +
		"<|im_start|>assistant\nhi<|im_end|>\n" +
		"<|im_start|>user\nagain<|im_end|>\n" +
		"<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("RenderChatPrompt() = %q, want %q", got, want)
	}
}

func TestRenderChatPromptCanDisableThinking(t *testing.T) {
	got := RenderChatPrompt([]Message{{Role: "user", Content: "2+2?"}}, "concise", false)
	for _, want := range []string{
		"Reasoning Mode: /no_think",
		"<|im_start|>user\n2+2?<|im_end|>",
		"<|im_start|>assistant\n<think>\n\n</think>\n",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("RenderChatPrompt() missing %q in %q", want, got)
		}
	}
}

func TestRenderChatPromptUsesDefaultSystemPrompt(t *testing.T) {
	got := RenderChatPrompt(nil, "", true)
	for _, want := range []string{
		"You are a helpful AI assistant named SmolLM",
		"Reasoning Mode: /think",
		"Please structure your response into two main sections: Thought and Solution",
		"<think> Thought section </think> Solution section",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("RenderChatPrompt() missing %q in %q", want, got)
		}
	}
}

func TestRenderChatPromptDoesNotAddDefaultThinkingPromptToCustomSystemPrompt(t *testing.T) {
	got := RenderChatPrompt(nil, "custom /think", false)
	for _, want := range []string{
		"Reasoning Mode: /think",
		"## Custom Instructions\n\ncustom\n",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("RenderChatPrompt() missing %q in %q", want, got)
		}
	}
}

func TestRenderChatPromptNoThinkFlagTakesPrecedence(t *testing.T) {
	got := RenderChatPrompt(nil, "/think /no_think", true)
	for _, want := range []string{
		"Reasoning Mode: /no_think",
		defaultSystemPrompt,
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("RenderChatPrompt() missing %q in %q", want, got)
		}
	}
}

func TestNormalizeGenerateOptions(t *testing.T) {
	got := normalizeGenerateOptions(GenerateOptions{MaxNewTokens: -1, Temperature: -1, TopP: 2})
	if got.MaxNewTokens != defaultMaxNewTokens {
		t.Fatalf("MaxNewTokens = %d, want %d", got.MaxNewTokens, defaultMaxNewTokens)
	}
	if got.Temperature != defaultTemperature {
		t.Fatalf("Temperature = %v, want %v", got.Temperature, defaultTemperature)
	}
	if got.TopP != defaultTopP {
		t.Fatalf("TopP = %v, want %v", got.TopP, defaultTopP)
	}
}

func TestValidateContextWindow(t *testing.T) {
	if err := validateContextWindow(3, 2, 5, 10); err != nil {
		t.Fatalf("validateContextWindow() returned error for exact fit: %v", err)
	}

	tests := []struct {
		name           string
		promptTokens   int
		existingTokens int
		maxNewTokens   int
		seqLen         int
		want           string
	}{
		{
			name:           "prompt exceeds context",
			promptTokens:   11,
			existingTokens: 0,
			maxNewTokens:   1,
			seqLen:         10,
			want:           "prompt uses 11 tokens with 0 existing context tokens, exceeding context length 10",
		},
		{
			name:           "max new exceeds remaining context",
			promptTokens:   4,
			existingTokens: 0,
			maxNewTokens:   7,
			seqLen:         10,
			want:           "max new tokens 7 exceeds remaining context 6 tokens",
		},
		{
			name:           "existing context reduces remaining context",
			promptTokens:   3,
			existingTokens: 5,
			maxNewTokens:   3,
			seqLen:         10,
			want:           "max new tokens 3 exceeds remaining context 2 tokens",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateContextWindow(tt.promptTokens, tt.existingTokens, tt.maxNewTokens, tt.seqLen)
			if err == nil {
				t.Fatal("validateContextWindow() returned nil error")
			}
			if err.Error() != tt.want {
				t.Fatalf("validateContextWindow() error = %q, want %q", err, tt.want)
			}
		})
	}
}

func TestNilClientGenerateAndChatReturnErrors(t *testing.T) {
	var c *Client
	if _, _, err := c.Generate(context.Background(), "hello", GenerateOptions{}); err == nil {
		t.Fatal("nil Client.Generate returned nil error")
	}
	if _, _, err := c.Chat(context.Background(), []Message{{Role: "user", Content: "hello"}}, ChatOptions{}); err == nil {
		t.Fatal("nil Client.Chat returned nil error")
	}
}

func TestGenerationStatsTokensPerSecond(t *testing.T) {
	stats := GenerationStats{GeneratedTokens: 4, Duration: 2 * time.Second}
	if got := stats.TokensPerSecond(); got != 2 {
		t.Fatalf("TokensPerSecond() = %v, want 2", got)
	}
}
