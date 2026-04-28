package main

import (
	"fmt"
	"os"
	"path/filepath"
	"smollm3go/internal/model"
	"smollm3go/internal/tokenizer"
	"strings"
	"testing"
)

func TestRenderChatPromptIncludesHistory(t *testing.T) {
	messages := []chatMessage{
		{role: "user", content: "hello"},
		{role: "assistant", content: "hi"},
		{role: "user", content: "again"},
	}
	got := renderChatPrompt(messages, "system", true)
	want := "<|im_start|>system\n## Metadata\n\nReasoning Mode: /think\n\n## Custom Instructions\n\nsystem\n<|im_end|>\n" +
		"<|im_start|>user\nhello<|im_end|>\n" +
		"<|im_start|>assistant\nhi<|im_end|>\n" +
		"<|im_start|>user\nagain<|im_end|>\n" +
		"<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("renderChatPrompt() = %q, want %q", got, want)
	}
}

func TestRenderChatPromptUsesDefaultSystemPrompt(t *testing.T) {
	got := renderChatPrompt(nil, "", true)
	if !strings.Contains(got, "You are a helpful AI assistant named SmolLM") {
		t.Fatalf("renderChatPrompt() = %q, want default system prompt", got)
	}
}

func TestRenderToolCallPromptIncludesBuiltinToolAndUserPrompt(t *testing.T) {
	got := renderToolCallPrompt("What time is it?")
	for _, want := range []string{
		`"name":"get_current_hour"`,
		`"name":"get_random_number_between"`,
		`"required":["min","max"]`,
		"Returns only the current hour of day in 24-hour format",
		`"type":"function"`,
		"<tool_call>[",
		"<|im_start|>user\nWhat time is it?<|im_end|>",
		"<|im_start|>assistant\n",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("renderToolCallPrompt() missing %q in %q", want, got)
		}
	}
}

func TestRenderToolResultPromptAsksForFinalAnswer(t *testing.T) {
	got := renderToolResultPrompt(
		"Can you give me the hour and a random number between 1 and 50?",
		`<tool_call>[{"name":"get_current_hour","arguments":{}},{"name":"get_random_number_between","arguments":{"min":1,"max":50}}]</tool_call>`,
		[]toolResultItem{
			{Name: "get_current_hour", Result: "13"},
			{Name: "get_random_number_between", Result: "42"},
		},
	)
	for _, want := range []string{
		"The conversation includes an assistant tool call followed by a tool result.",
		"Use the tool result to answer the user's original request.",
		"If there are multiple tool results, include all of them in the answer.",
		"Do not call tools again.",
		"<|im_start|>user\nCan you give me the hour and a random number between 1 and 50?<|im_end|>",
		"<|im_start|>assistant\nI called these tools: get_current_hour, get_random_number_between.<|im_end|>",
		"<|im_start|>tool\nget_current_hour result: 13\nget_random_number_between result: 42<|im_end|>",
		"<|im_start|>assistant\n",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("renderToolResultPrompt() missing %q in %q", want, got)
		}
	}
}

func TestParseToolCalls(t *testing.T) {
	got, err := parseToolCalls(`prefix <tool_call>[{"name":"get_current_hour","arguments":{}},{"name":"get_random_number_between","arguments":{"min":1,"max":50}}]</tool_call> suffix`)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].Name != "get_current_hour" || got[1].Name != "get_random_number_between" {
		t.Fatalf("parseToolCalls() = %#v", got)
	}
	if got[1].Arguments["min"] != float64(1) || got[1].Arguments["max"] != float64(50) {
		t.Fatalf("parseToolCalls() arguments = %#v", got[1].Arguments)
	}
}

func TestParseToolCallsAcceptsEmptyList(t *testing.T) {
	got, err := parseToolCalls(`<tool_call>[]</tool_call>`)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatalf("parseToolCalls() = %#v, want empty list", got)
	}
}

func TestParseToolCallsAcceptsSingleObject(t *testing.T) {
	got, err := parseToolCalls(`<tool_call>{"name":"get_current_hour","arguments":{}}</tool_call>`)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Name != "get_current_hour" {
		t.Fatalf("parseToolCalls() = %#v", got)
	}
}

func TestParseToolCallsRejectsMalformedOutput(t *testing.T) {
	tests := []string{
		`[{"name":"get_weather","arguments":{}}]`,
		`<tool_call>"get_weather"</tool_call>`,
		`<tool_call>[{"name":"","arguments":{}}]</tool_call>`,
		`<tool_call>[{"name":"get_weather"}]</tool_call>`,
	}
	for _, text := range tests {
		if _, err := parseToolCalls(text); err == nil {
			t.Fatalf("parseToolCalls(%q) returned nil error", text)
		}
	}
}

func TestRunTool(t *testing.T) {
	got, err := runTool(toolCallItem{Name: "get_current_hour", Arguments: map[string]any{}})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len("15") {
		t.Fatalf("runTool() = %q, want formatted hour", got)
	}
}

func TestRunRandomNumberTool(t *testing.T) {
	got, err := runTool(toolCallItem{
		Name:      "get_random_number_between",
		Arguments: map[string]any{"min": float64(1), "max": float64(50)},
	})
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	if _, err := fmt.Sscanf(got, "%d", &n); err != nil {
		t.Fatalf("runTool() = %q, want integer", got)
	}
	if n < 1 || n > 50 {
		t.Fatalf("runTool() = %d, want between 1 and 50", n)
	}
}

func TestRunToolRejectsUnknownTool(t *testing.T) {
	_, err := runTool(toolCallItem{Name: "get_weather", Arguments: map[string]any{}})
	if err == nil {
		t.Fatal("runTool() returned nil error for unknown tool")
	}
}

const benchPrompt = "Hello, my name is"

func benchPaths(b *testing.B) (string, string) {
	b.Helper()
	modelPath := filepath.Join("..", "..", "models", "smollm3-3b-f32.bin")
	tokenizerPath := filepath.Join("..", "..", "models", "smollm3-tokenizer.bin")
	if _, err := os.Stat(modelPath); err != nil {
		b.Skipf("model checkpoint not found: %s", modelPath)
	}
	if _, err := os.Stat(tokenizerPath); err != nil {
		b.Skipf("tokenizer not found: %s", tokenizerPath)
	}
	return modelPath, tokenizerPath
}

func BenchmarkInitialize(b *testing.B) {
	modelPath, tokenizerPath := benchPaths(b)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		t, err := model.Load(modelPath)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := tokenizer.Load(tokenizerPath, t.Config.VocabSize); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEncodePrompt(b *testing.B) {
	benchmarkEncode(b, benchPrompt)
}

func BenchmarkEncodeLongPrompt(b *testing.B) {
	benchmarkEncode(b, strings.Repeat(benchPrompt+". ", 512))
}

func benchmarkEncode(b *testing.B, prompt string) {
	modelPath, tokenizerPath := benchPaths(b)
	t, err := model.Load(modelPath)
	if err != nil {
		b.Fatal(err)
	}
	tok, err := tokenizer.Load(tokenizerPath, t.Config.VocabSize)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tok.Encode(prompt, false, false)
	}
}
