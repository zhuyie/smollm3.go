package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/zhuyie/smollm3.go"
)

func TestToolPrompts(t *testing.T) {
	got := toolCallSystemPrompt()
	for _, want := range []string{
		"get_product_price",
		"<tools>",
		"<tool_call>",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("toolCallSystemPrompt() missing %q in %q", want, got)
		}
	}

	if got := toolResultSystemPrompt(); !strings.Contains(got, "Do not call tools again.") {
		t.Fatalf("toolResultSystemPrompt() = %q, want final-answer instruction", got)
	}
}

func TestRenderToolResponse(t *testing.T) {
	got := renderToolResponse([]toolResultItem{{Name: "get_product_price", Result: "12"}})
	want := "<tool_response>\n{\"name\":\"get_product_price\",\"result\":\"12\"}\n</tool_response>"
	if got != want {
		t.Fatalf("renderToolResponse() = %q, want %q", got, want)
	}
}

func TestParseToolCalls(t *testing.T) {
	tests := []struct {
		name string
		text string
		want []toolCallItem
	}{
		{
			name: "array",
			text: `prefix <tool_call>[{"name":"get_product_price","arguments":{"product":"notebook"}}]</tool_call> suffix`,
			want: []toolCallItem{{Name: "get_product_price", Arguments: map[string]any{"product": "notebook"}}},
		},
		{
			name: "single object",
			text: `<tool_call>{"name":"get_product_price","arguments":{"product":"notebook"}}</tool_call>`,
			want: []toolCallItem{{Name: "get_product_price", Arguments: map[string]any{"product": "notebook"}}},
		},
		{
			name: "empty list",
			text: `<tool_call>[]</tool_call>`,
			want: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseToolCalls(tt.text)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(tt.want) {
				t.Fatalf("parseToolCalls() = %#v, want %#v", got, tt.want)
			}
			for i := range got {
				if got[i].Name != tt.want[i].Name {
					t.Fatalf("parseToolCalls()[%d].Name = %q, want %q", i, got[i].Name, tt.want[i].Name)
				}
				for key, want := range tt.want[i].Arguments {
					if got[i].Arguments[key] != want {
						t.Fatalf("parseToolCalls()[%d].Arguments[%q] = %#v, want %#v", i, key, got[i].Arguments[key], want)
					}
				}
			}
		})
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
	got, err := runTool(toolCallItem{Name: "get_product_price", Arguments: map[string]any{"product": "notebook"}})
	if err != nil {
		t.Fatal(err)
	}
	if got != "12" {
		t.Fatalf("runTool() = %q, want mock product price", got)
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
		if _, err := smollm3.Load(smollm3.Config{ModelPath: modelPath, TokenizerPath: tokenizerPath}); err != nil {
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
	client, err := smollm3.Load(smollm3.Config{ModelPath: modelPath, TokenizerPath: tokenizerPath})
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		client.Encode(prompt, false, false)
	}
}
