package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"strings"
	"time"

	"smollm3go/internal/model"
	"smollm3go/internal/sampler"
	"smollm3go/internal/tokenizer"
)

type chatMessage struct {
	role    string
	content string
}

type toolCallItem struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

type toolResultItem struct {
	Name   string
	Result string
}

const (
	ansiReset     = "\x1b[0m"
	ansiPrompt    = "\x1b[33m"
	ansiUserInput = "\x1b[1m\x1b[32m"
)

func main() {
	modelPath := flag.String("model", "", "SML3 model path")
	tokenizerPath := flag.String("tokenizer", "", "TOK3 tokenizer path")
	mode := flag.String("mode", "generate", "generate|chat|toolcall")
	prompt := flag.String("prompt", "", "input prompt")
	systemPrompt := flag.String("system", "", "optional system prompt for chat")
	thinking := flag.Bool("think", true, "enable SmolLM3 extended thinking chat template")
	maxNew := flag.Int("n", 256, "maximum new tokens")
	temperature := flag.Float64("temp", 1.0, "sampling temperature, 0 for greedy")
	topP := flag.Float64("top-p", 0.9, "top-p nucleus sampling")
	seed := flag.Int64("seed", time.Now().UnixNano(), "random seed")
	flag.Parse()

	if *modelPath == "" || *tokenizerPath == "" {
		flag.Usage()
		os.Exit(2)
	}

	transformer, err := model.Load(*modelPath)
	if err != nil {
		log.Fatal(err)
	}
	tok, err := tokenizer.Load(*tokenizerPath, transformer.Config.VocabSize)
	if err != nil {
		log.Fatal(err)
	}
	samp := sampler.New(float32(*temperature), float32(*topP), *seed)

	switch *mode {
	case "generate":
		generate(transformer, tok, samp, *prompt, *maxNew)
	case "chat":
		chat(transformer, tok, samp, *prompt, *systemPrompt, *thinking, *maxNew)
	case "toolcall":
		toolCall(transformer, tok, samp, *prompt, *maxNew)
	default:
		log.Fatalf("unknown mode %q", *mode)
	}
}

func generate(t *model.Transformer, tok *tokenizer.Tokenizer, samp *sampler.Sampler, prompt string, maxNew int) {
	ids := tok.Encode(prompt, false, false)
	if len(ids) == 0 {
		ids = append(ids, tok.EOS())
	}
	var logits []float32
	pos := 0
	// Prefill consumes the whole prompt and leaves logits for the next token.
	if len(ids) > 0 && pos < t.Config.SeqLen {
		end := min(len(ids), t.Config.SeqLen)
		logits = t.Prefill(ids[:end], pos)
		pos = end
	}
	generated := 0
	token := -1
	start := time.Time{}
	for generated < maxNew && pos < t.Config.SeqLen {
		next := samp.Sample(logits)
		if next == tok.EOS() {
			break
		}
		fmt.Print(tok.Decode(next))
		// Decode one token at a time, appending its KV entries to the cache.
		if token = next; token >= 0 {
			logits = t.Forward(token, pos)
			pos++
		}
		generated++
		if start.IsZero() {
			start = time.Now()
		}
	}
	fmt.Println()
	if generated > 1 && !start.IsZero() {
		tokPerSec := float64(generated) / time.Since(start).Seconds()
		fmt.Fprintf(os.Stderr, "achieved tok/s: %.6f\n", tokPerSec)
	}
}

func chat(t *model.Transformer, tok *tokenizer.Tokenizer, samp *sampler.Sampler, userPrompt string, systemPrompt string, thinking bool, maxNew int) {
	if userPrompt != "" {
		messages := []chatMessage{{role: "user", content: userPrompt}}
		printAssistantPrefix(os.Stdout)
		chatReply(t, tok, samp, renderChatPrompt(messages, systemPrompt, thinking), maxNew, os.Stdout)
		fmt.Println(ansiReset)
		return
	}

	pos := 0
	var logits []float32
	logits, pos = forwardTokens(t, tok.Encode(renderSystemPrompt(systemPrompt, thinking), false, false), pos)
	totalGenerated := 0
	totalDuration := time.Duration(0)
	scanner := bufio.NewScanner(os.Stdin)
	for {
		printUserPrefix(os.Stdout)
		if !scanner.Scan() {
			fmt.Print(ansiReset)
			break
		}
		fmt.Print(ansiReset)
		userPrompt := strings.TrimSpace(scanner.Text())
		if userPrompt == "" {
			continue
		}
		if userPrompt == "/exit" || userPrompt == "/quit" {
			fmt.Println()
			break
		}
		logits, pos = forwardTokens(t, tok.Encode(renderUserTurn(userPrompt, thinking), false, false), pos)
		printAssistantPrefix(os.Stdout)
		var generated int
		var duration time.Duration
		_, pos, generated, duration = generateAssistant(t, tok, samp, logits, pos, maxNew, os.Stdout)
		totalGenerated += generated
		totalDuration += duration
		fmt.Println(ansiReset)
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
	if totalGenerated > 1 && totalDuration > 0 {
		tokPerSec := float64(totalGenerated) / totalDuration.Seconds()
		fmt.Fprintf(os.Stderr, "achieved tok/s: %.6f\n", tokPerSec)
	}
}

func toolCall(t *model.Transformer, tok *tokenizer.Tokenizer, samp *sampler.Sampler, prompt string, maxNew int) {
	if strings.TrimSpace(prompt) == "" {
		log.Fatal("toolcall mode requires -prompt")
	}
	toolRequest := chatReply(t, tok, samp, renderToolCallPrompt(prompt), maxNew, io.Discard)
	calls, err := parseToolCalls(toolRequest)
	if err != nil {
		log.Fatal(err)
	}
	if len(calls) == 0 {
		log.Fatal("model returned no tool calls")
	}
	results, err := runTools(calls)
	if err != nil {
		log.Fatal(err)
	}
	for _, result := range results {
		fmt.Fprintf(os.Stderr, "tool: %s -> %s\n", result.Name, result.Result)
	}
	chatReply(t, tok, samp, renderToolResultPrompt(prompt, toolRequest, results), maxNew, os.Stdout)
	fmt.Println()
}

func printUserPrefix(w io.Writer) {
	fmt.Fprint(w, ansiPrompt, "User: ", ansiUserInput)
}

func printAssistantPrefix(w io.Writer) {
	fmt.Fprint(w, ansiPrompt, "Assistant: ", ansiReset)
}

func chatReply(t *model.Transformer, tok *tokenizer.Tokenizer, samp *sampler.Sampler, rendered string, maxNew int, w io.Writer) string {
	ids := tok.Encode(rendered, false, false)
	pos := 0
	// Chat mode differs from generate mode only in prompt rendering.
	logits, pos := forwardTokens(t, ids, pos)
	out, _, _, _ := generateAssistant(t, tok, samp, logits, pos, maxNew, w)
	return out
}

func generateAssistant(t *model.Transformer, tok *tokenizer.Tokenizer, samp *sampler.Sampler, logits []float32, pos int, maxNew int, w io.Writer) (string, int, int, time.Duration) {
	var out strings.Builder
	generated := 0
	start := time.Now()
	for generated < maxNew && pos < t.Config.SeqLen {
		next := samp.Sample(logits)
		if next == tok.EOS() {
			pos = closeAssistantTurn(t, tok, pos)
			break
		}
		piece := tok.Decode(next)
		fmt.Fprint(w, piece)
		out.WriteString(piece)
		logits = t.Forward(next, pos)
		pos++
		generated++
	}
	if generated == maxNew && pos < t.Config.SeqLen {
		pos = closeAssistantTurn(t, tok, pos)
	}
	duration := time.Duration(0)
	if generated > 0 {
		duration = time.Since(start)
	}
	return out.String(), pos, generated, duration
}

func closeAssistantTurn(t *model.Transformer, tok *tokenizer.Tokenizer, pos int) int {
	if pos < t.Config.SeqLen {
		t.Forward(tok.EOS(), pos)
		pos++
	}
	ids := tok.Encode("\n", false, false)
	for i := 0; i < len(ids) && pos < t.Config.SeqLen; i++ {
		t.Forward(ids[i], pos)
		pos++
	}
	return pos
}

func forwardTokens(t *model.Transformer, ids []int, pos int) ([]float32, int) {
	if len(ids) == 0 || pos >= t.Config.SeqLen {
		return nil, pos
	}
	end := min(len(ids), t.Config.SeqLen-pos)
	return t.Prefill(ids[:end], pos), pos + end
}

func renderChatPrompt(messages []chatMessage, systemPrompt string, thinking bool) string {
	var b strings.Builder
	b.WriteString(renderSystemPrompt(systemPrompt, thinking))
	for _, msg := range messages {
		b.WriteString("<|im_start|>")
		b.WriteString(msg.role)
		b.WriteByte('\n')
		b.WriteString(msg.content)
		b.WriteString("<|im_end|>\n")
	}
	b.WriteString("<|im_start|>assistant\n")
	if !thinking {
		b.WriteString("<think>\n\n</think>\n")
	}
	return b.String()
}

func renderSystemPrompt(systemPrompt string, thinking bool) string {
	if strings.Contains(systemPrompt, "/no_think") {
		thinking = false
		systemPrompt = strings.ReplaceAll(systemPrompt, "/no_think", "")
	}
	if strings.Contains(systemPrompt, "/think") {
		thinking = true
		systemPrompt = strings.ReplaceAll(systemPrompt, "/think", "")
	}
	systemPrompt = strings.TrimSpace(systemPrompt)
	if systemPrompt == "" {
		if thinking {
			systemPrompt = "You are a helpful AI assistant named SmolLM, trained by Hugging Face. Structure your response into two sections: <think> for reasoning and a concise final solution after </think>."
		} else {
			systemPrompt = "You are a helpful AI assistant named SmolLM, trained by Hugging Face."
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

func renderToolCallPrompt(userPrompt string) string {
	systemPrompt := `You are an expert in composing functions. You are given a question and a set of possible functions.
Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out and refuse to answer.
If the given question lacks the parameters required by the function, also point it out.

You have access to the following tools:
<tools>[{"type":"function","function":{"name":"get_random_number_between","description":"Gets a random integer between two numeric bounds.","parameters":{"type":"object","properties":{"min":{"type":"integer","description":"The minimum numeric bound."},"max":{"type":"integer","description":"The maximum numeric bound."}},"required":["min","max"]}}},{"type":"function","function":{"name":"get_current_hour","description":"Returns only the current hour of day in 24-hour format, such as 07 or 22.","parameters":{"type":"object","properties":{},"required":[]}}}]</tools>

The output MUST strictly adhere to the following format, and NO other text MUST be included.
The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make the tool calls an empty list '[]'.
<tool_call>[
{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}}
]</tool_call>`
	return renderChatPrompt([]chatMessage{{role: "user", content: userPrompt}}, systemPrompt, false)
}

func renderToolResultPrompt(userPrompt string, _ string, results []toolResultItem) string {
	systemPrompt := `You are a helpful AI assistant.
The conversation includes an assistant tool call followed by a tool result.
Use the tool result to answer the user's original request.
If there are multiple tool results, include all of them in the answer.
Do not call tools again.`
	names := make([]string, 0, len(results))
	for _, result := range results {
		names = append(names, result.Name)
	}
	messages := []chatMessage{
		{role: "user", content: userPrompt},
		{role: "assistant", content: "I called these tools: " + strings.Join(names, ", ") + "."},
	}
	var toolOut strings.Builder
	for _, result := range results {
		if toolOut.Len() > 0 {
			toolOut.WriteByte('\n')
		}
		toolOut.WriteString(result.Name)
		toolOut.WriteString(" result: ")
		toolOut.WriteString(result.Result)
	}
	messages = append(messages, chatMessage{role: "tool", content: toolOut.String()})
	return renderChatPrompt(messages, systemPrompt, false)
}

func parseToolCalls(text string) ([]toolCallItem, error) {
	const startTag = "<tool_call>"
	const endTag = "</tool_call>"
	start := strings.Index(text, startTag)
	if start < 0 {
		return nil, fmt.Errorf("missing %s in model output", startTag)
	}
	start += len(startTag)
	end := strings.Index(text[start:], endTag)
	if end < 0 {
		return nil, fmt.Errorf("missing %s in model output", endTag)
	}
	payload := strings.TrimSpace(text[start : start+end])
	var calls []toolCallItem
	if strings.HasPrefix(payload, "[") {
		if err := json.Unmarshal([]byte(payload), &calls); err != nil {
			return nil, fmt.Errorf("invalid tool call JSON: %w", err)
		}
	} else {
		var call toolCallItem
		if err := json.Unmarshal([]byte(payload), &call); err != nil {
			return nil, fmt.Errorf("invalid tool call JSON: %w", err)
		}
		calls = append(calls, call)
	}
	for i, call := range calls {
		if call.Name == "" {
			return nil, fmt.Errorf("tool call %d missing name", i)
		}
		if call.Arguments == nil {
			return nil, fmt.Errorf("tool call %d missing arguments object", i)
		}
	}
	return calls, nil
}

func runTools(calls []toolCallItem) ([]toolResultItem, error) {
	results := make([]toolResultItem, 0, len(calls))
	for _, call := range calls {
		result, err := runTool(call)
		if err != nil {
			return nil, err
		}
		results = append(results, toolResultItem{Name: call.Name, Result: result})
	}
	return results, nil
}

func runTool(call toolCallItem) (string, error) {
	switch call.Name {
	case "get_current_hour":
		if len(call.Arguments) != 0 {
			return "", fmt.Errorf("get_current_hour does not take arguments")
		}
		return time.Now().Format("15"), nil
	case "get_random_number_between":
		minVal, err := intArgument(call, "min")
		if err != nil {
			return "", err
		}
		maxVal, err := intArgument(call, "max")
		if err != nil {
			return "", err
		}
		if minVal > maxVal {
			return "", fmt.Errorf("min must be <= max")
		}
		return fmt.Sprint(rand.Intn(maxVal-minVal+1) + minVal), nil
	default:
		return "", fmt.Errorf("unknown tool %q", call.Name)
	}
}

func intArgument(call toolCallItem, name string) (int, error) {
	value, ok := call.Arguments[name]
	if !ok {
		return 0, fmt.Errorf("%s missing argument %q", call.Name, name)
	}
	switch v := value.(type) {
	case float64:
		if v != float64(int(v)) {
			return 0, fmt.Errorf("%s argument %q must be an integer", call.Name, name)
		}
		return int(v), nil
	case int:
		return v, nil
	default:
		return 0, fmt.Errorf("%s argument %q must be an integer", call.Name, name)
	}
}
