package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"strings"
	"time"

	"github.com/zhuyie/smollm3.go"
)

type toolCallItem struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

type toolResultItem struct {
	Name   string
	Result string
}

const builtinToolsJSON = `{"type":"function","function":{"name":"get_product_price","description":"Gets the unit price for a product in the local currency.","parameters":{"type":"object","properties":{"product":{"type":"string","description":"The product name to look up."}},"required":["product"]}}}`

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
	maxNew := flag.Int("n", 1024, "maximum new tokens")
	temperature := flag.Float64("temp", 1.0, "sampling temperature, 0 for greedy")
	topP := flag.Float64("top-p", 0.9, "top-p nucleus sampling")
	seed := flag.Int64("seed", time.Now().UnixNano(), "random seed")
	flag.Parse()

	if *modelPath == "" || *tokenizerPath == "" {
		flag.Usage()
		os.Exit(2)
	}
	if *maxNew <= 0 {
		*maxNew = 1024
	}
	if *temperature < 0 || math.IsNaN(*temperature) || math.IsInf(*temperature, 0) {
		*temperature = 1.0
	}
	if *topP < 0 || *topP > 1 || math.IsNaN(*topP) || math.IsInf(*topP, 0) {
		*topP = 0.9
	}

	client, err := smollm3.Load(smollm3.Config{
		ModelPath:     *modelPath,
		TokenizerPath: *tokenizerPath,
	})
	if err != nil {
		log.Fatal(err)
	}

	switch *mode {
	case "generate":
		generate(client, *prompt, *maxNew, *temperature, *topP, *seed)
	case "chat":
		chat(client, *prompt, *systemPrompt, *thinking, *maxNew, *temperature, *topP, *seed)
	case "toolcall":
		toolCall(client, *prompt, *maxNew, *temperature, *topP, *seed)
	default:
		log.Fatalf("unknown mode %q", *mode)
	}
}

func generate(client *smollm3.Client, prompt string, maxNew int, temperature float64, topP float64, seed int64) {
	_, stats, err := client.Generate(context.Background(), prompt, smollm3.GenerateOptions{
		MaxNewTokens: maxNew,
		Temperature:  temperature,
		TopP:         topP,
		Seed:         seed,
		TokenCallback: func(piece string) error {
			fmt.Print(piece)
			return nil
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println()
	fmt.Println()
	printGenerationStats(os.Stderr, stats)
}

func chat(client *smollm3.Client, userPrompt string, systemPrompt string, thinking bool, maxNew int, temperature float64, topP float64, seed int64) {
	opts := smollm3.ChatOptions{
		GenerateOptions: smollm3.GenerateOptions{
			MaxNewTokens: maxNew,
			Temperature:  temperature,
			TopP:         topP,
			Seed:         seed,
			TokenCallback: func(piece string) error {
				fmt.Fprint(os.Stdout, piece)
				return nil
			},
		},
		SystemPrompt: systemPrompt,
		Thinking:     thinking,
	}
	if userPrompt != "" {
		printAssistantPrefix(os.Stdout)
		_, stats, err := client.Chat(context.Background(), []smollm3.Message{{Role: "user", Content: userPrompt}}, opts)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(ansiReset)
		fmt.Fprintln(os.Stderr)
		printGenerationStats(os.Stderr, stats)
		return
	}

	session := client.NewChatSession(opts)
	totalStats := smollm3.GenerationStats{}
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
		printAssistantPrefix(os.Stdout)
		_, stats, err := session.Reply(context.Background(), userPrompt)
		if err != nil {
			log.Fatal(err)
		}
		totalStats.GeneratedTokens += stats.GeneratedTokens
		totalStats.Duration += stats.Duration
		fmt.Println(ansiReset)
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
	printGenerationStats(os.Stderr, totalStats)
}

func printGenerationStats(w io.Writer, stats smollm3.GenerationStats) {
	fmt.Fprintf(w, "achieved tok/s: %.6f\n", stats.TokensPerSecond())
}

func toolCall(client *smollm3.Client, prompt string, maxNew int, temperature float64, topP float64, seed int64) {
	if strings.TrimSpace(prompt) == "" {
		log.Fatal("toolcall mode requires -prompt")
	}
	toolRequest, _, err := client.Chat(context.Background(), []smollm3.Message{{Role: "user", Content: prompt}}, smollm3.ChatOptions{
		GenerateOptions: smollm3.GenerateOptions{
			MaxNewTokens: maxNew,
			Temperature:  temperature,
			TopP:         topP,
			Seed:         seed,
		},
		SystemPrompt: toolCallSystemPrompt(),
		Thinking:     false,
	})
	if err != nil {
		log.Fatal(err)
	}
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
	_, _, err = client.Chat(context.Background(), []smollm3.Message{
		{Role: "user", Content: prompt},
		{Role: "assistant", Content: strings.TrimSpace(toolRequest)},
		{Role: "user", Content: renderToolResponse(results)},
	}, smollm3.ChatOptions{
		GenerateOptions: smollm3.GenerateOptions{
			MaxNewTokens: maxNew,
			Temperature:  temperature,
			TopP:         topP,
			Seed:         seed,
			TokenCallback: func(piece string) error {
				fmt.Fprint(os.Stdout, piece)
				return nil
			},
		},
		SystemPrompt: toolResultSystemPrompt(),
		Thinking:     false,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println()
}

func printUserPrefix(w io.Writer) {
	fmt.Fprint(w, ansiPrompt, "User: ", ansiUserInput)
}

func printAssistantPrefix(w io.Writer) {
	fmt.Fprint(w, ansiPrompt, "Assistant: ", ansiReset)
}

func toolCallSystemPrompt() string {
	return `You are a helpful AI assistant named SmolLM, trained by Hugging Face.

### Tools

You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:

<tools>
` + builtinToolsJSON + `
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>`
}

func toolResultSystemPrompt() string {
	return `You are a helpful AI assistant.
Use the tool response to answer the user's request.
Write only the final answer in plain text.
Do not call tools again.`
}

func renderToolResponse(results []toolResultItem) string {
	var toolOut strings.Builder
	toolOut.WriteString("<tool_response>\n")
	for i, result := range results {
		if i > 0 {
			toolOut.WriteByte('\n')
		}
		data, err := json.Marshal(struct {
			Name   string `json:"name"`
			Result string `json:"result"`
		}{
			Name:   result.Name,
			Result: result.Result,
		})
		if err == nil {
			toolOut.Write(data)
		}
	}
	toolOut.WriteString("\n</tool_response>")
	return toolOut.String()
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
	case "get_product_price":
		product, err := stringArgument(call, "product")
		if err != nil {
			return "", err
		}
		return productPrice(product)
	default:
		return "", fmt.Errorf("unknown tool %q", call.Name)
	}
}

func productPrice(product string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(product)) {
	case "notebook", "notebooks":
		return "12", nil
	case "backpack", "backpacks":
		return "48.00", nil
	case "pen", "pens":
		return "1.20", nil
	default:
		return "", fmt.Errorf("unknown product %q", product)
	}
}

func stringArgument(call toolCallItem, name string) (string, error) {
	value, ok := call.Arguments[name]
	if !ok {
		return "", fmt.Errorf("%s missing argument %q", call.Name, name)
	}
	v, ok := value.(string)
	if !ok || strings.TrimSpace(v) == "" {
		return "", fmt.Errorf("%s argument %q must be a non-empty string", call.Name, name)
	}
	return v, nil
}
