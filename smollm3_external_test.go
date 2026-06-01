package smollm3_test

import (
	"context"
	"testing"

	"smollm3go"
)

func TestExternalPackageCanUsePublicAPI(t *testing.T) {
	_, _ = smollm3.Load(smollm3.Config{
		ModelPath:     "model.bin",
		TokenizerPath: "tokenizer.bin",
	})

	_ = smollm3.RenderChatPrompt([]smollm3.Message{
		{Role: "user", Content: "hello"},
	}, "", true)

	var client *smollm3.Client
	_, _, _ = client.Generate(context.Background(), "hello", smollm3.GenerateOptions{
		MaxNewTokens: 8,
		Temperature:  0,
	})
}
