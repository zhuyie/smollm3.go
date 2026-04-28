package tokenizer

import (
	"reflect"
	"testing"
)

func testTokenizer() *Tokenizer {
	vocab := []string{
		"<|bos|>",
		"<|eos|>",
		"<|unk|>",
		"a",
		"b",
		"c",
		"ab",
		"abc",
		"\u0120",
		"\u0120a",
		"<|im_start|>",
		"<|im_end|>",
		"<think>",
		"</think>",
	}
	tok := &Tokenizer{
		Vocab:      vocab,
		TokenToID:  make(map[string]int, len(vocab)),
		MergeRanks: make(map[[2]int]MergeRule),
		BOSID:      0,
		EOSID:      1,
		UNKID:      2,
		PADID:      -1,
	}
	for id, piece := range vocab {
		tok.TokenToID[piece] = id
	}
	tok.MergeRanks[[2]int{3, 4}] = MergeRule{Left: 3, Right: 4, Out: 6, Rank: 0}
	tok.MergeRanks[[2]int{6, 5}] = MergeRule{Left: 6, Right: 5, Out: 7, Rank: 1}
	tok.MergeRanks[[2]int{8, 3}] = MergeRule{Left: 8, Right: 3, Out: 9, Rank: 2}
	return tok
}

func TestEncodeAppliesMergeRanks(t *testing.T) {
	got := testTokenizer().Encode("abc", false, false)
	want := []int{7}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Encode() = %v, want %v", got, want)
	}
}

func TestEncodeHandlesSpecialAndOptionalBosEOS(t *testing.T) {
	got := testTokenizer().Encode("<|bos|>ab", true, true)
	want := []int{0, 0, 6, 1}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Encode() = %v, want %v", got, want)
	}
}

func TestEncodeKeepsLeadingSpaceWithFollowingWord(t *testing.T) {
	got := testTokenizer().Encode(" a", false, false)
	want := []int{9}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Encode() = %v, want %v", got, want)
	}
}

func TestDecodeRoundTripsByteLevelPieces(t *testing.T) {
	tok := testTokenizer()
	if got := tok.Decode(9); got != " a" {
		t.Fatalf("Decode() = %q, want %q", got, " a")
	}
	if got := tok.Decode(0); got != "<|bos|>" {
		t.Fatalf("Decode() = %q, want special token", got)
	}
}

func TestEncodeMatchesAllSpecialTokens(t *testing.T) {
	tok := testTokenizer()
	got := tok.Encode("<|im_start|>user<|im_end|><think></think>", false, false)
	want := []int{10, 2, 2, 2, 2, 11, 12, 13}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Encode() = %v, want %v", got, want)
	}
}
