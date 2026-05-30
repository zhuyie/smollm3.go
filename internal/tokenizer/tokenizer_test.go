package tokenizer

import (
	"encoding/binary"
	"os"
	"path/filepath"
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

func TestLoadRejectsNegativeCounts(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.bin")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	for _, v := range []any{
		tokenizerMagic, tokenizerVersion,
		int32(-1), int32(0), int32(0), int32(-1), int32(-1), int32(-1), int32(-1), int32(0),
	} {
		if err := binary.Write(f, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}

	if _, err := Load(path, -1); err == nil {
		t.Fatal("Load() succeeded for negative tokenizer counts")
	}
}

func TestLoadRejectsTokenLongerThanHeaderMax(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.bin")
	writeTokenizerFile(t, path, []string{"a"}, nil, nil, 0)

	if _, err := Load(path, 1); err == nil {
		t.Fatal("Load() succeeded for token length exceeding max_token_length")
	}
}

func TestLoadRejectsOutOfRangeMergeIDs(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.bin")
	writeTokenizerFile(t, path, []string{"a", "b"}, [][3]int32{{0, 1, 2}}, nil, 1)

	if _, err := Load(path, 2); err == nil {
		t.Fatal("Load() succeeded for out-of-range merge id")
	}
}

func TestLoadRejectsOutOfRangeSpecialIDs(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.bin")
	writeTokenizerFile(t, path, []string{"a"}, nil, []int32{1}, 1)

	if _, err := Load(path, 1); err == nil {
		t.Fatal("Load() succeeded for out-of-range special id")
	}
}

func writeTokenizerFile(t *testing.T, path string, vocab []string, merges [][3]int32, specials []int32, maxTokenLength int32) {
	t.Helper()

	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	header := []any{
		tokenizerMagic,
		tokenizerVersion,
		int32(len(vocab)),
		int32(len(merges)),
		maxTokenLength,
		int32(-1),
		int32(-1),
		int32(-1),
		int32(-1),
		int32(len(specials)),
	}
	for _, v := range header {
		if err := binary.Write(f, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := f.Write(make([]byte, tokenizerHeaderSize-40)); err != nil {
		t.Fatal(err)
	}
	for _, token := range vocab {
		data := []byte(token)
		if err := binary.Write(f, binary.LittleEndian, uint32(len(data))); err != nil {
			t.Fatal(err)
		}
		if _, err := f.Write(data); err != nil {
			t.Fatal(err)
		}
	}
	for _, merge := range merges {
		for _, id := range merge {
			if err := binary.Write(f, binary.LittleEndian, id); err != nil {
				t.Fatal(err)
			}
		}
	}
	for _, id := range specials {
		if err := binary.Write(f, binary.LittleEndian, id); err != nil {
			t.Fatal(err)
		}
	}
}
