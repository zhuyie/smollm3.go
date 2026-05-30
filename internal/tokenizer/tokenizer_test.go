package tokenizer

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
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
		"<tool_call>",
		"</tool_call>",
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
	got := tok.Encode("<|im_start|>user<|im_end|><think></think><tool_call></tool_call>", false, false)
	want := []int{10, 2, 2, 2, 2, 11, 12, 13, 14, 15}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Encode() = %v, want %v", got, want)
	}
}

func TestNextPieceUsesUnicodeNumberCategory(t *testing.T) {
	tok := testTokenizer()
	_, end := tok.nextPiece("ⅧⅨx", 0)
	if end != len("ⅧⅨ") {
		t.Fatalf("nextPiece() end = %d, want %d", end, len("ⅧⅨ"))
	}
}

func TestNextPieceOnlyAbsorbsASCIIBlankBeforeSymbols(t *testing.T) {
	tok := testTokenizer()

	_, end := tok.nextPiece(" !!!", 0)
	if end != len(" !!!") {
		t.Fatalf("nextPiece() end for space-prefixed symbols = %d, want %d", end, len(" !!!"))
	}

	_, end = tok.nextPiece("\t!!!", 0)
	if end != len("\t") {
		t.Fatalf("nextPiece() end for tab-prefixed symbols = %d, want %d", end, len("\t"))
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

func TestEncodeMatchesOfficialTokenizerSamples(t *testing.T) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("could not locate test file")
	}
	path := filepath.Join(filepath.Dir(filename), "..", "..", "models", "smollm3-tokenizer.bin")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("tokenizer not found: %s", path)
	}

	tok, err := Load(path, 128256)
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		text string
		want []int
	}{
		{"Hello, world!", []int{9906, 11, 1917, 0}},
		{"<|im_start|>user\nhi<|im_end|>", []int{128011, 882, 198, 6151, 128012}},
		{"<tool_call>{\"name\":\"x\"}</tool_call>", []int{128015, 5018, 609, 3332, 87, 9388, 128016}},
		{"<tool_response>ok</tool_response>", []int{128013, 564, 128014}},
		{"<code>print(1)</code>", []int{128017, 1374, 7, 16, 8, 128018}},
		{"abc1234567 def", []int{13997, 4513, 10961, 22, 711}},
		{"ⅧⅨx", []int{71567, 100, 71567, 101, 87}},
		{"\t!!!", []int{197, 12340}},
		{" !!!", []int{33970}},
		{"line  \n  next", []int{1074, 2355, 220, 1828}},
		{"中文 mixed １２3", []int{108891, 9709, 220, 20713, 25963, 18}},
	}
	for _, tt := range tests {
		got := tok.Encode(tt.text, false, false)
		if !reflect.DeepEqual(got, tt.want) {
			t.Fatalf("Encode(%q) = %v, want %v", tt.text, got, tt.want)
		}
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
