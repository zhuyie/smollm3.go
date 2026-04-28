package tokenizer

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"
	"unicode"
	"unicode/utf8"
)

const (
	tokenizerMagic      uint32 = 0x334b4f54 // TOK3
	tokenizerVersion           = int32(1)
	tokenizerHeaderSize        = int64(256)
)

type MergeRule struct {
	Left  int
	Right int
	Out   int
	Rank  int
}

type Tokenizer struct {
	Vocab []string
	// TokenToID avoids scanning the full vocab when byte-level pieces are
	// expanded into their initial token ids.
	TokenToID      map[string]int
	MergeRanks     map[[2]int]MergeRule
	BOSID          int
	EOSID          int
	PADID          int
	UNKID          int
	MaxTokenLength int
	SpecialIDs     []int
}

func Load(path string, expectedVocabSize int) (*Tokenizer, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var magic uint32
	var version int32
	if err := binary.Read(file, binary.LittleEndian, &magic); err != nil {
		return nil, err
	}
	if err := binary.Read(file, binary.LittleEndian, &version); err != nil {
		return nil, err
	}
	if magic != tokenizerMagic || version != tokenizerVersion {
		return nil, fmt.Errorf("bad tokenizer header: magic=%#x version=%d", magic, version)
	}

	var vocabSize, mergeCount, maxTokenLength, bosID, eosID, padID, unkID, specialCount int32
	for _, ptr := range []*int32{&vocabSize, &mergeCount, &maxTokenLength, &bosID, &eosID, &padID, &unkID, &specialCount} {
		if err := binary.Read(file, binary.LittleEndian, ptr); err != nil {
			return nil, err
		}
	}
	if int(vocabSize) != expectedVocabSize {
		return nil, fmt.Errorf("tokenizer vocab size %d does not match model vocab size %d", vocabSize, expectedVocabSize)
	}
	if _, err := file.Seek(tokenizerHeaderSize, io.SeekStart); err != nil {
		return nil, err
	}

	tok := &Tokenizer{
		Vocab:          make([]string, vocabSize),
		TokenToID:      make(map[string]int, vocabSize),
		MergeRanks:     make(map[[2]int]MergeRule, mergeCount),
		BOSID:          int(bosID),
		EOSID:          int(eosID),
		PADID:          int(padID),
		UNKID:          int(unkID),
		MaxTokenLength: int(maxTokenLength),
		SpecialIDs:     make([]int, 0, specialCount),
	}
	for i := range tok.Vocab {
		var n uint32
		if err := binary.Read(file, binary.LittleEndian, &n); err != nil {
			return nil, err
		}
		buf := make([]byte, n)
		if _, err := io.ReadFull(file, buf); err != nil {
			return nil, err
		}
		token := string(buf)
		tok.Vocab[i] = token
		tok.TokenToID[token] = i
	}

	// Merge ranks are the byte-level BPE rules. Lower rank means the pair was
	// learned earlier and should be merged first.
	for i := 0; i < int(mergeCount); i++ {
		var left, right, out int32
		if err := binary.Read(file, binary.LittleEndian, &left); err != nil {
			return nil, err
		}
		if err := binary.Read(file, binary.LittleEndian, &right); err != nil {
			return nil, err
		}
		if err := binary.Read(file, binary.LittleEndian, &out); err != nil {
			return nil, err
		}
		rule := MergeRule{Left: int(left), Right: int(right), Out: int(out), Rank: i}
		tok.MergeRanks[[2]int{rule.Left, rule.Right}] = rule
	}
	for i := 0; i < int(specialCount); i++ {
		var id int32
		if err := binary.Read(file, binary.LittleEndian, &id); err != nil {
			return nil, err
		}
		if id >= 0 && id < vocabSize {
			tok.SpecialIDs = append(tok.SpecialIDs, int(id))
		}
	}
	tok.ensureSpecialIDs()
	return tok, nil
}

func (t *Tokenizer) Encode(text string, bos bool, eos bool) []int {
	var tokens []int
	if bos && t.BOSID >= 0 {
		tokens = append(tokens, t.BOSID)
	}
	for i := 0; i < len(text); {
		if id, width, ok := t.matchSpecial(text, i); ok {
			tokens = append(tokens, id)
			i += width
			continue
		}
		start, end := t.nextPiece(text, i)
		tokens = append(tokens, t.encodePiece([]byte(text[start:end]))...)
		i = end
	}
	if eos && t.EOSID >= 0 {
		tokens = append(tokens, t.EOSID)
	}
	return tokens
}

func (t *Tokenizer) Decode(id int) string {
	if id < 0 || id >= len(t.Vocab) {
		return ""
	}
	piece := t.Vocab[id]
	if strings.HasPrefix(piece, "<|") {
		return piece
	}
	var out []byte
	for _, r := range piece {
		if b, ok := gpt2CodepointToByte(r); ok {
			out = append(out, b)
		}
	}
	return string(out)
}

func (t *Tokenizer) EOS() int {
	return t.EOSID
}

func (t *Tokenizer) matchSpecial(text string, pos int) (int, int, bool) {
	t.ensureSpecialIDs()
	bestID := 0
	bestWidth := 0
	for _, id := range t.SpecialIDs {
		if id < 0 || id >= len(t.Vocab) {
			continue
		}
		piece := t.Vocab[id]
		if len(piece) > bestWidth && strings.HasPrefix(text[pos:], piece) {
			bestID = id
			bestWidth = len(piece)
		}
	}
	if bestWidth > 0 {
		return bestID, bestWidth, true
	}
	return 0, 0, false
}

func (t *Tokenizer) ensureSpecialIDs() {
	if len(t.SpecialIDs) > 0 {
		return
	}
	seen := make(map[int]bool)
	for _, id := range []int{t.BOSID, t.EOSID, t.UNKID, t.PADID} {
		if id >= 0 && id < len(t.Vocab) && !seen[id] {
			t.SpecialIDs = append(t.SpecialIDs, id)
			seen[id] = true
		}
	}
	for id, piece := range t.Vocab {
		if (strings.HasPrefix(piece, "<|") || strings.HasPrefix(piece, "<think") || strings.HasPrefix(piece, "</think")) && !seen[id] {
			t.SpecialIDs = append(t.SpecialIDs, id)
			seen[id] = true
		}
	}
}

func (t *Tokenizer) encodePiece(bytes []byte) []int {
	// Initialize one token per byte using GPT-2's reversible byte-to-unicode
	// mapping, then repeatedly apply the best-ranked adjacent merge.
	tokens := make([]int, 0, len(bytes))
	for _, b := range bytes {
		piece := string(gpt2ByteToRunes(b))
		id, ok := t.TokenToID[piece]
		if !ok {
			id = t.UNKID
		}
		tokens = append(tokens, id)
	}
	for {
		bestIdx := -1
		bestOut := -1
		bestRank := int(^uint(0) >> 1)
		for i := 0; i < len(tokens)-1; i++ {
			if rule, ok := t.MergeRanks[[2]int{tokens[i], tokens[i+1]}]; ok && rule.Rank < bestRank {
				bestIdx = i
				bestOut = rule.Out
				bestRank = rule.Rank
			}
		}
		if bestIdx < 0 {
			break
		}
		tokens[bestIdx] = bestOut
		copy(tokens[bestIdx+1:], tokens[bestIdx+2:])
		tokens = tokens[:len(tokens)-1]
	}
	return tokens
}

// nextPiece applies the tokenizer's pre-tokenization step before BPE merging.
// It groups text into GPT-2-style pieces such as contractions, words with an
// optional leading symbol or space, short digit groups, symbol runs, and
// whitespace. Each piece is then expanded to reversible byte tokens and merged
// with the BPE rank table in encodePiece.
func (t *Tokenizer) nextPiece(text string, pos int) (int, int) {
	if end := matchContraction(text, pos); end > pos {
		return pos, end
	}
	r, width := utf8.DecodeRuneInString(text[pos:])
	if r == utf8.RuneError && width == 0 {
		return pos, pos
	}

	if isOptionalLetterPrefix(text, pos, r, width) {
		end := pos + width
		for end < len(text) {
			if _, _, ok := t.matchSpecial(text, end); ok {
				break
			}
			next, nextWidth := utf8.DecodeRuneInString(text[end:])
			if !unicode.IsLetter(next) {
				break
			}
			end += nextWidth
		}
		return pos, end
	}
	if unicode.IsLetter(r) {
		return pos, t.consumeWhile(text, pos, unicode.IsLetter)
	}
	if unicode.IsDigit(r) {
		end := pos
		for count := 0; count < 3 && end < len(text); count++ {
			if _, _, ok := t.matchSpecial(text, end); ok {
				break
			}
			next, nextWidth := utf8.DecodeRuneInString(text[end:])
			if !unicode.IsDigit(next) {
				break
			}
			end += nextWidth
		}
		return pos, end
	}
	if isSymbolPrefix(text, pos, r, width) {
		end := pos
		if unicode.IsSpace(r) && r != '\r' && r != '\n' {
			end += width
		}
		for end < len(text) {
			if _, _, ok := t.matchSpecial(text, end); ok {
				break
			}
			next, nextWidth := utf8.DecodeRuneInString(text[end:])
			if unicode.IsSpace(next) || unicode.IsLetter(next) || unicode.IsDigit(next) {
				break
			}
			end += nextWidth
		}
		for end < len(text) {
			next, nextWidth := utf8.DecodeRuneInString(text[end:])
			if next != '\r' && next != '\n' {
				break
			}
			end += nextWidth
		}
		return pos, end
	}
	return pos, t.consumeWhile(text, pos, unicode.IsSpace)
}

func (t *Tokenizer) consumeWhile(text string, pos int, pred func(rune) bool) int {
	end := pos
	for end < len(text) {
		if _, _, ok := t.matchSpecial(text, end); ok {
			break
		}
		r, width := utf8.DecodeRuneInString(text[end:])
		if !pred(r) {
			break
		}
		end += width
	}
	return end
}

func matchContraction(text string, pos int) int {
	for _, suffix := range []string{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"} {
		if len(text)-pos >= len(suffix) && strings.EqualFold(text[pos:pos+len(suffix)], suffix) {
			return pos + len(suffix)
		}
	}
	return pos
}

func isOptionalLetterPrefix(text string, pos int, r rune, width int) bool {
	if r == '\r' || r == '\n' || unicode.IsLetter(r) || unicode.IsDigit(r) {
		return false
	}
	nextPos := pos + width
	if nextPos >= len(text) {
		return false
	}
	next, _ := utf8.DecodeRuneInString(text[nextPos:])
	return unicode.IsLetter(next)
}

func isSymbolPrefix(text string, pos int, r rune, width int) bool {
	if !unicode.IsSpace(r) && !unicode.IsLetter(r) && !unicode.IsDigit(r) {
		return true
	}
	if !unicode.IsSpace(r) || r == '\r' || r == '\n' {
		return false
	}
	nextPos := pos + width
	if nextPos >= len(text) {
		return false
	}
	next, _ := utf8.DecodeRuneInString(text[nextPos:])
	return !unicode.IsSpace(next) && !unicode.IsLetter(next) && !unicode.IsDigit(next)
}

func gpt2ByteToRunes(b byte) []rune {
	cp := gpt2ByteToCodepoint(b)
	return []rune{cp}
}

func gpt2ByteToCodepoint(b byte) rune {
	if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255) {
		return rune(b)
	}
	n := 0
	for i := 0; i < 256; i++ {
		if (i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255) {
			continue
		}
		if byte(i) == b {
			return rune(256 + n)
		}
		n++
	}
	return rune(b)
}

func gpt2CodepointToByte(cp rune) (byte, bool) {
	if (cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255) {
		return byte(cp), true
	}
	n := 0
	for i := 0; i < 256; i++ {
		if (i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255) {
			continue
		}
		if cp == rune(256+n) {
			return byte(i), true
		}
		n++
	}
	return 0, false
}
