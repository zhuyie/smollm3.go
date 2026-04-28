package model

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestValidateConfig(t *testing.T) {
	valid := Config{
		Dim:        4,
		HiddenDim:  8,
		NLayers:    1,
		NHeads:     2,
		NKVHeads:   1,
		VocabSize:  3,
		SeqLen:     16,
		RopeTheta:  10000,
		RMSNormEps: 1e-6,
	}
	if err := validateConfig(valid); err != nil {
		t.Fatalf("validateConfig(valid) returned error: %v", err)
	}
	invalid := valid
	invalid.NHeads = 3
	if err := validateConfig(invalid); err == nil {
		t.Fatal("validateConfig(invalid) returned nil")
	}
}

func TestLoadReturnsErrorForTruncatedWeights(t *testing.T) {
	path := filepath.Join(t.TempDir(), "truncated.sml3")
	header := make([]byte, checkpointHeaderSize)
	off := 0
	binary.LittleEndian.PutUint32(header[off:], checkpointMagic)
	off += 4
	for _, value := range []int32{
		checkpointVersion,
		4,  // dim
		8,  // hidden_dim
		1,  // n_layers
		2,  // n_heads
		1,  // n_kv_heads
		3,  // vocab_size
		16, // seq_len
		1,  // shared_classifier
		0,  // bos_id
		1,  // eos_id
		-1, // pad_id
	} {
		binary.LittleEndian.PutUint32(header[off:], uint32(value))
		off += 4
	}
	binary.LittleEndian.PutUint32(header[off:], math.Float32bits(10000))
	off += 4
	binary.LittleEndian.PutUint32(header[off:], math.Float32bits(1e-6))
	off += 4
	binary.LittleEndian.PutUint32(header[off:], 1)
	off += 4 * checkpointMaxRopeHeaderLayers
	binary.LittleEndian.PutUint32(header[off:], uint32(weightTypeFP32))
	if err := os.WriteFile(path, header, 0o600); err != nil {
		t.Fatal(err)
	}

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Load panicked for truncated checkpoint: %v", r)
		}
	}()
	_, err := Load(path)
	if err == nil {
		t.Fatal("Load returned nil error for truncated checkpoint")
	}
	if !strings.Contains(err.Error(), "failed to load checkpoint weights") {
		t.Fatalf("Load error = %q, want checkpoint load context", err)
	}
}

func TestRMSNorm(t *testing.T) {
	out := make([]float32, 2)
	x := []float32{3, 4}
	weight := []float32{1, 2}
	rmsnorm(out, x, weight, 1e-6)

	scale := float32(1.0 / math.Sqrt(float64((3*3+4*4)/float32(2)+1e-6)))
	want := []float32{3 * scale, 8 * scale}
	for i := range want {
		if math.Abs(float64(out[i]-want[i])) > 1e-6 {
			t.Fatalf("out[%d] = %f, want %f", i, out[i], want[i])
		}
	}
}

func TestSoftmax(t *testing.T) {
	x := []float32{1, 2, 3}
	softmax(x)
	var sum float32
	for _, v := range x {
		sum += v
	}
	if math.Abs(float64(sum-1)) > 1e-6 {
		t.Fatalf("sum = %f, want 1", sum)
	}
	if !(x[0] < x[1] && x[1] < x[2]) {
		t.Fatalf("softmax probabilities not ordered: %v", x)
	}
}

func TestAddScaledF32(t *testing.T) {
	for _, n := range []int{1, 4, 7, 8, 9, 16, 17} {
		dst := make([]float32, n)
		src := make([]float32, n)
		want := make([]float32, n)
		for i := range dst {
			dst[i] = float32(i + 1)
			src[i] = float32(i*2 + 3)
			want[i] = dst[i] + 0.5*src[i]
		}
		addScaledF32(dst, src, 0.5)
		for i := range want {
			if dst[i] != want[i] {
				t.Fatalf("n=%d dst[%d] = %f, want %f", n, i, dst[i], want[i])
			}
		}
	}
}

func TestDotF32Batch4(t *testing.T) {
	for _, n := range []int{4, 8, 64, 65} {
		x0 := make([]float32, n)
		x1 := make([]float32, n)
		x2 := make([]float32, n)
		x3 := make([]float32, n)
		w := make([]float32, n)
		for i := range w {
			x0[i] = float32((i%7)-3) / 7
			x1[i] = float32((i%11)-5) / 11
			x2[i] = float32((i%13)-6) / 13
			x3[i] = float32((i%17)-8) / 17
			w[i] = float32((i%19)-9) / 19
		}

		got0, got1, got2, got3 := dotF32Batch4(x0, x1, x2, x3, w)
		want := []float32{
			dotF32Scalar(x0, w),
			dotF32Scalar(x1, w),
			dotF32Scalar(x2, w),
			dotF32Scalar(x3, w),
		}
		got := []float32{got0, got1, got2, got3}
		for i := range want {
			if math.Abs(float64(got[i]-want[i])) > 1e-4 {
				t.Fatalf("n=%d got[%d] = %f, want %f", n, i, got[i], want[i])
			}
		}
	}
}

func TestBuildRopeTables(t *testing.T) {
	seqLen := 4
	headSize := 8
	ropeTheta := float32(10000)
	cosTable, sinTable := buildRopeTables(seqLen, headSize, ropeTheta)
	headPairs := headSize / 2
	for pos := 0; pos < seqLen; pos++ {
		for pair := 0; pair < headPairs; pair++ {
			headDim := pair * 2
			freq := float32(1.0 / math.Pow(float64(ropeTheta), float64(headDim)/float64(headSize)))
			val := float32(pos) * freq
			idx := pos*headPairs + pair
			if got, want := cosTable[idx], float32(math.Cos(float64(val))); math.Abs(float64(got-want)) > 1e-6 {
				t.Fatalf("cosTable[%d] = %f, want %f", idx, got, want)
			}
			if got, want := sinTable[idx], float32(math.Sin(float64(val))); math.Abs(float64(got-want)) > 1e-6 {
				t.Fatalf("sinTable[%d] = %f, want %f", idx, got, want)
			}
		}
	}
}

func TestMatmul(t *testing.T) {
	x := []float32{2, 3, 5, 7, 11}
	w := []float32{
		1, 0, 1, 0, 1,
		0, 1, 0, 1, 0,
	}
	out := make([]float32, 2)
	matmul(out, x, w, 5, 2)
	want := []float32{18, 10}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("out[%d] = %f, want %f", i, out[i], want[i])
		}
	}
}

func TestDotF32Int8(t *testing.T) {
	x := []float32{1.5, -2, 0.25, 4, -3, 2.5, 1, -0.5, 0.75, -1.25, 3, -4, 2, 1.25, -0.75, 0.5}
	w := []int8{3, -4, 5, 6, -7, 8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18}
	got := dotF32Int8(x, w)
	want := dotF32Int8Scalar(x, w)
	if math.Abs(float64(got-want)) > 1e-5 {
		t.Fatalf("dotF32Int8 = %f, want %f", got, want)
	}
}

func TestMatmulInt8MatchesFloatMatmul(t *testing.T) {
	n := 16
	d := 8
	x := fillTestWeights(n)
	w := fillTestWeights(n * d)
	q := quantizeMatrixInt8(w, n, d)
	got := make([]float32, d)
	want := make([]float32, d)

	matmulInt8(got, x, q, n, d)
	matmul(want, x, w, n, d)

	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 0.001 {
			t.Fatalf("out[%d] = %f, want near %f", i, got[i], want[i])
		}
	}
}

func TestForwardInt8MatchesFloat(t *testing.T) {
	cfg := Config{
		Dim:        8,
		HiddenDim:  16,
		NLayers:    2,
		NHeads:     2,
		NKVHeads:   1,
		VocabSize:  17,
		SeqLen:     16,
		RopeTheta:  10000,
		RMSNormEps: 1e-6,
	}
	floatModel := newTestTransformer(cfg)
	quantModel := newTestTransformer(cfg)
	quantModel.quantizeInt8()

	floatLogits := floatModel.Forward(3, 0)
	quantLogits := quantModel.Forward(3, 0)
	for i := range floatLogits {
		if math.Abs(float64(quantLogits[i]-floatLogits[i])) > 0.05 {
			t.Fatalf("logit[%d] = %f, want near %f", i, quantLogits[i], floatLogits[i])
		}
	}
}

func TestPrefillInt8MatchesFloat(t *testing.T) {
	cfg := Config{
		Dim:        8,
		HiddenDim:  16,
		NLayers:    2,
		NHeads:     2,
		NKVHeads:   1,
		VocabSize:  17,
		SeqLen:     16,
		RopeTheta:  10000,
		RMSNormEps: 1e-6,
	}
	floatModel := newTestTransformer(cfg)
	quantModel := newTestTransformer(cfg)
	quantModel.quantizeInt8()
	tokens := []int{1, 5, 9, 3}

	floatLogits := floatModel.Prefill(tokens, 0)
	quantLogits := quantModel.Prefill(tokens, 0)
	for i := range floatLogits {
		if math.Abs(float64(quantLogits[i]-floatLogits[i])) > 0.05 {
			t.Fatalf("logit[%d] = %f, want near %f", i, quantLogits[i], floatLogits[i])
		}
	}
}

func TestPrefillMatchesForwardLoop(t *testing.T) {
	cfg := Config{
		Dim:        8,
		HiddenDim:  16,
		NLayers:    2,
		NHeads:     2,
		NKVHeads:   1,
		VocabSize:  17,
		SeqLen:     16,
		RopeTheta:  10000,
		RMSNormEps: 1e-6,
	}
	t1 := newTestTransformer(cfg)
	t2 := newTestTransformer(cfg)
	tokens := []int{1, 5, 9, 3, 7}

	var loopLogits []float32
	for pos, token := range tokens {
		loopLogits = t1.Forward(token, pos)
	}
	prefillLogits := t2.Prefill(tokens, 0)

	for i := range loopLogits {
		if math.Abs(float64(loopLogits[i]-prefillLogits[i])) > 1e-4 {
			t.Fatalf("logit[%d] = %f, want %f", i, prefillLogits[i], loopLogits[i])
		}
	}
}

func TestNoPELayerSkipsRotaryPosition(t *testing.T) {
	cfg := Config{
		Dim:        8,
		HiddenDim:  16,
		NLayers:    2,
		NHeads:     2,
		NKVHeads:   1,
		VocabSize:  17,
		SeqLen:     16,
		RopeTheta:  10000,
		RMSNormEps: 1e-6,
		RopeLayers: []bool{
			true,
			false,
		},
	}
	t1 := newTestTransformer(cfg)
	t2 := newTestTransformer(cfg)
	if !t1.layerUsesRope(0) || t1.layerUsesRope(1) {
		t.Fatalf("unexpected rope layer table: %v", cfg.RopeLayers)
	}
	tokens := []int{1, 5, 9, 3, 7}

	var loopLogits []float32
	for pos, token := range tokens {
		loopLogits = t1.Forward(token, pos)
	}
	prefillLogits := t2.Prefill(tokens, 0)

	for i := range loopLogits {
		if math.Abs(float64(loopLogits[i]-prefillLogits[i])) > 1e-4 {
			t.Fatalf("logit[%d] = %f, want %f", i, prefillLogits[i], loopLogits[i])
		}
	}
}

func newTestTransformer(cfg Config) *Transformer {
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	if len(cfg.RopeLayers) == 0 {
		cfg.RopeLayers = make([]bool, cfg.NLayers)
		for i := range cfg.RopeLayers {
			cfg.RopeLayers[i] = true
		}
	}
	kvDim := cfg.Dim * cfg.NKVHeads / cfg.NHeads
	weights := Weights{
		TokenEmbeddingTable: fillTestWeights(cfg.VocabSize * cfg.Dim),
		Layers:              make([]LayerWeights, cfg.NLayers),
		RMSFinalWeight:      fillTestWeights(cfg.Dim),
	}
	for i := range weights.Layers {
		lw := &weights.Layers[i]
		lw.RMSAttWeight = fillTestWeights(cfg.Dim)
		lw.RMSFFNWeight = fillTestWeights(cfg.Dim)
		lw.WQ = fillTestWeights(cfg.Dim * cfg.Dim)
		lw.WK = fillTestWeights(cfg.Dim * kvDim)
		lw.WV = fillTestWeights(cfg.Dim * kvDim)
		lw.WO = fillTestWeights(cfg.Dim * cfg.Dim)
		lw.W1 = fillTestWeights(cfg.Dim * cfg.HiddenDim)
		lw.W2 = fillTestWeights(cfg.HiddenDim * cfg.Dim)
		lw.W3 = fillTestWeights(cfg.Dim * cfg.HiddenDim)
	}
	weights.WCls = weights.TokenEmbeddingTable

	headSize := cfg.Dim / cfg.NHeads
	ropeCos, ropeSin := buildRopeTables(cfg.SeqLen, headSize, cfg.RopeTheta)
	return &Transformer{
		Config:  cfg,
		Weights: weights,
		State: State{
			X:          make([]float32, cfg.Dim),
			XB:         make([]float32, cfg.Dim),
			XB2:        make([]float32, cfg.Dim),
			HB:         make([]float32, cfg.HiddenDim),
			HB2:        make([]float32, cfg.HiddenDim),
			Q:          make([]float32, cfg.Dim),
			Att:        make([]float32, cfg.NHeads*cfg.SeqLen),
			Logits:     make([]float32, cfg.VocabSize),
			KeyCache:   make([]float32, cfg.NLayers*cfg.SeqLen*kvDim),
			ValueCache: make([]float32, cfg.NLayers*cfg.SeqLen*kvDim),
		},
		Tables: Tables{RopeCos: ropeCos, RopeSin: ropeSin},
	}
}

func (t *Transformer) quantizeInt8() {
	cfg := t.Config
	kvDim := cfg.Dim * cfg.NKVHeads / cfg.NHeads
	for i := range t.Weights.Layers {
		lw := &t.Weights.Layers[i]
		lw.QWQ = quantizeMatrixInt8(lw.WQ, cfg.Dim, cfg.Dim)
		lw.WQ = nil
		lw.QWK = quantizeMatrixInt8(lw.WK, cfg.Dim, kvDim)
		lw.WK = nil
		lw.QWV = quantizeMatrixInt8(lw.WV, cfg.Dim, kvDim)
		lw.WV = nil
		lw.QWO = quantizeMatrixInt8(lw.WO, cfg.Dim, cfg.Dim)
		lw.WO = nil
		lw.QW1 = quantizeMatrixInt8(lw.W1, cfg.Dim, cfg.HiddenDim)
		lw.W1 = nil
		lw.QW2 = quantizeMatrixInt8(lw.W2, cfg.HiddenDim, cfg.Dim)
		lw.W2 = nil
		lw.QW3 = quantizeMatrixInt8(lw.W3, cfg.Dim, cfg.HiddenDim)
		lw.W3 = nil
	}
	t.Weights.QWCls = quantizeMatrixInt8(t.Weights.WCls, cfg.Dim, cfg.VocabSize)
	if !t.Weights.SharedWeights {
		t.Weights.WCls = nil
	}
}

func quantizeMatrixInt8(w []float32, inputs int, rows int) *QuantizedMatrix {
	q := &QuantizedMatrix{
		Data:  make([]int8, inputs*rows),
		Scale: make([]float32, rows),
	}
	for row := 0; row < rows; row++ {
		src := w[row*inputs : (row+1)*inputs]
		var maxAbs float32
		for _, v := range src {
			abs := float32(math.Abs(float64(v)))
			if abs > maxAbs {
				maxAbs = abs
			}
		}
		scale := float32(1)
		if maxAbs > 0 {
			scale = maxAbs / 127
		}
		q.Scale[row] = scale
		dst := q.Data[row*inputs : (row+1)*inputs]
		invScale := float32(1) / scale
		for i, v := range src {
			quantized := int(math.Round(float64(v * invScale)))
			if quantized > 127 {
				quantized = 127
			} else if quantized < -127 {
				quantized = -127
			}
			dst[i] = int8(quantized)
		}
	}
	return q
}

func fillTestWeights(n int) []float32 {
	values := make([]float32, n)
	for i := range values {
		values[i] = float32((i%17)-8) / 100
	}
	return values
}

var benchmarkLogits []float32

func loadBenchmarkTransformerPath(b *testing.B, path string) *Transformer {
	b.Helper()
	if _, err := os.Stat(path); err != nil {
		b.Skipf("model checkpoint not found: %s", path)
	}
	t, err := Load(path)
	if err != nil {
		b.Fatal(err)
	}
	return t
}

func benchmarkModelPaths() []struct {
	name string
	path string
} {
	return []struct {
		name string
		path string
	}{
		{name: "fp32", path: filepath.Join("..", "..", "models", "smollm3-3b-f32.bin")},
		{name: "int8", path: filepath.Join("..", "..", "models", "smollm3-3b-int8.bin")},
	}
}

func benchmarkTokens(vocabSize int, count int) []int {
	tokens := make([]int, count)
	for i := range tokens {
		tokens[i] = (i*131 + 17) % vocabSize
	}
	return tokens
}

// BenchmarkPrefill measures batched prompt ingestion from position 0.
func BenchmarkPrefill(b *testing.B) {
	for _, model := range benchmarkModelPaths() {
		b.Run(model.name, func(b *testing.B) {
			for _, promptLen := range []int{128, 512} {
				b.Run(strconv.Itoa(promptLen), func(b *testing.B) {
					t := loadBenchmarkTransformerPath(b, model.path)
					if promptLen > t.Config.SeqLen {
						b.Skipf("prompt length %d exceeds sequence length %d", promptLen, t.Config.SeqLen)
					}
					tokens := benchmarkTokens(t.Config.VocabSize, promptLen)
					benchmarkLogits = t.Prefill(tokens, 0)

					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						benchmarkLogits = t.Prefill(tokens, 0)
					}
					b.ReportMetric(float64(b.N*promptLen)*1e9/float64(b.Elapsed().Nanoseconds()), "tok/s")
				})
			}
		})
	}
}

// BenchmarkDecode measures the cost of generating one token after an existing
// context has already populated the KV cache. Setup prefill is intentionally
// outside the timed region.
func BenchmarkDecode(b *testing.B) {
	for _, model := range benchmarkModelPaths() {
		b.Run(model.name, func(b *testing.B) {
			for _, contextLen := range []int{128, 512} {
				b.Run(strconv.Itoa(contextLen), func(b *testing.B) {
					t := loadBenchmarkTransformerPath(b, model.path)
					if contextLen >= t.Config.SeqLen {
						b.Skipf("context length %d leaves no decode position in sequence length %d", contextLen, t.Config.SeqLen)
					}
					tokens := benchmarkTokens(t.Config.VocabSize, contextLen+1)
					benchmarkLogits = t.Prefill(tokens[:contextLen], 0)
					decodeToken := tokens[contextLen]

					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						benchmarkLogits = t.Forward(decodeToken, contextLen)
					}
					b.ReportMetric(float64(b.N)*1e9/float64(b.Elapsed().Nanoseconds()), "tok/s")
				})
			}
		})
	}
}
