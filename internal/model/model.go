package model

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

const (
	checkpointMagic               uint32 = 0x334c4d53 // SML3
	checkpointVersion                    = int32(1)
	checkpointHeaderSize                 = int64(256)
	checkpointMaxRopeHeaderLayers        = 48
	weightTypeFP32                       = int32(0)
	weightTypeInt8                       = int32(1)
)

type Config struct {
	// Dim is the residual stream width. Every token position carries one vector
	// of this size through all transformer blocks.
	Dim int
	// HiddenDim is the intermediate width of the feed-forward network.
	HiddenDim int
	// NLayers is the number of transformer blocks.
	NLayers int
	// NHeads is the number of query heads.
	NHeads int
	// NKVHeads is the number of key/value heads. This value may be smaller
	// than NHeads when the model uses grouped-query attention.
	NKVHeads int
	// VocabSize is the number of tokenizer ids and output logits.
	VocabSize int
	// SeqLen is the maximum KV cache length supported by this checkpoint.
	SeqLen int
	// RopeTheta is the base frequency used by rotary positional embeddings.
	RopeTheta float32
	// RMSNormEps is the epsilon used by all RMSNorm layers.
	RMSNormEps float32
	BOSID      int
	EOSID      int
	PADID      int
	// RopeLayers marks which layers apply RoPE to Q/K. SmolLM3 uses a 3:1
	// RoPE/NoPE pattern, so this is part of model identity rather than tuning.
	RopeLayers []bool
}

type LayerWeights struct {
	// RMSNorm weights for the pre-attention and pre-FFN normalizations.
	RMSAttWeight []float32
	RMSFFNWeight []float32
	// Attention projections. WQ outputs Dim values, while WK/WV output kvDim
	// values because K/V only have NKVHeads heads.
	WQ []float32
	WK []float32
	WV []float32
	// WO projects concatenated attention head outputs back to Dim.
	WO []float32
	// Feed-forward projections for SwiGLU: W2(silu(W1(x)) * W3(x)).
	W1 []float32
	W2 []float32
	W3 []float32

	QWQ *QuantizedMatrix
	QWK *QuantizedMatrix
	QWV *QuantizedMatrix
	QWO *QuantizedMatrix
	QW1 *QuantizedMatrix
	QW2 *QuantizedMatrix
	QW3 *QuantizedMatrix
}

// Weights owns all model parameters.
type Weights struct {
	TokenEmbeddingTable []float32
	Layers              []LayerWeights
	RMSFinalWeight      []float32
	WCls                []float32
	QWCls               *QuantizedMatrix
	SharedWeights       bool
}

// State contains the reusable scratch buffers and KV cache.
type State struct {
	X           []float32
	XB          []float32
	XB2         []float32
	HB          []float32
	HB2         []float32
	Q           []float32
	K           []float32
	V           []float32
	Att         []float32
	Logits      []float32
	KeyCache    []float32
	ValueCache  []float32
	BatchDim    []float32
	BatchKVDim  []float32
	BatchHidden []float32
}

type Tables struct {
	RopeCos []float32
	RopeSin []float32
}

type Transformer struct {
	Config  Config
	Weights Weights
	State   State
	Tables  Tables
}

func Load(path string) (*Transformer, error) {
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
	if magic != checkpointMagic || version != checkpointVersion {
		return nil, fmt.Errorf("bad checkpoint header: magic=%#x version=%d", magic, version)
	}

	var fields [11]int32
	for i := range fields {
		if err := binary.Read(file, binary.LittleEndian, &fields[i]); err != nil {
			return nil, err
		}
	}
	var ropeTheta float32
	if err := binary.Read(file, binary.LittleEndian, &ropeTheta); err != nil {
		return nil, err
	}
	var rmsNormEps float32
	if err := binary.Read(file, binary.LittleEndian, &rmsNormEps); err != nil {
		return nil, err
	}
	var ropeLayerFields [checkpointMaxRopeHeaderLayers]int32
	for i := range ropeLayerFields {
		if err := binary.Read(file, binary.LittleEndian, &ropeLayerFields[i]); err != nil {
			return nil, err
		}
	}
	var weightType int32
	if err := binary.Read(file, binary.LittleEndian, &weightType); err != nil {
		return nil, err
	}
	if weightType != weightTypeFP32 && weightType != weightTypeInt8 {
		return nil, fmt.Errorf("unsupported checkpoint weight type %d", weightType)
	}

	cfg := Config{
		Dim:        int(fields[0]),
		HiddenDim:  int(fields[1]),
		NLayers:    int(fields[2]),
		NHeads:     int(fields[3]),
		NKVHeads:   int(fields[4]),
		VocabSize:  int(fields[5]),
		SeqLen:     int(fields[6]),
		RopeTheta:  ropeTheta,
		RMSNormEps: rmsNormEps,
		BOSID:      int(fields[8]),
		EOSID:      int(fields[9]),
		PADID:      int(fields[10]),
	}
	sharedWeights := fields[7] != 0
	if cfg.NLayers > checkpointMaxRopeHeaderLayers {
		return nil, fmt.Errorf("checkpoint has %d layers, max supported in header is %d", cfg.NLayers, checkpointMaxRopeHeaderLayers)
	}
	cfg.RopeLayers = make([]bool, cfg.NLayers)
	for i := range cfg.RopeLayers {
		cfg.RopeLayers[i] = ropeLayerFields[i] != 0
	}
	if err := validateConfig(cfg); err != nil {
		return nil, err
	}
	if _, err := file.Seek(checkpointHeaderSize, io.SeekStart); err != nil {
		return nil, err
	}

	kvDim := cfg.Dim * cfg.NKVHeads / cfg.NHeads

	// Weight order must match the export tools exactly. Matrices are stored
	// row-major: each output channel owns one contiguous row consumed by matmul.
	weights := Weights{SharedWeights: sharedWeights}
	weights.TokenEmbeddingTable = readFloat32s(file, cfg.VocabSize*cfg.Dim)
	weights.Layers = make([]LayerWeights, cfg.NLayers)
	if weightType == weightTypeInt8 {
		readInt8LayerWeights(file, cfg, kvDim, weights.Layers)
	} else {
		readFP32LayerWeights(file, cfg, kvDim, weights.Layers)
	}
	weights.RMSFinalWeight = readFloat32s(file, cfg.Dim)
	if weightType == weightTypeInt8 {
		weights.QWCls = readQuantizedMatrix(file, cfg.Dim, cfg.VocabSize)
		if sharedWeights {
			weights.WCls = weights.TokenEmbeddingTable
		}
	} else if sharedWeights {
		weights.WCls = weights.TokenEmbeddingTable
	} else {
		weights.WCls = readFloat32s(file, cfg.VocabSize*cfg.Dim)
	}

	state := State{
		X:      make([]float32, cfg.Dim),
		XB:     make([]float32, cfg.Dim),
		XB2:    make([]float32, cfg.Dim),
		HB:     make([]float32, cfg.HiddenDim),
		HB2:    make([]float32, cfg.HiddenDim),
		Q:      make([]float32, cfg.Dim),
		K:      make([]float32, kvDim),
		V:      make([]float32, kvDim),
		Att:    make([]float32, cfg.NHeads*cfg.SeqLen),
		Logits: make([]float32, cfg.VocabSize),
		// KV cache layout: [layer][position][kvDim]. It is append-only along the
		// position dimension during autoregressive decoding.
		KeyCache:   make([]float32, cfg.NLayers*cfg.SeqLen*kvDim),
		ValueCache: make([]float32, cfg.NLayers*cfg.SeqLen*kvDim),
	}

	headSize := cfg.Dim / cfg.NHeads
	ropeCos, ropeSin := buildRopeTables(cfg.SeqLen, headSize, cfg.RopeTheta)

	return &Transformer{
		Config:  cfg,
		Weights: weights,
		State:   state,
		Tables: Tables{
			RopeCos: ropeCos,
			RopeSin: ropeSin,
		},
	}, nil
}

func validateConfig(cfg Config) error {
	if cfg.Dim <= 0 || cfg.HiddenDim <= 0 || cfg.NLayers <= 0 || cfg.NHeads <= 0 ||
		cfg.NKVHeads <= 0 || cfg.VocabSize <= 0 || cfg.SeqLen <= 0 || cfg.RopeTheta <= 0 ||
		cfg.RMSNormEps <= 0 {
		return fmt.Errorf("invalid config: %+v", cfg)
	}
	if cfg.Dim%cfg.NHeads != 0 || cfg.NHeads%cfg.NKVHeads != 0 {
		return fmt.Errorf("invalid attention dimensions: %+v", cfg)
	}
	if len(cfg.RopeLayers) != 0 && len(cfg.RopeLayers) != cfg.NLayers {
		return fmt.Errorf("rope layer table length %d does not match layer count %d", len(cfg.RopeLayers), cfg.NLayers)
	}
	return nil
}

func readFloat32s(r io.Reader, count int) []float32 {
	data := make([]float32, count)
	if err := binary.Read(r, binary.LittleEndian, data); err != nil {
		panic(err)
	}
	return data
}

func readInt8s(r io.Reader, count int) []int8 {
	data := make([]int8, count)
	if err := binary.Read(r, binary.LittleEndian, data); err != nil {
		panic(err)
	}
	return data
}

func readQuantizedMatrix(r io.Reader, inputs int, rows int) *QuantizedMatrix {
	return &QuantizedMatrix{
		Data:  readInt8s(r, inputs*rows),
		Scale: readFloat32s(r, rows),
	}
}

func readFP32LayerWeights(r io.Reader, cfg Config, kvDim int, layers []LayerWeights) {
	for i := range layers {
		lw := &layers[i]
		lw.RMSAttWeight = readFloat32s(r, cfg.Dim)
		lw.WQ = readFloat32s(r, cfg.Dim*cfg.Dim)
		lw.WK = readFloat32s(r, cfg.Dim*kvDim)
		lw.WV = readFloat32s(r, cfg.Dim*kvDim)
		lw.WO = readFloat32s(r, cfg.Dim*cfg.Dim)
		lw.RMSFFNWeight = readFloat32s(r, cfg.Dim)
		lw.W1 = readFloat32s(r, cfg.Dim*cfg.HiddenDim)
		lw.W2 = readFloat32s(r, cfg.HiddenDim*cfg.Dim)
		lw.W3 = readFloat32s(r, cfg.Dim*cfg.HiddenDim)
	}
}

func readInt8LayerWeights(r io.Reader, cfg Config, kvDim int, layers []LayerWeights) {
	for i := range layers {
		lw := &layers[i]
		lw.RMSAttWeight = readFloat32s(r, cfg.Dim)
		lw.QWQ = readQuantizedMatrix(r, cfg.Dim, cfg.Dim)
		lw.QWK = readQuantizedMatrix(r, cfg.Dim, kvDim)
		lw.QWV = readQuantizedMatrix(r, cfg.Dim, kvDim)
		lw.QWO = readQuantizedMatrix(r, cfg.Dim, cfg.Dim)
		lw.RMSFFNWeight = readFloat32s(r, cfg.Dim)
		lw.QW1 = readQuantizedMatrix(r, cfg.Dim, cfg.HiddenDim)
		lw.QW2 = readQuantizedMatrix(r, cfg.HiddenDim, cfg.Dim)
		lw.QW3 = readQuantizedMatrix(r, cfg.Dim, cfg.HiddenDim)
	}
}

func buildRopeTables(seqLen int, headSize int, ropeTheta float32) ([]float32, []float32) {
	headPairs := headSize / 2
	cosTable := make([]float32, seqLen*headPairs)
	sinTable := make([]float32, seqLen*headPairs)
	for pair := 0; pair < headPairs; pair++ {
		headDim := pair * 2
		freq := float32(1.0 / math.Pow(float64(ropeTheta), float64(headDim)/float64(headSize)))
		for pos := 0; pos < seqLen; pos++ {
			val := float32(pos) * freq
			idx := pos*headPairs + pair
			cosTable[idx] = float32(math.Cos(float64(val)))
			sinTable[idx] = float32(math.Sin(float64(val)))
		}
	}
	return cosTable, sinTable
}

// Forward evaluates one autoregressive decoding step.
//
// The caller passes the token id at absolute sequence position pos. During
// prompt prefill, this function is called once per prompt token with increasing
// pos. During generation, it is called once for each sampled token. In both
// cases the function appends the current token's K/V vectors into the cache and
// returns logits for the next token.
//
// The high-level flow is:
//  1. Copy the token embedding into the residual stream s.X.
//  2. For each layer:
//     - RMS-normalize the residual stream for attention.
//     - Project Q, K, and V; write K/V directly into the current cache slot.
//     - Apply RoPE to Q and the newly written K.
//     - Run causal self-attention over cached positions [0, pos].
//     - Apply the attention output projection and residual add.
//     - Run the RMSNorm + SwiGLU feed-forward block and residual add.
//  3. Apply final RMSNorm and project to vocabulary logits.
//
// State buffers are reused in place; callers should treat the returned logits
// slice as owned by the Transformer until the next Forward call.
func (t *Transformer) Forward(token int, pos int) []float32 {
	cfg := t.Config
	w := t.Weights
	s := &t.State
	dim := cfg.Dim
	kvDim := cfg.Dim * cfg.NKVHeads / cfg.NHeads
	kvMul := cfg.NHeads / cfg.NKVHeads
	hiddenDim := cfg.HiddenDim
	headSize := dim / cfg.NHeads
	headPairs := headSize / 2
	attScale := float32(1.0 / math.Sqrt(float64(headSize)))
	ropeCos := t.Tables.RopeCos[pos*headPairs : (pos+1)*headPairs]
	ropeSin := t.Tables.RopeSin[pos*headPairs : (pos+1)*headPairs]

	// Start from the token embedding. The residual stream then flows through all
	// transformer blocks in s.X.
	copy(s.X, w.TokenEmbeddingTable[token*dim:(token+1)*dim])

	for layer := 0; layer < cfg.NLayers; layer++ {
		lw := w.Layers[layer]
		// Attention sublayer starts with pre-norm, following the Llama-style
		// transformer block layout.
		rmsnorm(s.XB, s.X, lw.RMSAttWeight, cfg.RMSNormEps)

		// Select the cache row for this layer and position. K/V matmul outputs
		// below are written straight into these slices.
		loff := layer * cfg.SeqLen * kvDim
		kcache := s.KeyCache[loff+pos*kvDim : loff+(pos+1)*kvDim]
		vcache := s.ValueCache[loff+pos*kvDim : loff+(pos+1)*kvDim]

		// Query is transient for the current token; K/V persist in the cache so
		// future positions can attend back to them.
		matmulWeight(s.Q, s.XB, lw.WQ, lw.QWQ, dim, dim)
		matmulWeight(kcache, s.XB, lw.WK, lw.QWK, dim, kvDim)
		matmulWeight(vcache, s.XB, lw.WV, lw.QWV, dim, kvDim)

		// Apply RoPE only on layers that use positional rotation. SmolLM3 leaves
		// every fourth layer as NoPE to improve long-context behavior.
		if t.layerUsesRope(layer) {
			for i := 0; i < dim; i += 2 {
				pair := (i % headSize) / 2
				fcr, fci := ropeCos[pair], ropeSin[pair]
				rotn := 1
				if i < kvDim {
					rotn = 2
				}
				for v := 0; v < rotn; v++ {
					vec := s.Q
					if v == 1 {
						vec = kcache
					}
					v0, v1 := vec[i], vec[i+1]
					vec[i] = v0*fcr - v1*fci
					vec[i+1] = v0*fci + v1*fcr
				}
			}
		}

		// Causal self-attention. Query heads share KV heads when NHeads>NKVHeads.
		for h := 0; h < cfg.NHeads; h++ {
			q := s.Q[h*headSize : (h+1)*headSize]
			att := s.Att[h*cfg.SeqLen : (h+1)*cfg.SeqLen]
			kvHead := h / kvMul
			headOff := kvHead * headSize
			// Score this query head against every cached key up to pos. This is
			// the only O(sequence length) part of a single-token decoding step.
			for ts := 0; ts <= pos; ts++ {
				k := s.KeyCache[loff+ts*kvDim+headOff : loff+ts*kvDim+headOff+headSize]
				att[ts] = dotF32(q, k) * attScale
			}
			softmax(att[:pos+1])

			// Weighted sum of cached values produces this head's slice of s.XB.
			xb := s.XB[h*headSize : (h+1)*headSize]
			clear(xb)
			for ts := 0; ts <= pos; ts++ {
				v := s.ValueCache[loff+ts*kvDim+headOff : loff+ts*kvDim+headOff+headSize]
				a := att[ts]
				addScaledF32(xb, v, a)
			}
		}

		// Attention output projection plus residual connection.
		matmulWeight(s.XB2, s.XB, lw.WO, lw.QWO, dim, dim)
		for i := 0; i < dim; i++ {
			s.X[i] += s.XB2[i]
		}

		// SwiGLU feed-forward block: W2(silu(W1(x)) * W3(x)).
		rmsnorm(s.XB, s.X, lw.RMSFFNWeight, cfg.RMSNormEps)
		matmulWeight(s.HB, s.XB, lw.W1, lw.QW1, dim, hiddenDim)
		matmulWeight(s.HB2, s.XB, lw.W3, lw.QW3, dim, hiddenDim)
		for i := 0; i < hiddenDim; i++ {
			val := s.HB[i]
			val *= 1.0 / (1.0 + float32(math.Exp(float64(-val))))
			s.HB[i] = val * s.HB2[i]
		}
		matmulWeight(s.XB, s.HB, lw.W2, lw.QW2, hiddenDim, dim)
		for i := 0; i < dim; i++ {
			s.X[i] += s.XB[i]
		}
	}

	rmsnorm(s.X, s.X, w.RMSFinalWeight, cfg.RMSNormEps)
	matmulWeight(s.Logits, s.X, w.WCls, w.QWCls, dim, cfg.VocabSize)
	return s.Logits
}

// Prefill consumes a contiguous prompt span and returns logits for the next
// token after the final prompt token. It keeps decode state compatible with
// Forward by filling the same KV cache.
func (t *Transformer) Prefill(tokens []int, startPos int) []float32 {
	if len(tokens) == 0 {
		return nil
	}
	if len(tokens) == 1 {
		return t.Forward(tokens[0], startPos)
	}

	cfg := t.Config
	w := t.Weights
	s := &t.State
	dim := cfg.Dim
	kvDim := cfg.Dim * cfg.NKVHeads / cfg.NHeads
	kvMul := cfg.NHeads / cfg.NKVHeads
	hiddenDim := cfg.HiddenDim
	headSize := dim / cfg.NHeads
	headPairs := headSize / 2
	attScale := float32(1.0 / math.Sqrt(float64(headSize)))
	batch := len(tokens)
	if startPos < 0 || startPos+batch > cfg.SeqLen {
		panic("prefill range exceeds sequence length")
	}
	t.ensureBatch(batch)

	batchDim := s.BatchDim[:4*batch*dim]
	x := batchDim[0 : batch*dim]
	xb := batchDim[batch*dim : 2*batch*dim]
	xb2 := batchDim[2*batch*dim : 3*batch*dim]
	qb := batchDim[3*batch*dim : 4*batch*dim]
	batchKVDim := s.BatchKVDim[:2*batch*kvDim]
	kb := batchKVDim[0 : batch*kvDim]
	vb := batchKVDim[batch*kvDim : 2*batch*kvDim]
	batchHidden := s.BatchHidden[:2*batch*hiddenDim]
	hb := batchHidden[0 : batch*hiddenDim]
	hb2 := batchHidden[batch*hiddenDim : 2*batch*hiddenDim]

	for b, token := range tokens {
		copy(x[b*dim:(b+1)*dim], w.TokenEmbeddingTable[token*dim:(token+1)*dim])
	}

	for layer := 0; layer < cfg.NLayers; layer++ {
		lw := w.Layers[layer]
		rmsnormBatch(xb, x, lw.RMSAttWeight, batch, dim, cfg.RMSNormEps)
		matmulBatchWeight(qb, xb, lw.WQ, lw.QWQ, batch, dim, dim)
		matmulBatchWeight(kb, xb, lw.WK, lw.QWK, batch, dim, kvDim)
		matmulBatchWeight(vb, xb, lw.WV, lw.QWV, batch, dim, kvDim)

		loff := layer * cfg.SeqLen * kvDim
		for b := 0; b < batch; b++ {
			pos := startPos + b
			q := qb[b*dim : (b+1)*dim]
			krow := kb[b*kvDim : (b+1)*kvDim]
			vrow := vb[b*kvDim : (b+1)*kvDim]
			kcache := s.KeyCache[loff+pos*kvDim : loff+(pos+1)*kvDim]
			vcache := s.ValueCache[loff+pos*kvDim : loff+(pos+1)*kvDim]
			copy(kcache, krow)
			copy(vcache, vrow)

			if t.layerUsesRope(layer) {
				ropeCos := t.Tables.RopeCos[pos*headPairs : (pos+1)*headPairs]
				ropeSin := t.Tables.RopeSin[pos*headPairs : (pos+1)*headPairs]
				for i := 0; i < dim; i += 2 {
					pair := (i % headSize) / 2
					fcr, fci := ropeCos[pair], ropeSin[pair]
					rotn := 1
					if i < kvDim {
						rotn = 2
					}
					for v := 0; v < rotn; v++ {
						vec := q
						if v == 1 {
							vec = kcache
						}
						v0, v1 := vec[i], vec[i+1]
						vec[i] = v0*fcr - v1*fci
						vec[i+1] = v0*fci + v1*fcr
					}
				}
			}
		}

		for b := 0; b < batch; b++ {
			pos := startPos + b
			qrow := qb[b*dim : (b+1)*dim]
			xbrow := xb[b*dim : (b+1)*dim]
			for h := 0; h < cfg.NHeads; h++ {
				q := qrow[h*headSize : (h+1)*headSize]
				att := s.Att[h*cfg.SeqLen : (h+1)*cfg.SeqLen]
				kvHead := h / kvMul
				headOff := kvHead * headSize
				for ts := 0; ts <= pos; ts++ {
					k := s.KeyCache[loff+ts*kvDim+headOff : loff+ts*kvDim+headOff+headSize]
					att[ts] = dotF32(q, k) * attScale
				}
				softmax(att[:pos+1])

				xbh := xbrow[h*headSize : (h+1)*headSize]
				clear(xbh)
				for ts := 0; ts <= pos; ts++ {
					v := s.ValueCache[loff+ts*kvDim+headOff : loff+ts*kvDim+headOff+headSize]
					addScaledF32(xbh, v, att[ts])
				}
			}
		}

		matmulBatchWeight(xb2, xb, lw.WO, lw.QWO, batch, dim, dim)
		for i := range x {
			x[i] += xb2[i]
		}

		rmsnormBatch(xb, x, lw.RMSFFNWeight, batch, dim, cfg.RMSNormEps)
		matmulBatchWeight(hb, xb, lw.W1, lw.QW1, batch, dim, hiddenDim)
		matmulBatchWeight(hb2, xb, lw.W3, lw.QW3, batch, dim, hiddenDim)
		for i := range hb {
			val := hb[i]
			val *= 1.0 / (1.0 + float32(math.Exp(float64(-val))))
			hb[i] = val * hb2[i]
		}
		matmulBatchWeight(xb, hb, lw.W2, lw.QW2, batch, hiddenDim, dim)
		for i := range x {
			x[i] += xb[i]
		}
	}

	last := x[(batch-1)*dim : batch*dim]
	copy(s.X, last)
	rmsnorm(s.X, s.X, w.RMSFinalWeight, cfg.RMSNormEps)
	matmulWeight(s.Logits, s.X, w.WCls, w.QWCls, dim, cfg.VocabSize)
	return s.Logits
}

func (t *Transformer) ensureBatch(batch int) {
	cfg := t.Config
	dim := cfg.Dim
	kvDim := cfg.Dim * cfg.NKVHeads / cfg.NHeads
	hiddenDim := cfg.HiddenDim
	s := &t.State
	if cap(s.BatchDim) < 4*batch*dim {
		s.BatchDim = make([]float32, 4*batch*dim)
	}
	if cap(s.BatchKVDim) < 2*batch*kvDim {
		s.BatchKVDim = make([]float32, 2*batch*kvDim)
	}
	if cap(s.BatchHidden) < 2*batch*hiddenDim {
		s.BatchHidden = make([]float32, 2*batch*hiddenDim)
	}
}

func (t *Transformer) layerUsesRope(layer int) bool {
	if len(t.Config.RopeLayers) == 0 {
		return true
	}
	return t.Config.RopeLayers[layer]
}

func rmsnorm(out []float32, x []float32, weight []float32, eps float32) {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	ss = ss/float32(len(x)) + eps
	scale := float32(1.0 / math.Sqrt(float64(ss)))
	for i := range x {
		out[i] = weight[i] * scale * x[i]
	}
}

func rmsnormBatch(out []float32, x []float32, weight []float32, batch int, dim int, eps float32) {
	for b := 0; b < batch; b++ {
		rmsnorm(out[b*dim:(b+1)*dim], x[b*dim:(b+1)*dim], weight, eps)
	}
}

func softmax(x []float32) {
	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	var sum float32
	for i, v := range x {
		x[i] = float32(math.Exp(float64(v - maxVal)))
		sum += x[i]
	}
	for i := range x {
		x[i] /= sum
	}
}
