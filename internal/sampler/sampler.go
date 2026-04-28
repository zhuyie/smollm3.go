package sampler

import (
	"math"
	"math/rand"
	"sort"
)

type Sampler struct {
	Temperature float32
	TopP        float32
	rng         *rand.Rand
}

func New(temp float32, topP float32, seed int64) *Sampler {
	return &Sampler{
		Temperature: temp,
		TopP:        topP,
		rng:         rand.New(rand.NewSource(seed)),
	}
}

func (s *Sampler) Sample(logits []float32) int {
	if s.Temperature == 0 {
		return argmax(logits)
	}
	probs := make([]float32, len(logits))
	for i, v := range logits {
		probs[i] = v / s.Temperature
	}
	softmax(probs)
	coin := s.rng.Float32()
	if s.TopP <= 0 || s.TopP >= 1 {
		return sampleMult(probs, coin)
	}
	return sampleTopP(probs, s.TopP, coin)
}

func argmax(x []float32) int {
	best := 0
	bestVal := x[0]
	for i, v := range x[1:] {
		if v > bestVal {
			best = i + 1
			bestVal = v
		}
	}
	return best
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

func sampleMult(probs []float32, coin float32) int {
	var cdf float32
	for i, p := range probs {
		cdf += p
		if coin < cdf {
			return i
		}
	}
	return len(probs) - 1
}

type probIndex struct {
	prob  float32
	index int
}

func sampleTopP(probs []float32, topP float32, coin float32) int {
	cutoff := (1 - topP) / float32(len(probs)-1)
	candidates := make([]probIndex, 0)
	for i, p := range probs {
		if p >= cutoff {
			candidates = append(candidates, probIndex{prob: p, index: i})
		}
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].prob > candidates[j].prob })
	var cumulative float32
	last := len(candidates) - 1
	for i, item := range candidates {
		cumulative += item.prob
		if cumulative > topP {
			last = i
			break
		}
	}
	r := coin * cumulative
	var cdf float32
	for i := 0; i <= last; i++ {
		cdf += candidates[i].prob
		if r < cdf {
			return candidates[i].index
		}
	}
	return candidates[last].index
}
