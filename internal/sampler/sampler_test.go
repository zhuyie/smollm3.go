package sampler

import "testing"

func TestGreedySampleReturnsArgmax(t *testing.T) {
	s := New(0, 0, 1)
	got := s.Sample([]float32{-1, 3, 2})
	if got != 1 {
		t.Fatalf("Sample() = %d, want 1", got)
	}
}

func TestSampleMultUsesCumulativeProbability(t *testing.T) {
	probs := []float32{0.2, 0.3, 0.5}
	tests := []struct {
		coin float32
		want int
	}{
		{coin: 0.1, want: 0},
		{coin: 0.25, want: 1},
		{coin: 0.75, want: 2},
		{coin: 1.0, want: 2},
	}
	for _, tt := range tests {
		if got := sampleMult(probs, tt.coin); got != tt.want {
			t.Fatalf("sampleMult(%v, %.2f) = %d, want %d", probs, tt.coin, got, tt.want)
		}
	}
}

func TestSampleTopPFiltersAndSamplesCandidates(t *testing.T) {
	probs := []float32{0.6, 0.25, 0.1, 0.05}
	if got := sampleTopP(probs, 0.8, 0.95); got != 1 {
		t.Fatalf("sampleTopP() = %d, want 1", got)
	}
}
