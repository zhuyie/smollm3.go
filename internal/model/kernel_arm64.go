//go:build arm64

package model

// simdMinN avoids paying assembly-call overhead for tiny vectors.
const simdMinN = 64

func dotF32ARM64(a []float32, b []float32) float32
func dotF32Batch4ARM64(x0 []float32, x1 []float32, x2 []float32, x3 []float32, w []float32) (float32, float32, float32, float32)
func dotInt8ARM64(x []int8, w []int8) int32
func addScaledF32ARM64(dst []float32, src []float32, scale float32)
func attentionValueARM64(out []float32, att []float32, values []float32, steps int, stride int, offset int)

func dotF32(a []float32, b []float32) float32 {
	n := min(len(a), len(b))
	if n >= simdMinN && n&3 == 0 {
		return dotF32ARM64(a[:n], b[:n])
	}
	return dotF32Scalar(a[:n], b[:n])
}

func dotF32Batch4(x0 []float32, x1 []float32, x2 []float32, x3 []float32, w []float32) (float32, float32, float32, float32) {
	n := min(len(x0), len(x1), len(x2), len(x3), len(w))
	if n >= simdMinN && n&3 == 0 {
		return dotF32Batch4ARM64(x0[:n], x1[:n], x2[:n], x3[:n], w[:n])
	}
	return dotF32Batch4Scalar(x0[:n], x1[:n], x2[:n], x3[:n], w[:n])
}

func dotInt8(x []int8, w []int8) int32 {
	n := min(len(x), len(w))
	if n >= simdMinN && n&15 == 0 {
		return dotInt8ARM64(x[:n], w[:n])
	}
	return dotInt8Scalar(x[:n], w[:n])
}

func dotInt8Batch4(x0 []int8, x1 []int8, x2 []int8, x3 []int8, w []int8) (int32, int32, int32, int32) {
	n := min(len(x0), len(x1), len(x2), len(x3), len(w))
	return dotInt8(x0[:n], w[:n]), dotInt8(x1[:n], w[:n]), dotInt8(x2[:n], w[:n]), dotInt8(x3[:n], w[:n])
}

func useDotInt8Batch4(int) bool {
	return false
}

func matmulF32(out []float32, x []float32, w []float32, n int, d int) {
	if n >= simdMinN && n&3 == 0 {
		out = out[:d]
		x = x[:n]
		w = w[:d*n]
		for i := range out {
			row := w[:n]
			w = w[n:]
			out[i] = dotF32ARM64(x, row)
		}
		return
	}
	matmulScalar(out, x, w, n, d)
}

func addScaledF32(dst []float32, src []float32, scale float32) {
	n := min(len(dst), len(src))
	vecN := n &^ 3
	if vecN > 0 {
		addScaledF32ARM64(dst[:vecN], src[:vecN], scale)
	}
	if vecN < n {
		addScaledF32Scalar(dst[vecN:n], src[vecN:n], scale)
	}
}

func attentionValue(out []float32, att []float32, values []float32, steps int, stride int, offset int) {
	if len(out) >= simdMinN && len(out)&3 == 0 {
		clear(out)
		attentionValueARM64(out, att[:steps], values, steps, stride, offset)
		return
	}
	attentionValueScalar(out, att, values, steps, stride, offset)
}

func attentionScores(out []float32, q []float32, keys []float32, steps int, stride int, offset int, scale float32) {
	attentionScoresBatch4(out, q, keys, steps, stride, offset, scale)
}
