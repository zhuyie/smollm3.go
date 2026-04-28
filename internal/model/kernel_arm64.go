//go:build arm64

package model

// simdMinN avoids paying assembly-call overhead for tiny vectors.
const simdMinN = 64

func dotF32ARM64(a []float32, b []float32) float32
func dotF32Batch4ARM64(x0 []float32, x1 []float32, x2 []float32, x3 []float32, w []float32) (float32, float32, float32, float32)
func dotF32Int8ARM64(x []float32, w []int8) float32
func addScaledF32ARM64(dst []float32, src []float32, scale float32)

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

func dotF32Int8(x []float32, w []int8) float32 {
	n := min(len(x), len(w))
	if n >= simdMinN && n&15 == 0 {
		return dotF32Int8ARM64(x[:n], w[:n])
	}
	return dotF32Int8Scalar(x[:n], w[:n])
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
