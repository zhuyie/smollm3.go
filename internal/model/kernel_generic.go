//go:build !arm64

package model

func dotF32(a []float32, b []float32) float32 {
	n := min(len(a), len(b))
	return dotF32Scalar(a[:n], b[:n])
}

func dotF32Batch4(x0 []float32, x1 []float32, x2 []float32, x3 []float32, w []float32) (float32, float32, float32, float32) {
	n := min(len(x0), len(x1), len(x2), len(x3), len(w))
	return dotF32Batch4Scalar(x0[:n], x1[:n], x2[:n], x3[:n], w[:n])
}

func dotF32Int8(x []float32, w []int8) float32 {
	n := min(len(x), len(w))
	return dotF32Int8Scalar(x[:n], w[:n])
}

func matmulF32(out []float32, x []float32, w []float32, n int, d int) {
	matmulScalar(out, x, w, n, d)
}

func addScaledF32(dst []float32, src []float32, scale float32) {
	addScaledF32Scalar(dst, src, scale)
}
