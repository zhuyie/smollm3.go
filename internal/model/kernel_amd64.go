//go:build amd64

package model

// simdMinN avoids paying assembly-call overhead for tiny vectors.
const simdMinN = 64

var useAVX2 = hasAVX2AMD64()

func hasAVX2AMD64() bool
func dotF32AMD64(a []float32, b []float32) float32
func dotF32Batch4AMD64(x0 []float32, x1 []float32, x2 []float32, x3 []float32, w []float32) (float32, float32, float32, float32)
func addScaledF32AMD64(dst []float32, src []float32, scale float32)

func dotF32(a []float32, b []float32) float32 {
	n := min(len(a), len(b))
	vecN := n &^ 7
	if useAVX2 && vecN >= simdMinN {
		val := dotF32AMD64(a[:vecN], b[:vecN])
		if vecN < n {
			val += dotF32Scalar(a[vecN:n], b[vecN:n])
		}
		return val
	}
	return dotF32Scalar(a[:n], b[:n])
}

func dotF32Batch4(x0 []float32, x1 []float32, x2 []float32, x3 []float32, w []float32) (float32, float32, float32, float32) {
	n := min(len(x0), len(x1), len(x2), len(x3), len(w))
	vecN := n &^ 7
	if useAVX2 && vecN >= simdMinN {
		v0, v1, v2, v3 := dotF32Batch4AMD64(x0[:vecN], x1[:vecN], x2[:vecN], x3[:vecN], w[:vecN])
		if vecN < n {
			r0, r1, r2, r3 := dotF32Batch4Scalar(x0[vecN:n], x1[vecN:n], x2[vecN:n], x3[vecN:n], w[vecN:n])
			v0 += r0
			v1 += r1
			v2 += r2
			v3 += r3
		}
		return v0, v1, v2, v3
	}
	return dotF32Batch4Scalar(x0[:n], x1[:n], x2[:n], x3[:n], w[:n])
}

func dotInt8(x []int8, w []int8) int32 {
	n := min(len(x), len(w))
	return dotInt8Scalar(x[:n], w[:n])
}

func matmulF32(out []float32, x []float32, w []float32, n int, d int) {
	if n >= simdMinN {
		out = out[:d]
		x = x[:n]
		w = w[:d*n]
		for i := range out {
			row := w[:n]
			w = w[n:]
			out[i] = dotF32(x, row)
		}
		return
	}
	matmulScalar(out, x, w, n, d)
}

func addScaledF32(dst []float32, src []float32, scale float32) {
	n := min(len(dst), len(src))
	vecN := n &^ 7
	if useAVX2 && vecN >= 8 {
		addScaledF32AMD64(dst[:vecN], src[:vecN], scale)
	} else {
		vecN = 0
	}
	if vecN < n {
		addScaledF32Scalar(dst[vecN:n], src[vecN:n], scale)
	}
}

func attentionValue(out []float32, att []float32, values []float32, steps int, stride int, offset int) {
	attentionValueScalar(out, att, values, steps, stride, offset)
}
