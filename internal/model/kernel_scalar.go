package model

func dotF32Scalar(a []float32, b []float32) float32 {
	var v0, v1, v2, v3 float32
	j := 0
	n := len(a)
	for ; j+3 < n; j += 4 {
		v0 += a[j] * b[j]
		v1 += a[j+1] * b[j+1]
		v2 += a[j+2] * b[j+2]
		v3 += a[j+3] * b[j+3]
	}
	val := v0 + v1 + v2 + v3
	for ; j < n; j++ {
		val += a[j] * b[j]
	}
	return val
}

func dotF32Batch4Scalar(x0 []float32, x1 []float32, x2 []float32, x3 []float32, w []float32) (float32, float32, float32, float32) {
	var v0, v1, v2, v3 float32
	for i, weight := range w {
		v0 += x0[i] * weight
		v1 += x1[i] * weight
		v2 += x2[i] * weight
		v3 += x3[i] * weight
	}
	return v0, v1, v2, v3
}

func dotF32Int8Scalar(x []float32, w []int8) float32 {
	var v0, v1, v2, v3 float32
	j := 0
	n := len(x)
	for ; j+3 < n; j += 4 {
		v0 += x[j] * float32(w[j])
		v1 += x[j+1] * float32(w[j+1])
		v2 += x[j+2] * float32(w[j+2])
		v3 += x[j+3] * float32(w[j+3])
	}
	val := v0 + v1 + v2 + v3
	for ; j < n; j++ {
		val += x[j] * float32(w[j])
	}
	return val
}

func matmulScalar(out []float32, x []float32, w []float32, n int, d int) {
	out = out[:d]
	x = x[:n]
	w = w[:d*n]
	for i := range out {
		// Keep row slicing explicit so the compiler's BCE pass can prove bounds.
		row := w[:n]
		w = w[n:]
		out[i] = dotF32Scalar(row, x)
	}
}

func addScaledF32Scalar(dst []float32, src []float32, scale float32) {
	n := min(len(dst), len(src))
	dst = dst[:n]
	src = src[:n]
	for i := range dst {
		dst[i] += scale * src[i]
	}
}
