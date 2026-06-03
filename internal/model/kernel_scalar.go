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

func dotInt8Scalar(x []int8, w []int8) int32 {
	var v0, v1, v2, v3 int32
	j := 0
	n := len(x)
	for ; j+3 < n; j += 4 {
		v0 += int32(x[j]) * int32(w[j])
		v1 += int32(x[j+1]) * int32(w[j+1])
		v2 += int32(x[j+2]) * int32(w[j+2])
		v3 += int32(x[j+3]) * int32(w[j+3])
	}
	val := v0 + v1 + v2 + v3
	for ; j < n; j++ {
		val += int32(x[j]) * int32(w[j])
	}
	return val
}

func dotInt8Batch4Scalar(x0 []int8, x1 []int8, x2 []int8, x3 []int8, w []int8) (int32, int32, int32, int32) {
	var v0, v1, v2, v3 int32
	for i, weight := range w {
		wi := int32(weight)
		v0 += int32(x0[i]) * wi
		v1 += int32(x1[i]) * wi
		v2 += int32(x2[i]) * wi
		v3 += int32(x3[i]) * wi
	}
	return v0, v1, v2, v3
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

func attentionValueScalar(out []float32, att []float32, values []float32, steps int, stride int, offset int) {
	clear(out)
	for ts := 0; ts < steps; ts++ {
		v := values[ts*stride+offset : ts*stride+offset+len(out)]
		a := att[ts]
		for i := range out {
			out[i] += a * v[i]
		}
	}
}

func attentionScoresScalar(out []float32, q []float32, keys []float32, steps int, stride int, offset int, scale float32) {
	for ts := 0; ts < steps; ts++ {
		k := keys[ts*stride+offset : ts*stride+offset+len(q)]
		out[ts] = dotF32Scalar(q, k) * scale
	}
}

func attentionScoresBatch4(out []float32, q []float32, keys []float32, steps int, stride int, offset int, scale float32) {
	ts := 0
	for ; ts+3 < steps; ts += 4 {
		k0 := keys[ts*stride+offset : ts*stride+offset+len(q)]
		k1 := keys[(ts+1)*stride+offset : (ts+1)*stride+offset+len(q)]
		k2 := keys[(ts+2)*stride+offset : (ts+2)*stride+offset+len(q)]
		k3 := keys[(ts+3)*stride+offset : (ts+3)*stride+offset+len(q)]
		v0, v1, v2, v3 := dotF32Batch4(k0, k1, k2, k3, q)
		out[ts] = v0 * scale
		out[ts+1] = v1 * scale
		out[ts+2] = v2 * scale
		out[ts+3] = v3 * scale
	}
	for ; ts < steps; ts++ {
		k := keys[ts*stride+offset : ts*stride+offset+len(q)]
		out[ts] = dotF32(q, k) * scale
	}
}
