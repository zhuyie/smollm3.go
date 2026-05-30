package model

import (
	"runtime"
	"sync"
)

const (
	matmulInt8MinParallelOps = 1 << 18
	matmulInt8RowsPerWorker  = 64
)

type QuantizedMatrix struct {
	Data  []int8
	Scale []float32
}

func matmulWeight(s *State, out []float32, x []float32, w []float32, q *QuantizedMatrix, n int, d int) {
	if q != nil {
		matmulInt8WithScratch(out, x, q, n, d, s.int8Activation(n))
		return
	}
	matmul(out, x, w, n, d)
}

func matmulBatchWeight(s *State, out []float32, x []float32, w []float32, q *QuantizedMatrix, batch int, n int, d int) {
	if q != nil {
		matmulBatchInt8WithScratch(out, x, q, batch, n, d, s.int8Activation(batch*n), s.int8Scales(batch))
		return
	}
	matmulBatch(out, x, w, batch, n, d)
}

func matmulInt8WithScratch(out []float32, x []float32, q *QuantizedMatrix, n int, d int, xq []int8) {
	xq = xq[:n]
	// Quantize the current activation vector once, then reuse it for every
	// output row. This turns the hot dot product from float32*int8 into
	// int8*int8, which maps directly to ARM SDOT on Apple Silicon.
	xScale := quantizeActivationInt8(xq, x[:n])

	workers := min(runtime.GOMAXPROCS(0), matmulMaxWorkers, (d+matmulInt8RowsPerWorker-1)/matmulInt8RowsPerWorker)
	if n*d < matmulInt8MinParallelOps || workers < 2 {
		matmulInt8Rows(out, xq, q, n, xScale, 0, d)
		return
	}
	startMatmulWorkers()

	out = out[:d]
	rowsPerWorker := (d + workers - 1) / workers
	wg := matmulWaitGroupPool.Get().(*sync.WaitGroup)
	for start := 0; start < d; start += rowsPerWorker {
		end := start + rowsPerWorker
		if end > d {
			end = d
		}
		wg.Add(1)
		matmulInt8Jobs <- matmulInt8Job{
			out:    out,
			x:      xq,
			q:      q,
			n:      n,
			xScale: xScale,
			row0:   start,
			row1:   end,
			wg:     wg,
		}
	}
	wg.Wait()
	matmulWaitGroupPool.Put(wg)
}

func matmulInt8Rows(out []float32, x []int8, q *QuantizedMatrix, n int, xScale float32, row0 int, row1 int) {
	x = x[:n]
	data := q.Data[:row1*n]
	for row := row0; row < row1; row++ {
		weights := data[row*n : (row+1)*n]
		out[row] = float32(dotInt8(x, weights)) * xScale * q.Scale[row]
	}
}

func matmulBatchInt8WithScratch(out []float32, x []float32, q *QuantizedMatrix, batch int, n int, d int, xq []int8, xScale []float32) {
	if batch == 1 {
		matmulInt8WithScratch(out[:d], x[:n], q, n, d, xq[:n])
		return
	}
	xq = xq[:batch*n]
	xScale = xScale[:batch]
	quantizeBatchActivationInt8(xq, xScale, x[:batch*n], batch, n)

	workers := min(runtime.GOMAXPROCS(0), matmulMaxWorkers, (d+matmulInt8RowsPerWorker-1)/matmulInt8RowsPerWorker)
	if batch*n*d < matmulInt8MinParallelOps || workers < 2 {
		matmulBatchInt8Rows(out, xq, xScale, q, batch, n, d, 0, d)
		return
	}
	startMatmulWorkers()

	out = out[:batch*d]
	rowsPerWorker := (d + workers - 1) / workers
	wg := matmulWaitGroupPool.Get().(*sync.WaitGroup)
	for start := 0; start < d; start += rowsPerWorker {
		end := start + rowsPerWorker
		if end > d {
			end = d
		}
		wg.Add(1)
		matmulBatchInt8Jobs <- matmulBatchInt8Job{
			out:    out,
			x:      xq,
			xScale: xScale,
			q:      q,
			batch:  batch,
			n:      n,
			d:      d,
			row0:   start,
			row1:   end,
			wg:     wg,
		}
	}
	wg.Wait()
	matmulWaitGroupPool.Put(wg)
}

func matmulBatchInt8Rows(out []float32, x []int8, xScale []float32, q *QuantizedMatrix, batch int, n int, d int, row0 int, row1 int) {
	out = out[:batch*d]
	x = x[:batch*n]
	for row := row0; row < row1; row++ {
		weights := q.Data[row*n : (row+1)*n]
		weightScale := q.Scale[row]
		for b := 0; b < batch; b++ {
			out[b*d+row] = float32(dotInt8(x[b*n:(b+1)*n], weights)) * xScale[b] * weightScale
		}
	}
}

func quantizeActivationInt8(dst []int8, x []float32) float32 {
	var m0, m1, m2, m3 float32
	i := 0
	for ; i+3 < len(x); i += 4 {
		v0 := x[i]
		v1 := x[i+1]
		v2 := x[i+2]
		v3 := x[i+3]
		if v0 < 0 {
			v0 = -v0
		}
		if v1 < 0 {
			v1 = -v1
		}
		if v2 < 0 {
			v2 = -v2
		}
		if v3 < 0 {
			v3 = -v3
		}
		if v0 > m0 {
			m0 = v0
		}
		if v1 > m1 {
			m1 = v1
		}
		if v2 > m2 {
			m2 = v2
		}
		if v3 > m3 {
			m3 = v3
		}
	}
	maxAbs := max(max(m0, m1), max(m2, m3))
	for ; i < len(x); i++ {
		v := x[i]
		if v < 0 {
			v = -v
		}
		if v > maxAbs {
			maxAbs = v
		}
	}
	if maxAbs == 0 {
		clear(dst[:len(x)])
		return 0
	}
	scale := maxAbs / 127
	invScale := 1 / scale
	i = 0
	for ; i+3 < len(x); i += 4 {
		dst[i] = roundClampInt8(x[i] * invScale)
		dst[i+1] = roundClampInt8(x[i+1] * invScale)
		dst[i+2] = roundClampInt8(x[i+2] * invScale)
		dst[i+3] = roundClampInt8(x[i+3] * invScale)
	}
	for ; i < len(x); i++ {
		v := x[i]
		dst[i] = roundClampInt8(v * invScale)
	}
	return scale
}

func roundClampInt8(v float32) int8 {
	q := int(v + 0.5)
	if v < 0 {
		q = int(v - 0.5)
	}
	if q > 127 {
		return 127
	}
	if q < -127 {
		return -127
	}
	return int8(q)
}

func quantizeBatchActivationInt8(dst []int8, scales []float32, x []float32, batch int, n int) {
	for b := 0; b < batch; b++ {
		scales[b] = quantizeActivationInt8(dst[b*n:(b+1)*n], x[b*n:(b+1)*n])
	}
}

func (s *State) int8Activation(size int) []int8 {
	if cap(s.Int8) < size {
		s.Int8 = make([]int8, size)
	}
	return s.Int8[:size]
}

func (s *State) int8Scales(size int) []float32 {
	if cap(s.Int8Scales) < size {
		s.Int8Scales = make([]float32, size)
	}
	return s.Int8Scales[:size]
}
