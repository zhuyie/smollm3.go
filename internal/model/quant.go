package model

import (
	"math"
	"runtime"
	"sync"
)

const (
	matmulInt8MinParallelOps = 1 << 18
	matmulInt8RowsPerWorker  = 64
)

var matmulInt8WorkPool sync.Pool

type QuantizedMatrix struct {
	Data  []int8
	Scale []float32
}

func (t *Transformer) quantizeInt8() {
	cfg := t.Config
	kvDim := cfg.Dim * cfg.NKVHeads / cfg.NHeads
	for i := range t.Weights.Layers {
		lw := &t.Weights.Layers[i]
		lw.QWQ = quantizeMatrixInt8(lw.WQ, cfg.Dim, cfg.Dim)
		lw.WQ = nil
		lw.QWK = quantizeMatrixInt8(lw.WK, cfg.Dim, kvDim)
		lw.WK = nil
		lw.QWV = quantizeMatrixInt8(lw.WV, cfg.Dim, kvDim)
		lw.WV = nil
		lw.QWO = quantizeMatrixInt8(lw.WO, cfg.Dim, cfg.Dim)
		lw.WO = nil
		lw.QW1 = quantizeMatrixInt8(lw.W1, cfg.Dim, cfg.HiddenDim)
		lw.W1 = nil
		lw.QW2 = quantizeMatrixInt8(lw.W2, cfg.HiddenDim, cfg.Dim)
		lw.W2 = nil
		lw.QW3 = quantizeMatrixInt8(lw.W3, cfg.Dim, cfg.HiddenDim)
		lw.W3 = nil
	}
	t.Weights.QWCls = quantizeMatrixInt8(t.Weights.WCls, cfg.Dim, cfg.VocabSize)
	if !t.Weights.SharedWeights {
		t.Weights.WCls = nil
	}
}

func quantizeMatrixInt8(w []float32, inputs int, rows int) *QuantizedMatrix {
	q := &QuantizedMatrix{
		Data:  make([]int8, inputs*rows),
		Scale: make([]float32, rows),
	}
	for row := 0; row < rows; row++ {
		src := w[row*inputs : (row+1)*inputs]
		var maxAbs float32
		for _, v := range src {
			abs := float32(math.Abs(float64(v)))
			if abs > maxAbs {
				maxAbs = abs
			}
		}
		scale := float32(1)
		if maxAbs > 0 {
			scale = maxAbs / 127
		}
		q.Scale[row] = scale
		dst := q.Data[row*inputs : (row+1)*inputs]
		invScale := float32(1) / scale
		for i, v := range src {
			quantized := int(math.Round(float64(v * invScale)))
			if quantized > 127 {
				quantized = 127
			} else if quantized < -127 {
				quantized = -127
			}
			dst[i] = int8(quantized)
		}
	}
	return q
}

func matmulWeight(out []float32, x []float32, w []float32, q *QuantizedMatrix, n int, d int) {
	if q != nil {
		matmulInt8(out, x, q, n, d)
		return
	}
	matmul(out, x, w, n, d)
}

func matmulBatchWeight(out []float32, x []float32, w []float32, q *QuantizedMatrix, batch int, n int, d int) {
	if q != nil {
		matmulBatchInt8(out, x, q, batch, n, d)
		return
	}
	matmulBatch(out, x, w, batch, n, d)
}

func matmulInt8(out []float32, x []float32, q *QuantizedMatrix, n int, d int) {
	workers := min(runtime.GOMAXPROCS(0), matmulMaxWorkers, (d+matmulInt8RowsPerWorker-1)/matmulInt8RowsPerWorker)
	if n*d < matmulInt8MinParallelOps || workers < 2 {
		matmulInt8Rows(out, x, q, n, 0, d)
		return
	}
	startMatmulWorkers()

	out = out[:d]
	x = x[:n]
	rowsPerWorker := (d + workers - 1) / workers
	wg := matmulWaitGroupPool.Get().(*sync.WaitGroup)
	for start := 0; start < d; start += rowsPerWorker {
		end := start + rowsPerWorker
		if end > d {
			end = d
		}
		wg.Add(1)
		matmulInt8Jobs <- matmulInt8Job{
			out:  out,
			x:    x,
			q:    q,
			n:    n,
			row0: start,
			row1: end,
			wg:   wg,
		}
	}
	wg.Wait()
	matmulWaitGroupPool.Put(wg)
}

func matmulInt8Rows(out []float32, x []float32, q *QuantizedMatrix, n int, row0 int, row1 int) {
	x = x[:n]
	data := q.Data[:row1*n]
	for row := row0; row < row1; row++ {
		weights := data[row*n : (row+1)*n]
		out[row] = dotF32Int8(x, weights) * q.Scale[row]
	}
}

func matmulBatchInt8(out []float32, x []float32, q *QuantizedMatrix, batch int, n int, d int) {
	if batch == 1 {
		matmulInt8(out[:d], x[:n], q, n, d)
		return
	}
	workers := min(runtime.GOMAXPROCS(0), matmulMaxWorkers, (d+matmulInt8RowsPerWorker-1)/matmulInt8RowsPerWorker)
	if batch*n*d < matmulInt8MinParallelOps || workers < 2 {
		work := getMatmulInt8Work(n)
		defer putMatmulInt8Work(work)
		matmulBatchInt8Rows(out, x, q, work, batch, n, d, 0, d)
		return
	}
	startMatmulWorkers()

	out = out[:batch*d]
	x = x[:batch*n]
	rowsPerWorker := (d + workers - 1) / workers
	work := getMatmulInt8Work(workers * n)
	defer putMatmulInt8Work(work)
	wg := matmulWaitGroupPool.Get().(*sync.WaitGroup)
	worker := 0
	for start := 0; start < d; start += rowsPerWorker {
		end := start + rowsPerWorker
		if end > d {
			end = d
		}
		workRow := work[worker*n : (worker+1)*n]
		worker++
		wg.Add(1)
		matmulBatchInt8Jobs <- matmulBatchInt8Job{
			out:   out,
			x:     x,
			q:     q,
			work:  workRow,
			batch: batch,
			n:     n,
			d:     d,
			row0:  start,
			row1:  end,
			wg:    wg,
		}
	}
	wg.Wait()
	matmulWaitGroupPool.Put(wg)
}

func matmulBatchInt8Rows(out []float32, x []float32, q *QuantizedMatrix, work []float32, batch int, n int, d int, row0 int, row1 int) {
	out = out[:batch*d]
	x = x[:batch*n]
	work = work[:n]
	for row := row0; row < row1; row++ {
		weights := q.Data[row*n : (row+1)*n]
		scale := q.Scale[row]
		for i, weight := range weights {
			work[i] = float32(weight) * scale
		}
		b := 0
		for ; b+3 < batch; b += 4 {
			x0 := x[b*n : (b+1)*n]
			x1 := x[(b+1)*n : (b+2)*n]
			x2 := x[(b+2)*n : (b+3)*n]
			x3 := x[(b+3)*n : (b+4)*n]
			v0, v1, v2, v3 := dotF32Batch4(x0, x1, x2, x3, work)
			out[b*d+row] = v0
			out[(b+1)*d+row] = v1
			out[(b+2)*d+row] = v2
			out[(b+3)*d+row] = v3
		}
		for ; b < batch; b++ {
			out[b*d+row] = dotF32(x[b*n:(b+1)*n], work)
		}
	}
}

func getMatmulInt8Work(size int) []float32 {
	if value := matmulInt8WorkPool.Get(); value != nil {
		work := value.([]float32)
		if cap(work) >= size {
			return work[:size]
		}
	}
	return make([]float32, size)
}

func putMatmulInt8Work(work []float32) {
	matmulInt8WorkPool.Put(work[:0])
}
