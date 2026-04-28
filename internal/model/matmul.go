package model

import (
	"runtime"
	"sync"
)

const (
	matmulMinParallelOps = 1 << 20
	matmulRowsPerWorker  = 256
	matmulMaxWorkers     = 16
)

type matmulJob struct {
	out []float32
	x   []float32
	w   []float32
	n   int
	d   int
	wg  *sync.WaitGroup
}

type matmulBatchJob struct {
	out   []float32
	x     []float32
	w     []float32
	batch int
	n     int
	d     int
	row0  int
	row1  int
	wg    *sync.WaitGroup
}

var (
	matmulJobs          = make(chan matmulJob, matmulMaxWorkers)
	matmulBatchJobs     = make(chan matmulBatchJob, matmulMaxWorkers)
	matmulStartOnce     sync.Once
	matmulWaitGroupPool = sync.Pool{New: func() any { return new(sync.WaitGroup) }}
)

func matmul(out []float32, x []float32, w []float32, n int, d int) {
	workers := min(runtime.GOMAXPROCS(0), matmulMaxWorkers, d/matmulRowsPerWorker)
	if n*d < matmulMinParallelOps || workers < 2 {
		matmulF32(out, x, w, n, d)
		return
	}
	startMatmulWorkers()

	out = out[:d]
	x = x[:n]
	w = w[:d*n]

	rowsPerWorker := (d + workers - 1) / workers
	wg := matmulWaitGroupPool.Get().(*sync.WaitGroup)
	for start := 0; start < d; start += rowsPerWorker {
		end := start + rowsPerWorker
		if end > d {
			end = d
		}
		wg.Add(1)
		matmulJobs <- matmulJob{
			out: out[start:end],
			x:   x,
			w:   w[start*n : end*n],
			n:   n,
			d:   end - start,
			wg:  wg,
		}
	}
	wg.Wait()
	matmulWaitGroupPool.Put(wg)
}

func matmulBatch(out []float32, x []float32, w []float32, batch int, n int, d int) {
	if batch == 1 {
		matmul(out[:d], x[:n], w, n, d)
		return
	}
	workers := min(runtime.GOMAXPROCS(0), matmulMaxWorkers, d/matmulRowsPerWorker)
	if batch*n*d < matmulMinParallelOps || workers < 2 {
		matmulBatchRows(out, x, w, batch, n, d, 0, d)
		return
	}

	out = out[:batch*d]
	x = x[:batch*n]
	w = w[:d*n]

	rowsPerWorker := (d + workers - 1) / workers
	startMatmulWorkers()
	wg := matmulWaitGroupPool.Get().(*sync.WaitGroup)
	for start := 0; start < d; start += rowsPerWorker {
		end := start + rowsPerWorker
		if end > d {
			end = d
		}
		wg.Add(1)
		matmulBatchJobs <- matmulBatchJob{
			out:   out,
			x:     x,
			w:     w,
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

func matmulBatchRows(out []float32, x []float32, w []float32, batch int, n int, d int, row0 int, row1 int) {
	for row := row0; row < row1; row++ {
		weight := w[row*n : (row+1)*n]
		b := 0
		for ; b+3 < batch; b += 4 {
			x0 := x[b*n : (b+1)*n]
			x1 := x[(b+1)*n : (b+2)*n]
			x2 := x[(b+2)*n : (b+3)*n]
			x3 := x[(b+3)*n : (b+4)*n]
			v0, v1, v2, v3 := dotF32Batch4(x0, x1, x2, x3, weight)
			out[b*d+row] = v0
			out[(b+1)*d+row] = v1
			out[(b+2)*d+row] = v2
			out[(b+3)*d+row] = v3
		}
		for ; b < batch; b++ {
			out[b*d+row] = dotF32(x[b*n:(b+1)*n], weight)
		}
	}
}

func startMatmulWorkers() {
	matmulStartOnce.Do(func() {
		for i := 0; i < matmulMaxWorkers; i++ {
			go func() {
				for {
					select {
					case job := <-matmulJobs:
						matmulF32(job.out, job.x, job.w, job.n, job.d)
						job.wg.Done()
					case job := <-matmulBatchJobs:
						matmulBatchRows(job.out, job.x, job.w, job.batch, job.n, job.d, job.row0, job.row1)
						job.wg.Done()
					}
				}
			}()
		}
	})
}
