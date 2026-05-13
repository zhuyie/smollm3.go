//go:build amd64

#include "textflag.h"

// func hasAVX2AMD64() bool
TEXT ·hasAVX2AMD64(SB), NOSPLIT, $0-1
	MOVL $1, AX
	XORL CX, CX
	CPUID
	TESTL $(1<<27), CX // OSXSAVE
	JZ no_avx2
	TESTL $(1<<28), CX // AVX
	JZ no_avx2

	XORL CX, CX
	BYTE $0x0f; BYTE $0x01; BYTE $0xd0 // XGETBV
	ANDL $6, AX
	CMPL AX, $6 // XMM and YMM state enabled by the OS
	JNE no_avx2

	MOVL $7, AX
	XORL CX, CX
	CPUID
	TESTL $(1<<5), BX // AVX2
	JZ no_avx2
	MOVB $1, ret+0(FP)
	RET

no_avx2:
	MOVB $0, ret+0(FP)
	RET

// func dotF32AMD64(a, b []float32) float32
// The Go wrapper calls this with a length that is a multiple of 8.
TEXT ·dotF32AMD64(SB), NOSPLIT, $0-52
	MOVQ a_base+0(FP), AX
	MOVQ a_len+8(FP), CX
	MOVQ b_base+24(FP), BX

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	SHRQ $4, CX
	JZ dot_f32_rem8

dot_f32_loop16:
	VMOVUPS (AX), Y2
	VMOVUPS 32(AX), Y3
	VMOVUPS (BX), Y4
	VMOVUPS 32(BX), Y5
	VMULPS Y4, Y2, Y2
	VMULPS Y5, Y3, Y3
	VADDPS Y2, Y0, Y0
	VADDPS Y3, Y1, Y1
	ADDQ $64, AX
	ADDQ $64, BX
	DECQ CX
	JNZ dot_f32_loop16

dot_f32_rem8:
	MOVQ a_len+8(FP), CX
	ANDQ $15, CX
	CMPQ CX, $8
	JB dot_f32_reduce
	VMOVUPS (AX), Y2
	VMOVUPS (BX), Y4
	VMULPS Y4, Y2, Y2
	VADDPS Y2, Y0, Y0

dot_f32_reduce:
	VADDPS Y1, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
	VMOVSS X0, ret+48(FP)
	VZEROUPPER
	RET

// func dotF32Batch4AMD64(x0, x1, x2, x3, w []float32) (float32, float32, float32, float32)
// The Go wrapper calls this with a length that is a multiple of 8.
TEXT ·dotF32Batch4AMD64(SB), NOSPLIT, $0-136
	MOVQ x0_base+0(FP), AX
	MOVQ x0_len+8(FP), CX
	MOVQ x1_base+24(FP), BX
	MOVQ x2_base+48(FP), DX
	MOVQ x3_base+72(FP), SI
	MOVQ w_base+96(FP), DI

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	SHRQ $3, CX
	JZ batch4_reduce

batch4_loop8:
	VMOVUPS (DI), Y4
	VMOVUPS (AX), Y5
	VMOVUPS (BX), Y6
	VMOVUPS (DX), Y7
	VMOVUPS (SI), Y8
	VMULPS Y4, Y5, Y5
	VMULPS Y4, Y6, Y6
	VMULPS Y4, Y7, Y7
	VMULPS Y4, Y8, Y8
	VADDPS Y5, Y0, Y0
	VADDPS Y6, Y1, Y1
	VADDPS Y7, Y2, Y2
	VADDPS Y8, Y3, Y3
	ADDQ $32, AX
	ADDQ $32, BX
	ADDQ $32, DX
	ADDQ $32, SI
	ADDQ $32, DI
	DECQ CX
	JNZ batch4_loop8

batch4_reduce:
	VEXTRACTF128 $1, Y0, X4
	VADDPS X4, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
	VMOVSS X0, ret+120(FP)

	VEXTRACTF128 $1, Y1, X4
	VADDPS X4, X1, X1
	VHADDPS X1, X1, X1
	VHADDPS X1, X1, X1
	VMOVSS X1, ret1+124(FP)

	VEXTRACTF128 $1, Y2, X4
	VADDPS X4, X2, X2
	VHADDPS X2, X2, X2
	VHADDPS X2, X2, X2
	VMOVSS X2, ret2+128(FP)

	VEXTRACTF128 $1, Y3, X4
	VADDPS X4, X3, X3
	VHADDPS X3, X3, X3
	VHADDPS X3, X3, X3
	VMOVSS X3, ret3+132(FP)
	VZEROUPPER
	RET

// func dotF32Int8AMD64(x []float32, w []int8) float32
// Computes sum(x[i] * float32(w[i])). The Go wrapper calls this for lengths
// that are multiples of 32.
TEXT ·dotF32Int8AMD64(SB), NOSPLIT, $0-52
	MOVQ x_base+0(FP), AX
	MOVQ x_len+8(FP), CX
	MOVQ w_base+24(FP), BX

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	SHRQ $5, CX
	JZ dot_i8_reduce

dot_i8_loop32:
	VPMOVSXBD (BX), Y4
	VPMOVSXBD 8(BX), Y5
	VPMOVSXBD 16(BX), Y6
	VPMOVSXBD 24(BX), Y7
	VCVTDQ2PS Y4, Y4
	VCVTDQ2PS Y5, Y5
	VCVTDQ2PS Y6, Y6
	VCVTDQ2PS Y7, Y7
	VMOVUPS (AX), Y8
	VMOVUPS 32(AX), Y9
	VMOVUPS 64(AX), Y10
	VMOVUPS 96(AX), Y11
	VMULPS Y8, Y4, Y4
	VMULPS Y9, Y5, Y5
	VMULPS Y10, Y6, Y6
	VMULPS Y11, Y7, Y7
	VADDPS Y4, Y0, Y0
	VADDPS Y5, Y1, Y1
	VADDPS Y6, Y2, Y2
	VADDPS Y7, Y3, Y3
	ADDQ $128, AX
	ADDQ $32, BX
	DECQ CX
	JNZ dot_i8_loop32

dot_i8_reduce:
	VADDPS Y1, Y0, Y0
	VADDPS Y3, Y2, Y2
	VADDPS Y2, Y0, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
	VMOVSS X0, ret+48(FP)
	VZEROUPPER
	RET

// func addScaledF32AMD64(dst, src []float32, scale float32)
// The Go wrapper calls this with a length that is a multiple of 8.
TEXT ·addScaledF32AMD64(SB), NOSPLIT, $0-52
	MOVQ dst_base+0(FP), AX
	MOVQ dst_len+8(FP), CX
	MOVQ src_base+24(FP), BX
	VBROADCASTSS scale+48(FP), Y0
	SHRQ $3, CX
	JZ addscaled_done

addscaled_loop8:
	VMOVUPS (AX), Y1
	VMOVUPS (BX), Y2
	VMULPS Y0, Y2, Y2
	VADDPS Y2, Y1, Y1
	VMOVUPS Y1, (AX)
	ADDQ $32, AX
	ADDQ $32, BX
	DECQ CX
	JNZ addscaled_loop8

addscaled_done:
	VZEROUPPER
	RET
