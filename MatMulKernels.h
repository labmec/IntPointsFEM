#include "pzreal.h"

//#define TILE_WIDTH 16

//extern "C" {
//__global__ void matrixMultiply(REAL * A, REAL * B, REAL * C,
//  		       int numARows, int numAColumns,
//			       int numBRows, int numBColumns,
//			       int numCRows, int numCColumns) {
//    //@@ Insert code to implement matrix multiplication here
//    __shared__ REAL ds_M[TILE_WIDTH][TILE_WIDTH];
//    __shared__ REAL ds_N[TILE_WIDTH][TILE_WIDTH];
//    int bx = blockIdx.x, by = blockIdx.y,
//       tx = threadIdx.x, ty = threadIdx.y,
//       Row = by * TILE_WIDTH + ty,
//       Col = bx * TILE_WIDTH + tx;
//    REAL Pvalue = 0;
//
//    for (int m = 0; m < (numAColumns-1)/TILE_WIDTH+1; ++m) {
//       if (Row < numARows && m*TILE_WIDTH+tx < numAColumns)
//          ds_M[ty][tx] = A[Row + m*TILE_WIDTH+tx*numARows];
//       else
//          ds_M[ty][tx] = 0;
//       if (Col < numBColumns && m*TILE_WIDTH+ty < numBRows)
//          ds_N[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
//       else
//          ds_N[ty][tx] = 0;
//
//       __syncthreads();
//       for (int k = 0; k < TILE_WIDTH; ++k)
//          Pvalue += ds_M[ty][k] * ds_N[k][tx];
//       __syncthreads();
//    }
//    if (Row < numCRows && Col < numCColumns)
//       C[Row*numCColumns+Col] = Pvalue;
//}
//}

extern "C" {
__global__ void MatMulcuBLASKernel(cublasOperation_t trans, int64_t nelem,
		REAL *A, int *rowsizes, int *colsizes, int *matrixpos,
		int *rowfirstindex, int* colfirstindex, int npts, int nphis, REAL *B,
		REAL *C) {

	int iel = blockIdx.x * blockDim.x + threadIdx.x;

	REAL alpha;
	REAL beta;

	int lda, ldb, ldc;
	int Bpos, Cpos;
	int Boffset, Coffset;
	int m, n, k;
	int Apos;

	if (iel < nelem) {
		cublasHandle_t cnpHandle; //each thread must have its own handle
		cublasCreate(&cnpHandle);

		Apos = matrixpos[iel];

		if (trans == CUBLAS_OP_N) {
			m = rowsizes[iel];
			n = 1;
			k = colsizes[iel];

			alpha = 1.;
			beta = 0;

			lda = m;
			ldb = k;
			ldc = m;

			Bpos = colfirstindex[iel];
			Boffset = nphis;

			Cpos = rowfirstindex[iel];
			Coffset = npts;

		} else if (trans == CUBLAS_OP_T) {
			m = colsizes[iel];
			n = 1;
			k = rowsizes[iel];

			alpha = -1.;
			beta = 0;

			lda = k;
			ldb = k;
			ldc = m;

			Bpos = rowfirstindex[iel];
			Boffset = npts;

			Cpos = colfirstindex[iel];
			Coffset = nphis;
		}
		cublasDgemm(cnpHandle, trans, CUBLAS_OP_N, m, n, k, &alpha, &A[Apos],
				lda, &B[Bpos], ldb, &beta, &C[Cpos], ldc);

//		cublasDgemm(cnpHandle, trans, CUBLAS_OP_N, m, n, k, &alpha, &A[Apos],
//				lda, &B[Bpos + Boffset], ldb, &beta, &C[Cpos + Coffset], ldc);

		__syncthreads();
		cublasDestroy(cnpHandle);

	}
}
}

__global__ void MatMulKernel(bool trans, int64_t nelem, REAL *A, int *rowsizes, int *colsizes, int *matrixpos, int *rowfirstindex, int* colfirstindex, int npts, int nphis, REAL *B, REAL *C) {
	int iel = blockIdx.x;

	__shared__ REAL alpha;
	__shared__ REAL a;

	int Apos, Bpos, Cpos;
	int m, n, k;
	int aux1;
	int aux2;

	__shared__ REAL Bs1[128];


	if (iel < nelem) {
		Apos = matrixpos[iel];

		if (trans == false) {
			m = rowsizes[iel];
			k = colsizes[iel];

			aux1 = rowsizes[iel];
			aux2 = 1;

			alpha = 1.;

			Bpos = colfirstindex[iel];
			Cpos = rowfirstindex[iel];

		} else if (trans == true) {
			m = colsizes[iel];
			k = rowsizes[iel];

			aux1 = 1;
			aux2 = rowsizes[iel];

			alpha = -1.;

			Bpos = rowfirstindex[iel];
			Cpos = colfirstindex[iel];
		}

//		if (threadIdx.x < k) {
//			Bs1[threadIdx.x] = B[threadIdx.x + Bpos];
//			Bs2[threadIdx.x] = B[threadIdx.x + Bpos + Boffset];
//		}
		for(int i = 0; i < k; i++) {
			Bs1[i] = B[i + Bpos];
		}
		__syncthreads();

#pragma unroll
		for (int i = 0; i < m; i++) {
#pragma unroll
			for (int j = 0; j < k; j++) {
				a = A[j * aux1 + i * aux2 + Apos];
				C[i + Cpos] += alpha * A[j * aux1 + i * aux2 + Apos] * Bs1[j];
			}
		}

	}
}

//__global__ void MatMulKernel2(bool trans, int64_t nelem, REAL *A, int *rowsizes, int *colsizes, int *matrixpos, int *rowfirstindex, int* colfirstindex, int npts, int nphis, REAL *B, REAL *C) {
//
//	int iel = blockIdx.x * blockDim.x + threadIdx.x;
//
//	if (iel < nelem) {
//		int Apos = matrixpos[iel];
//		int m = rowsizes[iel];
//		int n = 1;
//		int k = colsizes[iel];
//
//		int Bpos = colfirstindex[iel];
//
//		int Cpos = rowfirstindex[iel];
//
//		dim3 dimGrid(k, m, 1);
//		dim3 dimBlock(1, 1, 1);
//
//	    matrixMultiply<<<dimGrid, dimBlock>>>(&A[Apos], &B[Bpos], &C[Cpos], m, k, k, n, m, n);
//		cudaDeviceSynchronize();
//	}
//}


