#include "CudaCalls.h"
#include "pzreal.h"
#include "pzvec.h"

#ifdef USING_CUDA
#include "SpectralDecompKernels.h"
#include "StressStrainKernels.h"
#include "SigmaProjectionKernels.h"
#include "MatMulKernels.h"
#endif

	CudaCalls::CudaCalls() {

	}

	CudaCalls::~CudaCalls() {
		if(handle_cublas) {
			cublasDestroy(handle_cublas);
		}
		if(handle_cusparse) {
			cusparseDestroy(handle_cusparse);			
		}
	}

	void CudaCalls::Multiply(bool trans, int *m, int *n, int *k, REAL *A, int *strideA, 
		REAL *B, int *strideB,  REAL *C, int *strideC, REAL alpha, int nmatrices) {

		matrixMultiplication<<<nmatrices,1>>> (trans, m, n, k, A, strideA, B, strideB, C, strideC, alpha, nmatrices);
		if (cudaGetLastError() != cudaSuccess) {
			throw std::runtime_error("failed to perform Multiply kernel");      
		}

	}

	void CudaCalls::GatherOperation(int n, REAL *x, REAL *y, int *id) {
		if(!handle_cusparse) {
			cusparseStatus_t result = cusparseCreate(&handle_cusparse);
			if (result != CUSPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("failed to initialize cuSparse");      
       		}			
		}
		cusparseStatus_t result = cusparseDgthr(handle_cusparse, n, x, y, id, CUSPARSE_INDEX_BASE_ZERO);
		if (result != CUSPARSE_STATUS_SUCCESS) {
			throw std::runtime_error("failed to perform GatherOperation");      
		}	
	}