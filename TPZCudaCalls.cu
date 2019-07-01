#include "TPZCudaCalls.h"
// #include "pzreal.h"
#include "pzvec.h"

#include "SpectralDecompKernels.h"
#include "StressStrainKernels.h"
#include "SigmaProjectionKernels.h"
#include "MatMulKernels.h"

#define NT 512

TPZCudaCalls::TPZCudaCalls() {
	cusparse_h = false;
	cublas_h = false;
}

TPZCudaCalls::~TPZCudaCalls() {
	if(cublas_h == true) {
		cublasDestroy(handle_cublas);
	}
	if(cusparse_h == true) {
		cusparseDestroy(handle_cusparse);			
	}
}

void TPZCudaCalls::Multiply(bool trans, int *m, int *n, int *k, double *A, int *strideA, 
	double *B, int *strideB,  double *C, int *strideC, double alpha, int nmatrices) {

	MatrixMultiplicationKernel<<<nmatrices,1>>> (trans, m, n, k, A, strideA, B, strideB, C, strideC, alpha, nmatrices);
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaSuccess) {
		throw std::runtime_error("failed to perform MatrixMultiplicationKernel");      
	}

}

void TPZCudaCalls::GatherOperation(int n, double *x, double *y, int *id) {
	if(cusparse_h == false) {
		cusparse_h = true;
		cusparseStatus_t result = cusparseCreate(&handle_cusparse);
		if (result != CUSPARSE_STATUS_SUCCESS) {
			throw std::runtime_error("failed to initialize cuSparse");      
		}			
	}
	cusparseStatus_t result = cusparseDgthr(handle_cusparse, n, x, y, id, CUSPARSE_INDEX_BASE_ZERO);
	if (result != CUSPARSE_STATUS_SUCCESS) {
		throw std::runtime_error("failed to perform cusparseDgthr");      
	}	
}

void TPZCudaCalls::ScatterOperation(int n, double *x, double *y, int *id) {
	if(cusparse_h == false) {
		cusparse_h = true;
		cusparseStatus_t result = cusparseCreate(&handle_cusparse);
		if (result != CUSPARSE_STATUS_SUCCESS) {
			throw std::runtime_error("failed to initialize cuSparse");      
		}			
	}
	cusparseStatus_t result = cusparseDsctr(handle_cusparse, n, x, id, y, CUSPARSE_INDEX_BASE_ZERO);
	if (result != CUSPARSE_STATUS_SUCCESS) {
		throw std::runtime_error("failed to perform cusparseDsctr");      
	}	
}

void TPZCudaCalls::DaxpyOperation(int n, double alpha, double *x, double *y) {
	if(cublas_h == false) {
		cublas_h = true;
		cublasStatus_t result = cublasCreate(&handle_cublas);
		if (result != CUBLAS_STATUS_SUCCESS) {
			throw std::runtime_error("failed to initialize cuBLAS");      
		}			
	}
	cublasStatus_t result = cublasDaxpy(handle_cublas, n, &alpha, x, 1., y, 1.);
	if (result != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("failed to perform cublasDaxpy");      
	}	
}

void TPZCudaCalls::SpMV(int opt, int m, int k, int nnz, double alpha, double *csrVal, int *csrRowPtr, int *csrColInd, double *B, double *C) {
	if(cusparse_h == false) {
		cusparse_h = true;
		cusparseStatus_t result = cusparseCreate(&handle_cusparse);
		if (result != CUSPARSE_STATUS_SUCCESS) {
			throw std::runtime_error("failed to initialize cuSparse");      
		}			
	}
	cusparseMatDescr_t descr;
	cusparseCreateMatDescr(&descr);
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

	double beta = 0.;
	cusparseStatus_t result;
	if(opt == 0) {
		result = cusparseDcsrmv(handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, m, k, nnz, &alpha, descr, csrVal, csrRowPtr, csrColInd, B, &beta, C);
	} else {
		result = cusparseDcsrmv(handle_cusparse, CUSPARSE_OPERATION_TRANSPOSE, m, k, nnz, &alpha, descr, csrVal, csrRowPtr, csrColInd, B, &beta, C);
	}
	if (result != CUSPARSE_STATUS_SUCCESS) {
		throw std::runtime_error("failed to perform cusparseDcsrmv");      
	}	
}

void TPZCudaCalls::SpMSpM(int opt, int m, int n, int k, int nnzA, double *csrValA, int *csrRowPtrA, int *csrColIndA, 
														int nnzB, double *csrValB, int *csrRowPtrB, int *csrColIndB, 
														int nnzC, double *csrValC, int *csrRowPtrC) {
	if(cusparse_h == false) {
		cusparse_h = true;
		cusparseStatus_t result = cusparseCreate(&handle_cusparse);
		if (result != CUSPARSE_STATUS_SUCCESS) {
			throw std::runtime_error("failed to initialize cuSparse");      
		}			
	}

	cusparseMatDescr_t descr;
	cusparseCreateMatDescr(&descr);
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

	cusparseOperation_t trans;
	if(opt == 0) {
		trans = CUSPARSE_OPERATION_NON_TRANSPOSE;

	} else {
		trans = CUSPARSE_OPERATION_TRANSPOSE;
	}

	int *csrColIndC;
	cudaMalloc((void**)&csrColIndC, sizeof(int)*nnzC);

	cusparseStatus_t result = cusparseDcsrgemm(handle_cusparse, trans, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, 
					descr, nnzA, csrValA, csrRowPtrA, csrColIndA, 
					descr, nnzB, csrValB, csrRowPtrB, csrColIndB,
					descr, csrValC, csrRowPtrC, csrColIndC);
	if (result != CUSPARSE_STATUS_SUCCESS) {
		throw std::runtime_error("failed to perform cusparseDcsrgemm");      
	}	
}