#include "TPZCudaCalls.h"
#include "pzreal.h"
#include "pzvec.h"

// #include "MatMulKernels.h"
#include "ComputeSigmaKernel.h"
#include "MatrixAssembleKernel.h"


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

void TPZCudaCalls::Multiply(bool trans, int *m, int *n, int *k, REAL *A, int *strideA, 
	REAL *B, int *strideB,  REAL *C, int *strideC, REAL alpha, int nmatrices) {

	MatrixMultiplicationKernel<<<nmatrices,1>>> (trans, m, n, k, A, strideA, B, strideB, C, strideC, alpha, nmatrices);
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaSuccess) {
		throw std::runtime_error("failed to perform MatrixMultiplicationKernel");      
	}

}

void TPZCudaCalls::GatherOperation(int n, REAL *x, REAL *y, int *id) {
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

void TPZCudaCalls::ScatterOperation(int n, REAL *x, REAL *y, int *id) {
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

void TPZCudaCalls::SpMV(int opt, int m, int k, int nnz, REAL alpha, REAL *csrVal, int *csrRowPtr, int *csrColInd, REAL *B, REAL *C) {
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

	REAL beta = 0.;
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

void TPZCudaCalls::SpMSpM(int opt, int m, int n, int k, int nnzA, REAL *csrValA, int *csrRowPtrA, int *csrColIndA, 
														int nnzB, REAL *csrValB, int *csrRowPtrB, int *csrColIndB, 
														int nnzC, REAL *csrValC, int *csrRowPtrC) {
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

void TPZCudaCalls::ComputeSigma(int npts, REAL *delta_strain, REAL *sigma, REAL lambda, REAL mu, REAL mc_phi, REAL mc_psi, REAL mc_cohesion, REAL *plastic_strain,  REAL *m_type, REAL *alpha, REAL *weight){
	int numBlocks = (npts + NT - 1) / NT;
	ComputeSigmaKernel<<<numBlocks,NT>>> (npts, delta_strain, sigma, lambda, mu, mc_phi, mc_psi, mc_cohesion, plastic_strain, m_type, alpha, weight);
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaSuccess) {
		throw std::runtime_error("failed to perform ComputeSigmaKernel");      
	}
}

void TPZCudaCalls::MatrixAssemble(REAL *Kg, int first_el, int last_el, int64_t *el_color_index, REAL *weight, int *dof_indexes,
						REAL *storage, int *rowsizes, int *colsizes, int *rowfirstindex, int *colfirstindex, int *matrixposition, int64_t *ia_to_sequence, int64_t *ja_to_sequence) {
	int nel = last_el - first_el;
	int numBlocks = (nel + NT - 1) / NT;
	std::cout << nel << std::endl;
	MatrixAssembleKernel<<<nel,1>>> (nel, Kg, first_el, el_color_index, weight, dof_indexes, storage, rowsizes, colsizes, rowfirstindex, colfirstindex, matrixposition, ia_to_sequence, ja_to_sequence);
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaSuccess) {
		throw std::runtime_error("failed to perform MatrixAssembleKernel");      
	}


}