#include "CudaCalls.h"
#include "pzreal.h"
#include "pzvec.h"

#ifdef USING_CUDA
#include "SpectralDecompKernels.h"
#include "StressStrainKernels.h"
#include "SigmaProjectionKernels.h"
#include "MatMulKernels.h"
#endif

#define NT 512

CudaCalls::CudaCalls() {
	cusparse_h = false;
	cublas_h = false;
}

CudaCalls::~CudaCalls() {
	if(cublas_h == true) {
		cublasDestroy(handle_cublas);
	}
	if(cusparse_h == true) {
		cusparseDestroy(handle_cusparse);			
	}
}

void CudaCalls::Multiply(bool trans, int *m, int *n, int *k, REAL *A, int *strideA, 
	REAL *B, int *strideB,  REAL *C, int *strideC, REAL alpha, int nmatrices) {

	MatrixMultiplicationKernel<<<nmatrices,1>>> (trans, m, n, k, A, strideA, B, strideB, C, strideC, alpha, nmatrices);
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaSuccess) {
		throw std::runtime_error("failed to perform MatrixMultiplicationKernel kernel");      
	}

}

void CudaCalls::GatherOperation(int n, REAL *x, REAL *y, int *id) {
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

void CudaCalls::ScatterOperation(int n, REAL *x, REAL *y, int *id) {
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

void CudaCalls::DaxpyOperation(int n, REAL alpha, REAL *x, REAL *y) {
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







void CudaCalls::ElasticStrain(REAL *delta_strain, REAL *elastic_strain, int64_t n) {
	cudaMemcpy(elastic_strain, &delta_strain[0], n * sizeof(REAL), cudaMemcpyDeviceToDevice);
	REAL *plastic_strain;
	cudaMalloc(&plastic_strain, n * sizeof(REAL));
	cudaMemset(plastic_strain, 0, n * sizeof(REAL));

	REAL a = -1.;
	if(cublas_h == false) {
		cublas_h = true;
		cublasStatus_t result = cublasCreate(&handle_cublas);
		if (result != CUBLAS_STATUS_SUCCESS) {
			throw std::runtime_error("failed to initialize cuBLAS");      
		}			
	}
	cublasStatus_t result = cublasDaxpy(handle_cublas, n, &a, &plastic_strain[0], 1, &elastic_strain[0], 1);
	if (result != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("failed to perform cublasDaxpy");      
	}
	cudaFree(plastic_strain);
}

void CudaCalls::ComputeStress(REAL *elastic_strain, REAL *sigma, int64_t n, REAL mu, REAL lambda) {
	int numBlocks = (n + NT - 1) / NT;

	ComputeStressKernel<<<numBlocks, NT>>>(elastic_strain, sigma, n, mu, lambda);
	cudaDeviceSynchronize();
	if (cudaGetLastError() != cudaSuccess) {
		throw std::runtime_error("failed to perform ComputeStressKernel");      
	}
}

void CudaCalls::SpectralDecomposition(REAL *sigma_trial, REAL *eigenvalues, REAL *eigenvectors, int64_t n) {
	int numBlocks = (n + NT - 1) / NT;

	SpectralDecompositionKernel<<<numBlocks, NT>>>(sigma_trial, eigenvalues, eigenvectors, n);
	cudaDeviceSynchronize();
		// if (cudaGetLastError() != cudaSuccess) {
		// 	throw std::runtime_error("failed to perform SpectralDecompositionKernel");      
		// }
}

void CudaCalls::ProjectSigma(REAL *eigenvalues, REAL *sigma_projected, int64_t n, REAL mc_phi, REAL mc_psi, REAL mc_cohesion, REAL K, REAL G) {
	REAL *m_type;
	cudaMalloc((void**) &m_type, n * sizeof(REAL));

	REAL *alpha;
	cudaMalloc((void**) &alpha, n * sizeof(REAL));

	int numBlocks = (n + NT - 1) / NT;

	ProjectSigmaKernel<<<numBlocks, NT>>>(eigenvalues, sigma_projected, m_type, alpha, n, mc_phi, mc_psi, mc_cohesion, K, G);	
	cudaDeviceSynchronize();
	// if (cudaGetLastError() != cudaSuccess) {
	// 	throw std::runtime_error("failed to perform ProjectSigmaKernel");      
	// }

}