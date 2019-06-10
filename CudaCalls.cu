#include "CudaCalls.h"
#include "pzreal.h"
#include "pzvec.h"

__global__ void matrixMultiplication (bool trans, int *m, int *n, int *k, REAL *A, int *strideA, REAL *B, int *strideB, REAL *C, int *strideC, REAL alpha, int nmatrices) {

	int imatrix = blockIdx.x;

	if (imatrix < nmatrices) {
		int m_i = m[imatrix];
		int n_i = n[imatrix];
		int k_i = k[imatrix];
		int strideA_i = strideA[imatrix]; 
		int strideB_i = strideB[imatrix]; 
		int strideC_i = strideC[imatrix]; 

		int aux1, aux2;

		if (trans == false) {
			aux1 = m_i;
			aux2 = 1;

		} else {
			aux1 = 1;
			aux2 = m_i;
		}


		for (int i = 0; i < m_i; i++) {
			for (int j = 0; j < n_i; j++) {
				C[j * m_i + i] = 0;
				for (int l = 0; l < k_i; l++) {
					C[j * m_i + i + strideC_i] += alpha * A[l * aux1 + i * aux2 + strideA_i] * B[j * k_i + l + strideB_i];
				}
			}
		}
	}
}

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

	void CudaCalls::Multiply(bool trans, TPZVecGPU<int> m, TPZVecGPU<int> n, TPZVecGPU<int> k, TPZVecGPU<REAL> A, TPZVecGPU<int> strideA, 
		TPZVecGPU<REAL> B, TPZVecGPU<int> strideB,  TPZVecGPU<REAL> C, TPZVecGPU<int> strideC, REAL alpha, int nmatrices) {

		matrixMultiplication<<<nmatrices,1>>> (trans, m.getData(), n.getData(), k.getData(), A.getData(), strideA.getData(), 
		B.getData(), strideB.getData(), C.getData(), strideC.getData(), alpha, nmatrices);

	}

	void CudaCalls::GatherOperation(int n, TPZVecGPU<REAL> &x, TPZVecGPU<REAL> &y, TPZVecGPU<int> &id) {
		if(!handle_cusparse) {
			cusparseStatus_t result = cusparseCreate(&handle_cusparse);
			if (result != CUSPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("failed to initialize cuSparse");      
       		}			
		}
		cusparseStatus_t result = cusparseDgthr(handle_cusparse, n, x.getData(), &y.getData()[0], &id.getData()[0], CUSPARSE_INDEX_BASE_ZERO);
		if (result != CUSPARSE_STATUS_SUCCESS) {
			throw std::runtime_error("failed to perform GatherOperation");      
		}	

	}