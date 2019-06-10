#include "TPZVecGPU.h"
#include "pzreal.h"
#include <cublas_v2.h>
#include <cusparse.h>



class CudaCalls {
public:
	CudaCalls();

	~CudaCalls();

	void Multiply(bool trans, TPZVecGPU<int> m, TPZVecGPU<int> n, TPZVecGPU<int> k, TPZVecGPU<REAL> A, TPZVecGPU<int> strideA, 
		TPZVecGPU<REAL> B, TPZVecGPU<int> strideB,  TPZVecGPU<REAL> C, TPZVecGPU<int> strideC, REAL alpha, int nmatrices);

	void GatherOperation(int n, TPZVecGPU<REAL> x, TPZVecGPU<REAL> y, TPZVecGPU<int> id);

private:
	cusparseHandle_t handle_cusparse;
	cublasHandle_t handle_cublas;

};