#include "TPZVecGPU.h"
#include "pzreal.h"
#include <cublas_v2.h>
#include <cusparse.h>



class CudaCalls {
public:
	CudaCalls();

	~CudaCalls();

	//PASSAR A ----------REFERENCIA!!!!!!!!!-------
	void Multiply(bool trans, int *m, int *n, int *k, REAL *A, int *strideA, 
		REAL *B, int *strideB,  REAL *C, int *strideC, REAL alpha, int nmatrices);

	void GatherOperation(int n, TPZVecGPU<REAL> &x, TPZVecGPU<REAL> &y, TPZVecGPU<int> &id);

private:
	cusparseHandle_t handle_cusparse;
	cublasHandle_t handle_cublas;

};