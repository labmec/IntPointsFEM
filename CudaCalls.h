#ifndef CUDACALLS_H
#define CUDACALLS_H

#include "TPZVecGPU.h"
#include "pzreal.h"
#include <cublas_v2.h>
#include <cusparse.h>




class CudaCalls {
public:
	CudaCalls();

	~CudaCalls();

	CudaCalls &operator=(const CudaCalls &copy) {
		if(&copy == this){
			return *this;
		}
		handle_cusparse = copy.handle_cusparse;
		handle_cublas = copy.handle_cublas;

    	return *this;
	}

	//PASSAR A ----------REFERENCIA!!!!!!!!!-------
	void Multiply(bool trans, int *m, int *n, int *k, REAL *A, int *strideA, 
		REAL *B, int *strideB,  REAL *C, int *strideC, REAL alpha, int nmatrices);

	void GatherOperation(int n, REAL *x, REAL *y, int *id);

private:
	cusparseHandle_t handle_cusparse;
	cublasHandle_t handle_cublas;

};
#endif