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
		cusparse_h = copy.cusparse_h;
		handle_cublas = copy.handle_cublas;
		cublas_h = copy.cublas_h;

		return *this;
	}

	void Multiply(bool trans, int *m, int *n, int *k, REAL *A, int *strideA, 
		REAL *B, int *strideB,  REAL *C, int *strideC, REAL alpha, int nmatrices);

	void GatherOperation(int n, REAL *x, REAL *y, int *id);

	void ScatterOperation(int n, REAL *x, REAL *y, int *id);

	void DaxpyOperation(int n, REAL alpha, REAL *x, REAL *y); 

	void ElasticStrain(REAL *delta_strain, REAL *elastic_strain, int64_t n);

	void ComputeStress(REAL *elastic_strain, REAL *sigma, int64_t n, REAL mu, REAL lambda);

	void SpectralDecomposition(REAL *sigma_trial, REAL *eigenvalues, REAL *eigenvectors, int64_t n);

	void ProjectSigma(REAL *eigenvalues, REAL *sigma_projected, int64_t npts, REAL mc_phi, REAL mc_psi, REAL mc_cohesion, REAL K, REAL G);

private:
	cusparseHandle_t handle_cusparse;
	bool cusparse_h;
	
	cublasHandle_t handle_cublas;
	bool cublas_h;

};
#endif