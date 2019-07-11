#ifndef TPZCudaCalls_H
#define TPZCudaCalls_H

#include "TPZVecGPU.h"
#include "pzreal.h"

#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverSp.h>


class TPZCudaCalls {
public:
	TPZCudaCalls();

	~TPZCudaCalls();

	TPZCudaCalls &operator=(const TPZCudaCalls &copy) {
		if(&copy == this){
			return *this;
		}
		handle_cusparse = copy.handle_cusparse;
		cusparse_h = copy.cusparse_h;
		handle_cublas = copy.handle_cublas;
		cublas_h = copy.cublas_h;

		return *this;
	}

	void Multiply(bool trans, int *m, int *n, int *k, REAL *A, int *strideA, REAL *B, int *strideB,  REAL *C, int *strideC, REAL alpha, int nmatrices);

	void GatherOperation(int n, REAL *x, REAL *y, int *id);

	void ScatterOperation(int n, REAL *x, REAL *y, int *id);

	void DaxpyOperation(int n, REAL alpha, REAL *x, REAL *y); 

	void SpMSpM(int opt, int sym, int m, int n, int k, int nnzA, REAL *csrValA, int *csrRowPtrA, int *csrColIndA, int nnzB, REAL *csrValB, int *csrRowPtrB, int *csrColIndB, int nnzC, REAL *csrValC, int *csrRowPtrC);

	void SpMV(int opt, int sym, int m, int k, int nnz, REAL alpha, REAL *csrVal, int *csrRowPtr, int *csrColInd, REAL *B, REAL *C) ; 

	void ComputeSigma(bool update_mem, int npts, REAL *glob_delta_strain, REAL *glob_sigma, REAL lambda, REAL mu, REAL mc_phi, REAL mc_psi, REAL mc_cohesion, REAL *dPlasticStrain,  
		REAL *dMType, REAL *dAlpha, REAL *dSigma, REAL *dStrain, REAL *weight);

	void MatrixAssemble(int nnz, REAL *K, int first_el, int last_el, int64_t *el_color_index, REAL *weight, int *dof_indexes,
		REAL *storage, int *rowsizes, int *colsizes, int *rowfirstindex, int *colfirstindex, int *matrixposition, int *ia_to_sequence, int *ja_to_sequence);

	void MatrixAssemble(REAL *K, int nnz, REAL *Kg, int first_el, int last_el, int64_t *el_color_index, int *dof_indexes,
	int *colsizes, int *colfirstindex, int *ia_to_sequence, int *ja_to_sequence);

	void SolveCG(int n, int nnzA, REAL *csrValA, int *csrRowPtrA, int *csrColIndA, REAL *b, REAL *x);

	void SetHeapSize() {
		if(heap_q == false) {
			heap_q = true;
			size_t size = 120000000;
			cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);
		}
	}

private:
	cusparseHandle_t handle_cusparse;
	bool cusparse_h;
	
	cublasHandle_t handle_cublas;
	bool cublas_h;

	bool heap_q;

};
#endif