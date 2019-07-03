#ifndef TPZCudaCalls_H
#define TPZCudaCalls_H

#include "TPZVecGPU.h"
#include "pzreal.h"

#include <cublas_v2.h>
#include <cusparse.h>


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

	void SpMSpM(int opt, int m, int n, int k, int nnzA, REAL *csrValA, int *csrRowPtrA, int *csrColIndA, int nnzB, REAL *csrValB, int *csrRowPtrB, int *csrColIndB, int nnzC, REAL *csrValC, int *csrRowPtrC);

	void SpMV(int opt, int m, int k, int nnz, REAL alpha, REAL *csrVal, int *csrRowPtr, int *csrColInd, REAL *B, REAL *C); 

	void ComputeSigma(int npts, REAL *delta_strain, REAL *sigma, REAL lambda, REAL mu, REAL mc_phi, REAL mc_psi, REAL mc_cohesion, REAL *plastic_strain,  REAL *m_type, REAL *alpha, REAL *weight);

	void MatrixAssemble(REAL *K, int first_el, int last_el, int64_t *el_color_index, REAL *weight, int *dof_indexes,
						REAL *storage, int *rowsizes, int *colsizes, int *rowfirstindex, int *colfirstindex, int *matrixposition, int64_t *ia_to_sequence, int64_t *ja_to_sequence);

private:
	cusparseHandle_t handle_cusparse;
	bool cusparse_h;
	
	cublasHandle_t handle_cublas;
	bool cublas_h;

};
#endif