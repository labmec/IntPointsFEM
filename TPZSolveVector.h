//
// Created by natalia on 18/10/18.
//

#ifndef INTEGRATIONPOINTSEXPERIMENT_TPZSOLVEVECTOR_H
#define INTEGRATIONPOINTSEXPERIMENT_TPZSOLVEVECTOR_H


#include "pzmatrix.h"
#include "pzfmatrix.h"
#include <mkl.h>
#include "pzinterpolationspace.h"
#include "pzcmesh.h"
#ifdef USING_MKL
#include "mkl.h"
#endif
#ifdef __CUDACC__
#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include "mkl.h"
#endif

class TPZSolveVector : public TPZMatrix<STATE> {

public:

    TPZSolveVector() : TPZMatrix<STATE>(), fIndexes(), fColSizes(), fRowSizes(), fMatrixPosition(), fStorageVec() {

    }

    TPZSolveVector(int64_t rows, int64_t cols, TPZVec<int64_t> rowsizes, TPZVec<int64_t> colsizes) : TPZMatrix(rows,
                                                                                                               cols) {
        SetParameters(rowsizes, colsizes);
        cuSparseHandle();
        cuBlasHandle();
    }


    ~TPZSolveVector() {

    }

    TPZSolveVector(const TPZSolveVector &copy) : TPZMatrix<STATE>(copy),
                                                 fStorageVec(copy.fStorageVec), fIndexes(copy.fIndexes),
                                                 fColSizes(copy.fColSizes), fRowSizes(copy.fRowSizes),
                                                 fMatrixPosition(copy.fMatrixPosition),
                                                 fRowFirstIndex(copy.fRowFirstIndex),
                                                 fElemColor(copy.fElemColor), fIndexesColor(copy.fIndexesColor),
                                                 fColFirstIndex(copy.fColFirstIndex), dglobal_solution(copy.dglobal_solution),
                                                 dindexes(copy.dindexes), dstoragevec(copy.dstoragevec), dexpandsolution(copy.dexpandsolution),
                                                 dresult(copy.dresult), dweight(copy.dweight), dsigma(copy.dsigma), dnodal_forces_vec(copy.dnodal_forces_vec),
                                                 dindexescolor(copy.dindexescolor), dnodal_forces_global(copy.dnodal_forces_global) {

    }

    TPZSolveVector &operator=(const TPZSolveVector &copy) {
        TPZMatrix::operator=(copy);
        fStorageVec = copy.fStorageVec;
        fIndexes = copy.fIndexes;
        fColSizes = copy.fColSizes;
        fRowSizes = copy.fRowSizes;
        fElemColor = copy.fElemColor;
        fIndexesColor = copy.fIndexesColor;
        fMatrixPosition = copy.fMatrixPosition;
        fRowFirstIndex = copy.fRowFirstIndex;
        fColFirstIndex = copy.fColFirstIndex;
        dglobal_solution = copy.dglobal_solution;
        dindexes = copy.dindexes;
        dstoragevec = copy.dstoragevec;
        dexpandsolution = copy.dexpandsolution;
        dresult = copy.dresult;
        dweight = copy.dweight;
        dsigma = copy.dsigma;
        dnodal_forces_vec = copy.dnodal_forces_vec;
        dindexescolor = copy.dindexescolor;
        dnodal_forces_global = copy.dnodal_forces_global;

        return *this;
    }

    TPZMatrix<STATE> *Clone() const {
        return new TPZSolveVector(*this);
    }

    void SetParameters(TPZVec<int64_t> rowsize, TPZVec<int64_t> colsize) {
        int64_t nelem = rowsize.size();

        fRowSizes.resize(nelem);
        fColSizes.resize(nelem);
        fMatrixPosition.resize(nelem + 1);
        fRowFirstIndex.resize(nelem + 1);
        fColFirstIndex.resize(nelem + 1);

        fRowSizes = rowsize;
        fColSizes = colsize;

        fMatrixPosition[0] = 0;
        fRowFirstIndex[0] = 0;
        fColFirstIndex[0] = 0;

        for (int64_t iel = 0; iel < nelem; ++iel) {
            fMatrixPosition[iel + 1] = fMatrixPosition[iel] + fRowSizes[iel] * fColSizes[iel];
            fRowFirstIndex[iel + 1] = fRowFirstIndex[iel] + fRowSizes[iel];
            fColFirstIndex[iel + 1] = fColFirstIndex[iel] + fColSizes[iel];
        }
        fStorageVec.resize(fMatrixPosition[nelem]);
        fElemColor.resize(nelem);
        fElemColor.Fill(-1);
    }

    void SetElementMatrix(int iel, TPZFMatrix<REAL> &elmat) {
        int64_t rows = fRowSizes[iel];
        int64_t cols = fColSizes[iel];
        int64_t pos = fMatrixPosition[iel];
        int64_t nelem = fRowSizes.size();

        int cont = 0;
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < cols; j++) {
                int k = (j + i) % cols;
                int id1 = iel + j*nelem + cont;
                int id2 = iel + j*nelem + cont + nelem*cols;
                fStorageVec[id1] =  elmat(j, k); //primeira metade da matriz
                fStorageVec[id2] =  elmat(j + rows/2, k); //segunda metade da matriz
            }
            cont += 2*cols*nelem;
        }
    }

    void SetIndexes(TPZVec<MKL_INT> indexes) {
        int64_t indsize = indexes.size();
        fIndexes.resize(indsize);
        fIndexes = indexes;
        fIndexesColor.resize(indsize);
    }

    /** @brief Solve procedure */

    //USING CUDA
    void AllocateMemory(TPZCompMesh *cmesh);

    void FreeMemory();

    void cuSparseHandle();

    void cuBlasHandle();


    void Multiply(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const;

    void MultiplyCUDA(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const;


    void TraditionalAssemble(TPZFMatrix<STATE> &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) const;

    void ColoringElements(TPZCompMesh *cmesh) const;

protected:

/// vector containing the matrix coefficients
    TPZVec<REAL> fStorageVec;

/// number of rows of each block matrix
    TPZVec<int64_t> fRowSizes;

/// number of columns of each block matrix
    TPZVec<int64_t> fColSizes;

/// indexes vector in x and y direction
    TPZVec<MKL_INT> fIndexes;

/// indexes vector in x and y direction by color
    TPZVec<MKL_INT> fIndexesColor;

/// color indexes of each element
    TPZVec<int64_t> fElemColor;

/// position of the matrix of the elements
    TPZVec<int64_t> fMatrixPosition;

/// position of the result vector
    TPZVec<int64_t> fRowFirstIndex;

/// position in the fIndex vector of each element
    TPZVec<int64_t> fColFirstIndex;

/// Parameters stored on device
    double *dglobal_solution;
    int *dindexes;
    double *dstoragevec;
    double *dexpandsolution;
    double *dresult;
    double *dweight;
    double *dsigma;
    double *dnodal_forces_vec;
    int *dindexescolor;
    double *dnodal_forces_global;

//Libraries handles
#ifdef __CUDACC__
    cusparseHandle_t handle_cusparse;
    cublasHandle_t handle_cublas;
#endif

};


#endif //INTEGRATIONPOINTSEXPERIMENT_TPZSOLVEVECTOR_H
