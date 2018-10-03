/**
 * @file
 * @brief Contains the TPZSolveMatrix class which implements a solution based on a matrix procedure.
 */

#ifndef TPZSolveMatrix_h
#define TPZSolveMatrix_h

#include "pzmatrix.h"
#include "pzfmatrix.h"
#include <mkl.h>
#include "pzinterpolationspace.h"
#include "pzcmesh.h"

#ifdef USING_MKL
#include "mkl.h"
#endif

class  TPZSolveMatrix : public TPZMatrix<STATE> {

public:

TPZSolveMatrix() : TPZMatrix<STATE>(), fStorage(), fIndexes(), fColSizes(), fRowSizes(), fMatrixPosition()
{

}

TPZSolveMatrix(int64_t rows, int64_t cols, TPZVec<int64_t> rowsizes, TPZVec<int64_t> colsizes) : TPZMatrix(rows,cols)
{
    SetParameters(rowsizes, colsizes);
}


~TPZSolveMatrix()
{

}

TPZSolveMatrix(const TPZSolveMatrix &copy) : TPZMatrix<STATE>(copy),
    fStorage(copy.fStorage),fIndexes(copy.fIndexes),
    fColSizes(copy.fColSizes), fRowSizes(copy.fRowSizes),
    fMatrixPosition(copy.fMatrixPosition),fRowFirstIndex(copy.fRowFirstIndex),
    fElemColor(copy.fElemColor), fIndexesColor(copy.fIndexesColor),
    fColFirstIndex(copy.fColFirstIndex), dfStorage(copy.dfStorage),
    dfColSizes(copy.dfColSizes), dfRowSizes(copy.dfRowSizes), dfIndexes(copy.dfIndexes),
    dfMatrixPosition(copy.dfMatrixPosition),dfRowFirstIndex(copy.dfRowFirstIndex),
    dfColFirstIndex(copy.dfColFirstIndex)

{

}

TPZSolveMatrix &operator=(const TPZSolveMatrix &copy)
{
    TPZMatrix::operator=(copy);
    fStorage = copy.fStorage;
    fIndexes = copy.fIndexes;
    fColSizes = copy.fColSizes;
    fRowSizes = copy.fRowSizes;
    fElemColor = copy.fElemColor;
    fIndexesColor = copy.fIndexesColor;
    fMatrixPosition = copy.fMatrixPosition;
    fRowFirstIndex = copy.fRowFirstIndex;
    fColFirstIndex = copy.fColFirstIndex;
    dfStorage = copy.dfStorage;
    dfColSizes = copy.dfColSizes;
    dfRowSizes = copy.dfRowSizes;
    dfIndexes = copy.dfIndexes;
    dfMatrixPosition = copy.dfMatrixPosition;
    dfRowFirstIndex = copy.dfRowFirstIndex;
    dfColFirstIndex = copy.dfColFirstIndex;

    return *this;
}

TPZMatrix<STATE> *Clone() const
{
    return new TPZSolveMatrix(*this);
}

void SetParameters(TPZVec<int64_t> rowsize, TPZVec<int64_t> colsize)
{
int64_t nelem = rowsize.size();

fRowSizes.resize(nelem);
fColSizes.resize(nelem);
fMatrixPosition.resize(nelem+1);
fRowFirstIndex.resize(nelem+1);
fColFirstIndex.resize(nelem+1);

fRowSizes = rowsize;
fColSizes = colsize;

fMatrixPosition[0] = 0;
fRowFirstIndex[0] = 0;
fColFirstIndex[0] = 0;

for (int64_t iel = 0; iel < nelem; ++iel) {
    fMatrixPosition[iel+1] = fMatrixPosition[iel] + fRowSizes[iel]*fColSizes[iel];
    fRowFirstIndex[iel+1]= fRowFirstIndex[iel]+fRowSizes[iel];
    fColFirstIndex[iel+1]= fColFirstIndex[iel]+fColSizes[iel];
}
fStorage.resize(fMatrixPosition[nelem]);
fElemColor.resize(nelem);
fElemColor.Fill(-1);
}

void SetElementMatrix(int iel, TPZFMatrix<REAL> &elmat)
{
TPZFMatrix<REAL> elmatloc(fRowSizes[iel],fColSizes[iel],&fStorage[fMatrixPosition[iel]],fRowSizes[iel]*fColSizes[iel]);
elmatloc = elmat;
}

void SetIndexes(TPZVec<MKL_INT> indexes)
{
int64_t indsize = indexes.size();
fIndexes.resize(indsize);
fIndexes = indexes;
fIndexesColor.resize(indexes.size());
}

    /** @brief Solve procedure */

void SolveWithCUDA(TPZCompMesh *cmesh, const TPZFMatrix<STATE>  &global_solution, TPZStack<REAL> &weight, TPZFMatrix<REAL> &nodal_forces_global) const;

void Multiply(const TPZFMatrix<STATE>  &global_solution, TPZFMatrix<STATE> &result) const;

void ComputeSigma(TPZStack<REAL> &weight, TPZFMatrix<REAL> &result, TPZFMatrix<REAL> &sigma);

void MultiplyTranspose(TPZFMatrix<STATE>  &intpoint_solution, TPZFMatrix<STATE> &nodal_forces_vec);

void TraditionalAssemble(TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) const;

void ColoringElements(TPZCompMesh * cmesh) const;

void ColoredAssemble(TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global);

protected:

/// vector containing the matrix coefficients
TPZVec<REAL> fStorage;

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
double *dfStorage;
int *dfRowSizes;
int *dfColSizes;
int *dfIndexes;
int *dfMatrixPosition;
int *dfRowFirstIndex;
int *dfColFirstIndex;


};

#endif /* TPZSolveMatrix_h */
