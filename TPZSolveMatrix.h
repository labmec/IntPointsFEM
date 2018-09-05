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

#ifdef USING_MKL
#include "mkl.h"
#endif

class  TPZSolveMatrix : public TPZMatrix<STATE> {

public:

    /** @brief Default constructor */
    TPZSolveMatrix();

    /** @brief Set solve parameters
     * @param A [in] is the input matrix
     * @param coef [in] is the coeffient's matrix
     * @param ind_x [in] is the inxedes vector associated to the solution x
     * @param ind_y [in] is the inxedes vector associated to the solution y
     */
    TPZSolveMatrix(int64_t rows, int64_t cols, TPZVec<MKL_INT> rowsizes, TPZVec<MKL_INT> colsizes) : TPZMatrix(rows,cols)
    {
        SetParameters(rowsizes, colsizes);
    }

    /** @brief Default destructor */
    ~TPZSolveMatrix();

    TPZSolveMatrix(const TPZSolveMatrix &copy) : TPZMatrix<STATE>(copy),
        fStorage(copy.fStorage),fIndexes(copy.fIndexes),
        fColSizes(copy.fColSizes), fRowSizes(copy.fRowSizes),
        fMatrixPosition(copy.fMatrixPosition),fRowFirstIndex(copy.fRowFirstIndex),fColFirstIndex(copy.fColFirstIndex)
    {

    }

    TPZSolveMatrix &operator=(const TPZSolveMatrix &copy)
    {
        TPZMatrix::operator=(copy);
        fStorage = copy.fStorage;
        fIndexes = copy.fIndexes;
        fColSizes = copy.fColSizes;
        fRowSizes = copy.fRowSizes;
        fMatrixPosition = copy.fMatrixPosition;
        fRowFirstIndex = copy.fRowFirstIndex;
        fColFirstIndex = copy.fColFirstIndex;

        return *this;
    }

    TPZMatrix<STATE> *Clone() const
    {
        return new TPZSolveMatrix(*this);
    }
    /** @brief Set solve parameters
     * @param A [in] is the input matrix
     * @param coef [in] is the coeffient's matrix
     * @param ind_x [in] is the inxedes vector associated to the solution x
     * @param ind_y [in] is the inxedes vector associated to the solution y
     */
    void SetParameters(TPZVec<MKL_INT> rowsize, TPZVec<MKL_INT> colsize)
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
    };

    /// return the element matrix
    void SetElementMatrix(int iel, TPZFMatrix<REAL> &elmat){
        TPZFMatrix<REAL> elmatloc(fRowSizes[iel],fColSizes[iel],&fStorage[fMatrixPosition[iel]],fRowSizes[iel]*fColSizes[iel]);
        elmatloc = elmat;
    }

    /// set indexes vector
    void SetIndexes(TPZVec<MKL_INT> indexes){
        int64_t indsize = indexes.size();
        fIndexes.resize(indsize);
        fIndexes = indexes;
    }

    /** @brief Solve procedure */

    void Multiply(const TPZFMatrix<STATE>  &global_solution, TPZFMatrix<STATE> &result, int transpose = 0) const;

    void ComputeSigma(TPZStack<REAL> &weight, TPZFMatrix<REAL> &result, TPZFMatrix<REAL> &sigma);

    void MultiplyTranspose(TPZFMatrix<STATE>  &intpoint_solution, TPZFMatrix<STATE> &nodal_forces_vec);

    void TraditionalAssemble(TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) const;

    void ColoredAssemble(TPZCompMesh * cmesh, TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) const;

protected:

    /// vector containing the matrix coeficients
    TPZVec<REAL> fStorage;

    /// number of rows of each block matrix
    TPZVec<MKL_INT> fRowSizes;

    /// number of columns of each block matrix
    TPZVec<MKL_INT> fColSizes;

    /// indexes vector in x direction and the y direction
    TPZManVector<MKL_INT> fIndexes;

    /// position of the matrix of the elements
    TPZVec<int64_t> fMatrixPosition;

    /// position of the result vector
    TPZVec<MKL_INT> fRowFirstIndex;

    /// position in the fIndex vector of each element
    TPZVec<MKL_INT> fColFirstIndex;


};

#endif /* TPZSolveMatrix_h */
