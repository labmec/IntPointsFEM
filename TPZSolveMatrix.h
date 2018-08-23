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

//#ifdef USING_MKL
//#include "mkl.h"
//#endif

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
    TPZSolveMatrix(int64_t rows, int64_t cols, const TPZManVector<TPZFMatrix<REAL>> &ElementMatrices, const TPZVec<MKL_INT> &indexes)
    {
        SetParameters(rows, cols, ElementMatrices, indexes);
    }

    /** @brief Default destructor */
    ~TPZSolveMatrix();

    TPZSolveMatrix(const TPZSolveMatrix &copy) : TPZMatrix<STATE>(copy),
        fElementMatrices(copy.fElementMatrices), fIndexes(copy.fIndexes),
        fMatrixPosition(copy.fMatrixPosition),fColSize(copy.fColSize), fRowSize(copy.fRowSize),
        fColFirstIndex(copy.fColFirstIndex), fRowFirstIndex(copy.fRowFirstIndex)
    {

    }

    TPZSolveMatrix &operator=(const TPZSolveMatrix &copy)
    {
        TPZMatrix::operator=(copy);
        fElementMatrices = copy.fElementMatrices;
        fIndexes = copy.fIndexes;
        fMatrixPosition = copy.fMatrixPosition;
        fColSize = copy.fColSize;
        fRowSize = copy.fRowSize;
        fColFirstIndex = copy.fColFirstIndex;
        fRowFirstIndex = copy.fRowFirstIndex;
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
    void SetParameters(int64_t rows, int64_t cols, const TPZVec<TPZFMatrix<REAL>> &ElementMatrices, const TPZVec<MKL_INT> &indexes)
    {
        TPZMatrix<STATE>::Resize(rows, cols);
        fRow = rows;
        fCol = cols;
        fElementMatrices = ElementMatrices;
        fIndexes = indexes;
        ComputeElementFirstIndex();
    };

    /** @brief Solve procedure */
    void Multiply(const TPZFMatrix<STATE>  &global_solution, TPZFMatrix<STATE> &result, int transpose = 0) const;

    void ComputeSigma(TPZStack<REAL> &weight, TPZFMatrix<REAL> &result, TPZFMatrix<REAL> &sigma);

    void MultiplyTranspose(const TPZFMatrix<STATE>  &intpoint_solution, TPZFMatrix<STATE> &nodal_forces_vec) const;

    void TraditionalAssemble(TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) const;

    void ColoredAssemble(TPZCompMesh * cmesh, TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) const;

private:
    /** @brief Order the coef vectors

     * @param A [in] is the input matrix
     * @param coef [in] is the coeffient's matrix
     * @param ind_x [in] is the inxedes vector associated to the solution x
     * @param ind_y [in] is the inxedes vector associated to the solution y
     */
    void OrderGlobalSolution (TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &global_solution_x, TPZFMatrix<STATE> &global_solution_y);

    /// compute the first index of each element
    void ComputeElementFirstIndex();

    /** @brief Multiply with the transpose matrix */




protected:

//    /** @brief number of rows */
//    int64_t fRows;
//
//    /** @brief number of columns */
//    int64_t fCols;

    /** @brief Matrix */
    TPZManVector<TPZFMatrix<REAL>> fElementMatrices;

    /** @brief Indexes vector in x direction and the y direction */
    TPZManVector<MKL_INT> fIndexes;

    /// position of the matrix of the elements
    TPZVec<int64_t> fMatrixPosition;

    /// number of columns of each block matrix
    // the size of the vector is the number of blocks
    TPZVec<MKL_INT> fColSize;

    /// number of rows of each block matrix
    // the size of the vector is the number of blocks
    TPZVec<MKL_INT> fRowSize;

    /// position in the fIndex vector of each element
    TPZVec<MKL_INT> fColFirstIndex;

    /// position of the result vector
    TPZVec<MKL_INT> fRowFirstIndex;

};

#endif /* TPZSolveMatrix_h */
