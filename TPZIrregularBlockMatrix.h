//
// Created by natalia on 14/05/19.
//
#include "pzcmesh.h"

#ifndef INTPOINTSFEM_TPZIRREGULARBLOCKMATRIX_H
#define INTPOINTSFEM_TPZIRREGULARBLOCKMATRIX_H


class TPZIrregularBlockMatrix {
public:
    /** @brief Default constructor */
    TPZIrregularBlockMatrix();

    /** @brief Creates the object based on cmesh
     * @param cmesh : computational mesh
     * */
    TPZIrregularBlockMatrix(TPZCompMesh *cmesh);

    /** @brief Default destructor */
    ~TPZIrregularBlockMatrix();

    /** @brief Creates a irregular block matrix with copy constructor
     * @param copy : original irregular block matrix
     */
    TPZIrregularBlockMatrix(const TPZIrregularBlockMatrix &copy);

    /** @brief operator= */
    TPZIrregularBlockMatrix &operator=(const TPZIrregularBlockMatrix &copy);

    /** @brief Sets the computational mesh
     * @param cmesh : computational mesh
     */
    void SetCompMesh(TPZCompMesh *cmesh);

    /** @brief Sets a matrix to the irregular block matrix
     * @param iel : matrix index
     * @param elmat : matrix to be set
     */
    void SetElementMatrix(int iel, TPZFMatrix<REAL> &elmat);

    /** @brief Sets blocks information */
    void BlocksInfo();

    /** @brief Sets blocks information for CSR format*/
    void CSRInfo();

    /** @brief Multiply the irrgular block matrix by a matrix A
     * @param A : matrix that will be multipied
     * @param res : result of the multiplication
     * @param opt : indicates if transpose or not
     */
    void Multiply(TPZFMatrix<REAL> &A, TPZFMatrix<REAL> &res, REAL alpha, REAL beta, bool transpose = false);

    int Dimension() {
        return fDim;
    }

    TPZCompMesh *CompMesh() {
        return fCmesh;
    }

    int64_t NumBlocks() {
        return fNumBlocks;
    }

    TPZVec<REAL> Storage() {
        return fStorage;
    }

    TPZVec<int> RowSizes() {
        return fRowSizes;
    }

    TPZVec<int> ColSizes() {
        return fColSizes;
    }

    TPZVec<int> MatrixPosition() {
        return fMatrixPosition;
    }

    TPZVec<int> RowFirstIndex() {
        return fRowFirstIndex;
    }

    TPZVec<int> ColFirstIndex() {
        return fColFirstIndex;
    }

    int Rows() {
        return fRow;
    }

    int Cols() {
        return fCol;
    }

    TPZVec<int> RowPtr() {
        return fRowPtr;
    }

    TPZVec<int> ColInd() {
        return fColInd;
    }

    TPZStack<int64_t> ElemIndexes() {
        return fElemIndex;
    }

    TPZStack<int64_t> BoundaryElemIndexes() {
        return fBoundaryElemIndex;
    }

private:
    /** @brief Mesh Dimension */
    int fDim;

    /** @brief Computational mesh */
    TPZCompMesh *fCmesh;

    /** @brief Number of blocks */
    int64_t fNumBlocks;

    /** @brief Vector with all matrices values */
    TPZVec<REAL> fStorage;

    /** @brief Vector with number of rows of each matrix */
    TPZVec<int> fRowSizes;

    /** @brief Vector with number of columns of each matrix */
    TPZVec<int> fColSizes;

    /** @brief Vector with matrix position in fStorage */
    TPZVec<int> fMatrixPosition;

    /** @brief Vector with first row index of each matrix */
    TPZVec<int> fRowFirstIndex;

    /** @brief Vector with first column index of each matrix */
    TPZVec<int> fColFirstIndex;

    /** @brief Number of rows of the irregular block matrix */
    int64_t fRow;

    /** @brief Number of columns of the irregular block matrix */
    int64_t fCol;

    /** @brief Vector with the start of every row and the end of the last row plus one (this is for CSR format) */
    TPZVec<int> fRowPtr;

    /** @brief Vector with column indices for each non-zero element of the matrix (this is for CSR format)*/
    TPZVec<int> fColInd;

    /** @brief Vector with the indexes of the domain elements*/
    TPZStack<int64_t> fElemIndex;

    /** @brief Vector with the indexes of the boundary elements*/
    TPZStack<int64_t> fBoundaryElemIndex;
};


#endif //INTPOINTSFEM_TPZIRREGULARBLOCKMATRIX_H