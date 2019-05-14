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
    void SetBlocksInfo();

    /** @brief Sets blocks information for CSR format*/
    void CSRInfo();

private:
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
};


#endif //INTPOINTSFEM_TPZIRREGULARBLOCKMATRIX_H
