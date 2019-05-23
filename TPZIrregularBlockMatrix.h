//
// Created by natalia on 14/05/19.
//
#include "pzcmesh.h"
#include "TPZPlasticStepPV.h"
#include "TPZYCMohrCoulombPV.h"
#include "TPZElastoPlasticMem.h"
#include "TPZMatElastoPlastic2D.h"

#ifndef INTPOINTSFEM_TPZIRREGULARBLOCKMATRIX_H
#define INTPOINTSFEM_TPZIRREGULARBLOCKMATRIX_H
#include "pzmatrix.h"

class TPZIrregularBlockMatrix : public TPZMatrix<REAL> {
public:
    struct IrregularBlocks {
        int64_t fNumBlocks;
        TPZVec<REAL> fStorage;
        TPZVec<int> fRowSizes;
        TPZVec<int> fColSizes;
        TPZVec<int> fMatrixPosition;
        TPZVec<int> fRowFirstIndex;
        TPZVec<int> fColFirstIndex;
        TPZVec<int> fRowPtr;
        TPZVec<int> fColInd;
    };

    /** @brief Default constructor */
    TPZIrregularBlockMatrix();

    TPZIrregularBlockMatrix(const int64_t rows,const int64_t cols);

    /** @brief Default destructor */
    ~TPZIrregularBlockMatrix();

    virtual TPZMatrix<REAL> * Clone() const {
        return new TPZIrregularBlockMatrix(*this);
    }
    /** @brief Creates a irregular block matrix with copy constructor
     * @param copy : original irregular block matrix
     */
    TPZIrregularBlockMatrix(const TPZIrregularBlockMatrix &copy);

    /** @brief operator= */
    TPZIrregularBlockMatrix &operator=(const TPZIrregularBlockMatrix &copy);

    /** @brief Performs the following operation: res = alpha * BMatrix * A + beta * res
     * @param A : matrix that will be multipied
     * @param res : result of the multiplication
     * @param alpha : scalar parameter
     * @param beta : scalar parameter
     * @param opt : indicates if transpose or not
     */
    void Multiply(REAL *A, REAL *res, int opt);

    /** @brief Access methods */
    void SetBlocks(struct IrregularBlocks blocks) {
        fBlocksInfo = blocks;
    }

    struct IrregularBlocks Blocks() {
        return fBlocksInfo;
    }

protected:
    struct IrregularBlocks fBlocksInfo;
};


#endif //INTPOINTSFEM_TPZIRREGULARBLOCKMATRIX_H