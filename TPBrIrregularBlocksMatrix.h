//
//  TPBrIrregularBlocksMatrix.h
//  IntPointsFEM
//
//  Created by Omar Durán on 9/5/19.
//

#ifndef TPBrIrregularBlocksMatrix_h
#define TPBrIrregularBlocksMatrix_h

#include "pzmatrix.h"

class TPBrIrregularBlocksMatrix : public TPZMatrix<REAL> {
public:

    /** @brief Irregular blocks information */
    struct IrregularBlocks {
        int64_t fNumBlocks; //number of blocks
        TPZVec<REAL> fStorage; // blocks values
        TPZVec<int> fRowSizes; // blocks row sizes
        TPZVec<int> fColSizes; // blocks columns sizes
        TPZVec<int> fMatrixPosition; // blocks start position in fStorage vector
        TPZVec<int> fRowFirstIndex; // blocks first row index
        TPZVec<int> fColFirstIndex; // blocks first column index
    };

    TPBrIrregularBlocksMatrix();

    TPBrIrregularBlocksMatrix(const int64_t rows, const int64_t cols);

    ~TPBrIrregularBlocksMatrix();

    virtual TPZMatrix<REAL> * Clone() const;

    TPBrIrregularBlocksMatrix(const TPBrIrregularBlocksMatrix &copy);

    TPBrIrregularBlocksMatrix &operator=(const TPBrIrregularBlocksMatrix &copy);

    void MultiplyVector(TPZFMatrix <REAL> &A, REAL *res, int opt);

    void BlocksMultiplication(bool trans, TPZVec<int> &m, TPZVec<int> &k, TPZVec<REAL> &A, TPZVec<int> &strideA, TPZFMatrix<REAL> &B, TPZVec<int> &strideB, REAL *C, TPZVec<int> &strideC, REAL alpha, int nmatrices);

    void SetBlocks(struct IrregularBlocks & blocks) {
        fBlocksInfo = blocks;
    }

    struct IrregularBlocks & Blocks() {
        return fBlocksInfo;
    }

private:
    struct IrregularBlocks fBlocksInfo;
};

#endif /* TPBrIrregularBlocksMatrix_h */
