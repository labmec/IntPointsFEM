//
// Created by natalia on 24/05/19.
//

#include "TPZIrregularBlocksMatrix.h"

#ifdef USING_CUDA
#include "TPZVecGPU.h"
#include "CudaCalls.h"
#endif


#ifdef USING_MKL
#include <mkl.h>
#endif

#ifndef INTPOINTSFEM_TPZCOEFTOGRADSOL_H
#define INTPOINTSFEM_TPZCOEFTOGRADSOL_H


class TPZCoefToGradSol {

public:
    TPZCoefToGradSol();

    TPZCoefToGradSol(TPZIrregularBlocksMatrix irregularBlocksMatrix);

    ~TPZCoefToGradSol();

    void SetIrregularBlocksMatrix(TPZIrregularBlocksMatrix & irregularBlocksMatrix);

    void Multiply(TPZFMatrix<REAL> &coef, TPZFMatrix<REAL> &grad_u);

    void MultiplyTranspose(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &res);

    void SetIndexes(TPZVec<int> indexes) {
        fIndexes = indexes;
    }

    void SetIndexesColor(TPZVec<int> indexescolor) {
        fIndexesColor = indexescolor;
    }

    void SetNColors(int ncolor) {
        fNColor = ncolor;
    }

    TPZIrregularBlocksMatrix & IrregularBlocksMatrix() {
        return fBlockMatrix;
    }

    void TransferDataToGPU();


private:
    TPZIrregularBlocksMatrix fBlockMatrix;

    int64_t fNColor; //needed to do the assembly

    TPZVec<int> fIndexes; //needed to do the gather operation

    TPZVec<int> fIndexesColor; //nedeed to scatter operation

    TPZVecGPU<REAL> dStorage;
    TPZVecGPU<int> dRowSizes;
    TPZVecGPU<int> dColSizes;
    TPZVecGPU<int> dRowFirstIndex;
    TPZVecGPU<int> dColFirstIndex;
    TPZVecGPU<int> dMatrixPosition;
    TPZVecGPU<int> dIndexes;
    TPZVecGPU<int> dIndexesColor;
    CudaCalls fCudaCalls;


};


#endif //INTPOINTSFEM_TPZCOEFTOGRADSOL_H
