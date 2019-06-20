//
// Created by natalia on 24/05/19.
//

#include "TPZIrregularBlocksMatrix.h"

#ifdef USING_CUDA
#include "TPZVecGPU.h"
#include "TPZCudaCalls.h"
#endif


#ifdef USING_MKL
#include <mkl.h>
#endif

#ifndef INTPOINTSFEM_TPZCOEFTOGRADSOL_H
#define INTPOINTSFEM_TPZCOEFTOGRADSOL_H


class TPZCoefToGradSol {

public:
    
    TPZCoefToGradSol();

    TPZCoefToGradSol(TPZIrregularBlocksMatrix &irregularBlocksMatrix);

    ~TPZCoefToGradSol();

    void SetIrregularBlocksMatrix(TPZIrregularBlocksMatrix & irregularBlocksMatrix);

    void Multiply(TPZFMatrix<REAL> &coef, TPZFMatrix<REAL> &grad_u);

#ifdef USING_CUDA
    void Multiply(TPZVecGPU<REAL> &coef, TPZVecGPU<REAL> &grad_u);
    void MultiplyTranspose(TPZVecGPU<REAL> &sigma, TPZVecGPU<REAL> &res); 
#endif

    void MultiplyTranspose(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &res);

    void SetDoFIndexes(TPZVec<int> dof_indexes) {
        fDoFIndexes = dof_indexes;
    }

    void SetColorIndexes(TPZVec<int> color_indexes) {
        fColorIndexes = color_indexes;
    }
    
    TPZVec<int> & DoFIndexes() {
        return fDoFIndexes;
    }
    
    TPZVec<int> & ColorIndexes() {
       return fColorIndexes;
    }

    void SetNColors(int ncolor) {
        fNColor = ncolor;
    }

    TPZIrregularBlocksMatrix & IrregularBlocksMatrix() {
        return fBlockMatrix;
    }

    void TransferDataToGPU();

private:
    
    /// Irregular block matrix containing spatial gradients for scalar basis functions of order k
    TPZIrregularBlocksMatrix fBlockMatrix;

    /// Number of colors grouping no adjacent elements
    int64_t fNColor; //needed to do the assembly

    /// Degree of Freedom indexes organized element by element with stride ndof
    TPZVec<int> fDoFIndexes; // needed to do the gather operation

    /// Color indexes organized element by element with stride ndof
    TPZVec<int> fColorIndexes; //nedeed to scatter operation

#ifdef USING_CUDA
    TPZVecGPU<int> dIndexes;
    TPZVecGPU<int> dIndexesColor;
    TPZCudaCalls fCudaCalls;
#endif


};


#endif //INTPOINTSFEM_TPZCOEFTOGRADSOL_H
