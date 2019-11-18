//
// Created by natalia on 24/05/19.
//

#include "TPZIrregularBlocksMatrix.h"
#include "TPZConstitutiveLawProcessor.h"

#ifdef USING_CUDA
#include "TPZVecGPU.h"
#include "TPZCudaCalls.h"
#endif


#ifdef USING_MKL
#include <mkl.h>
#endif

#ifndef INTPOINTSFEM_TPZNUMERICALINTEGRATOR_H
#define INTPOINTSFEM_TPZNUMERICALINTEGRATOR_H


class TPZNumericalIntegrator {

public:

    TPZNumericalIntegrator();

    ~TPZNumericalIntegrator();

    void Multiply(TPZFMatrix<REAL> &coef, TPZFMatrix<REAL> &delta_strain);

    void MultiplyTranspose(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &res);

    void ResidualIntegration(TPZFMatrix<REAL> & solution ,TPZFMatrix<REAL> &rhs);

    void ComputeTangentMatrix(int64_t iel, TPZFMatrix<REAL> &Dep, TPZFMatrix<REAL> &K);

    void SetUpIrregularBlocksData(TPZCompMesh * cmesh);

    int StressRateVectorSize(int dim);

    void SetUpIndexes(TPZCompMesh * cmesh);

    void SetUpColoredIndexes(TPZCompMesh * cmesh);

    void FillLIndexes(TPZVec<int64_t> & IA, TPZVec<int64_t> & JA);

    int64_t me(TPZVec<int64_t> &IA, TPZVec<int64_t> &JA, int64_t & i_dest, int64_t & j_dest);

    bool isBuilt() {
        if(fBlockMatrix.Rows() != 0) return true;
        else return false;
    }

    void KAssembly(TPZFMatrix<REAL> & solution, TPZVec<STATE> & Kg, TPZFMatrix<STATE> & rhs);

    void SetElementIndexes(TPZVec<int> element_indexes) {
        fElementIndex = element_indexes;
    }

    TPZIrregularBlocksMatrix &IrregularBlockMatrix() {
        return fBlockMatrix;
    }

#ifdef USING_CUDA
    void Multiply(TPZVecGPU<REAL> &coef, TPZVecGPU<REAL> &delta_strain);
    
    void MultiplyTranspose(TPZVecGPU<REAL> &sigma, TPZVecGPU<REAL> &res); 

    void ResidualIntegration(TPZFMatrix<REAL> & solution ,TPZVecGPU<REAL> &rhs);

    void KAssembly(TPZFMatrix<REAL> & solution, TPZVecGPU<STATE> & Kg, TPZVecGPU<STATE> & rhs);

    void TransferDataToGPU();
#endif

private:

    TPZVec<int> fElementIndex;

    /// Irregular block matrix containing spatial gradients for scalar basis functions of order k
    TPZIrregularBlocksMatrix fBlockMatrix;

    /// Number of colors grouping no adjacent elements
    int64_t fNColor;

    /// Degree of Freedom indexes organized element by element with stride ndof
    TPZVec<int> fDoFIndexes;

    /// Color indexes organized element by element with stride ndof
    TPZVec<int> fColorIndexes;

    TPZConstitutiveLawProcessor fConstitutiveLawProcessor;

    TPZVec<int> fElColorIndex;

    TPZVec<int64_t> fFirstColorIndex;

    TPZVec<int> fColorLSequence;

    TPZVec<int> fFirstColorLIndex;
    
#ifdef USING_CUDA
    TPZCudaCalls fCudaCalls;

    TPZVecGPU<int> dDoFIndexes;

    TPZVecGPU<int> dColorIndexes;

    TPZVecGPU<int> dElColorIndex;

    TPZVecGPU<int> dColorLSequence;
#endif
};


#endif //INTPOINTSFEM_TPZNUMERICALINTEGRATOR_H
