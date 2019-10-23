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

    TPZNumericalIntegrator(TPZIrregularBlocksMatrix &irregularBlocksMatrix);

    ~TPZNumericalIntegrator();

    void Multiply(TPZFMatrix<REAL> &coef, TPZFMatrix<REAL> &delta_strain);

    void MultiplyTranspose(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &res);

    void ResidualIntegration(TPZFMatrix<REAL> & solution ,TPZFMatrix<REAL> &rhs);

    void ComputeConstitutiveMatrix(int64_t point_index, TPZFMatrix<STATE> &De);

    void ComputeTangentMatrix(int64_t iel, TPZFMatrix<REAL> &Dep, TPZFMatrix<REAL> &K);

    void ComputeTangentMatrix(int64_t iel, TPZFMatrix<REAL> &K);

    void SetUpIrregularBlocksData(TPZCompMesh * cmesh);

    int StressRateVectorSize(int dim);

    void SetUpIndexes(TPZCompMesh * cmesh);

    void SetUpColoredIndexes(TPZCompMesh * cmesh);

    void FillLIndexes(TPZVec<int64_t> & IA, TPZVec<int64_t> & JA);

    int64_t me(TPZVec<int64_t> &IA, TPZVec<int64_t> &JA, int64_t & i_dest, int64_t & j_dest);

#ifdef USING_CUDA
    void Multiply(TPZVecGPU<REAL> &coef, TPZVecGPU<REAL> &delta_strain);
    
    void MultiplyTranspose(TPZVecGPU<REAL> &sigma, TPZVecGPU<REAL> &res); 

    void ResidualIntegration(TPZFMatrix<REAL> & solution ,TPZVecGPU<REAL> &rhs);
#endif

    void SetElementIndexes(TPZVec<int> & elemindex) {
        fElementIndex = elemindex;
    }

    TPZVec<int> & ElementIndexes() {
        return fElementIndex;
    }

    void SetIrregularBlocksMatrix(TPZIrregularBlocksMatrix & irregularBlocksMatrix) {
        fBlockMatrix = irregularBlocksMatrix;
    }

    TPZIrregularBlocksMatrix & IrregularBlocksMatrix() {
        return fBlockMatrix;
    }

    void SetConstitutiveLawProcessor(TPZConstitutiveLawProcessor & processor){
        fConstitutiveLawProcessor = processor;
    }

    TPZConstitutiveLawProcessor & ConstitutiveLawProcessor(){
        return fConstitutiveLawProcessor;
    }

    void SetDoFIndexes(TPZVec<int> dof_indexes) {
        fDoFIndexes = dof_indexes;
    }

    TPZVec<int> & DoFIndexes() {
        return fDoFIndexes;
    }

    void SetColorIndexes(TPZVec<int> color_indexes) {
        fColorIndexes = color_indexes;
    }
    
    TPZVec<int> & ColorIndexes() {
     return fColorIndexes;
    }

    void SetNColors(int ncolor) {
        fNColor = ncolor;
    }

    int NColors() {
        return fNColor;
    }

    void SetElColorIndexes(TPZVec<int64_t> &el_color_indexes){
        m_el_color_indexes = el_color_indexes;
    }

    TPZVec<int64_t> &ElColorIndexes() {
        return m_el_color_indexes;
    }

    void SetFirstColorIndex(TPZVec<int64_t> &first_color_index){
        m_first_color_index = first_color_index;
    }

    TPZVec<int64_t> &FirstColorIndex() {
        return m_first_color_index;
    }

    void SetColorLSequence(TPZVec<int> &color_l_sequence){
        m_color_l_sequence = color_l_sequence;
    }

    TPZVec<int> &ColorLSequence() {
        return m_color_l_sequence;
    }

    void SetFirstColorLIndex(TPZVec<int> &first_color_l_index){
        m_first_color_l_index = first_color_l_index;
    }

    TPZVec<int> &FirstColorLIndex() {
        return m_first_color_l_index;
    }

#ifdef USING_CUDA
    TPZVecGPU<int> & DoFIndexesDev() {
        return dDoFIndexes;
    }

    TPZVecGPU<int> & ColorIndexesDev() {
        return dColorIndexes;
    }

    void TransferDataToGPU();

#endif

private:

    TPZVec<int> fElementIndex;

    /// Irregular block matrix containing spatial gradients for scalar basis functions of order k
    TPZIrregularBlocksMatrix fBlockMatrix;

    /// Number of colors grouping no adjacent elements
    int64_t fNColor; //needed to do the assembly

    /// Degree of Freedom indexes organized element by element with stride ndof
    TPZVec<int> fDoFIndexes; // needed to do the gather operation

    /// Color indexes organized element by element with stride ndof
    TPZVec<int> fColorIndexes; //nedeed to scatter operation

    TPZConstitutiveLawProcessor fConstitutiveLawProcessor;

    TPZVec<int64_t> m_el_color_indexes;

    TPZVec<int64_t> m_first_color_index;

    TPZVec<int> m_color_l_sequence;

    TPZVec<int> m_first_color_l_index;
    
#ifdef USING_CUDA
    TPZVecGPU<int> dDoFIndexes;
    TPZVecGPU<int> dColorIndexes;
    TPZCudaCalls fCudaCalls;
#endif


};


#endif //INTPOINTSFEM_TPZNUMERICALINTEGRATOR_H
