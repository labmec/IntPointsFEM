//
//  TPBrNumericalIntegrator.hpp
//  IntPointsFEM
//
//  Created by Omar Durán on 9/5/19.
//

#ifndef TPBrNumericalIntegrator_h
#define TPBrNumericalIntegrator_h

#include "TPZElastoPlasticMem.h"
#include "TPBrIrregularBlocksMatrix.h"
#include "TPBrConstitutiveLawProcessor.h"

template <class T, class MEM = TPZElastoPlasticMem>
class TPBrNumericalIntegrator {

public:

    TPBrNumericalIntegrator();

    TPBrNumericalIntegrator(TPBrIrregularBlocksMatrix &irregularBlocksMatrix);

    ~TPBrNumericalIntegrator();

    void Multiply(TPZFMatrix<REAL> &coef, TPZFMatrix<REAL> &delta_strain);

    void MultiplyTranspose(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &res);

    void ResidualIntegration(TPZFMatrix<REAL> & solution ,TPZFMatrix<REAL> &rhs);

    void ComputeConstitutiveMatrix(TPZFMatrix<REAL> &De);

    void ComputeTangentMatrix(int64_t iel, TPZFMatrix<REAL> &K);

    void SetIrregularBlocksMatrix(TPBrIrregularBlocksMatrix & irregularBlocksMatrix) {
        fBlockMatrix = irregularBlocksMatrix;
    }

    TPBrIrregularBlocksMatrix & IrregularBlocksMatrix() {
        return fBlockMatrix;
    }

    void SetMaterial(TPZMatElastoPlastic2D < T, MEM > * material){
        fMaterial = material;
    }
    
    TPZMatElastoPlastic2D < T, MEM > * Material(){
        return fMaterial;
    }
    
    void SetConstitutiveLawProcessor(TPBrConstitutiveLawProcessor<T, MEM> & processor){
        fConstitutiveLawProcessor = processor;
    }

    TPBrConstitutiveLawProcessor<T, MEM> & ConstitutiveLawProcessor(){
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
   ;
    
    void KAssembly(TPZVec<STATE> &Kg, TPZVec<int64_t> & IAToSequence, TPZVec<int64_t> & JAToSequence);
    
    int64_t me(TPZVec<int64_t> &IA, TPZVec<int64_t> &JA, int64_t & i_dest, int64_t & j_dest);

public:
    
    TPZVec<int64_t> fElColorIndexes;
    
    TPZVec<int64_t> fFirstColorIndex;
    
private:

    /// Irregular block matrix containing spatial gradients for scalar basis functions of order k
    TPBrIrregularBlocksMatrix fBlockMatrix;

    /// Number of colors grouping no adjacent elements
    int64_t fNColor; //needed to do the assembly

    /// Degree of Freedom indexes organized element by element with stride ndof
    TPZVec<int> fDoFIndexes; // needed to do the gather operation

    /// Color indexes organized element by element with stride ndof
    TPZVec<int> fColorIndexes; //nedeed to scatter operation
    
//    TPZVec<int64_t> fElColorIndexes;
//    
//    TPZVec<int64_t> fFirstColorIndex;

    TPBrConstitutiveLawProcessor<T, MEM> fConstitutiveLawProcessor;
    
    TPZMatElastoPlastic2D < T, MEM > * fMaterial;
};


#endif /* TPBrNumericalIntegrator_h */
