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

    ~TPBrNumericalIntegrator();

    TPBrNumericalIntegrator(const TPBrNumericalIntegrator &copy);

    TPBrNumericalIntegrator &operator=(const TPBrNumericalIntegrator &copy);

    bool isBuilt();

    void Multiply(TPZFMatrix<REAL> &coef, TPZFMatrix<REAL> &delta_strain);

    void MultiplyTranspose(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &res);

    void ResidualIntegration(TPZFMatrix<REAL> & solution ,TPZFMatrix<REAL> &rhs);

    void KAssembly(TPZFMatrix<REAL> &solution, TPZFMatrix<REAL> &rhs, TPZVec<STATE> &Kg, TPZVec<int64_t> &IAToSequence, TPZVec<int64_t> &JAToSequence);

    int64_t me(TPZVec<int64_t> &IA, TPZVec<int64_t> &JA, int64_t & i_dest, int64_t & j_dest);

    void ComputeConstitutiveMatrix(TPZFMatrix<REAL> &De);

    void ComputeTangentMatrix(TPZFMatrix<REAL> &glob_dep, int64_t iel, TPZFMatrix<REAL> &K);

    int StressRateVectorSize(int dim);

    void SetUpIrregularBlocksMatrix(TPZCompMesh * cmesh);

    void SetUpIndexes(TPZCompMesh * cmesh);

    void ColoredIndexes(TPZCompMesh * cmesh);

    void SetElementIndexes(TPZVec<int> &element_vec) {
        fElementIndexes = element_vec;
    }

    TPZVec<int> & ElementIndexes() {
        return fElementIndexes;
    }

    void SetIrregularBlocksMatrix(TPBrIrregularBlocksMatrix & blocksMatrix) {
        fBlockMatrix = blocksMatrix;
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

    void SetMaterialRegionElColorIndexes(TPZVec<int64_t> & elcolorindexes) {
        fMaterialRegionElColorIndexes = elcolorindexes;
    }

    TPZVec<int64_t> & MaterialRegionElColorIndexes() {
        return fMaterialRegionElColorIndexes;
    }

    void SetMaterialRegionFirstColorIndex(TPZVec<int64_t> & firstcolorindexes) {
        fMaterialRegionFirstColorIndex = firstcolorindexes;
    }

    TPZVec<int64_t> & MaterialRegionFirstColorIndex() {
        return fMaterialRegionFirstColorIndex;
    }

private:
    /// Element indexes
    TPZVec<int> fElementIndexes;

    /// Irregular block matrix containing spatial gradients for scalar basis functions of order k
    TPBrIrregularBlocksMatrix fBlockMatrix;

    /// Material type
    TPZMatElastoPlastic2D < T, MEM > * fMaterial;

    /// Constitutive law processor
    TPBrConstitutiveLawProcessor<T, MEM> fConstitutiveLawProcessor;

    /// Degree of Freedom indexes organized element by element with stride ndof
    TPZVec<int> fDoFIndexes;

    /// Color indexes organized element by element with stride ndof
    TPZVec<int> fColorIndexes;

    /// Number of colors grouping no adjacent elements
    int64_t fNColor;

    TPZVec<int64_t> fMaterialRegionElColorIndexes;

    TPZVec<int64_t> fMaterialRegionFirstColorIndex;
};


#endif /* TPBrNumericalIntegrator_h */
