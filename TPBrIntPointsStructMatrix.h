//
// Created by natalia on 03/09/2019.
//

#ifndef INTPOINTSFEM_TPBRINTPOINTSSTRUCTMATRIX_H
#define INTPOINTSFEM_TPBRINTPOINTSSTRUCTMATRIX_H

#include "TPZSpStructMatrix.h"
#include "TPZIrregularBlocksMatrix.h"
#include "TPBrIrregularBlocksMatrix.h"
#include "TPZNumericalIntegrator.h"
#include "TPBrNumericalIntegrator.h"
#include "tpzverysparsematrix.h"
#include "TPZElastoPlasticMem.h"

template <class T, class MEM = TPZElastoPlasticMem>
class TPBrIntPointsStructMatrix : public TPZSpStructMatrix {

public:

    // Default methods of a struct matrix

    TPBrIntPointsStructMatrix();

    TPBrIntPointsStructMatrix(TPZCompMesh *cmesh);

    ~TPBrIntPointsStructMatrix();

    TPZStructMatrix *Clone();

    TPZMatrix<STATE> *Create();

    TPZMatrix<STATE> *CreateAssemble(TPZFMatrix<STATE> &rhs, TPZAutoPointer<TPZGuiInterface> guiInterface);

    void Assemble(TPZMatrix<STATE> & mat, TPZFMatrix<STATE> & rhs, TPZAutoPointer<TPZGuiInterface> guiInterface);

    void Assemble(TPZFMatrix<STATE> & rhs, TPZAutoPointer<TPZGuiInterface> guiInterface);

private:
    // Methods necessary to use IntPoints

    void SetUpDataStructure();

    void ClassifyMaterialsByDimension();

    void ComputeDomainElementIndexes(TPZVec<int> &element_indexes, int mat_id);

    void AssembleBoundaryData();

    void SetNMaterials(int nmaterials) {
        fNMaterials = nmaterials;
    }

    int NMaterials() {
        return fNMaterials;
    }

    void SetIntegratorVector(TPZVec< TPBrNumericalIntegrator<T,MEM> > & integrator) {
        fIntegrator = integrator;
    }

    TPZVec< TPBrNumericalIntegrator<T,MEM> > & IntegratorVector() {
        return fIntegrator;
    }

    void SetBCMatrix(TPZVerySparseMatrix<STATE> & bcMatrix) {
        fSparseMatrixLinear = bcMatrix;
    }

    TPZVerySparseMatrix<STATE> & BCMatrix() {
        return fSparseMatrixLinear;
    }

    void SetBCRhs (TPZFMatrix<STATE> &BCrhs) {
        fRhsLinear = BCrhs;
    }

    TPZFMatrix<STATE> & BCRhs() {
        return fRhsLinear;
    }

    void SetBCMaterialIds(std::set<int> &matids) {
        fBCMaterialIds = matids;
    }

    std::set<int> & BCMaterialIds() {
        return fBCMaterialIds;
    }

    void SetIA(TPZVec<int64_t>& ia) {
        fIAToSequence = ia;
    }

    TPZVec<int64_t> &IA() {
        return fIAToSequence;
    }

    void SetJA(TPZVec<int64_t>& ja) {
        fJAToSequence = ja;
    }

    TPZVec<int64_t> &JA() {
        return fJAToSequence;
    }

private:

    // Class members

    int fNMaterials;

    TPZVec< TPBrNumericalIntegrator<T,MEM> > fIntegrator;

    TPZVerySparseMatrix<STATE> fSparseMatrixLinear; //-> BC data

    TPZFMatrix<STATE> fRhsLinear; //-> BC data

    std::set<int> fBCMaterialIds;

    TPZVec<int64_t> fIAToSequence;

    TPZVec<int64_t> fJAToSequence;
};


#endif //INTPOINTSFEM_TPBRINTPOINTSSTRUCTMATRIX_H
