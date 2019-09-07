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

    void SetPlasticModel();

    void AssembleBoundaryData();

    int StressRateVectorSize();


private:

    // Class members

    int fDimension;
    
    int fNMaterials;

    TPZVec< TPBrNumericalIntegrator<T,MEM> > fIntegrator;

    TPZVerySparseMatrix<STATE> fSparseMatrixLinear; //-> BC data

    TPZFMatrix<STATE> fRhsLinear; //-> BC data

    std::set<int> fBCMaterialIds;

    TPZVec<int64_t> fIAToSequence;

    TPZVec<int64_t> fJAToSequence;

//    TPZVec<int64_t> fElColorIndexes;

//    TPZVec<int64_t> fFirstColorIndex;
};


#endif //INTPOINTSFEM_TPBRINTPOINTSSTRUCTMATRIX_H
