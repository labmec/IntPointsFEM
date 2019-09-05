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

    bool isBuilt();

    void ClassifyMaterialsByDimension();

    void ComputeDomainElementIndexes(TPZVec<int> &element_indexes);

    void AssembleBoundaryData();

    void SetUpIrregularBlocksData(TPZVec<int> &element_indexes, TPBrIrregularBlocksMatrix::IrregularBlocks &blocksData);

    void SetUpIndexes(TPZVec<int> &element_indexes, TPZVec<int> & dof_indexes);

    void ColoredIndexes(TPZVec<int> &element_indexes, TPZVec<int> &indexes, TPZVec<int> &coloredindexes, int &ncolor);

    int StressRateVectorSize();

    int64_t me(TPZVec<int64_t> &IA, TPZVec<int64_t> &JA, int64_t & i_dest, int64_t & j_dest);

private:

    // Class members

    int fDimension;

    TPBrNumericalIntegrator<T,MEM> fIntegrator;

    TPZVerySparseMatrix<STATE> fSparseMatrixLinear; //-> BC data

    TPZFMatrix<STATE> fRhsLinear; //-> BC data

    std::set<int> fBCMaterialIds;

    TPZVec<int64_t> fIAToSequence;

    TPZVec<int64_t> fJAToSequence;

    TPZVec<int64_t> fElColorIndexes;

    TPZVec<int64_t> fFirstColorIndex;
};


#endif //INTPOINTSFEM_TPBRINTPOINTSSTRUCTMATRIX_H
