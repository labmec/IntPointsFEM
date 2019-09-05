//
// Created by natalia on 03/09/2019.
//

#ifndef INTPOINTSFEM_TPZINTPOINTSSTRUCTMATRIX_H
#define INTPOINTSFEM_TPZINTPOINTSSTRUCTMATRIX_H

#include "TPZSpStructMatrix.h"
#include "TPZIrregularBlocksMatrix.h"
#include "TPZNumericalIntegrator.h"
#include "tpzverysparsematrix.h"

template <class PlasticModel_t>
class TPZIntPointsStructMatrix : public TPZSpStructMatrix {

public:

    // Default methods of a struct matrix

    TPZIntPointsStructMatrix();

    TPZIntPointsStructMatrix(TPZCompMesh *cmesh);

    ~TPZIntPointsStructMatrix();

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

    void SetUpIrregularBlocksData(TPZVec<int> &element_indexes, TPZIrregularBlocksMatrix::IrregularBlocks &blocksData);

    void SetUpIndexes(TPZVec<int> &element_indexes, TPZVec<int> & dof_indexes);

    void ColoredIndexes(TPZVec<int> &element_indexes, TPZVec<int> &indexes, TPZVec<int> &coloredindexes, int &ncolor);

    int StressRateVectorSize();

    int64_t me(TPZVec<int64_t> &IA, TPZVec<int64_t> &JA, int64_t & i_dest, int64_t & j_dest);

private:

    // Class members

    int fDimension;

    TPZNumericalIntegrator fIntegrator;

    TPZVerySparseMatrix<STATE> fSparseMatrixLinear; //-> BC data

    TPZFMatrix<STATE> fRhsLinear; //-> BC data

    std::set<int> fBCMaterialIds;

    TPZVec<int64_t> fIAToSequence;

    TPZVec<int64_t> fJAToSequence;

    TPZVec<int64_t> fElColorIndexes;

    TPZVec<int64_t> fFirstColorIndex;

    PlasticModel_t fPlasticModel;
};


#endif //INTPOINTSFEM_TPZINTPOINTSSTRUCTMATRIX_H
