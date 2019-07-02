/**
 * @file
 * @brief Contains the TPZSolveMatrix class which implements a solution based on a matrix procedure.
 */

#ifndef TPZIntPointsFEM_h
#define TPZIntPointsFEM_h

//#include <StrMatrix/TPZSSpStructMatrix.h>
#include <TPZSSpStructMatrix.h>
#include "pzsysmp.h"
#include "tpzverysparsematrix.h"
#include "TPZIrregularBlocksMatrix.h"
#include "TPZNumericalIntegrator.h"
#include <unordered_map>


class TPZElastoPlasticIntPointsStructMatrix : public TPZSymetricSpStructMatrix {
public:
    /** @brief Default constructor */
    TPZElastoPlasticIntPointsStructMatrix();

    /** @brief Creates the object based on a Compmesh
     * @param Compmesh : Computational mesh */
    TPZElastoPlasticIntPointsStructMatrix(TPZCompMesh *cmesh);

    /** @brief Default destructor */
    ~TPZElastoPlasticIntPointsStructMatrix();

    /** @brief Clone */
    TPZStructMatrix *Clone();
    
    TPZMatrix<STATE> * Create();

    // need help
    TPZMatrix<STATE> *CreateAssemble(TPZFMatrix<STATE> &rhs, TPZAutoPointer<TPZGuiInterface> guiInterface);

    void SetUpDataStructure();

    void Assemble(TPZMatrix<STATE> & mat, TPZFMatrix<STATE> & rhs, TPZAutoPointer<TPZGuiInterface> guiInterface);
    
    void Assemble(TPZFMatrix<STATE> & rhs, TPZAutoPointer<TPZGuiInterface> guiInterface);
    
    bool isBuilt() {
        if(fIntegrator.IrregularBlocksMatrix().Rows() != 0) return true;
        else return false;
    }

private:
    
    int StressRateVectorSize();
    
    void ComputeDomainElementIndexes(TPZVec<int> &element_indexes);
    
    void ClassifyMaterialsByDimension();
    
    void AssembleBoundaryData();

    void SetUpIrregularBlocksData(TPZVec<int> &element_indexes, TPZIrregularBlocksMatrix::IrregularBlocks &blocksData);

    void SetUpIndexes(TPZVec<int> &element_indexes, TPZVec<int> & dof_indexes);

    void ColoredIndexes(TPZVec<int> &element_indexes, TPZVec<int> &indexes, TPZVec<int> &coloredindexes, int &ncolor);
    
    int fDimension;
    

    TPZNumericalIntegrator fIntegrator;

    TPZVerySparseMatrix<STATE> fSparseMatrixLinear; //-> BC data

    TPZFMatrix<STATE> fRhsLinear; //-> BC data
    
    std::set<int> fBCMaterialIds;
    
    // @TODO:: Replace it for a matrix object
//    std::unordered_map<int64_t,std::unordered_map<int64_t,int64_t>> m_i_j_to_sequence;
    std::vector<std::vector<int64_t>> m_i_j_to_sequence;
    
};

#endif /* TPZIntPointsFEM_h */
