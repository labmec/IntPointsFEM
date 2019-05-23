/**
 * @file
 * @brief Contains the TPZSolveMatrix class which implements a solution based on a matrix procedure.
 */

#ifndef TPZIntPointsFEM_h
#define TPZIntPointsFEM_h
#include "TPZIrregularBlockMatrix.h"
#include "TPZIntPointsData.h"
#include "pzstrmatrix.h"

class TPZIntPointsStructMatrix : public TPZStructMatrix {
public:
    /** @brief Default constructor */
    TPZIntPointsStructMatrix();

    /** @brief Creates the object based on a TPZIrregularBlockMatrix
     * @param Bmatrix : Irregular block matrix
     */
    TPZIntPointsStructMatrix(TPZCompMesh *cmesh);

    /** @brief Default destructor */
    ~TPZIntPointsStructMatrix();

    /** @brief Creates a TPZIntPointsStructMatrix object with copy constructor
     * @param copy : original IntPointFEM object
     */
    TPZIntPointsStructMatrix(const TPZIntPointsStructMatrix &copy);

    /** @brief operator= */
    TPZIntPointsStructMatrix &operator=(const TPZIntPointsStructMatrix &copy);

    void SetMaterialIds(TPZVec<int> materialids) {
        fMaterialIds = materialids;
    }

    void ElementsToAssemble();

    void BlocksInfo(int imat);

    void FillBlocks(int imat);

    void IntPointsInfo(int imat);

    void Assemble();

    void ColoringElements(int imat);

    void AssembleRhsBoundary();

    void Initialize();

    TPZIrregularBlockMatrix BlockMatrix(int imat) {
        return fBlockMatrix[imat];
    }

    TPZIntPointsData IntPointsData(int imat) {
        return fIntPointsData[imat];
    }
    TPZFMatrix<REAL> Rhs() {
        return fRhs;
    }

protected:
    TPZFMatrix<REAL> fRhs;

    TPZFMatrix<REAL> fRhsBoundary;

    /** @brief Vector of material ids */
    TPZVec<int> fMaterialIds;

    /** @brief Irregular blocks matrix */
    TPZVec<TPZIrregularBlockMatrix> fBlockMatrix;

    TPZVec<TPZIntPointsData> fIntPointsData;

    /** @brief Vector of indexes of the domain elements for each material id*/
    TPZVec<TPZStack<int64_t>> fElemIndexes;
};

#endif /* TPZIntPointsFEM_h */
