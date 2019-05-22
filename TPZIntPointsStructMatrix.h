/**
 * @file
 * @brief Contains the TPZSolveMatrix class which implements a solution based on a matrix procedure.
 */

#ifndef TPZIntPointsFEM_h
#define TPZIntPointsFEM_h
#include "TPZIrregularBlockMatrix.h"
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

    void BlocksInfo(TPZStack<int64_t> elemindexes);

    void FillBlocks(TPZStack<int64_t> elemindexes);

    void IntPointsInfo(TPZStack<int64_t> elemindexes);

    void GatherSolution(TPZFMatrix<REAL> &solution, TPZFMatrix<REAL> &gather_solution);

    void ColoredAssemble(TPZFMatrix<REAL>  &nodal_forces);

    void Assemble(TPZFMatrix<REAL> &solution);

    void ColoringElements(TPZStack<int64_t> elemindexes);

    void AssembleRhsBoundary();

    void InitializeMatrix();

    TPZIrregularBlockMatrix BlockMatrix() {
        return fBlockMatrix;
    }

    TPZVec<REAL> Weight() {
        return fWeight;
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
    TPZIrregularBlockMatrix fBlockMatrix;

    /** @brief Vector of indexes of the domain elements for each material id*/
    TPZVec<TPZStack<int64_t>> fElemIndexes;

    /** @brief Number of colors of colored mesh */
    int64_t fNColor;

    /** @brief DOF indexes vector ordered by element */
	TPZVec<int> fIndexes;

    /** @brief Colored DOF indexes vector ordered by element */
	TPZVec<int> fIndexesColor;

    /** @brief Weight Vector */
	TPZVec<REAL> fWeight;


};

#endif /* TPZIntPointsFEM_h */
