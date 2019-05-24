/**
 * @file
 * @brief Contains the TPZSolveMatrix class which implements a solution based on a matrix procedure.
 */

#ifndef TPZIntPointsFEM_h
#define TPZIntPointsFEM_h
#include "TPZIrregularBlocksMatrix.h"
#include "TPZIntPointsData.h"
#include "pzstrmatrix.h"

class TPZIntPointsStructMatrix : public TPZStructMatrix {
public:
    /** @brief Default constructor */
    TPZIntPointsStructMatrix();

    /** @brief Creates the object based on a Compmesh
     * @param Compmesh : Computational mesh */
    TPZIntPointsStructMatrix(TPZCompMesh *cmesh);

    TPZIntPointsStructMatrix(TPZAutoPointer<TPZCompMesh> cmesh);

    /** @brief Default destructor */
    ~TPZIntPointsStructMatrix();

    /** @brief Clone */
    TPZStructMatrix *Clone();

    /** @brief Creates a TPZIntPointsStructMatrix object with copy constructor
     * @param copy : original TPZIntPointsStructMatrix object */
    TPZIntPointsStructMatrix(const TPZIntPointsStructMatrix &copy);

    /** @brief operator= */
    TPZIntPointsStructMatrix &operator=(const TPZIntPointsStructMatrix &copy);

    /** @brief Defines which elements must be assembled */
    void ElementsToAssemble();

    /** @brief Defines integration points information */
    void IntPointsInfo(TPZIrregularBlocksMatrix &blockMatrix);

    /** @brief Assemble the load vector for domain elements*/
    void Assemble(TPZFMatrix<REAL> & rhs);

    /** @brief Performs elements coloring */
    void ColoringElements(TPZIrregularBlocksMatrix &blockMatrix);

    /** @brief Assemble the load vector for boundary elements */
    void AssembleRhsBoundary();

    /** @brief Creates a TPZIrregularBlockMatrix */
    TPZMatrix<REAL> * Create();

    TPZIntPointsData IntPointsData() {
        return fIntPointsData;
    }

protected:
    TPZIrregularBlocksMatrix *fBlockMatrix;

    /** @brief Load vector for boundary elements */
    TPZFMatrix<REAL> fRhsBoundary;

    /** @brief Vector of integration points info (each position of the vector represents one material) */
    TPZIntPointsData fIntPointsData;

    /** @brief Vector of indexes of the domain elements (each position of the vector represents one material) */
    TPZStack<int64_t> fElemIndexes;
};

#endif /* TPZIntPointsFEM_h */
