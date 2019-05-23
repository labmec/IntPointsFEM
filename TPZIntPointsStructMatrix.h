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

    /** @brief Creates the object based on a Compmesh
     * @param Compmesh : Computational mesh
     */
    TPZIntPointsStructMatrix(TPZCompMesh *cmesh);

    TPZIntPointsStructMatrix(TPZAutoPointer<TPZCompMesh> cmesh);

    /** @brief Default destructor */
    ~TPZIntPointsStructMatrix();

    /** @brief Clone */
    TPZStructMatrix *Clone();

    /** @brief Creates a TPZIntPointsStructMatrix object with copy constructor
     * @param copy : original TPZIntPointsStructMatrix object
     */
    TPZIntPointsStructMatrix(const TPZIntPointsStructMatrix &copy);

    /** @brief operator= */
    TPZIntPointsStructMatrix &operator=(const TPZIntPointsStructMatrix &copy);

    /** @brief Defines with elements must be assembled */
    void ElementsToAssemble();

    /** @brief Defines matrices rows and columns sizes, first row and column indexes and CSR parameters */
    void BlocksInfo();

    /** @brief Fill matrices values */
    void FillBlocks();

    /** @brief Defines integration points information */
    void IntPointsInfo();

    /** @brief Assemble the load vector */
    void Assemble(TPZFMatrix<REAL> & rhs);

    /** @brief Performs elements coloring */
    void ColoringElements();

    /** @brief Assemble the load vector for boundary elements */
    void AssembleRhsBoundary();

    /** @brief Initialize fBlockMatrix and fIntPointsData */
    void Initialize();

    /** @brief Access methods */
    TPZIrregularBlockMatrix BlockMatrix() {
        return fBlockMatrix;
    }

    TPZIntPointsData IntPointsData() {
        return fIntPointsData;
    }

protected:
    /** @brief Load vector for boundary elements */
    TPZFMatrix<REAL> fRhsBoundary;

    /** @brief Vector of irregular blocks matrix (each position of the vector represents one material) */
    TPZIrregularBlockMatrix fBlockMatrix;

    /** @brief Vector of integration points info (each position of the vector represents one material) */
    TPZIntPointsData fIntPointsData;

    /** @brief Vector of indexes of the domain elements (each position of the vector represents one material) */
    TPZStack<int64_t> fElemIndexes;
};

#endif /* TPZIntPointsFEM_h */
