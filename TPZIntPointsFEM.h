/**
 * @file
 * @brief Contains the TPZSolveMatrix class which implements a solution based on a matrix procedure.
 */

#ifndef TPZIntPointsFEM_h
#define TPZIntPointsFEM_h
#include "pzmatrix.h"
#include "pzfmatrix.h"
#include "pzinterpolationspace.h"
#include "pzcmesh.h"
#include "Timer.h"

#ifdef USING_MKL
#include "mkl.h"
#endif
#include "TPZIrregularBlockMatrix.h"

#ifdef __CUDACC__
#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda.h>
#endif

class TPZIntPointsFEM {

public:
    /** @brief Default constructor */
    TPZIntPointsFEM();

    /** @brief Creates the object based on a TPZIrregularBlockMatrix
     * @param Bmatrix : Irregular block matrix
     */
    TPZIntPointsFEM(TPZIrregularBlockMatrix *Bmatrix);

    /** @brief Default destructor */
    ~TPZIntPointsFEM();

    /** @brief Creates a IntPointFEM object with copy constructor
     * @param copy : original IntPointFEM object
     */
    TPZIntPointsFEM(const TPZIntPointsFEM &copy);

    /** @brief operator= */
    TPZIntPointsFEM &operator=(const TPZIntPointsFEM &copy);

    /** @brief Sets the irregular block matrix
     * @param Bmatrix : irregular block matrix
     */
    void SetBMatrix(TPZIrregularBlockMatrix *Bmatrix) {
        fBMatrix = Bmatrix;
    }

    /** @brief Sets integration points information (weight vector and indexes vector) */
    void SetIntPointsInfo();

    /** @brief Coloring elements process */
    void ColoringElements();

    /** @brief Assemble Rhs of boundary elements */
    void AssembleRhsBoundary();

    /** @brief Gathers the global solution in a solution vector ordered by the indexes vector */
    void GatherSolution(TPZFMatrix<REAL> &global_solution, TPZFMatrix<REAL> &gather_solution);

    /** @brief Assemble the domain elements residual */
    void ColoredAssemble(TPZFMatrix<REAL> &nodal_forces);

    /** @brief Access methods  */

    int64_t NumColors() {
        return fNColor;
    }

    TPZFMatrix<REAL> & Rhs() {
        return fRhs;
    }

    TPZFMatrix<REAL> RhsBoundary() {
        return fRhsBoundary;
    }

    TPZVec<int> Indexes() {
        return fIndexes;
    }

    TPZVec<int> ColoredIndexes() {
        return fIndexesColor;
    }

    TPZVec<REAL> Weight() {
        return fWeight;
    }

    TPZIrregularBlockMatrix *BMatrix() {
        return fBMatrix;
    }

protected:
    int64_t fNColor;

    TPZFMatrix<REAL> fRhs;

    TPZFMatrix<REAL> fRhsBoundary;

	TPZVec<int> fIndexes;

	TPZVec<int> fIndexesColor;

	TPZVec<REAL> fWeight;

    TPZIrregularBlockMatrix *fBMatrix;
};

#endif /* TPZIntPointsFEM_h */
