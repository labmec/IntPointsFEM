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

    TPZIntPointsFEM();

    TPZIntPointsFEM(TPZIrregularBlockMatrix *Bmatrix);

    ~TPZIntPointsFEM();

    TPZIntPointsFEM(const TPZIntPointsFEM &copy);

    TPZIntPointsFEM &operator=(const TPZIntPointsFEM &copy);

    void SetBMatrix(TPZIrregularBlockMatrix *Bmatrix) {
        fBMatrix = Bmatrix;
    }

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

    void SetIntPointsInfo();

    void ColoringElements();

    void AssembleRhsBoundary();

    void GatherSolution(TPZFMatrix<REAL> &global_solution, TPZFMatrix<REAL> &gather_solution);

    void ColoredAssemble(TPZFMatrix<REAL> &nodal_forces);

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
