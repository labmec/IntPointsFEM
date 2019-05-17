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
#include "TPZPlasticStepPV.h"
#include "TPZYCMohrCoulombPV.h"
#include "TPZElastoPlasticMem.h"
#include "TPZMatElastoPlastic2D.h"
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

    TPZIntPointsFEM(TPZIrregularBlockMatrix *Bmatrix, int materialid);

    ~TPZIntPointsFEM();

    TPZIntPointsFEM(const TPZIntPointsFEM &copy);

    TPZIntPointsFEM &operator=(const TPZIntPointsFEM &copy);

    void SetBMatrix(TPZIrregularBlockMatrix *Bmatrix) {
        fBMatrix = Bmatrix;
    }

    void SetMaterialId (int materialid) {
        TPZMaterial *material = fBMatrix->CompMesh()->FindMaterial(materialid);
        fMaterial = dynamic_cast<TPZMatElastoPlastic2D<TPZPlasticStepPV<TPZYCMohrCoulombPV,TPZElasticResponse> , TPZElastoPlasticMem> *>(material);
    }

     TPZFMatrix<REAL> & Rhs() {
        return fRhs;
    }

    void SetTimerConfig(Timer::WhichUnit unit);

    void AssembleResidual();
    void SetDataStructure();
    void ColoringElements();
    void AssembleRhsBoundary();
    void ComputeSigma(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &sigma);
#ifdef __CUDACC__
    void TransferDataStructure();

    void GatherSolution(REAL *gather_solution);
    void DeltaStrain(REAL *gather_solution, REAL *delta_strain);
    void ElasticStrain(REAL *delta_strain, REAL *plastic_strain, REAL *elastic_strain);
    void ComputeStress(REAL *elastic_strain, REAL *sigma);
    void SpectralDecomposition(REAL *sigma_trial, REAL *eigenvalues, REAL *eigenvectors);
    void ProjectSigma(REAL *eigenvalues, REAL *sigma_projected);
    void StressCompleteTensor(REAL *sigma_projected, REAL *eigenvectors, REAL *sigma);
    void NodalForces(REAL *sigma, REAL *nodal_forces);
    void ColoredAssemble(REAL *nodal_forces, REAL *residual);
    void ComputeStrain(REAL *sigma, REAL *elastic_strain);
    void PlasticStrain(REAL *delta_strain, REAL *elastic_strain, REAL *plastic_strain);
#else
    void GatherSolution(TPZFMatrix<REAL> &global_solution, TPZFMatrix<REAL> &gather_solution);
    void ElasticStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &plastic_strain, TPZFMatrix<REAL> &elastic_strain);
    void ComputeStress(TPZFMatrix<REAL> &elastic_strain, TPZFMatrix<REAL> &sigma);
    void SpectralDecomposition(TPZFMatrix<REAL> &sigma_trial, TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &eigenvectors);
    void ProjectSigma(TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &sigma_projected);
    void StressCompleteTensor(TPZFMatrix<REAL> &sigma_projected, TPZFMatrix<REAL> &eigenvectors, TPZFMatrix<REAL> &sigma);
    void ColoredAssemble(TPZFMatrix<REAL> &nodal_forces);
    void ComputeStrain(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &elastic_strain);
    void PlasticStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &elastic_strain, TPZFMatrix<REAL> &plastic_strain);
#endif

protected:
    int fDim;
    int64_t fNpts;
    int64_t fNphis;
    int64_t fNColor;
    TPZMatElastoPlastic2D<TPZPlasticStepPV<TPZYCMohrCoulombPV,TPZElasticResponse>, TPZElastoPlasticMem> *fMaterial;
    TPZFMatrix<REAL> fRhs;
    TPZFMatrix<REAL> fRhsBoundary;
	TPZVec<int> fIndexes;
	TPZVec<int> fIndexesColor;
	TPZVec<REAL> fWeight;
    TPZIrregularBlockMatrix *fBMatrix;

    TPZFMatrix<REAL> fPlasticStrain;

    Timer fTimer;



#ifdef __CUDACC__
	cusparseHandle_t handle_cusparse;
	cublasHandle_t handle_cublas;
#endif


    REAL *dRhs;
    REAL *dRhsBoundary;
    REAL *dSolution;
    REAL *dPlasticStrain;
    REAL *dStorage;
    int *dRowSizes;
    int *dColSizes;
    int *dMatrixPosition;
    int *dRowFirstIndex;
    int *dColFirstIndex;
    int *dIndexes;
    int *dIndexesColor;
    REAL *dWeight;

    int *dRowPtr;
    int *dColInd;
};

#endif /* TPZIntPointsFEM_h */
