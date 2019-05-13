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


#ifdef __CUDACC__
#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda.h>
#endif

class TPZIntPointsFEM {

public:

    TPZIntPointsFEM();

    TPZIntPointsFEM(TPZCompMesh *cmesh, int materialid);

    ~TPZIntPointsFEM();

    TPZIntPointsFEM(const TPZIntPointsFEM &copy);

    TPZIntPointsFEM &operator=(const TPZIntPointsFEM &copy);

    void SetRowandColSizes(TPZVec<int> rowsize, TPZVec<int> colsize) {
        int64_t nelem = rowsize.size();

        fRowSizes.resize(nelem);
        fColSizes.resize(nelem);
        fMatrixPosition.resize(nelem + 1);
        fRowFirstIndex.resize(nelem + 1);
        fColFirstIndex.resize(nelem + 1);
        fMatrixPosition[0] = 0;
        fRowFirstIndex[0] = 0;
        fColFirstIndex[0] = 0;

        for (int64_t iel = 0; iel < nelem; ++iel) {
        	fRowSizes[iel] = rowsize[iel];
        	fColSizes[iel] = colsize[iel];
            fMatrixPosition[iel + 1] = fMatrixPosition[iel] + fRowSizes[iel] * fColSizes[iel];
            fRowFirstIndex[iel + 1] = fRowFirstIndex[iel] + fRowSizes[iel];
            fColFirstIndex[iel + 1] = fColFirstIndex[iel] + fColSizes[iel];
        }
        fStorage.resize(fMatrixPosition[nelem]);
        fElemColor.resize(nelem);
        fElemColor.Fill(-1);
    }

    void CSRInfo() {
        int64_t nelem = fRowSizes.size();
        int64_t nnz = fStorage.size();

        fRowPtr.resize(fNpts + 1); //m+1
        fColInd.resize(nnz);
        for (int iel = 0; iel < nelem; ++iel) {
            for (int irow = 0; irow < fRowSizes[iel]; ++irow) {
                fRowPtr[irow + fRowFirstIndex[iel]] = fMatrixPosition[iel] + irow*fColSizes[iel];

                 for (int icol = 0; icol < fColSizes[iel]; ++icol) {
                    fColInd[icol + fMatrixPosition[iel] + irow*fColSizes[iel]] = icol + fColFirstIndex[iel];
                }
            }
        }
        fRowPtr[fNpts] = fMatrixPosition[nelem];
    }

    void SetElementMatrix(int iel, TPZFMatrix<REAL> &elmat) {
        int64_t rows = fRowSizes[iel];
        int64_t cols = fColSizes[iel];
        int64_t pos = fMatrixPosition[iel];

        TPZFMatrix<REAL> elmatloc(rows, cols, &fStorage[pos], rows*cols);
        elmatloc = elmat;
    }

    void SetIndexes(TPZVec<int> indexes) {
        int64_t indsize = indexes.size();
        fIndexes.resize(indsize);
        fIndexes = indexes;
        fIndexesColor.resize(indsize);
    }

    void SetNumberofIntPoints(int64_t npts) {
        fNpts = npts;
    }

    void SetNumberofPhis(int64_t nphis) {
        fNphis = nphis;
    }

    void SetCompMesh(TPZCompMesh *cmesh) {
        fCmesh = cmesh;
    }

    void SetWeightVector (TPZVec<REAL> &wvec) {
        fWeight = wvec;
    }

    void SetMaterialId (int materialid) {
        TPZMaterial *material = fCmesh->FindMaterial(materialid);
        fMaterial = dynamic_cast<TPZMatElastoPlastic2D<TPZPlasticStepPV<TPZYCMohrCoulombPV,TPZElasticResponse> , TPZElastoPlasticMem> *>(material);
    }

    void LoadSolution (TPZFMatrix<REAL> & sol) {
        fSolution = sol;
    }

     TPZFMatrix<REAL> & Rhs() {
        return fRhs;
    }

    void SetMeshDimension(int dim) {
        fDim = dim;
    }

    void SetTimerConfig(Timer::WhichUnit unit);

    void AssembleResidual();
    void SetDataStructure();
    void ColoringElements() const;
    void AssembleRhsBoundary();

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
    void DeltaStrain(TPZFMatrix<REAL> &gather_solution, TPZFMatrix<REAL> &delta_strain);
    void ElasticStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &plastic_strain, TPZFMatrix<REAL> &elastic_strain);
    void ComputeStress(TPZFMatrix<REAL> &elastic_strain, TPZFMatrix<REAL> &sigma);
    void SpectralDecomposition(TPZFMatrix<REAL> &sigma_trial, TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &eigenvectors);
    void ProjectSigma(TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &sigma_projected);
    void StressCompleteTensor(TPZFMatrix<REAL> &sigma_projected, TPZFMatrix<REAL> &eigenvectors, TPZFMatrix<REAL> &sigma);
    void NodalForces(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &nodal_forces);
    void ColoredAssemble(TPZFMatrix<REAL> &nodal_forces, TPZFMatrix<REAL> &residual);
    void ComputeStrain(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &elastic_strain);
    void PlasticStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &elastic_strain, TPZFMatrix<REAL> &plastic_strain);
#endif

protected:
    int fDim;
    TPZStack<int64_t> fBoundaryElements;
    TPZCompMesh *fCmesh;
    int64_t fNpts;
    int64_t fNphis;
    TPZVec<int64_t> fElemColor;
    TPZMatElastoPlastic2D<TPZPlasticStepPV<TPZYCMohrCoulombPV,TPZElasticResponse>, TPZElastoPlasticMem> *fMaterial;

    TPZFMatrix<REAL> fRhs;
    TPZFMatrix<REAL> fRhsBoundary;
	TPZFMatrix<REAL> fSolution;
    TPZFMatrix<REAL> fPlasticStrain;
	TPZVec<REAL> fStorage;
	TPZVec<int> fRowSizes;
	TPZVec<int> fColSizes;
	TPZVec<int> fMatrixPosition;
	TPZVec<int> fRowFirstIndex;
	TPZVec<int> fColFirstIndex;
	TPZVec<int> fIndexes;
	TPZVec<int> fIndexesColor;
	TPZVec<REAL> fWeight;

    TPZVec<int> fRowPtr;
    TPZVec<int> fColInd;

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
