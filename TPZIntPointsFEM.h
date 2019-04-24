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


#ifdef USING_MKL
#include "mkl.h"
#endif

#ifdef __CUDACC__
#include "TPZVecGPU.h"
#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda.h>
#endif

class TPZIntPointsFEM {

public:

    TPZIntPointsFEM() {
    }

    TPZIntPointsFEM(TPZCompMesh *cmesh, int materialid) {
        SetCompMesh(cmesh);
        SetMaterialId(materialid);
    }

    ~TPZIntPointsFEM() {
#ifdef __CUDACC__
    	DestroyHandles();
#endif

    }

    TPZIntPointsFEM(const TPZIntPointsFEM &copy) {
        fDim = copy.fDim;
        fRhs = copy.fRhs;
        fRhsBoundary = copy.fRhsBoundary;
        fBoundaryElements = copy.fBoundaryElements;
        fPlasticStrain = copy.fPlasticStrain;
        fSolution = copy.fSolution;
        fCmesh = copy.fCmesh;
        fNpts = copy.fNpts;
        fNphis = copy.fNphis;
        fStorage = copy.fStorage;
        fColSizes = copy.fColSizes;
        fRowSizes = copy.fRowSizes;
        fMatrixPosition = copy.fMatrixPosition;
        fColFirstIndex = copy.fColFirstIndex;
        fRowFirstIndex = copy.fRowFirstIndex;
        fElemColor = copy.fElemColor;
        fIndexes = copy.fIndexes;
        fIndexesColor = copy.fIndexesColor;
        fWeight = copy.fWeight;
        fMaterial = copy.fMaterial;

#ifdef __CUDACC__
        handle_cusparse = copy.handle_cusparse;
        handle_cublas = copy.handle_cublas;

        dRhs = copy.dRhs;
        dRhsBoundary = copy.dRhsBoundary;
        dSolution = copy.dSolution;
        dPlasticStrain = copy.dPlasticStrain;
        dStorage = copy.dStorage;
        dRowSizes = copy.dRowSizes;
        dColSizes = copy.dColSizes;
        dMatrixPosition = copy.dMatrixPosition;
        dRowFirstIndex = copy.dRowFirstIndex;
        dColFirstIndex = copy.dColFirstIndex;
        dIndexes = copy.dIndexes;
        dIndexesColor = copy.dIndexesColor;
        dWeight = copy.dWeight;
#endif
    }

    TPZIntPointsFEM &operator=(const TPZIntPointsFEM &copy) {
        fDim = copy.fDim;
        fRhs = copy.fRhs;
        fRhsBoundary = copy.fRhsBoundary;
        fBoundaryElements = copy.fBoundaryElements;
        fPlasticStrain = copy.fPlasticStrain;
        fSolution = copy.fSolution;
        fCmesh = copy.fCmesh;
        fStorage = copy.fStorage;
        fColSizes = copy.fColSizes;
        fRowSizes = copy.fRowSizes;
        fMatrixPosition = copy.fMatrixPosition;
        fColFirstIndex = copy.fColFirstIndex;
        fRowFirstIndex = copy.fRowFirstIndex;
        fElemColor = copy.fElemColor;
        fIndexes = copy.fIndexes;
        fIndexesColor = copy.fIndexesColor;
        fWeight = copy.fWeight;
        fMaterial = copy.fMaterial;

#ifdef __CUDACC__
        handle_cusparse = copy.handle_cusparse;
        handle_cublas = copy.handle_cublas;

        dRhs = copy.dRhs;
        dRhsBoundary = copy.dRhsBoundary;
        dSolution = copy.dSolution;
        dPlasticStrain = copy.dPlasticStrain;
        dStorage = copy.dStorage;
        dRowSizes = copy.dRowSizes;
        dColSizes = copy.dColSizes;
        dMatrixPosition = copy.dMatrixPosition;
        dRowFirstIndex = copy.dRowFirstIndex;
        dColFirstIndex = copy.dColFirstIndex;
        dIndexes = copy.dIndexes;
        dIndexesColor = copy.dIndexesColor;
        dWeight = copy.dWeight;
#endif
        return *this;
    }

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

    int64_t NumberofIntPoints() {
        return fNpts;
    }

    void SetNumberofPhis(int64_t nphis) {
        fNphis = nphis;
    }

    void SetCompMesh(TPZCompMesh *cmesh) {
        fCmesh = cmesh;
    }

    void SetWeightVector (TPZStack<REAL> wvec) {
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

    void SetDataStructure();

    void GatherSolution(TPZFMatrix<REAL> &global_solution, TPZFMatrix<REAL> &gather_solution);
    void DeltaStrain(TPZFMatrix<REAL> &gather_solution, TPZFMatrix<REAL> &delta_strain);

    void ElasticStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &plastic_strain, TPZFMatrix<REAL> &elastic_strain);
    void PlasticStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &elastic_strain, TPZFMatrix<REAL> &plastic_strain);

    void ComputeStress(TPZFMatrix<REAL> &elastic_strain, TPZFMatrix<REAL> &sigma);

    void SpectralDecomposition(TPZFMatrix<REAL> &sigma_trial, TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &eigenvectors);
    void Normalize(double *sigma, double &maxel);
    void Interval(double *sigma, double *interval);
    void NewtonIterations(double *interval, double *sigma, double *eigenvalues, double &maxel);
    void Eigenvectors(double *sigma, double *eigenvalue, double *eigenvector, double &maxel);
    void Multiplicity1(double *sigma, double eigenvalue, double *eigenvector);
    void Multiplicity2(double *sigma, double eigenvalue, double *eigenvector1, double *eigenvector2);

    void ProjectSigma(TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &sigma_projected);
    bool PhiPlane(double *eigenvalues, double *sigma_projected);
    bool ReturnMappingMainPlane(double *eigenvalues, double *sigma_projected, double &m_hardening);
    bool ReturnMappingRightEdge(double *eigenvalues, double *sigma_projected, double &m_hardening);
    bool ReturnMappingLeftEdge(double *eigenvalues, double *sigma_projected, double &m_hardening);
    void ReturnMappingApex(double *eigenvalues, double *sigma_projected, double &m_hardening);

    void StressCompleteTensor(TPZFMatrix<REAL> &sigma_projected, TPZFMatrix<REAL> &eigenvectors, TPZFMatrix<REAL> &sigma);

    void ComputeStrain(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &elastic_strain);

    void NodalForces(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &nodal_forces);

    void ColoredAssemble(TPZFMatrix<REAL> &nodal_forces, TPZFMatrix<REAL> &residual);

    void AssembleResidual();

    void ColoringElements() const;

    void AssembleRhsBoundary();

    void DestroyHandles();

#ifdef __CUDACC__
    void GatherSolutionGPU(TPZVecGPU<REAL> &global_solution, TPZVecGPU<REAL> &gather_solution);
    void DeltaStrainGPU(TPZVecGPU<REAL> &gather_solution, TPZVecGPU<REAL> &delta_strain);
    void ElasticStrainGPU(TPZVecGPU<REAL> &delta_strain, TPZVecGPU<REAL> &plastic_strain, TPZVecGPU<REAL> &elastic_strain);
    void ComputeStressGPU(TPZVecGPU<REAL> &elastic_strain, TPZVecGPU<REAL> &sigma);
    void SpectralDecompositionGPU(TPZVecGPU<REAL> &sigma_trial, TPZVecGPU<REAL> &eigenvalues, TPZVecGPU<REAL> &eigenvectors);
    void ProjectSigmaGPU(TPZVecGPU<REAL> &eigenvalues, TPZVecGPU<REAL> &sigma_projected);
    void StressCompleteTensorGPU(TPZVecGPU<REAL> &sigma_projected, TPZVecGPU<REAL> &eigenvectors, TPZVecGPU<REAL> &sigma);
    void NodalForcesGPU(TPZVecGPU<REAL> &sigma, TPZVecGPU<REAL> &nodal_forces);
    void ColoredAssembleGPU(TPZVecGPU<REAL> &nodal_forces, TPZVecGPU<REAL> &residual);
    void ComputeStrainGPU(TPZVecGPU<REAL> &sigma, TPZVecGPU<REAL> &elastic_strain);
    void PlasticStrainGPU(TPZVecGPU<REAL> &delta_strain, TPZVecGPU<REAL> &elastic_strain, TPZVecGPU<REAL> &plastic_strain);

    void TransferDataStructure();

    void cuSparseHandleCreate() {
        cusparseCreate (&handle_cusparse);
    }

    void cuBlasHandleCreate() {
        cublasCreate (&handle_cublas);
    }

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
	TPZVec<MKL_INT> fIndexesColor;
	TPZStack<REAL> fWeight;

#ifdef __CUDACC__
    cusparseHandle_t handle_cusparse;
    cublasHandle_t handle_cublas;

    TPZVecGPU<REAL> dRhs;
    TPZVecGPU<REAL> dRhsBoundary;
    TPZVecGPU<REAL> dSolution;
    TPZVecGPU<REAL> dPlasticStrain;
    TPZVecGPU<REAL> dStorage;
    TPZVecGPU<int> dRowSizes;
    TPZVecGPU<int> dColSizes;
    TPZVecGPU<int> dMatrixPosition;
    TPZVecGPU<int> dRowFirstIndex;
    TPZVecGPU<int> dColFirstIndex;
    TPZVecGPU<int> dIndexes;
    TPZVecGPU<int> dIndexesColor;
    TPZVecGPU<REAL> dWeight;
#endif

};

#endif /* TPZIntPointsFEM_h */
