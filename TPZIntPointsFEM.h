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
#include <cublas_v2.h>
#include <cusparse.h>
#endif

class TPZIntPointsFEM {

public:

    TPZIntPointsFEM() {
    }

    TPZIntPointsFEM(TPZCompMesh *cmesh, int materialid) {
        SetCompMesh(cmesh);
        SetMaterialId(materialid);
        SetDataStructure();
        std::cout << "oioioi" << std::endl;
        AssembleRhsBoundary();



        fDim = fCmesh->Dimension();
#ifdef __CUDACC__
        cuSparseHandle();
        cuBlasHandle();

		cudaMalloc(&fPlasticStrain, fDim * fNpts * sizeof(REAL));
		cudaMemset(fPlasticStrain, 0, fDim * fNpts * sizeof(REAL));
#else
        fPlasticStrain.Resize(fDim * fNpts, 1);
        fPlasticStrain.Zero();
#endif
    }

    ~TPZIntPointsFEM() {

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
        return *this;
    }

    void SetRowandColSizes(TPZVec<int> rowsize, TPZVec<int> colsize) {
        int64_t nelem = rowsize.size();

#ifdef __CUDACC__
        cudaMallocManaged(&fRowSizes, nelem*sizeof(int));
        cudaMallocManaged(&fColSizes, nelem*sizeof(int));
        cudaMallocManaged(&fMatrixPosition, (nelem + 1)*sizeof(int));
        cudaMallocManaged(&fRowFirstIndex, (nelem + 1)*sizeof(int));
        cudaMallocManaged(&fColFirstIndex, (nelem + 1)*sizeof(int));
 #else
        fRowSizes.resize(nelem);
        fColSizes.resize(nelem);
        fMatrixPosition.resize(nelem + 1);
        fRowFirstIndex.resize(nelem + 1);
        fColFirstIndex.resize(nelem + 1);
#endif

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
#ifdef __CUDACC__
        cudaMallocManaged(&fStorage, fMatrixPosition[nelem]*sizeof(REAL));
 #else
        fStorage.resize(fMatrixPosition[nelem]);
#endif
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

    void SetIndexes(TPZVec<MKL_INT> indexes) {
        int64_t indsize = indexes.size();

#ifdef __CUDACC__
    	cudaMalloc(&fIndexes, indsize*sizeof(int));
    	cudaMemcpyAsync	(fIndexes, &indexes[0], indsize*sizeof(int),cudaMemcpyHostToDevice,0);
    	cudaMalloc(&fIndexesColor, indsize*sizeof(int));
 #else
        fIndexes.resize(indsize);
        fIndexes = indexes;
        fIndexesColor.resize(indsize);
#endif


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

    void SetDataStructure();

    void GatherSolution(TPZFMatrix<REAL> &global_solution, TPZFMatrix<REAL> &gather_solution);

    void DeltaStrain(TPZFMatrix<REAL> &global_solution, TPZFMatrix<REAL> &deltastrain);

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

    void ColoredAssemble(TPZFMatrix<REAL> &nodal_forces_vec, TPZFMatrix<REAL> &nodal_forces_global);

    void AssembleResidual();

    void ColoringElements() const;

    void AssembleRhsBoundary();

    void cuSparseHandle() {
#ifdef __CUDACC__
        cusparseCreate (&handle_cusparse);
#endif
    }

    void cuBlasHandle() {
#ifdef __CUDACC__
        cublasCreate (&handle_cublas);
#endif
    }


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
//	TPZFMatrix<REAL> fPlasticStrain;

#ifdef __CUDACC__
    REAL *fPlasticStrain;
    REAL *fStorage;
    int *fColSizes;
    int *fRowSizes;
    int *fMatrixPosition;
    int *fColFirstIndex;
    int *fRowFirstIndex;
    int *fIndexes;
    int *fIndexesColor;
    REAL *fWeight;

    cusparseHandle_t handle_cusparse;
    cublasHandle_t handle_cublas;
#else
    TPZFMatrix<REAL> fPlasticStrain;
	TPZVec<REAL> fStorage;
	TPZVec<int> fRowSizes;
	TPZVec<int> fColSizes;
	TPZVec<int> fMatrixPosition;
	TPZVec<int> fRowFirstIndex;
	TPZVec<int> fColFirstIndex;
	TPZVec<MKL_INT> fIndexes;
	TPZVec<MKL_INT> fIndexesColor;
	TPZStack<REAL> fWeight;
#endif

};

#endif /* TPZIntPointsFEM_h */
