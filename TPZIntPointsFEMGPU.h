/*
 * TPZIntPointsFEMGPU.h
 *
 *  Created on: 16/04/2019
 *      Author: natalia
 */

#ifndef TPZINTPOINTSFEMGPU_H_
#define TPZINTPOINTSFEMGPU_H_

class TPZIntPointsFEMGPU {
public:
	TPZIntPointsFEMGPU(TPZCompMesh *cmesh, int materialid) {
        SetCompMesh(cmesh);
        SetMaterialId(materialid);
        SetDataStructure();
        AssembleRhsBoundary();

        fDim = fCmesh->Dimension();
        fPlasticStrain.Resize(fDim * fNpts, 1);
        fPlasticStrain.Zero();

        TransferDataStructure();
    }

    ~TPZIntPointsFEMGPU() {
    }

    TPZIntPointsFEMGPU(const TPZIntPointsFEM &copy) {
        fDim = copy.fDim;
        fRhs = copy.fRhs;
        fBoundaryElements = copy.fBoundaryElements;
        fCmesh = copy.fCmesh;
        fNpts = copy.fNpts;
        fNphis = copy.fNphis;
        fElemColor = copy.fElemColor;
        fMaterial = copy.fMaterial;

        dSolution = copy.dSolution;
        dRhsBoundary = copy.dRhsBoundary;
        dPlasticStrain = copy.dPlasticStrain;
        dStorage = copy.dStorage;
        dColSizes = copy.dColSizes;
        dRowSizes = copy.dRowSizes;
        dMatrixPosition = copy.dMatrixPosition;
        dColFirstIndex = copy.dColFirstIndex;
        dRowFirstIndex = copy.dRowFirstIndex;
        dIndexes = copy.dIndexes;
        dIndexesColor = copy.dIndexesColor;
        dWeight = copy.dWeight;
    }

    TPZIntPointsFEMGPU &operator=(const TPZIntPointsFEM &copy) {
        fDim = copy.fDim;
        fRhs = copy.fRhs;
        fBoundaryElements = copy.fBoundaryElements;
        fCmesh = copy.fCmesh;
        fNpts = copy.fNpts;
        fNphis = copy.fNphis;
        fElemColor = copy.fElemColor;
        fMaterial = copy.fMaterial;

        dSolution = copy.dSolution;
        dRhsBoundary = copy.dRhsBoundary;
        dPlasticStrain = copy.dPlasticStrain;
        dStorage = copy.dStorage;
        dColSizes = copy.dColSizes;
        dRowSizes = copy.dRowSizes;
        dMatrixPosition = copy.dMatrixPosition;
        dColFirstIndex = copy.dColFirstIndex;
        dRowFirstIndex = copy.dRowFirstIndex;
        dIndexes = copy.dIndexes;
        dIndexesColor = copy.dIndexesColor;
        dWeight = copy.dWeight;
        return *this;
    }

protected:
/// Data stored on host
/// dim
    int fDim;
/// rhs
    TPZFMatrix<REAL> fRhs;

/// boundary elements vector
    TPZStack<int64_t> fBoundaryElements;

/// computational mesh
    TPZCompMesh *fCmesh;

/// total number of int points
    int64_t fNpts;

/// total number of phis
    int64_t fNphis;

/// material
    TPZMatElastoPlastic2D<TPZPlasticStepPV<TPZYCMohrCoulombPV,TPZElasticResponse>, TPZElastoPlasticMem> *fMaterial;

/// color indexes of each element
    TPZVec<int64_t> fElemColor;

/// Data stored on device
/// solution
    REAL *dSolution;

/// rhs boundary
    REAL *dRhsBoundary;

/// plastic strain
    REAL *dPlasticStrain;

/// vector containing the matrix coefficients
    REAL *dStorage;

/// number of columns of each block matrix
    int *dColSizes;

/// number of rows of each block matrix
    int *dRowSizes;

/// indexes vector in x and y direction
    int *dIndexes;

/// position of the matrix of the elements
    int *dMatrixPosition;

/// position in the fIndex vector of each element
    int *dColFirstIndex;

/// position in the fIndex vector of each element
    int *dRowFirstIndex;

/// indexes vector in x and y direction by color
    int *dIndexesColor;

/// weight Vector
    REAL * dWeight;

/// library handles
    cusparseHandle_t handle_cusparse;
    cublasHandle_t handle_cublas;

};
#endif /* TPZINTPOINTSFEMGPU_H_ */

