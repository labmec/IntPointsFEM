//
// Created by natalia on 14/05/19.
//
#include "pzintel.h"
#include "TPZIrregularBlockMatrix.h"

#ifdef USING_MKL
#include <mkl.h>
#endif

TPZIrregularBlockMatrix::TPZIrregularBlockMatrix() : fDim(-1), fCmesh(), fMaterial(), fNumBlocks(-1), fStorage(0), fRowSizes(0), fColSizes(0),
                                                     fMatrixPosition(0), fRowFirstIndex(0), fColFirstIndex(0),
                                                     fRow(-1), fCol(-1), fRowPtr(0), fColInd(0), fElemIndex(),
                                                     fBoundaryElemIndex() {}

TPZIrregularBlockMatrix::TPZIrregularBlockMatrix(TPZCompMesh *cmesh, int materialid) : fDim(-1), fCmesh(), fMaterial(), fNumBlocks(-1), fStorage(0), fRowSizes(0), fColSizes(0),
                                                                       fMatrixPosition(0), fRowFirstIndex(0), fColFirstIndex(0),
                                                                       fRow(-1), fCol(-1), fRowPtr(0), fColInd(0), fElemIndex(),
                                                                       fBoundaryElemIndex() {
    SetCompMesh(cmesh);
    SetMaterialId(materialid);
}

TPZIrregularBlockMatrix::~TPZIrregularBlockMatrix() {
}

TPZIrregularBlockMatrix::TPZIrregularBlockMatrix(const TPZIrregularBlockMatrix &copy) {
    fDim = copy.fDim;
    fCmesh = copy.fCmesh;
    fMaterial = copy.fMaterial;
    fNumBlocks = copy.fNumBlocks;
    fStorage = copy.fStorage;
    fRowSizes = copy.fRowSizes;
    fColSizes = copy.fColSizes;
    fMatrixPosition = copy.fMatrixPosition;
    fRowFirstIndex = copy.fRowFirstIndex;
    fColFirstIndex = copy.fColFirstIndex;
    fRow = copy.fRow;
    fCol = copy.fCol;
    fRowPtr = copy.fRowPtr;
    fColInd = copy.fColInd;
    fElemIndex = copy.fElemIndex;
    fBoundaryElemIndex = copy.fBoundaryElemIndex;
}

TPZIrregularBlockMatrix &TPZIrregularBlockMatrix::operator=(const TPZIrregularBlockMatrix &copy) {
    if(&copy == this){
        return *this;
    }

    fDim = copy.fDim;
    fCmesh = copy.fCmesh;
    fMaterial = copy.fMaterial;
    fNumBlocks = copy.fNumBlocks;
    fStorage = copy.fStorage;
    fRowSizes = copy.fRowSizes;
    fColSizes = copy.fColSizes;
    fMatrixPosition = copy.fMatrixPosition;
    fRowFirstIndex = copy.fRowFirstIndex;
    fColFirstIndex = copy.fColFirstIndex;
    fRow = copy.fRow;
    fCol = copy.fCol;
    fRowPtr = copy.fRowPtr;
    fColInd = copy.fColInd;
    fElemIndex = copy.fElemIndex;
    fBoundaryElemIndex = copy.fBoundaryElemIndex;

    return *this;
}

void TPZIrregularBlockMatrix::BlocksInfo() {
    fDim = fCmesh->Reference()->Dimension(); // Mesh dimension

    //Vector with domain elements and boundary elements
    for (int64_t i = 0; i < fCmesh->NElements(); i++) {
        TPZCompEl *cel = fCmesh->Element(i);
        if (!cel) continue;
        TPZGeoEl *gel = fCmesh->Element(i)->Reference();
        if (!gel) continue;
        if(cel->Material() == fMaterial) fElemIndex.Push(cel->Index());
        else if (cel->Material() != fMaterial && gel->Dimension() < fDim) fBoundaryElemIndex.Push(cel->Index());
    }
    if (fElemIndex.size() == 0) DebugStop();

    // RowSizes and ColSizes vectors
    fNumBlocks = fElemIndex.size();
    fRowSizes.resize(fNumBlocks);
    fColSizes.resize(fNumBlocks);
    fMatrixPosition.resize(fNumBlocks + 1);
    fRowFirstIndex.resize(fNumBlocks + 1);
    fColFirstIndex.resize(fNumBlocks + 1);

    fMatrixPosition[0] = 0;
    fRowFirstIndex[0] = 0;
    fColFirstIndex[0] = 0;

    fRow = 0;
    fCol = 0;

    int it = 0;
    for (auto iel : fElemIndex) {
        TPZCompEl *cel = fCmesh->Element(iel);
        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement*>(cel);
        if (!cel_inter) DebugStop();
        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());

        int64_t npts = int_rule->NPoints(); // number of integration points of the element
        int64_t dim = cel_inter->Dimension(); //dimension of the element
        int64_t nf = cel_inter->NShapeF(); // number of shape functions of the element

        fRowSizes[it] = dim * npts;
        fColSizes[it] = nf;
        fMatrixPosition[it + 1] = fMatrixPosition[it] + fRowSizes[it] * fColSizes[it];
        fRowFirstIndex[it + 1] = fRowFirstIndex[it] + fRowSizes[it];
        fColFirstIndex[it + 1] = fColFirstIndex[it] + fColSizes[it];

        fRow += fRowSizes[it];
        fCol += fColSizes[it];
        it++;
    }

    fStorage.resize(fMatrixPosition[fNumBlocks]);

    it = 0;
    for(auto iel : fElemIndex) {
        TPZCompEl *cel = fCmesh->Element(iel);
        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement*>(cel);
        if (!cel_inter) DebugStop();
        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());

        int64_t npts = int_rule->NPoints(); // number of integration points of the element
        int64_t dim = cel_inter->Dimension(); //dimension of the element
        int64_t nf = cel_inter->NShapeF(); // number of shape functions of the element

        TPZFMatrix<REAL> elmatrix;
        elmatrix.Resize(fRowSizes[it], fColSizes[it]);

        TPZMaterialData data;
        cel_inter->InitMaterialData(data);

        for (int64_t ipts = 0; ipts < npts; ipts++) {
            TPZVec<REAL> qsi(dim);
            REAL w;
            int_rule->Point(ipts, qsi, w);
            cel_inter->ComputeRequiredData(data, qsi);

            TPZFMatrix<REAL> axes = data.axes;
            TPZFMatrix<REAL> dphix = data.dphix;
            TPZFMatrix<REAL> dphiXY;

            axes.Transpose();
            axes.Multiply(dphix,dphiXY);

            for (int inf = 0; inf < nf; inf++) {
                for (int idim = 0; idim < dim; idim++)
                    elmatrix(ipts * dim + idim, inf) = dphiXY(idim, inf);
            }
        }
        elmatrix.Transpose(); // Using CSR format
        TPZFMatrix<REAL> elmatloc(fRowSizes[it], fColSizes[it], &fStorage[fMatrixPosition[it]], fRowSizes[it] * fColSizes[it]);
        elmatloc = elmatrix;
        it++;
    }

    CSRInfo();
}

void TPZIrregularBlockMatrix::CSRInfo() {
    int64_t nelem = fRowSizes.size();
    int64_t nnz = fStorage.size();

    fRowPtr.resize(fRow + 1); //m+1
    fColInd.resize(nnz);
    for (int iel = 0; iel < nelem; ++iel) {
        for (int irow = 0; irow < fRowSizes[iel]; ++irow) {
            fRowPtr[irow + fRowFirstIndex[iel]] = fMatrixPosition[iel] + irow*fColSizes[iel];

            for (int icol = 0; icol < fColSizes[iel]; ++icol) {
                fColInd[icol + fMatrixPosition[iel] + irow*fColSizes[iel]] = icol + fColFirstIndex[iel];
            }
        }
    }
    fRowPtr[fRow] = fMatrixPosition[nelem];
}

void TPZIrregularBlockMatrix::Multiply(TPZFMatrix<REAL> &A, TPZFMatrix<REAL> &res, REAL alpha, REAL beta, bool transpose) {
    char trans;
    char matdescra[] = {'G',' ',' ','C'};

    int Apos, respos;
    if (transpose == false) {
        trans = 'N';
        Apos = fCol;
        respos = fRow;
        res.Resize(fDim * fRow, A.Cols());
        res.Zero();
    }
    else if (transpose == true) {
        trans = 'T';
        Apos = fRow;
        respos = fCol;
        res.Resize(fDim * fCol, A.Cols());
        res.Zero();
    }

    const int m = fRow;
    const int n = A.Cols();
    const int k = fCol;

//    mkl_dcsrmm (&trans, &m, &n, &k, &alpha, matdescra, &fStorage[0], &fColInd[0], &fRowPtr[0], &fRowPtr[1], &A(0,0), &n, &beta, &res(0,0), &n);
    mkl_dcsrmv(&trans, &m, &k, &alpha, matdescra , &fStorage[0], &fColInd[0], &fRowPtr[0], &fRowPtr[1], &A(0,0) , &beta, &res(0,0));
    mkl_dcsrmv(&trans, &m, &k, &alpha, matdescra , &fStorage[0], &fColInd[0], &fRowPtr[0], &fRowPtr[1], &A(Apos,0) , &beta, &res(respos,0));
}