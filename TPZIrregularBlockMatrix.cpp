//
// Created by natalia on 14/05/19.
//
#include "pzintel.h"

#include "TPZIrregularBlockMatrix.h"

TPZIrregularBlockMatrix::TPZIrregularBlockMatrix() : fCmesh(), fNumBlocks(-1), fStorage(0), fRowSizes(0), fColSizes(0),
                                                    fMatrixPosition(0), fRowFirstIndex(0), fColFirstIndex(0),
                                                    fRow(-1), fCol(-1), fRowPtr(0), fColInd(0) {}

TPZIrregularBlockMatrix::TPZIrregularBlockMatrix(TPZCompMesh *cmesh) : fCmesh(), fNumBlocks(-1), fStorage(0), fRowSizes(0), fColSizes(0),
                                                                       fMatrixPosition(0), fRowFirstIndex(0), fColFirstIndex(0),
                                                                       fRow(-1), fCol(-1), fRowPtr(0), fColInd(0) {
    SetCompMesh(cmesh);
}

TPZIrregularBlockMatrix::~TPZIrregularBlockMatrix() {
}

TPZIrregularBlockMatrix::TPZIrregularBlockMatrix(const TPZIrregularBlockMatrix &copy) {
    fCmesh = copy.fCmesh;
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
}

TPZIrregularBlockMatrix &TPZIrregularBlockMatrix::operator=(const TPZIrregularBlockMatrix &copy) {
    if(&copy == this){
        return *this;
    }

    fCmesh = copy.fCmesh;
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

    return *this;
}

void TPZIrregularBlockMatrix::SetCompMesh(TPZCompMesh *cmesh) {
    fCmesh = cmesh;
}

void TPZIrregularBlockMatrix::SetElementMatrix(int iel, TPZFMatrix<REAL> &elmat) {
    int64_t rows = fRowSizes[iel];
    int64_t cols = fColSizes[iel];
    int64_t pos = fMatrixPosition[iel];

    TPZFMatrix<REAL> elmatloc(rows, cols, &fStorage[pos], rows*cols);
    elmatloc = elmat;
}

void TPZIrregularBlockMatrix::SetBlocks() {
    int dim_mesh = (fCmesh->Reference())->Dimension(); // Mesh dimension

    TPZStack<int64_t> cel_indexes; // Indexes of domain elements
    for (int64_t i = 0; i < fCmesh->NElements(); i++) {
        TPZCompEl *cel = fCmesh->Element(i);
        if (!cel) continue;
        TPZGeoEl *gel = fCmesh->Element(i)->Reference();
        if (!gel) continue;
        if(gel->Dimension() == dim_mesh) cel_indexes.Push(cel->Index());
    }
    if (cel_indexes.size() == 0) DebugStop();

    // RowSizes and ColSizes vectors
    fNumBlocks = cel_indexes.size();
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
    for (auto iel : cel_indexes) {
        TPZCompEl *cel = fCmesh->Element(iel);

        //Integration rule
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

        //Dphi element matrix
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
        this->SetElementMatrix(it, elmatrix);

        it++;
    }
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


