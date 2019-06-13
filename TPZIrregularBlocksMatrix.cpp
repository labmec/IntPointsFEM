//
// Created by natalia on 14/05/19.
//
#include "TPZIrregularBlocksMatrix.h"

#include "MatMul.h"

#ifdef USING_MKL
#include <mkl.h>
#endif

TPZIrregularBlocksMatrix::TPZIrregularBlocksMatrix() : TPZMatrix<REAL>(), fBlocksInfo() {

#ifdef USING_CUDA
    dStorage.resize(0);
    dRowSizes.resize(0);
    dColSizes.resize(0);
    dRowFirstIndex.resize(0);
    dColFirstIndex.resize(0);
    dMatrixPosition.resize(0);
    fCudaCalls = new TPZCudaCalls();
#endif
    this->Resize(0,0);
    fBlocksInfo.fNumBlocks = -1;
    fBlocksInfo.fStorage.resize(0);
    fBlocksInfo.fRowSizes.resize(0);
    fBlocksInfo.fColSizes.resize(0);
    fBlocksInfo.fMatrixPosition.resize(0);
    fBlocksInfo.fRowFirstIndex.resize(0);
    fBlocksInfo.fColFirstIndex.resize(0);
    fBlocksInfo.fRowPtr.resize(0);
    fBlocksInfo.fColInd.resize(0);
}

TPZIrregularBlocksMatrix::TPZIrregularBlocksMatrix(const int64_t rows,const int64_t cols) : TPZMatrix<REAL>(rows,cols), fBlocksInfo() {
#ifdef USING_CUDA
    dStorage.resize(0);
    dRowSizes.resize(0);
    dColSizes.resize(0);
    dRowFirstIndex.resize(0);
    dColFirstIndex.resize(0);
    dMatrixPosition.resize(0);
    fCudaCalls = new TPZCudaCalls();
#endif
    fBlocksInfo.fNumBlocks = -1;
    fBlocksInfo.fStorage.resize(0);
    fBlocksInfo.fRowSizes.resize(0);
    fBlocksInfo.fColSizes.resize(0);
    fBlocksInfo.fMatrixPosition.resize(0);
    fBlocksInfo.fRowFirstIndex.resize(0);
    fBlocksInfo.fColFirstIndex.resize(0);
    fBlocksInfo.fRowPtr.resize(0);
    fBlocksInfo.fColInd.resize(0);
}

TPZIrregularBlocksMatrix::~TPZIrregularBlocksMatrix() {
}

TPZIrregularBlocksMatrix::TPZIrregularBlocksMatrix(const TPZIrregularBlocksMatrix &copy) {
    TPZMatrix<REAL>::operator=(copy);
    fBlocksInfo = copy.fBlocksInfo;

#ifdef USING_CUDA
    dStorage = copy.dStorage;
    dRowSizes = copy.dRowSizes;
    dColSizes = copy.dColSizes;
    dRowFirstIndex = copy.dRowFirstIndex;
    dColFirstIndex = copy.dColFirstIndex;
    dMatrixPosition = copy.dMatrixPosition;
    fCudaCalls = copy.fCudaCalls;
#endif
}

TPZIrregularBlocksMatrix &TPZIrregularBlocksMatrix::operator=(const TPZIrregularBlocksMatrix &copy) {
    if(&copy == this){
        return *this;
    }
    TPZMatrix<REAL>::operator=(copy);
    fBlocksInfo = copy.fBlocksInfo;

#ifdef USING_CUDA
    dStorage = copy.dStorage;
    dRowSizes = copy.dRowSizes;
    dColSizes = copy.dColSizes;
    dRowFirstIndex = copy.dRowFirstIndex;
    dColFirstIndex = copy.dColFirstIndex;
    dMatrixPosition = copy.dMatrixPosition;
    fCudaCalls = copy.fCudaCalls;
#endif

    return *this;
}

void TPZIrregularBlocksMatrix::Multiply(TPZFMatrix<REAL> &A, TPZFMatrix<REAL> &res, TPZVec<int> ColsA, int opt) {
    int nblocks = fBlocksInfo.fNumBlocks;

    if(opt == 0) {
        MatrixMultiplication(opt, &fBlocksInfo.fRowSizes[0], &ColsA[0], &fBlocksInfo.fColSizes[0], &fBlocksInfo.fStorage[0], &fBlocksInfo.fMatrixPosition[0], &A(0,0), &fBlocksInfo.fColFirstIndex[0], &res(0,0), &fBlocksInfo.fRowFirstIndex[0], 1., nblocks);
    } else {
        MatrixMultiplication(opt, &fBlocksInfo.fColSizes[0], &ColsA[0], &fBlocksInfo.fRowSizes[0], &fBlocksInfo.fStorage[0], &fBlocksInfo.fMatrixPosition[0], &A(0,0), &fBlocksInfo.fRowFirstIndex[0], &res(0,0), &fBlocksInfo.fColFirstIndex[0], -1., nblocks);
    }
}

void TPZIrregularBlocksMatrix::Multiply(REAL *A, REAL *res, int *ColsA, int opt) {
#ifdef USING_CUDA
    int nblocks = fBlocksInfo.fNumBlocks;

    if(opt == 0) {
        fCudaCalls->Multiply(opt, dRowSizes.getData(), ColsA, dColSizes.getData(), dStorage.getData(), dMatrixPosition.getData(), A, dColFirstIndex.getData(), res, dRowFirstIndex.getData(), 1., nblocks); 
    } else {
        fCudaCalls->Multiply(opt, dColSizes.getData(), ColsA, dRowSizes.getData(), dStorage.getData(), dMatrixPosition.getData(), A, dRowFirstIndex.getData(), res, dColFirstIndex.getData(), -1., nblocks); 
    }
#endif
}

void TPZIrregularBlocksMatrix::MultiplyMatrix(TPZIrregularBlocksMatrix &A, TPZIrregularBlocksMatrix &res, int opt) {
    int nblocks = fBlocksInfo.fNumBlocks;

    res.Blocks().fMatrixPosition.resize(nblocks + 1);
    res.Blocks().fMatrixPosition[0] = 0;
    if (opt == 0) {
        for (int i = 0; i < nblocks; ++i) {
            res.Blocks().fMatrixPosition[i + 1] = res.Blocks().fMatrixPosition[i] + fBlocksInfo.fRowSizes[i] * A.Blocks().fColSizes[i];
        }
    } else {
        for (int i = 0; i < nblocks; ++i) {
            res.Blocks().fMatrixPosition[i + 1] = res.Blocks().fMatrixPosition[i] + fBlocksInfo.fColSizes[i] * A.Blocks().fColSizes[i];
        }
    }

    res.Blocks().fStorage.resize(res.Blocks().fMatrixPosition[nblocks]);

    if(opt == 0) {
        MatrixMultiplication(opt, &fBlocksInfo.fRowSizes[0], &A.Blocks().fColSizes[0], &fBlocksInfo.fColSizes[0], &fBlocksInfo.fStorage[0], &fBlocksInfo.fMatrixPosition[0], &A.Blocks().fStorage[0], &A.Blocks().fMatrixPosition[0], &res.Blocks().fStorage[0], &res.Blocks().fMatrixPosition[0], 1., nblocks);
    } else {
        MatrixMultiplication(opt, &fBlocksInfo.fColSizes[0], &A.Blocks().fColSizes[0], &fBlocksInfo.fRowSizes[0], &fBlocksInfo.fStorage[0], &fBlocksInfo.fMatrixPosition[0], &A.Blocks().fStorage[0], &A.Blocks().fMatrixPosition[0], &res.Blocks().fStorage[0], &res.Blocks().fMatrixPosition[0], 1., nblocks);
    }
    res.Blocks().fRowSizes = fBlocksInfo.fRowSizes;
    res.Blocks().fColSizes = A.Blocks().fColSizes;
    res.Blocks().fRowFirstIndex = fBlocksInfo.fRowFirstIndex;
    res.Blocks().fColFirstIndex = A.Blocks().fColFirstIndex;
}

void TPZIrregularBlocksMatrix::TransferDataToGPU() {
#ifdef USING_CUDA
    dStorage.resize(fBlocksInfo.fStorage.size());
    dStorage.set(&fBlocksInfo.fStorage[0], fBlocksInfo.fStorage.size());

    dRowSizes.resize(fBlocksInfo.fRowSizes.size());
    dRowSizes.set(&fBlocksInfo.fRowSizes[0], fBlocksInfo.fRowSizes.size());

    dColSizes.resize(fBlocksInfo.fColSizes.size());
    dColSizes.set(&fBlocksInfo.fColSizes[0], fBlocksInfo.fColSizes.size());

    dMatrixPosition.resize(fBlocksInfo.fMatrixPosition.size());
    dMatrixPosition.set(&fBlocksInfo.fMatrixPosition[0], fBlocksInfo.fMatrixPosition.size());

    dRowFirstIndex.resize(fBlocksInfo.fRowFirstIndex.size());
    dRowFirstIndex.set(&fBlocksInfo.fRowFirstIndex[0], fBlocksInfo.fRowFirstIndex.size());

    dColFirstIndex.resize(fBlocksInfo.fColFirstIndex.size());
    dColFirstIndex.set(&fBlocksInfo.fColFirstIndex[0], fBlocksInfo.fColFirstIndex.size());
#endif
}

void TPZIrregularBlocksMatrix::CSRVectors() {
    int64_t nblocks = fBlocksInfo.fNumBlocks;
    int64_t rows = fBlocksInfo.fRowFirstIndex[nblocks];

    fBlocksInfo.fRowPtr.resize(rows + 1);
    fBlocksInfo.fColInd.resize(fBlocksInfo.fMatrixPosition[nblocks]);

    for (int iel = 0; iel < nblocks; ++iel) {
        for (int irow = 0; irow < fBlocksInfo.fRowSizes[iel]; ++irow) {
            fBlocksInfo.fRowPtr[irow + fBlocksInfo.fRowFirstIndex[iel]] = fBlocksInfo.fMatrixPosition[iel] + irow * fBlocksInfo.fColSizes[iel];

            for (int icol = 0; icol < fBlocksInfo.fColSizes[iel]; ++icol) {
                fBlocksInfo.fColInd[icol + fBlocksInfo.fMatrixPosition[iel] + irow * fBlocksInfo.fColSizes[iel]] = icol + fBlocksInfo.fColFirstIndex[iel];
            }
        }
    }
    fBlocksInfo.fRowPtr[rows] = fBlocksInfo.fMatrixPosition[nblocks];
}