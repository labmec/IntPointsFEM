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
    dBlocksInfo.dStorage.resize(0);
    dBlocksInfo.dRowSizes.resize(0);
    dBlocksInfo.dColSizes.resize(0);
    dBlocksInfo.dRowFirstIndex.resize(0);
    dBlocksInfo.dColFirstIndex.resize(0);
    dBlocksInfo.dMatrixPosition.resize(0);
    dBlocksInfo.dRowPtr.resize(0);
    dBlocksInfo.dColInd.resize(0);
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
    dBlocksInfo.dStorage.resize(0);
    dBlocksInfo.dRowSizes.resize(0);
    dBlocksInfo.dColSizes.resize(0);
    dBlocksInfo.dRowFirstIndex.resize(0);
    dBlocksInfo.dColFirstIndex.resize(0);
    dBlocksInfo.dMatrixPosition.resize(0);
    dBlocksInfo.dRowPtr.resize(0);
    dBlocksInfo.dColInd.resize(0);
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
    dBlocksInfo = copy.dBlocksInfo;
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
    dBlocksInfo = copy.dBlocksInfo;
    fCudaCalls = copy.fCudaCalls;
#endif

    return *this;
}

void TPZIrregularBlocksMatrix::MultiplyVector(REAL *A, REAL *res, int opt) {
    int nblocks = fBlocksInfo.fNumBlocks;

    TPZVec<int> one(nblocks);
    one.Fill(1);

#ifdef USING_CUDA
    TPZVecGPU<int> dOne(nblocks);
    dOne.set(&one[0], nblocks);

    if(opt == 0) {
        fCudaCalls->Multiply(opt, dBlocksInfo.dRowSizes.getData(), dOne.getData(), dBlocksInfo.dColSizes.getData(), dBlocksInfo.dStorage.getData(), dBlocksInfo.dMatrixPosition.getData(), A, dBlocksInfo.dColFirstIndex.getData(), res, dBlocksInfo.dRowFirstIndex.getData(), 1., nblocks); 
    } else {
        fCudaCalls->Multiply(opt, dBlocksInfo.dColSizes.getData(), dOne.getData(), dBlocksInfo.dRowSizes.getData(), dBlocksInfo.dStorage.getData(), dBlocksInfo.dMatrixPosition.getData(), A, dBlocksInfo.dRowFirstIndex.getData(), res, dBlocksInfo.dColFirstIndex.getData(), -1., nblocks); 
    }
 #else
    if(opt == 0) {
        MatrixMultiplication(opt, &fBlocksInfo.fRowSizes[0], &one[0], &fBlocksInfo.fColSizes[0], &fBlocksInfo.fStorage[0], &fBlocksInfo.fMatrixPosition[0], A, &fBlocksInfo.fColFirstIndex[0], res, &fBlocksInfo.fRowFirstIndex[0], 1., nblocks);
    } else {
        MatrixMultiplication(opt, &fBlocksInfo.fColSizes[0], &one[0], &fBlocksInfo.fRowSizes[0], &fBlocksInfo.fStorage[0], &fBlocksInfo.fMatrixPosition[0], A, &fBlocksInfo.fRowFirstIndex[0], res, &fBlocksInfo.fColFirstIndex[0], -1., nblocks);
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

#ifdef USING_CUDA
    res.BlocksDev().dMatrixPosition.resize(nblocks + 1);
    res.BlocksDev().dMatrixPosition.set(&res.Blocks().fMatrixPosition[0], nblocks + 1);
    res.BlocksDev().dStorage.resize(res.Blocks().fMatrixPosition[nblocks]);

    if(opt == 0) {
        fCudaCalls->Multiply(opt, dBlocksInfo.dRowSizes.getData(), A.BlocksDev().dColSizes.getData(), dBlocksInfo.dColSizes.getData(), dBlocksInfo.dStorage.getData(), dBlocksInfo.dMatrixPosition.getData(), A.BlocksDev().dStorage.getData(), A.BlocksDev().dMatrixPosition.getData(), res.BlocksDev().dStorage.getData(), res.BlocksDev().dMatrixPosition.getData(), 1., nblocks);
    } else {
        fCudaCalls->Multiply(opt, dBlocksInfo.dColSizes.getData(), A.BlocksDev().dColSizes.getData(), dBlocksInfo.dRowSizes.getData(), dBlocksInfo.dStorage.getData(), dBlocksInfo.dMatrixPosition.getData(), A.BlocksDev().dStorage.getData(), A.BlocksDev().dMatrixPosition.getData(), res.BlocksDev().dStorage.getData(), res.BlocksDev().dMatrixPosition.getData(), 1., nblocks);
    }
#else
    res.Blocks().fStorage.resize(res.Blocks().fMatrixPosition[nblocks]);

    if(opt == 0) {
        MatrixMultiplication(opt, &fBlocksInfo.fRowSizes[0], &A.Blocks().fColSizes[0], &fBlocksInfo.fColSizes[0], &fBlocksInfo.fStorage[0], &fBlocksInfo.fMatrixPosition[0], &A.Blocks().fStorage[0], &A.Blocks().fMatrixPosition[0], &res.Blocks().fStorage[0], &res.Blocks().fMatrixPosition[0], 1., nblocks);
    } else {
        MatrixMultiplication(opt, &fBlocksInfo.fColSizes[0], &A.Blocks().fColSizes[0], &fBlocksInfo.fRowSizes[0], &fBlocksInfo.fStorage[0], &fBlocksInfo.fMatrixPosition[0], &A.Blocks().fStorage[0], &A.Blocks().fMatrixPosition[0], &res.Blocks().fStorage[0], &res.Blocks().fMatrixPosition[0], 1., nblocks);
    }
#endif
    res.Blocks().fRowSizes = fBlocksInfo.fRowSizes;
    res.Blocks().fColSizes = A.Blocks().fColSizes;
    res.Blocks().fRowFirstIndex = fBlocksInfo.fRowFirstIndex;
    res.Blocks().fColFirstIndex = A.Blocks().fColFirstIndex;

    res.BlocksDev().dRowSizes.resize(nblocks);
    res.BlocksDev().dColSizes.resize(nblocks);
    res.BlocksDev().dRowFirstIndex.resize(nblocks + 1);
    res.BlocksDev().dColFirstIndex.resize(nblocks + 1);

    res.BlocksDev().dRowSizes.set(&res.Blocks().fRowSizes[0], nblocks);
    res.BlocksDev().dColSizes.set(&res.Blocks().fColSizes[0], nblocks);
    res.BlocksDev().dRowFirstIndex.set(&res.Blocks().fRowFirstIndex[0], nblocks + 1);
    res.BlocksDev().dColFirstIndex.set(&res.Blocks().fColFirstIndex[0], nblocks + 1);
}

#ifdef USING_CUDA
void TPZIrregularBlocksMatrix::TransferDataToGPU() {
    dBlocksInfo.dStorage.resize(fBlocksInfo.fStorage.size());
    dBlocksInfo.dStorage.set(&fBlocksInfo.fStorage[0], fBlocksInfo.fStorage.size());

    dBlocksInfo.dRowSizes.resize(fBlocksInfo.fRowSizes.size());
    dBlocksInfo.dRowSizes.set(&fBlocksInfo.fRowSizes[0], fBlocksInfo.fRowSizes.size());

    dBlocksInfo.dColSizes.resize(fBlocksInfo.fColSizes.size());
    dBlocksInfo.dColSizes.set(&fBlocksInfo.fColSizes[0], fBlocksInfo.fColSizes.size());

    dBlocksInfo.dMatrixPosition.resize(fBlocksInfo.fMatrixPosition.size());
    dBlocksInfo.dMatrixPosition.set(&fBlocksInfo.fMatrixPosition[0], fBlocksInfo.fMatrixPosition.size());

    dBlocksInfo.dRowFirstIndex.resize(fBlocksInfo.fRowFirstIndex.size());
    dBlocksInfo.dRowFirstIndex.set(&fBlocksInfo.fRowFirstIndex[0], fBlocksInfo.fRowFirstIndex.size());

    dBlocksInfo.dColFirstIndex.resize(fBlocksInfo.fColFirstIndex.size());
    dBlocksInfo.dColFirstIndex.set(&fBlocksInfo.fColFirstIndex[0], fBlocksInfo.fColFirstIndex.size());

    dBlocksInfo.dRowPtr.resize(fBlocksInfo.fRowPtr.size());
    dBlocksInfo.dRowPtr.set(&fBlocksInfo.fRowPtr[0], fBlocksInfo.fRowPtr.size());

    dBlocksInfo.dColInd.resize(fBlocksInfo.fColInd.size());
    dBlocksInfo.dColInd.set(&fBlocksInfo.fColInd[0], fBlocksInfo.fColInd.size());
}
#endif

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