//
// Created by natalia on 14/05/19.
//
#include "TPZIrregularBlocksMatrix.h"

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

void TPZIrregularBlocksMatrix::Multiply(TPZFMatrix<REAL> &A, TPZFMatrix<REAL> &res, int opt) {
    char trans;
    char matdescra[] = {'G',' ',' ','C'};
    REAL alpha, beta;

    if (opt == false) {
        trans = 'N';
        alpha = 1.;
    }
    else if (opt == true) {
        trans = 'T';
        alpha = -1.;
    }

    const int m = fRow;
    const int n = 1;
    const int k = fCol;
    beta = 0.;
//    mkl_dcsrmm (&trans, &m, &n, &k, &alpha, matdescra, &fStorage[0], &fColInd[0], &fRowPtr[0], &fRowPtr[1], &A(0,0), &n, &beta, &res(0,0), &n);
    mkl_dcsrmv(&trans, &m, &k, &alpha, matdescra , &fBlocksInfo.fStorage[0], &fBlocksInfo.fColInd[0], &fBlocksInfo.fRowPtr[0], &fBlocksInfo.fRowPtr[1], A, &beta, &res(0,0));
}

void TPZIrregularBlocksMatrix::Multiply(REAL *A, REAL *res, int opt) {
#ifdef USING_CUDA
    int nblocks = fBlocksInfo.fNumBlocks;

    TPZVec<int> one(nblocks);
    one.Fill(1);

    TPZVecGPU<int> dOne(nblocks);
    dOne.set(&one[0], nblocks);

    if(opt == 0) {
        fCudaCalls->Multiply(opt, dRowSizes.getData(), dOne.getData(), dColSizes.getData(), dStorage.getData(), dMatrixPosition.getData(), A, dColFirstIndex.getData(), res, dRowFirstIndex.getData(), 1., nblocks); 
    } else {
        fCudaCalls->Multiply(opt, dColSizes.getData(), dOne.getData(), dRowSizes.getData(), dStorage.getData(), dMatrixPosition.getData(), A, dRowFirstIndex.getData(), res, dColFirstIndex.getData(), -1., nblocks); 
    }
#endif
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