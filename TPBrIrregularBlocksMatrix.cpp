//
//  TPBrIrregularBlocksMatrix.cpp
//  IntPointsFEM
//
//  Created by Omar Durán on 9/5/19.
//

#include "TPBrIrregularBlocksMatrix.h"
#include "mkl.h"

TPBrIrregularBlocksMatrix::TPBrIrregularBlocksMatrix() : fBlocksInfo() {
    this->Resize(0,0);
    fBlocksInfo.fNumBlocks = -1;
    fBlocksInfo.fStorage.resize(0);
    fBlocksInfo.fRowSizes.resize(0);
    fBlocksInfo.fColSizes.resize(0);
    fBlocksInfo.fMatrixPosition.resize(0);
    fBlocksInfo.fRowFirstIndex.resize(0);
    fBlocksInfo.fColFirstIndex.resize(0);
}

TPBrIrregularBlocksMatrix::TPBrIrregularBlocksMatrix(const int64_t rows, const int64_t cols) : TPZMatrix<REAL>(rows,cols), fBlocksInfo() {
    fBlocksInfo.fNumBlocks = -1;
    fBlocksInfo.fStorage.resize(0);
    fBlocksInfo.fRowSizes.resize(0);
    fBlocksInfo.fColSizes.resize(0);
    fBlocksInfo.fMatrixPosition.resize(0);
    fBlocksInfo.fRowFirstIndex.resize(0);
    fBlocksInfo.fColFirstIndex.resize(0);
}

TPBrIrregularBlocksMatrix::~TPBrIrregularBlocksMatrix() {

}

TPZMatrix<REAL> * TPBrIrregularBlocksMatrix::Clone() const {
    return new TPBrIrregularBlocksMatrix(*this);
}

TPBrIrregularBlocksMatrix::TPBrIrregularBlocksMatrix(const TPBrIrregularBlocksMatrix &copy) {
    TPZMatrix<REAL>::operator=(copy);
    fBlocksInfo = copy.fBlocksInfo;
}

TPBrIrregularBlocksMatrix &TPBrIrregularBlocksMatrix::operator=(const TPBrIrregularBlocksMatrix &copy) {
    if(&copy == this){
        return *this;
    }
    TPZMatrix<REAL>::operator=(copy);
    fBlocksInfo = copy.fBlocksInfo;

    return *this;
}

void TPBrIrregularBlocksMatrix::MultiplyVector(TPZFMatrix <REAL> &A, REAL *res, int opt) {
    int nblocks = fBlocksInfo.fNumBlocks;

    TPZVec<int> one(nblocks);
    one.Fill(1);

    if(opt == 0) {
        BlocksMultiplication(opt, fBlocksInfo.fRowSizes, fBlocksInfo.fColSizes, fBlocksInfo.fStorage, fBlocksInfo.fMatrixPosition, A, fBlocksInfo.fColFirstIndex, res, fBlocksInfo.fRowFirstIndex, 1., nblocks);
    } else {
        BlocksMultiplication(opt, fBlocksInfo.fColSizes, fBlocksInfo.fRowSizes, fBlocksInfo.fStorage, fBlocksInfo.fMatrixPosition, A, fBlocksInfo.fRowFirstIndex, res, fBlocksInfo.fColFirstIndex, -1., nblocks);
        }

}

void TPBrIrregularBlocksMatrix::BlocksMultiplication(bool trans, TPZVec<int> &m, TPZVec<int> &k, TPZVec<REAL> &A, TPZVec<int> &strideA, TPZFMatrix<REAL> &B, TPZVec<int> &strideB, REAL *C, TPZVec<int> &strideC, REAL alpha, int nmatrices) {
#ifdef USING_TBB
    tbb::parallel_for(size_t(0),size_t(nmatrices),size_t(1),[&](size_t imatrix)
#else
    for (int imatrix = 0; imatrix < nmatrices; imatrix++)
#endif
    {
        int m_i = m[imatrix];
        int k_i = k[imatrix];

        int strideA_i = strideA[imatrix];
        int strideB_i = strideB[imatrix];
        int strideC_i = strideC[imatrix];

        int lda_i;
        CBLAS_TRANSPOSE transpose;
        if (trans == false) {
            lda_i = k_i;
            transpose = CblasNoTrans;
        } else {
            lda_i = m_i;
            transpose = CblasTrans;
        }

        cblas_dgemm(CblasRowMajor, transpose, CblasNoTrans, m_i, 1, k_i, alpha, &A[strideA_i], lda_i,  &B[strideB_i], 1, 0., &C[strideC_i], 1);

    }
#ifdef USING_TBB
    );
#endif
}

void TPBrIrregularBlocksMatrix::SetBlocks(struct IrregularBlocks & blocks) {
    fBlocksInfo = blocks;
}