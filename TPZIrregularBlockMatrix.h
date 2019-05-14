//
// Created by natalia on 14/05/19.
//
#include "pzcmesh.h"

#ifndef INTPOINTSFEM_TPZIRREGULARBLOCKMATRIX_H
#define INTPOINTSFEM_TPZIRREGULARBLOCKMATRIX_H


class TPZIrregularBlockMatrix {
public:
    TPZIrregularBlockMatrix();

    TPZIrregularBlockMatrix(TPZCompMesh *cmesh);

    ~TPZIrregularBlockMatrix();

    TPZIrregularBlockMatrix(const TPZIrregularBlockMatrix &copy);

    TPZIrregularBlockMatrix &operator=(const TPZIrregularBlockMatrix &copy);

    void SetCompMesh(TPZCompMesh *cmesh);

    void SetElementMatrix(int iel, TPZFMatrix<REAL> &elmat);

    void SetBlocks();

    void CSRInfo();

private:
    TPZCompMesh *fCmesh;
    int64_t fNumBlocks;
    TPZVec<REAL> fStorage;
    TPZVec<int> fRowSizes;
    TPZVec<int> fColSizes;
    TPZVec<int> fMatrixPosition;
    TPZVec<int> fRowFirstIndex;
    TPZVec<int> fColFirstIndex;
    int64_t fRow;
    int64_t fCol;
    TPZVec<int> fRowPtr;
    TPZVec<int> fColInd;
};


#endif //INTPOINTSFEM_TPZIRREGULARBLOCKMATRIX_H
