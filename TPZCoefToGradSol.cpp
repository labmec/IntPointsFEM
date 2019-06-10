//
// Created by natalia on 24/05/19.
//

#include "TPZCoefToGradSol.h"
#include "pzcmesh.h"

TPZCoefToGradSol::TPZCoefToGradSol() : fBlockMatrix(0,0), fNColor(-1), fIndexes(0), fIndexesColor(0) {
#ifdef USING_CUDA
    // fCudaCalls.Initialize();
    dStorage.resize(0);
    dRowSizes.resize(0);
    dColSizes.resize(0);
    dRowFirstIndex.resize(0);
    dColFirstIndex.resize(0);
    dMatrixPosition.resize(0);
    dIndexes.resize(0);
    dIndexesColor.resize(0);
#endif
}

TPZCoefToGradSol::TPZCoefToGradSol(TPZIrregularBlocksMatrix irregularBlocksMatrix) : fBlockMatrix(0,0), fNColor(-1), fIndexes(0), fIndexesColor(0) {
    SetIrregularBlocksMatrix(irregularBlocksMatrix);
}

TPZCoefToGradSol::~TPZCoefToGradSol() {

}

void TPZCoefToGradSol::SetIrregularBlocksMatrix(TPZIrregularBlocksMatrix & irregularBlocksMatrix) {
    fBlockMatrix = irregularBlocksMatrix;
}


void TPZCoefToGradSol::Multiply(TPZFMatrix<REAL> &coef, TPZFMatrix<REAL> &grad_u) {
    int dim = 2;
    int64_t rows = fBlockMatrix.Rows();
    int64_t cols = fBlockMatrix.Cols();
    int nelem = fBlockMatrix.Blocks().fNumBlocks;

    #ifdef USING_CUDA

    // TPZVecGPU<REAL> dcoef;
    // dcoef.set(&coef(0,0),coef.Rows());

    // TPZVecGPU<REAL> dgather_solution(dim * cols);
    // dgather_solution.fill(0., dim * cols);

    int n = nelem;
    TPZVec<int> teste(n);
    dRowSizes.get(&teste[0]);
    std::cout << "indexes: " << teste << std::endl;
    std::cout << "indexes: " << fBlockMatrix.Blocks().fRowSizes << std::endl;

    // fCudaCalls.GatherOperation(dim * cols, dcoef, dgather_solution, dIndexes);


    
    // TPZVecGPU<int> dOne;
    // dOne.fill(1, nelem);



    // TPZVecGPU<REAL> dgrad_u;
    // dgrad_u.fill(0, dim * rows);

    // fCudaCalls.Multiply(false, dRowSizes, dOne, dColSizes, dStorage, dMatrixPosition, 
    //     TPZVecGPU<REAL> B, dColFirstIndex, TPZVecGPU<REAL> C, dRowFirstIndex, 1., nelem); 
    #else
    TPZFMatrix<REAL> gather_solution(dim * cols, 1);
    grad_u.Resize(dim * rows, 1);

    // Gather operation
    cblas_dgthr(dim * cols, coef, &gather_solution(0, 0), &fIndexes[0]);

    // Multiply: BMatrix * gather = grad_u
    TPZFMatrix<REAL> gather_x(cols, 1, &gather_solution(0, 0), cols);
    TPZFMatrix<REAL> gather_y(cols, 1, &gather_solution(cols, 0), cols);

    TPZFMatrix<REAL> grad_u_x(rows, 1, &grad_u(0, 0), rows);
    TPZFMatrix<REAL> grad_u_y(rows, 1, &grad_u(rows, 0), rows);

    fBlockMatrix.Multiply(gather_x, grad_u_x, false);
    fBlockMatrix.Multiply(gather_y, grad_u_y, false);
    #endif
    
}

void TPZCoefToGradSol::MultiplyTranspose(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &res) {
    int dim = 2;
    int64_t rows = fBlockMatrix.Rows();
    int64_t cols = fBlockMatrix.Cols();

    int64_t ncolor = fNColor;
    int64_t neq = res.Rows();

    TPZFMatrix<REAL> forces(dim * cols, 1);
    res.Resize(ncolor * neq, 1);
    res.Zero();

    // Multiply: transpose(BMatrix) * sigma = forces
    TPZFMatrix<REAL> sigma_x(rows, 1, &sigma(0, 0), rows);
    TPZFMatrix<REAL> sigma_y(rows, 1, &sigma(rows, 0), rows);

    TPZFMatrix<REAL> forces_x(cols, 1, &forces(0, 0), cols);
    TPZFMatrix<REAL> forces_y(cols, 1, &forces(cols, 0), cols);

    fBlockMatrix.Multiply(sigma_x, forces_x, true);
    fBlockMatrix.Multiply(sigma_y, forces_y, true);

    // Assemble forces
    cblas_dsctr(dim * cols, forces, &fIndexesColor[0], &res(0,0));

    int64_t colorassemb = ncolor / 2.;
    while (colorassemb > 0) {

        int64_t firsteq = (ncolor - colorassemb) * neq;
        cblas_daxpy(colorassemb * neq, 1., &res(firsteq, 0), 1., &res(0, 0), 1.);

        ncolor -= colorassemb;
        colorassemb = ncolor/2;
    }
    res.Resize(neq, 1);
}

void TPZCoefToGradSol::TransferDataToGPU() {
    #ifdef USING_CUDA
    dStorage.resize(fBlockMatrix.Blocks().fStorage.size());
    dStorage.set(&fBlockMatrix.Blocks().fStorage[0]);

    dRowSizes.resize(fBlockMatrix.Blocks().fRowSizes.size());
    dRowSizes.set(&fBlockMatrix.Blocks().fRowSizes[0]);

    dColSizes.resize(fBlockMatrix.Blocks().fColSizes.size());
    dColSizes.set(&fBlockMatrix.Blocks().fColSizes[0]);

    dMatrixPosition.resize(fBlockMatrix.Blocks().fMatrixPosition.size());
    dMatrixPosition.set(&fBlockMatrix.Blocks().fMatrixPosition[0]);

    dRowFirstIndex.resize(fBlockMatrix.Blocks().fRowFirstIndex.size());
    dRowFirstIndex.set(&fBlockMatrix.Blocks().fRowFirstIndex[0]);

    dColFirstIndex.resize(fBlockMatrix.Blocks().fColFirstIndex.size());
    dColFirstIndex.set(&fBlockMatrix.Blocks().fColFirstIndex[0]);

    dIndexes.resize(fIndexes.size());
    dIndexes.set(&fIndexes[0]);

    dIndexesColor.resize(fIndexesColor.size());
    dIndexesColor.set(&fIndexesColor[0]);
    #endif
}
