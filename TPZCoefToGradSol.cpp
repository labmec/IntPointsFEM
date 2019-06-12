//
// Created by natalia on 24/05/19.
//

#include "TPZCoefToGradSol.h"
#include "pzcmesh.h"

TPZCoefToGradSol::TPZCoefToGradSol() : fBlockMatrix(0,0), fNColor(-1), fIndexes(0), fIndexesColor(0) {
#ifdef USING_CUDA
    dIndexes.resize(0);
    dIndexesColor.resize(0);
#endif
}

TPZCoefToGradSol::TPZCoefToGradSol(TPZIrregularBlocksMatrix &irregularBlocksMatrix) : fBlockMatrix(0,0), fNColor(-1), fIndexes(0), fIndexesColor(0) {
    SetIrregularBlocksMatrix(irregularBlocksMatrix);
}

TPZCoefToGradSol::~TPZCoefToGradSol() {

}

void TPZCoefToGradSol::SetIrregularBlocksMatrix(TPZIrregularBlocksMatrix & irregularBlocksMatrix) {
    fBlockMatrix = irregularBlocksMatrix;
}

#ifdef USING_CUDA
void TPZCoefToGradSol::Multiply(TPZVecGPU<REAL> &coef, TPZVecGPU<REAL> &grad_u) {
    int dim = 2;
    int64_t rows = fBlockMatrix.Rows();
    int64_t cols = fBlockMatrix.Cols();

    TPZVecGPU<REAL> gather_solution(dim * cols);
    fCudaCalls.GatherOperation(dim * cols, coef.getData(), gather_solution.getData(), dIndexes.getData());

    grad_u.resize(dim * rows);

    // cols of gather_solution (one because it is a vector!!)
    int nblocks = fBlockMatrix.Blocks().fNumBlocks;
    TPZVec<int> one(nblocks);
    one.Fill(1);

    TPZVecGPU<int> dOne(nblocks);
    dOne.set(&one[0], nblocks);

    fBlockMatrix.Multiply(&gather_solution.getData()[0], &grad_u.getData()[0], dOne.getData(), false);
    fBlockMatrix.Multiply(&gather_solution.getData()[cols], &grad_u.getData()[rows], dOne.getData(), false);   
}
#endif 

void TPZCoefToGradSol::Multiply(TPZFMatrix<REAL> &coef, TPZFMatrix<REAL> &grad_u) {
    int dim = 2;
    int64_t rows = fBlockMatrix.Rows();
    int64_t cols = fBlockMatrix.Cols();

    TPZFMatrix<REAL> gather_solution(dim * cols, 1);
    cblas_dgthr(dim * cols, coef, &gather_solution(0, 0), &fIndexes[0]);

    grad_u.Resize(dim * rows, 1);

    // Multiply: BMatrix * gather = grad_u
    TPZFMatrix<REAL> gather_x(cols, 1, &gather_solution(0, 0), cols);
    TPZFMatrix<REAL> gather_y(cols, 1, &gather_solution(cols, 0), cols);

    TPZFMatrix<REAL> grad_u_x(rows, 1, &grad_u(0, 0), rows);
    TPZFMatrix<REAL> grad_u_y(rows, 1, &grad_u(rows, 0), rows);

    fBlockMatrix.Multiply(gather_x, grad_u_x, false);
    fBlockMatrix.Multiply(gather_y, grad_u_y, false);   
}

#ifdef USING_CUDA
void TPZCoefToGradSol::MultiplyTranspose(TPZVecGPU<REAL> &sigma, TPZVecGPU<REAL> &res) {
    int dim = 2;
    int64_t rows = fBlockMatrix.Rows();
    int64_t cols = fBlockMatrix.Cols();

    int64_t ncolor = fNColor;
    int64_t neq = res.getSize();    

    TPZVecGPU<REAL> forces(dim * cols);
    res.resize(ncolor * neq);
    res.Zero();

    // cols of sigma (one because it is a vector!!)
    int nblocks = fBlockMatrix.Blocks().fNumBlocks;
    TPZVec<int> one(nblocks);
    one.Fill(1);

    TPZVecGPU<int> dOne(nblocks);
    dOne.set(&one[0], nblocks);

    fBlockMatrix.Multiply(&sigma.getData()[0], &forces.getData()[0], dOne.getData(), true);
    fBlockMatrix.Multiply(&sigma.getData()[rows], &forces.getData()[cols], dOne.getData(), true); 

    // Assemble forces
    fCudaCalls.ScatterOperation(dim * cols, forces.getData(), res.getData(), dIndexesColor.getData());

    int64_t colorassemb = ncolor / 2.;
    while (colorassemb > 0) {

        int64_t firsteq = (ncolor - colorassemb) * neq;
        fCudaCalls.DaxpyOperation(colorassemb * neq, 1., &res.getData()[firsteq], &res.getData()[0]); 

        ncolor -= colorassemb;
        colorassemb = ncolor/2;
    }
    // res.resize(neq);
}
#endif


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
    fBlockMatrix.TransferDataToGPU();

    dIndexes.resize(fIndexes.size());
    dIndexes.set(&fIndexes[0], fIndexes.size());

    dIndexesColor.resize(fIndexesColor.size());
    dIndexesColor.set(&fIndexesColor[0], fIndexesColor.size());
#endif
}
