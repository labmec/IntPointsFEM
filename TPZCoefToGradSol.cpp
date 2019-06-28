//
// Created by natalia on 24/05/19.
//

#include "TPZCoefToGradSol.h"
#include "pzcmesh.h"

TPZCoefToGradSol::TPZCoefToGradSol() : fBlockMatrix(0,0), fNColor(-1), fDoFIndexes(0), fColorIndexes(0), fConstitutiveLawProcessor() {
#ifdef USING_CUDA
    dIndexes.resize(0);
    dIndexesColor.resize(0);
#endif
}

TPZCoefToGradSol::TPZCoefToGradSol(TPZIrregularBlocksMatrix &irregularBlocksMatrix) : fBlockMatrix(0,0), fNColor(-1), fDoFIndexes(0), fColorIndexes(0), fConstitutiveLawProcessor() {
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

    fBlockMatrix.MultiplyVector(&gather_solution.getData()[0], &grad_u.getData()[0], false);
    fBlockMatrix.MultiplyVector(&gather_solution.getData()[cols], &grad_u.getData()[rows], false);   
}
#endif 

void TPZCoefToGradSol::Multiply(TPZFMatrix<REAL> &coef, TPZFMatrix<REAL> &delta_strain) {

    int64_t rows = fBlockMatrix.Rows();
    int64_t cols = fBlockMatrix.Cols();

    TPZFMatrix<REAL> gather_solution(rows, 1);
    gather_solution.Zero();
    cblas_dgthr(cols, coef, &gather_solution(0, 0), &fDoFIndexes[0]);
    
//    gather_solution.Print("u = ",std::cout,EMathematicaInput);

    delta_strain.Resize(rows, 1);
    fBlockMatrix.MultiplyVector(&gather_solution(0,0), &delta_strain(0, 0), false);
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

    fBlockMatrix.MultiplyVector(&sigma.getData()[0], &forces.getData()[0], true);
    fBlockMatrix.MultiplyVector(&sigma.getData()[rows], &forces.getData()[cols], true); 

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

void TPZCoefToGradSol::ResidualIntegration(TPZFMatrix<REAL> & solution ,TPZFMatrix<REAL> &rhs)
{
    TPZFMatrix<REAL> delta_strain;
    TPZFMatrix<REAL> sigma;
    Multiply(solution, delta_strain);
    fConstitutiveLawProcessor.ComputeSigma(delta_strain, sigma);
    MultiplyTranspose(sigma, rhs); // Perform Residual integration using a global linear application B
}

void TPZCoefToGradSol::MultiplyTranspose(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &res) {
    int64_t cols = fBlockMatrix.Cols();

    int64_t ncolor = fNColor;
    int64_t neq = res.Rows();

    TPZFMatrix<REAL> forces(cols, 1);
    res.Resize(ncolor * neq, 1);
    res.Zero();

    fBlockMatrix.MultiplyVector(&sigma(0, 0), &forces(0, 0), true);

//    forces.Print("F = ",std::cout, EMathematicaInput);

    // Assemble forces
    cblas_dsctr(cols, forces, &fColorIndexes[0], &res(0,0));

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


void TPZCoefToGradSol::ComputeConstitutiveMatrix(int64_t point_index, TPZFMatrix<STATE> &De){
    
    De.Zero();
    REAL lambda = 555.555555555556;
    REAL mu = 833.333333333333;
    
    De(0,0) = lambda + 2.0*mu;
    De(1,1) = mu;
    De(2,2) = lambda + 2.0*mu;
    De(0,2) = lambda;
    De(2,0) = lambda;
}

void TPZCoefToGradSol::ComputeTangetMatrix(int64_t iel, TPZFMatrix<REAL> &K){
    
    int n_sigma_comps = 3;
    int el_npts = fBlockMatrix.Blocks().fRowSizes[iel]/n_sigma_comps;
    int el_dofs = fBlockMatrix.Blocks().fColSizes[iel];
    int first_el_ip = fBlockMatrix.Blocks().fRowFirstIndex[iel]/n_sigma_comps;
    
    K.Resize(el_dofs, el_dofs);
    K.Zero();

    int pos = fBlockMatrix.Blocks().fMatrixPosition[iel];
    TPZFMatrix<STATE> De(3,3);
    int c = 0;
    for (int ip = 0; ip < el_npts; ip++) {
    
        TPZFMatrix<STATE> Bip(n_sigma_comps,el_dofs,0.0);
        for (int i = 0; i < n_sigma_comps; i++) {
            for (int j = 0; j < el_dofs; j++) {
                REAL val  = fBlockMatrix.Blocks().fStorage[pos + c];
                Bip(i,j) = val;
                c++;
            }
        }
        
        REAL omega = fConstitutiveLawProcessor.fWeight[first_el_ip + ip];
        ComputeConstitutiveMatrix(ip,De);
        TPZFMatrix<STATE> DeBip;
        De.Multiply(Bip, DeBip);
        Bip.Transpose();
        Bip.MultAdd(DeBip, K, K, omega, 1.0);
    }
}

void TPZCoefToGradSol::SetConstitutiveLawProcessor(TPZConstitutiveLawProcessor & processor){
    fConstitutiveLawProcessor = processor;
}

TPZConstitutiveLawProcessor & TPZCoefToGradSol::ConstitutiveLawProcessor(){
    return fConstitutiveLawProcessor;
}
