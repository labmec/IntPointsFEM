//
//  TPBrNumericalIntegrator.hpp
//  IntPointsFEM
//
//  Created by Omar Durán on 9/5/19.
//

#include "TPBrNumericalIntegrator.h"
#include "mkl.h"

template <class T, class MEM>
TPBrNumericalIntegrator<T, MEM>::TPBrNumericalIntegrator() : fBlockMatrix(0,0), fNColor(-1), fDoFIndexes(0), fColorIndexes(0), fConstitutiveLawProcessor(){
    
}

template <class T, class MEM>
TPBrNumericalIntegrator<T, MEM>::TPBrNumericalIntegrator(TPBrIrregularBlocksMatrix &irregularBlocksMatrix) : fBlockMatrix(0,0), fNColor(-1), fDoFIndexes(0), fColorIndexes(0), fConstitutiveLawProcessor() {
    SetIrregularBlocksMatrix(irregularBlocksMatrix);
}

template <class T, class MEM>
TPBrNumericalIntegrator<T, MEM>::~TPBrNumericalIntegrator() {
    
}

template <class T, class MEM>
void TPBrNumericalIntegrator<T, MEM>::Multiply(TPZFMatrix<REAL> &coef, TPZFMatrix<REAL> &delta_strain) {
    int64_t rows = fBlockMatrix.Rows();
    int64_t cols = fBlockMatrix.Cols();

    TPZFMatrix<REAL> gather_solution(cols, 1);
    gather_solution.Zero();
    cblas_dgthr(cols, coef, &gather_solution(0, 0), &fDoFIndexes[0]);

    delta_strain.Resize(rows, 1);
    fBlockMatrix.MultiplyVector(gather_solution, &delta_strain(0,0), false);
}

template <class T, class MEM>
void TPBrNumericalIntegrator<T, MEM>::MultiplyTranspose(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &res) {
    int64_t cols = fBlockMatrix.Cols();

    int64_t ncolor = fNColor;
    int64_t neq = res.Rows();

    TPZFMatrix<REAL> forces(cols, 1);
    res.Resize(ncolor * neq, 1);
    res.Zero();

    fBlockMatrix.MultiplyVector(sigma, &forces(0,0), true);

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

template <class T, class MEM>
void TPBrNumericalIntegrator<T, MEM>::ResidualIntegration(TPZFMatrix<REAL> & solution ,TPZFMatrix<REAL> &rhs) {
    TPZFMatrix<REAL> delta_strain;
    TPZFMatrix<REAL> sigma;

    Multiply(solution, delta_strain);
    ofstream out("tpbr.txt");
    delta_strain.Print(out);
    out.flush();
    fConstitutiveLawProcessor.ComputeSigma(delta_strain, sigma);
    MultiplyTranspose(sigma, rhs); // Perform Residual integration using a global linear application B
}

template <class T, class MEM>
void TPBrNumericalIntegrator<T, MEM>::ComputeConstitutiveMatrix(TPZFMatrix<REAL> &De) {
//  @TODO : get lambda and mu from the material

    De.Zero();
    REAL lambda = fConstitutiveLawProcessor.GetPlasticModel().fER.Lambda();
    REAL mu = fConstitutiveLawProcessor.GetPlasticModel().fER.Mu();

    De(0,0) = lambda + 2.0*mu;
    De(1,1) = mu;
    De(2,2) = lambda + 2.0*mu;
    De(0,2) = lambda;
    De(2,0) = lambda;
}

template <class T, class MEM>
void TPBrNumericalIntegrator<T, MEM>::ComputeTangentMatrix(int64_t iel, TPZFMatrix<REAL> &K) {
    TPZFMatrix<REAL> De(3,3);
    ComputeConstitutiveMatrix(De);

    int n_sigma_comps = 3;
    int el_npts = fBlockMatrix.Blocks().fRowSizes[iel]/n_sigma_comps;
    int el_dofs = fBlockMatrix.Blocks().fColSizes[iel];
    int first_el_ip = fBlockMatrix.Blocks().fRowFirstIndex[iel]/n_sigma_comps;

    K.Resize(el_dofs, el_dofs);
    K.Zero();

    int pos = fBlockMatrix.Blocks().fMatrixPosition[iel];
    TPZFMatrix<STATE> Bip(n_sigma_comps,el_dofs,0.0);
    TPZFMatrix<STATE> DeBip;
    int c = 0;
    for (int ip = 0; ip < el_npts; ip++) {
        for (int i = 0; i < n_sigma_comps; i++) {
            for (int j = 0; j < el_dofs; j++) {
                Bip(i,j) = fBlockMatrix.Blocks().fStorage[pos + c];
                c++;
            }
        }

        REAL omega = fConstitutiveLawProcessor.WeightVector()[first_el_ip + ip];
        De.Multiply(Bip, DeBip);
        Bip.MultAdd(DeBip, K, K, omega, 1.0, 1);
    }
}
