//
// Created by natalia on 18/10/18.
//

#include "TPZSolveVector.h"
#include "pzmatrix.h"
#include <mkl.h>
#include <stdlib.h>

#include <chrono>
using namespace std::chrono;


#ifdef USING_MKL
#include <mkl.h>
#include <algorithm>
#endif

#ifdef USING_TBB
#include "tbb/parallel_for_each.h"
using namespace tbb;
#endif

void multvec(int n, double *a, double *b, double *c){
    for (int i = 0; i < n; i++) {
        c[i] += a[i]*b[i];
    }
}

void sigmaxx(int nelem, double *weight, double *dudx, double *dvdy, double *sigxx){
    REAL E = 200000000.;
    REAL nu =0.30;
    for (int i = 0; i < nelem; i++) {
        sigxx[i] = weight[i] * (dudx[i] * E * (1. - nu) / ((1. - 2 * nu) * (1. + nu))  + dvdy[i] * E * nu / ((1. - 2 * nu) * (1. + nu))); //plane strain
//        sigxx[i] = weight[i]*E/(1.-nu*nu)*(dudx[i] + nu*dvdy[i]);
    }
}

void sigmayy(int nelem, double *weight, double *dudx, double *dvdy, double *sigyy){
    REAL E = 200000000.;
    REAL nu =0.30;
    for (int i = 0; i < nelem; i++) {
        sigyy[i] = weight[i] * (dudx[i] * E * nu / ((1. - 2 * nu) * (1. + nu)) + dvdy[i] * E * (1. - nu) / ((1. - 2 * nu) * (1. + nu))); //plane strain
//        sigyy[i] = weight[i]*E/(1.-nu*nu)*(nu*dudx[i] + dvdy[i]);
    }
}

void sigmaxy(int nelem, double *weight, double *dvdx, double *dudy, double *sigxy, double *sigyx){
    REAL E = 200000000.;
    REAL nu =0.30;
    for (int i = 0; i < nelem; i++) {
        sigxy[i] = weight[i] * E / (2 * (1. + nu)) * (dvdx[i] + dudy[i]); //plane strain
//        sigxy[i] = weight[i]*E/(1.+nu)*(dvdx[i] + dudy[i]);
        sigyx[i] = sigxy[i];
    }
}

void TPZSolveVector::Multiply(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const {
    int64_t n_globalsol = fIndexes.size(); //o vetor de indices esta duplicado
    int64_t nelem = fRowSizes.size();
    int rows = fRowSizes[0];
    int cols = fColSizes[0];

    TPZFMatrix<REAL> expandsolution(n_globalsol,1); //vetor solucao duplicado

    cblas_dgthr(n_globalsol, global_solution, &expandsolution(0,0), &fIndexes[0]);

    TPZFMatrix<REAL> expand_x(n_globalsol/2,1,&expandsolution(0,0),n_globalsol/2);
    expand_x.Resize(n_globalsol,1);
    expand_x.AddSub(n_globalsol/2,0,expand_x);

    TPZFMatrix<REAL> expand_y(n_globalsol/2,1,&expandsolution(n_globalsol/2,0),n_globalsol/2);
    expand_y.Resize(n_globalsol,1);
    expand_y.AddSub(n_globalsol/2,0,expand_y);

    result.Resize(2*n_globalsol,1);
    result.Zero();

        for (int i = 0; i < cols; i++) {
            multvec(nelem * rows / 2, &fStorageVec[i * nelem * rows], &expand_x(i * nelem, 0), &result(0,0));
            multvec(nelem * rows / 2, &fStorageVec[i * nelem * rows + nelem * rows / 2], &expand_x(i * nelem, 0), &result(nelem * rows / 2,0));

            multvec(nelem * rows / 2, &fStorageVec[i * nelem * rows], &expand_y(i * nelem, 0), &result(n_globalsol,0));
            multvec(nelem * rows / 2, &fStorageVec[i * nelem * rows + nelem * rows / 2], &expand_y(i * nelem, 0), &result(n_globalsol + nelem * rows / 2,0));
    }
}

void TPZSolveVector::ComputeSigma( TPZVec<REAL> &weight, TPZFMatrix<REAL> &result, TPZFMatrix<STATE> &sigma) {
    int nelem = fRowSizes.size();
    int npts_el = fRow/nelem;
    sigma.Resize(2*fRow,1);
    sigma.Zero();

    for (int64_t ipts=0; ipts< npts_el/2; ipts++) {
          sigmaxx(nelem, &weight[ipts*nelem], &result(2*ipts*nelem ,0), &result((2*ipts + npts_el + 1)*nelem ,0), &sigma(2*ipts*nelem,0));
          sigmayy(nelem, &weight[ipts*nelem], &result((2*ipts + npts_el + 1)*nelem,0), &result(2*ipts*nelem,0), &sigma((2*ipts+npts_el+1)*nelem,0));
          sigmaxy(nelem, &weight[ipts*nelem], &result((2*ipts + npts_el)*nelem,0), &result((2*ipts + 1)*nelem,0), &sigma((2*ipts+1)*nelem,0), &sigma((2*ipts+npts_el)*nelem,0));

//        cblas_daxpy(nelem, nu, &result((2*ipts + npts_el + 1)*nelem ,0), 1, &aux(0,0), 1);
//        cblas_daxpy(nelem, 1, &result(2*ipts*nelem ,0), 1, &aux(0,0), 1);
//        multvec(nelem, &weight[ipts*nelem], &aux(0,0), &sigma(2*ipts*nelem,0));
//        cblas_dscal (nelem, E/(1.-nu*nu), &sigma(2*ipts*nelem,0), 1);
//        aux.Zero();
//
////        sigma yy
//        cblas_daxpy(nelem, nu, &result(2*ipts*nelem,0), 1, &aux(0,0), 1);
//        cblas_daxpy(nelem, 1, &result((2*ipts + npts_el + 1)*nelem,0), 1, &aux(0,0), 1);
//        multvec(nelem, &weight[ipts*nelem], &aux(0,0), &sigma((2*ipts+npts_el+1)*nelem,0));
//        cblas_dscal (nelem, E/(1.-nu*nu), &sigma((2*ipts+npts_el+1)*nelem,0), 1);
//        aux.Zero();
//
////        sigma xy
//        cblas_daxpy(nelem, 1, &result((2*ipts + 1)*nelem,0), 1, &aux(0,0), 1);
//        cblas_daxpy(nelem, 1, &result((2*ipts + npts_el)*nelem,0), 1, &aux(0,0), 1);
//        multvec(nelem, &weight[ipts*nelem], &aux(0,0), &sigma((2*ipts+1)*nelem,0));
//        cblas_dscal (nelem, E/(1.-nu*nu)*(1.-nu), &sigma((2*ipts+1)*nelem,0), 1);
//        aux.Zero();
//
//        cblas_daxpy(nelem, 1, &sigma((2*ipts+1)*nelem,0), 1, &sigma((2*ipts+npts_el)*nelem,0), 1);
    }
}

void TPZSolveVector::MultiplyTranspose(TPZFMatrix<STATE>  &sigma, TPZFMatrix<STATE> &nodal_forces_vec) {
    int64_t npts_tot = fRow;
    nodal_forces_vec.Resize(npts_tot,1);
    nodal_forces_vec.Zero();

    int cols = fColSizes[0];
    int rows = fRowSizes[0];
    int  nelem = fRowSizes.size();

    for (int i = 0; i < cols; i++) {
        //Fx
        multvec((cols-i)*nelem, &fStorageVec[(((cols-i)%cols)*rows + i)*nelem], &sigma(i * nelem, 0), &nodal_forces_vec(0, 0));
        multvec(i*nelem, &fStorageVec[((cols-i)%cols)*rows*nelem], &sigma(0, 0), &nodal_forces_vec(nelem*((cols - i)%cols), 0));

        multvec((cols-i)*nelem, &fStorageVec[(((cols-i)%cols)*rows + i)*nelem + cols*nelem], &sigma(i * nelem + cols*nelem, 0), &nodal_forces_vec(0, 0));
        multvec(i*nelem, &fStorageVec[((cols-i)%cols)*rows*nelem + cols*nelem], &sigma(cols*nelem, 0), &nodal_forces_vec(nelem*((cols - i)%cols), 0));

        //Fy
        multvec((cols-i)*nelem, &fStorageVec[(((cols-i)%cols)*rows + i)*nelem], &sigma(i * nelem + fRow, 0),  &nodal_forces_vec(npts_tot/2, 0));
        multvec(i*nelem, &fStorageVec[((cols-i)%cols)*rows*nelem], &sigma(fRow, 0), &nodal_forces_vec(npts_tot/2 + nelem*((cols - i)%cols), 0));

        multvec((cols-i)*nelem, &fStorageVec[(((cols-i)%cols)*rows + i)*nelem + cols*nelem], &sigma(i * nelem + cols*nelem + fRow, 0), &nodal_forces_vec(npts_tot/2, 0));
        multvec(i*nelem, &fStorageVec[((cols-i)%cols)*rows*nelem + cols*nelem], &sigma(cols*nelem + fRow, 0), &nodal_forces_vec(npts_tot/2 + nelem*((cols - i)%cols), 0));
    }




}

void TPZSolveVector::ColoredAssemble(TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) {
    int64_t ncolor = *std::max_element(fElemColor.begin(), fElemColor.end())+1;
    int64_t sz = fIndexesColor.size();
    int64_t neq = nodal_forces_global.Rows();
    nodal_forces_global.Resize(neq*ncolor,1);

    cblas_dsctr(sz, nodal_forces_vec, &fIndexesColor[0], &nodal_forces_global(0,0));

    int64_t colorassemb = ncolor / 2.;
    while (colorassemb > 0) {

        int64_t firsteq = (ncolor - colorassemb) * neq;
        cblas_daxpy(firsteq, 1., &nodal_forces_global(firsteq, 0), 1., &nodal_forces_global(0, 0), 1.);

        ncolor -= colorassemb;
        colorassemb = ncolor/2;
    }
    nodal_forces_global.Resize(neq, 1);
}


void TPZSolveVector::MultiplyCUDA(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const {
    return;
}

void TPZSolveVector::ComputeSigmaCUDA( TPZVec<REAL> &weight, TPZFMatrix<REAL> &result, TPZFMatrix<STATE> &sigma){
    return;
}

void TPZSolveVector::MultiplyTransposeCUDA(TPZFMatrix<STATE>  &intpoint_solution, TPZFMatrix<STATE> &nodal_forces_vec){
    return;
}

void TPZSolveVector::ColoredAssembleCUDA(TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global){
    return;
}

void TPZSolveVector::AllocateMemory(TPZCompMesh *cmesh){
    return;
}

void TPZSolveVector::FreeMemory(){
    return;
}

void TPZSolveVector::cuSparseHandle(){
    return;
}

void TPZSolveVector::cuBlasHandle(){
    return;
}


void TPZSolveVector::TraditionalAssemble(TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) const {
    for (int64_t ir=0; ir<fRow/2; ir++) {
        nodal_forces_global(fIndexes[ir], 0) += nodal_forces_vec(ir, 0);
        nodal_forces_global(fIndexes[ir+fRow/2], 0) += nodal_forces_vec(ir+fRow/2, 0);
    }
}

void TPZSolveVector::ColoringElements(TPZCompMesh * cmesh) const {
    int64_t nelem_c = cmesh->NElements();
    int64_t nconnects = cmesh->NConnects();
    TPZVec<int64_t> connects_vec(nconnects,0);

    int64_t contcolor = 0;
    bool needstocontinue = true;

    while (needstocontinue)
    {
        needstocontinue = false;
        for (int64_t iel = 0; iel < nelem_c; iel++) {
            TPZCompEl *cel = cmesh->Element(iel);
            if (!cel || cel->Dimension() != cmesh->Dimension()) continue;

            if (fElemColor[iel] != -1) continue;
            TPZStack<int64_t> connectlist;
            cmesh->Element(iel)->BuildConnectList(connectlist);
            int64_t ncon = connectlist.size();

            int64_t icon;
            for (icon = 0; icon < ncon; icon++) {
                if (connects_vec[connectlist[icon]] != 0) break;
            }
            if (icon != ncon) {
                needstocontinue = true;
                continue;
            }
            fElemColor[iel] = contcolor;
            //cel->Reference()->SetMaterialId(contcolor);

            for (icon = 0; icon < ncon; icon++) {
                connects_vec[connectlist[icon]] = 1;
            }
        }
        contcolor++;
        connects_vec.Fill(0);
    }

    int64_t nelem = fRowSizes.size();
    int64_t neq = cmesh->NEquations();
    for (int64_t iel = 0; iel < nelem; iel++) {
        int64_t cols = fColSizes[iel];

        for (int64_t icols = 0; icols < cols; icols++) {
            fIndexesColor[iel + icols*nelem] = fIndexes[iel + icols*nelem] + fElemColor[iel]*neq;
            fIndexesColor[iel + icols*nelem + fRow/2] = fIndexes[iel + icols*nelem + fRow/2] + fElemColor[iel]*neq;
        }
    }
}





