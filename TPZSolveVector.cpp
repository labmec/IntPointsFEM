//
// Created by natalia on 18/10/18.
//

#include "TPZSolveVector.h"
#include "pzmatrix.h"
#include <mkl.h>
#include <stdlib.h>

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

void TPZSolveVector::Multiply(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const {
    int64_t n_globalsol = fIndexes.size()/2; //o vetor de indices esta duplicado
    int64_t nelem = fRowSizes.size();
    int rows = fRowSizes[0];
    int cols = fColSizes[0];

    TPZFMatrix<REAL> expandsolution(2*n_globalsol,1); //vetor solucao duplicado

    cblas_dgthr(2*n_globalsol, global_solution, &expandsolution(0,0), &fIndexes[0]);

    result.Resize(2*n_globalsol,1);
    result.Zero();
    //Multiplicacao vetor-vetor:
    //Usando o metodo criado
        for (int i = 0; i < cols; i++) {
            multvec(nelem * rows / 2, &fStorageVec[i * nelem * rows], &expandsolution(i * nelem, 0), &result(0,0));
            multvec(nelem * rows / 2, &fStorageVec[i * nelem * rows + nelem * rows / 2], &expandsolution(i * nelem, 0), &result(nelem * rows / 2,0));

            multvec(nelem * rows / 2, &fStorageVec[i * nelem * rows], &expandsolution(i * nelem + n_globalsol, 0), &result(n_globalsol,0));
            multvec(nelem * rows / 2, &fStorageVec[i * nelem * rows + nelem * rows / 2], &expandsolution(i * nelem + n_globalsol, 0), &result(n_globalsol + nelem * rows / 2,0));
    }

    //Usando dsbmv (multiplicacao matriz-vetor com matriz de banda 0)
//    for (int i = 0; i < cols; i++) {
//        cblas_dsbmv(CblasColMajor, CblasUpper, nelem * rows / 2, 0, 1., &fStorageVec[i * nelem * rows], 1, &expandsolution(i * nelem, 0), 1, 1., &result(0,0), 1);
//        cblas_dsbmv(CblasColMajor, CblasUpper, nelem * rows / 2, 0, 1., &fStorageVec[i * nelem * rows + nelem * rows / 2], 1, &expandsolution(i * nelem, 0), 1, 1., &result(nelem * rows / 2,0), 1);
//
//        cblas_dsbmv(CblasColMajor, CblasUpper, nelem * rows / 2, 0, 1., &fStorageVec[i * nelem * rows], 1, &expandsolution(i * nelem + n_globalsol, 0), 1, 1., &result(n_globalsol,0), 1);
//        cblas_dsbmv(CblasColMajor, CblasUpper, nelem * rows / 2, 0, 1., &fStorageVec[i * nelem * rows + nelem * rows / 2], 1, &expandsolution(i * nelem + n_globalsol, 0), 1, 1., &result(n_globalsol + nelem * rows / 2,0), 1);
//
//    }
//
//    TPZVec<int64_t> solpos(rows*cols/2);
//    for (int i = 0; i < cols; i++) {
//        for (int j = 0; j < cols; j++) {
//            solpos[i*cols + j] = nelem * ((j + i) % cols);
//        }
//    }
//
    //Usando daxpy (multiplicacao escalar-vetor) obs: apenas 1 matriz para todos os elementos
//    for (int i = 0; i < rows * cols / 2; i++) {
//        cblas_daxpy(nelem, fStorageVec[i], &expandsolution(solpos[i],0), 1, &result((i%cols)*nelem,0), 1);
//        cblas_daxpy(nelem, fStorageVec[i + rows*cols/2], &expandsolution(solpos[i],0), 1, &result((i%cols)*nelem + nelem*rows/2,0), 1);
//
//        cblas_daxpy(nelem, fStorageVec[i], &expandsolution(solpos[i] + n_globalsol,0), 1, &result((i%cols)*nelem + n_globalsol,0), 1);
//        cblas_daxpy(nelem, fStorageVec[i + rows*cols/2], &expandsolution(solpos[i] + n_globalsol,0), 1, &result((i%cols)*nelem + n_globalsol + nelem*rows/2,0), 1);
//    }

}

void TPZSolveVector::ComputeSigma( TPZStack<REAL> &weight, TPZFMatrix<REAL> &result, TPZFMatrix<STATE> &sigma) {
    REAL E = 200000000.;
    REAL nu =0.30;
    int nelem = fRowSizes.size();
    int npts_el = fRow/nelem;
    sigma.Resize(2*fRow,1);

        for (int64_t ipts=0; ipts< npts_el/2; ipts++) {
            //sigma xx
            cblas_daxpy(nelem, nu, &result((2*ipts + npts_el + 1)*nelem ,0), 1, &sigma(2*ipts*nelem,0), 1);
            cblas_daxpy(nelem, 1, &result(2*ipts*nelem ,0), 1, &sigma(2*ipts*nelem,0), 1);
            cblas_dscal (nelem, weight[0]*E/(1.-nu*nu) , &sigma(2*ipts*nelem,0), 1);

            //sigma yy
            cblas_daxpy(nelem, nu, &result(2*ipts*nelem,0), 1, &sigma((2*ipts+npts_el+1)*nelem,0), 1);
            cblas_daxpy(nelem, 1, &result((2*ipts + npts_el + 1)*nelem,0), 1, &sigma((2*ipts+npts_el+1)*nelem,0), 1);
            cblas_dscal (nelem, weight[0]*E/(1.-nu*nu) , &sigma((2*ipts+npts_el+1)*nelem,0), 1);

            //sigma xy
            cblas_daxpy(nelem, 1, &result((2*ipts + 1)*nelem,0), 1, &sigma((2*ipts+1)*nelem,0), 1);
            cblas_daxpy(nelem, 1, &result((2*ipts + npts_el)*nelem,0), 1, &sigma((2*ipts+1)*nelem,0), 1);
            cblas_dscal (nelem, weight[0]*E/(1.-nu*nu)*(1.-nu), &sigma((2*ipts+1)*nelem,0), 1);

            cblas_daxpy(nelem, 1, &sigma((2*ipts+1)*nelem,0), 1, &sigma((2*ipts+npts_el)*nelem,0), 1);

    }
}

void TPZSolveVector::MultiplyTranspose(TPZFMatrix<STATE>  &sigma, TPZFMatrix<STATE> &nodal_forces_vec) {
    int64_t npts_tot = fRow;
    nodal_forces_vec.Resize(npts_tot,1);
    nodal_forces_vec.Zero();

    TPZFMatrix<REAL> sigx(fRow, 1, &sigma(0, 0), fRow);
    TPZFMatrix<REAL> sigx_2(2 * fRow);
    sigx_2.AddSub(0, 0, sigx);
    sigx_2.AddSub(fRow, 0, fRow);

    TPZFMatrix<REAL> sigy(fRow, 1, &sigma(fRow, 0), fRow);
    TPZFMatrix<REAL> sigy_2(2 * fRow);
    sigy_2.AddSub(0, 0, sigy);
    sigy_2.AddSub(fRow, 0, sigy);

    int cols = fColSizes[0];
    int rows = fRowSizes[0];
    int  nelem = fRowSizes.size();

    for (int i = 0; i < cols; i++) {
        //Fx
        cblas_dsbmv(CblasColMajor, CblasUpper, (cols-i)*nelem, 0, 1., &fStorageVec[(((cols-i)%cols)*rows + i)*nelem], 1, &sigx_2(i * nelem, 0), 1, 1., &nodal_forces_vec(0, 0), 1);
        cblas_dsbmv(CblasColMajor, CblasUpper, i*nelem, 0, 1., &fStorageVec[((cols-i)%cols)*rows*nelem], 1, &sigx_2(0, 0), 1, 1., &nodal_forces_vec(nelem*((cols - i)%cols), 0), 1);

        cblas_dsbmv(CblasColMajor, CblasUpper, (cols-i)*nelem, 0, 1., &fStorageVec[(((cols-i)%cols)*rows + i)*nelem + cols*nelem], 1, &sigx_2(i * nelem + cols*nelem, 0), 1, 1., &nodal_forces_vec(0, 0), 1);
        cblas_dsbmv(CblasColMajor, CblasUpper, i*nelem, 0, 1., &fStorageVec[((cols-i)%cols)*rows*nelem + cols*nelem], 1, &sigx_2(cols*nelem, 0), 1, 1., &nodal_forces_vec(nelem*((cols - i)%cols), 0), 1);

        //Fy
        cblas_dsbmv(CblasColMajor, CblasUpper, (cols-i)*nelem, 0, 1., &fStorageVec[(((cols-i)%cols)*rows + i)*nelem], 1, &sigy_2(i * nelem, 0), 1, 1., &nodal_forces_vec(npts_tot/2, 0), 1);
        cblas_dsbmv(CblasColMajor, CblasUpper, i*nelem, 0, 1., &fStorageVec[((cols-i)%cols)*rows*nelem], 1, &sigy_2(0, 0), 1, 1., &nodal_forces_vec(npts_tot/2 + nelem*((cols - i)%cols), 0), 1);

        cblas_dsbmv(CblasColMajor, CblasUpper, (cols-i)*nelem, 0, 1., &fStorageVec[(((cols-i)%cols)*rows + i)*nelem + cols*nelem], 1, &sigy_2(i * nelem + cols*nelem, 0), 1, 1., &nodal_forces_vec(npts_tot/2, 0), 1);
        cblas_dsbmv(CblasColMajor, CblasUpper, i*nelem, 0, 1., &fStorageVec[((cols-i)%cols)*rows*nelem + cols*nelem], 1, &sigy_2(cols*nelem, 0), 1, 1., &nodal_forces_vec(npts_tot/2 + nelem*((cols - i)%cols), 0), 1);
    }
}


void TPZSolveVector::MultiplyCUDA(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const {
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
#ifdef USING_TBB
    parallel_for(size_t(0),size_t(fRow),size_t(1),[&](size_t ir)
             {
                 nodal_forces_global(fIndexes[ir], 0) += nodal_forces_vec(ir, 0);
             }
);
#else
    for (int64_t ir=0; ir<fRow; ir++) {
        nodal_forces_global(fIndexes[ir], 0) += nodal_forces_vec(ir, 0);
    }
#endif
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
        int64_t cont_cols = fColFirstIndex[iel];

        for (int64_t icols = 0; icols < cols; icols++) {
            fIndexesColor[cont_cols + icols] = fIndexes[cont_cols + icols] + fElemColor[iel]*neq;
            fIndexesColor[cont_cols+fRow/2 + icols] = fIndexes[cont_cols + fRow/2 + icols] + fElemColor[iel]*neq;
        }
    }
}





