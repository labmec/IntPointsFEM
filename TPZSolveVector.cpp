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

void TPZSolveVector::Multiply(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const {
    int64_t n_globalsol = fIndexes.size()/2; //o vetor de indices esta duplicado
    int64_t nelem = fRowSizes.size();
    int rows = fRowSizes[0];
    int cols = fColSizes[0];

    TPZFMatrix<REAL> expandsolution(2*n_globalsol,1); //vetor solucao duplicado

    cblas_dgthr(2*n_globalsol, global_solution, &expandsolution(0,0), &fIndexes[0]);

    result.Resize(2*n_globalsol,1);
    result.Zero();

    for (int i = 0; i < cols; i++) {
        cblas_dsbmv(CblasColMajor, CblasUpper, nelem * rows / 2, 0, 1., &fStorageVec[i * nelem * rows], 1, &expandsolution(i * nelem, 0), 1, 1., &result(0,0), 1);
        cblas_dsbmv(CblasColMajor, CblasUpper, nelem * rows / 2, 0, 1., &fStorageVec[i * nelem * rows + nelem * rows / 2], 1, &expandsolution(i * nelem, 0), 1, 1., &result(nelem * rows / 2,0), 1);

        cblas_dsbmv(CblasColMajor, CblasUpper, nelem * rows / 2, 0, 1., &fStorageVec[i * nelem * rows], 1, &expandsolution(i * nelem + n_globalsol, 0), 1, 1., &result(n_globalsol,0), 1);
        cblas_dsbmv(CblasColMajor, CblasUpper, nelem * rows / 2, 0, 1., &fStorageVec[i * nelem * rows + nelem * rows / 2], 1, &expandsolution(i * nelem + n_globalsol, 0), 1, 1., &result(n_globalsol + nelem * rows / 2,0), 1);

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





