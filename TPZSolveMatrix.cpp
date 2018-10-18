#include "TPZSolveMatrix.h"
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

void TPZSolveMatrix::Multiply(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const {
int64_t nelem = fRowSizes.size();

int64_t n_globalsol = fIndexes.size();

result.Resize(2*n_globalsol,1);
result.Zero();

TPZVec<REAL> expandsolution(n_globalsol);

/// gather operation
cblas_dgthr(n_globalsol, global_solution, &expandsolution[0], &fIndexes[0]);

#ifdef USING_TBB
parallel_for(size_t(0),size_t(nelem),size_t(1),[&](size_t iel)
             {
                int64_t pos = fMatrixPosition[iel];
                int64_t cols = fColSizes[iel];
                int64_t rows = fRowSizes[iel];
                TPZFMatrix<REAL> elmatrix(rows,cols,&fStorage[pos],rows*cols);

                int64_t cont_cols = fColFirstIndex[iel];
                int64_t cont_rows = fRowFirstIndex[iel];

                 TPZFMatrix<REAL> element_solution_x(cols,1,&expandsolution[cont_cols],cols);
                 TPZFMatrix<REAL> element_solution_y(cols,1,&expandsolution[cont_cols+fColFirstIndex[nelem]],cols);

                TPZFMatrix<REAL> solx(rows,1,&result(cont_rows,0),rows);
                TPZFMatrix<REAL> soly(rows,1,&result(cont_rows+fRowFirstIndex[nelem],0),rows);

                elmatrix.Multiply(element_solution_x,solx);
                elmatrix.Multiply(element_solution_y,soly);
             }
             );

#else
for (int64_t iel=0; iel<nelem; iel++) {
    int64_t pos = fMatrixPosition[iel];
    int64_t cols = fColSizes[iel];
    int64_t rows = fRowSizes[iel];
    TPZFMatrix<REAL> elmatrix(rows,cols,&fStorage[pos],rows*cols);

    int64_t cont_cols = fColFirstIndex[iel];
    int64_t cont_rows = fRowFirstIndex[iel];

    TPZFMatrix<REAL> element_solution_x(cols,1,&expandsolution[cont_cols],cols);
    TPZFMatrix<REAL> element_solution_y(cols,1,&expandsolution[cont_cols+fColFirstIndex[nelem]],cols);

    TPZFMatrix<REAL> solx(rows,1,&result(cont_rows,0),rows); //du
    TPZFMatrix<REAL> soly(rows,1,&result(cont_rows+fRowFirstIndex[nelem],0),rows); //dv

    elmatrix.Multiply(element_solution_x,solx);
    elmatrix.Multiply(element_solution_y,soly);
}
#endif
}

void TPZSolveMatrix::ComputeSigma( TPZStack<REAL> &weight, TPZFMatrix<REAL> &result, TPZFMatrix<STATE> &sigma) {
REAL E = 200000000.;
REAL nu =0.30;
int npts_tot = fRow;
sigma.Resize(2*npts_tot,1);

#ifdef USING_TBB
parallel_for(size_t(0),size_t(npts_tot/2),size_t(1),[&](size_t ipts)
                      {
                            sigma(2*ipts,0) = weight[ipts]*E/(1.-nu*nu)*(result(2*ipts,0)+nu*result(2*ipts+npts_tot+1,0)); // Sigma x
                            sigma(2*ipts+1,0) = weight[ipts]*E/(1.-nu*nu)*(1.-nu)/2*(result(2*ipts+1,0)+result(2*ipts+npts_tot,0))*0.5; // Sigma xy
                            sigma(2*ipts+npts_tot,0) = sigma(2*ipts+1,0); //Sigma xy
                            sigma(2*ipts+npts_tot+1,0) = weight[ipts]*E/(1.-nu*nu)*(result(2*ipts+npts_tot+1,0)+nu*result(2*ipts,0)); // Sigma y
                      }
                      );
#else

for (int64_t ipts=0; ipts< npts_tot/2; ipts++) {
    sigma(2*ipts,0) = weight[ipts]*E/(1.-nu*nu)*(result(2*ipts,0)+nu*result(2*ipts+npts_tot+1,0)); // Sigma x
    sigma(2*ipts+1,0) = weight[ipts]*E/(1.-nu*nu)*(1.-nu)/2*(result(2*ipts+1,0)+result(2*ipts+npts_tot,0))*0.5; // Sigma xy
    sigma(2*ipts+npts_tot,0) = sigma(2*ipts+1,0); //Sigma xy
    sigma(2*ipts+npts_tot+1,0) = weight[ipts]*E/(1.-nu*nu)*(result(2*ipts+npts_tot+1,0)+nu*result(2*ipts,0)); // Sigma y
}
#endif
}

void TPZSolveMatrix::MultiplyTranspose(TPZFMatrix<STATE>  &intpoint_solution, TPZFMatrix<STATE> &nodal_forces_vec) {
int64_t nelem = fRowSizes.size();
int64_t npts_tot = fRow;
nodal_forces_vec.Resize(npts_tot,1);
nodal_forces_vec.Zero();

#ifdef USING_TBB
parallel_for(size_t(0),size_t(nelem),size_t(1),[&](size_t iel)
                  {
                        int64_t pos = fMatrixPosition[iel];
                        int64_t rows = fRowSizes[iel];
                        int64_t cols = fColSizes[iel];
                        int64_t cont_rows = fRowFirstIndex[iel];
                        int64_t cont_cols = fColFirstIndex[iel];
                        TPZFMatrix<REAL> elmatrix(rows,cols,&fStorage[pos],rows*cols);

                        // Forças nodais na direção x
                        TPZFMatrix<REAL> fvx(rows,1, &intpoint_solution(cont_rows,0),rows);
                        TPZFMatrix<STATE> nodal_forcex(cols, 1, &nodal_forces_vec(cont_cols, 0), cols);
                        elmatrix.MultAdd(fvx,nodal_forcex,nodal_forcex,1,0,1);

                        // Forças nodais na direção y
                        TPZFMatrix<REAL> fvy(rows,1, &intpoint_solution(cont_rows+npts_tot,0),rows);
                        TPZFMatrix<STATE> nodal_forcey(cols, 1, &nodal_forces_vec(cont_cols + npts_tot/2, 0), cols);
                        elmatrix.MultAdd(fvy,nodal_forcey,nodal_forcey,1,0,1);
                  }
                  );
#else
for (int64_t iel = 0; iel < nelem; iel++) {
    int64_t pos = fMatrixPosition[iel];
    int64_t rows = fRowSizes[iel];
    int64_t cols = fColSizes[iel];
    int64_t cont_rows = fRowFirstIndex[iel];
    int64_t cont_cols = fColFirstIndex[iel];
    TPZFMatrix<REAL> elmatrix(rows,cols,&fStorage[pos],rows*cols);

    // Nodal forces in x direction
    TPZFMatrix<REAL> fvx(rows,1, &intpoint_solution(cont_rows,0),rows);
    TPZFMatrix<STATE> nodal_forcex(cols, 1, &nodal_forces_vec(cont_cols, 0), cols);
    elmatrix.MultAdd(fvx,nodal_forcex,nodal_forcex,1,0,1);

    // Nodal forces in y direction
    TPZFMatrix<REAL> fvy(rows,1, &intpoint_solution(cont_rows+npts_tot,0),rows);
    TPZFMatrix<STATE> nodal_forcey(cols, 1, &nodal_forces_vec(cont_cols + npts_tot/2, 0), cols);
    elmatrix.MultAdd(fvy,nodal_forcey,nodal_forcey,1,0,1);
}
#endif
}

void TPZSolveMatrix::ColoredAssemble(TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) {
    int64_t ncolor = *std::max_element(fElemColor.begin(), fElemColor.end())+1;
    int64_t sz = fIndexes.size();
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


void TPZSolveMatrix::TraditionalAssemble(TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) const {
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

void TPZSolveMatrix::ColoringElements(TPZCompMesh * cmesh) const {
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


void TPZSolveMatrix::AllocateMemory(TPZCompMesh *cmesh){
    return;
}

void TPZSolveMatrix::FreeMemory(){
    return;
}

void TPZSolveMatrix::cuSparseHandle(){
    return;
}

void TPZSolveMatrix::cuBlasHandle(){
    return;
}

void TPZSolveMatrix::MultiplyCUDA(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const{
    return;
}

void TPZSolveMatrix::ComputeSigmaCUDA(TPZStack<REAL> &weight, TPZFMatrix<REAL> &result, TPZFMatrix<REAL> &sigma){
    return;
}

void TPZSolveMatrix::MultiplyTransposeCUDA(TPZFMatrix<STATE> &intpoint_solution, TPZFMatrix<STATE> &nodal_forces_vec){
    return;
}

void TPZSolveMatrix::ColoredAssembleCUDA(TPZFMatrix<STATE> &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global){
    return;
}


