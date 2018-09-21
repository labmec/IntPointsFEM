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

void TPZSolveMatrix::HostToDevice()
{
    DebugStop();
}

void TPZSolveMatrix::SolveWithCUDA(const TPZFMatrix<STATE>  &global_solution, TPZStack<REAL> &weight, TPZFMatrix<REAL> &nodal_forces_global) const
{
    DebugStop();
}

void TPZSolveMatrix::FreeDeviceMemory()
{
    DebugStop();
}

void TPZSolveMatrix::Multiply(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const
{
int64_t nelem = fRowSizes.size();

MKL_INT n_globalsol = fIndexes.size();

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

void TPZSolveMatrix::ComputeSigma( TPZStack<REAL> &weight, TPZFMatrix<REAL> &result, TPZFMatrix<STATE> &sigma)
{
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

void TPZSolveMatrix::MultiplyTranspose(TPZFMatrix<STATE>  &intpoint_solution, TPZFMatrix<STATE> &nodal_forces_vec)
{
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

void TPZSolveMatrix::TraditionalAssemble(TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) const
{
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

void TPZSolveMatrix::ColoredAssemble(TPZCompMesh * cmesh, TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) const
{
    int nelem = fRowSizes.size();
    int nelem_c = cmesh->NElements();
    int dim_mesh = cmesh->Dimension();

    // INÍCIO DA ASSEMBLAGEM
    int64_t nnodes_tot = cmesh->Reference()->NNodes();
    TPZVec<int> nnodes_vec(nnodes_tot,0.);

    int cont_elem = 0;

    TPZVec<int> nelem_cor(nelem,-1); // vetor de cores
    TPZVec<int> elem_neighbour(100,0.);

    for (int64_t iel1=0; iel1<nelem_c; iel1++) {
        if(!cmesh->Element(iel1)) continue;
        TPZGeoEl * gel1 = cmesh->Element(iel1)->Reference();
        if(!gel1 ||  gel1->Dimension() != dim_mesh) continue;

        TPZVec<int64_t> nodeindices;
        gel1->GetNodeIndices(nodeindices); // Armazena os nós do elemento finito

        // ** Início da verificação de qual coord é repetida:
        TPZGeoEl * gel2;
        // contadores
        int64_t cont_elem_cor = 0;
        int64_t cont_elem_neighbour = 0;

        // inicializa com nnodes_vec nulo, e preenche com 1 os nós repetidos
        nnodes_vec.Fill(0);
        for (int64_t iel2=0; iel2<nelem_c; iel2++) {
            if(!cmesh->Element(iel2)) continue;
            gel2 = cmesh->Element(iel2)->Reference();
            if(!gel2 ||  gel2->Dimension() != dim_mesh) continue;

            for (int64_t inode=0; inode<gel2->NNodes(); inode++) {
                if(std::find (nodeindices.begin(), nodeindices.end(), gel2->NodeIndex(inode)) != nodeindices.end()){
                    nnodes_vec[gel2->NodeIndex(inode)] = 1; // preenchendo nnodes_vec
                    elem_neighbour[cont_elem_neighbour] = nelem_cor[cont_elem_cor]; // preenche o vetor de elementos vizinhos ao elemento de análise
                    cont_elem_neighbour++;
                }
            }
            cont_elem_cor++;
        }
        // ** fim da verificação

        // Preenche a cor
        for (int64_t inodes_tot=0; inodes_tot<nnodes_tot; inodes_tot++) {
            cont_elem_cor = cont_elem;
            if (nnodes_vec[inodes_tot] == 1){
                for (int64_t iel2=iel1; iel2<nelem_c; iel2++) {
                    if(!cmesh->Element(iel2)) continue;
                    gel2 = cmesh->Element(iel2)->Reference();
                    if(!gel2 ||  gel2->Dimension() != dim_mesh) continue;

                    gel2->GetNodeIndices(nodeindices);
                    if (std::find(nodeindices.begin(), nodeindices.end(), inodes_tot) != nodeindices.end()){
                        nelem_cor[cont_elem_cor] = 1+nelem_cor[cont_elem];
                    }
                }
            }

            // Verifica se pode ser uma cor menor
            for (int64_t icolor=0; icolor<nelem_cor[cont_elem_cor]; icolor++) {
                if (std::find(elem_neighbour.begin(), elem_neighbour.end(), icolor) == elem_neighbour.end())
                    nelem_cor[cont_elem_cor] = icolor;
                if (cont_elem==0)
                    nelem_cor[cont_elem_cor] = 0;
            }
            cont_elem_cor++;
        }
        cont_elem++;
    }
    // -----------------------------------------------------------------------
    // ASSEMBLAGEM POR COR
    int64_t ncoef = cmesh->Solution().Rows();
    int nf_tot = fCol;
    int neq = cmesh->NEquations();
    int ncolor = *std::max_element(nelem_cor.begin(), nelem_cor.end())+1;

    nodal_forces_global.Resize(ncolor*neq,1);
    int pos = 0;
    for (int icolor=0; icolor < ncolor; icolor++) {
        for (int iel = 0; iel < nelem; iel++) {
            if (icolor == nelem_cor[iel]) {
                for (int ipts = 0; ipts < fRowSizes[iel]/2; ipts++) {
                    int idx = fIndexes[iel * fRowSizes[iel]/2 + ipts];
                    int idy = fIndexes[iel * fRowSizes[iel]/2 + ipts + fRow / 2];
                    nodal_forces_global(idx + pos, 0) += nodal_forces_vec(iel * fRowSizes[iel] / 2 + ipts, 0);
                    nodal_forces_global(idy + pos, 0) += nodal_forces_vec(iel * fRowSizes[iel] / 2 + ipts + fRow / 2, 0);
                }
            }
        }
        pos += neq;
    }

    int colorid;

    if(ncolor%2 != 0 && ncolor > 1){ //se o numero de cores da malha eh impar, adiciona a ultima cor nas posicoes 0 a neq
        colorid = ncolor - 1;
        TPZFMatrix<REAL> assemblecolorid(neq, 1, &nodal_forces_global(colorid * neq, 0), neq);
        nodal_forces_global.AddSub(0, 0, assemblecolorid);
        ncolor -= 1; //menos uma cor para assemblar
    }

    double colorassemb = ncolor/2.;
    while (colorassemb >= 1) {
            for (int icolor = 0; icolor < colorassemb; icolor++) {
                colorid = icolor + colorassemb;
                TPZFMatrix<REAL> assemblecolorid(neq, 1, &nodal_forces_global(colorid * neq, 0), neq);
                nodal_forces_global.AddSub(icolor * neq, 0, assemblecolorid);
            }
            colorassemb = colorassemb / 2;
            if (std::fmod(colorassemb,2) != 0 && colorassemb > 1) { //se depois de "dobrar" o vetor, restar um numero impar de cores, adiciona ultima cor nas posicoes 0 a neq
                colorid = 2 * colorassemb - 1;
                TPZFMatrix<REAL> assemblecolorid(neq, 1, &nodal_forces_global(colorid * neq, 0), neq);
                nodal_forces_global.AddSub(0, 0, assemblecolorid);
                colorassemb = ceil(colorassemb)/2;
            }
    }
    nodal_forces_global.Resize(neq,1);
}
