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


TPZSolveMatrix::TPZSolveMatrix() : TPZMatrix<STATE>(), fElementMatrices(), fIndexes()
{
}

TPZSolveMatrix::~TPZSolveMatrix(){

}

void TPZSolveMatrix::Multiply(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result, int opt) const
{
    if(opt != 0) DebugStop();
    int64_t nelem = fElementMatrices.size();

    MKL_INT n_globalsol = fIndexes.size();

   result.Resize(2*n_globalsol,1);
   result.Zero();

    if (result.Rows() != 2*n_globalsol) {
        DebugStop();
    }

    TPZVec<REAL> expand_solution(n_globalsol);

    /// gather operation
    cblas_dgthr(n_globalsol, global_solution, &expand_solution[0], &fIndexes[0]);

#ifdef USING_TBB
    parallel_for(size_t(0),size_t(nelem),size_t(1),[&](size_t iel)
                 {
                     int64_t nrows_element = fElementMatrices[iel].Rows();
                     int64_t ncols_element = fElementMatrices[iel].Cols();
                     int64_t cont;

                     cont = fColFirstIndex[iel];
                     int64_t position_in_resultmatrix = fRowFirstIndex[iel];
                     TPZFMatrix<REAL> element_solution_x(ncols_element,1,&expand_solution[cont],ncols_element);
                     TPZFMatrix<REAL> element_solution_y(ncols_element,1,&expand_solution[cont+fColFirstIndex[nelem]],ncols_element);

                     TPZFMatrix<REAL> solx(nrows_element,1,&result(position_in_resultmatrix,0),nrows_element);
                     TPZFMatrix<REAL> soly(nrows_element,1,&result(position_in_resultmatrix+fCol*2,0),nrows_element);

                     fElementMatrices[iel].Multiply(element_solution_x,solx);
                     fElementMatrices[iel].Multiply(element_solution_y,soly);

                 }
                 );

#else
    for (int64_t iel=0; iel<nelem; iel++) {
        int64_t nrows_element = fElementMatrices[iel].Rows();
        int64_t ncols_element = fElementMatrices[iel].Cols();
        int64_t cont;

        cont = fColFirstIndex[iel];
        int64_t position_in_resultmatrix = fRowFirstIndex[iel];

        TPZFMatrix<REAL> element_solution_x(ncols_element,1,&expand_solution[cont],ncols_element);
        TPZFMatrix<REAL> element_solution_y(ncols_element,1,&expand_solution[cont+fColFirstIndex[nelem]],ncols_element);

        TPZFMatrix<REAL> solx(nrows_element,1,&result(position_in_resultmatrix,0),nrows_element);
        TPZFMatrix<REAL> soly(nrows_element,1,&result(position_in_resultmatrix+fCol*2,0),nrows_element);

        fElementMatrices[iel].Multiply(element_solution_x,solx);
        fElementMatrices[iel].Multiply(element_solution_y,soly);
    }
#endif
}

void TPZSolveMatrix::ComputeSigma( TPZStack<REAL> &weight, TPZFMatrix<REAL> &result, TPZFMatrix<STATE> &sigma){
    REAL E = 200000000.;
    REAL nu =0.30;
    int npts_tot = fRow;
    sigma.Resize(2*npts_tot,2);
//    sigma.Resize(3*npts_tot,1);
    sigma.Zero();

#ifdef USING_TBB
    parallel_for(size_t(0),size_t(npts_tot),size_t(1),[&](size_t ipts)
                      {
                            sigma(2*ipts,0) = weight[ipts]*E/((1.-2.*nu)*(1.+nu))*((1.-nu)*result(2*ipts,0)+nu*result(2*ipts+2*npts_tot+1,0)); // Sigma x
                            sigma(2*ipts+1,0) = weight[ipts]*2.*E/(2.*(1.+nu))*(result(2*ipts+1,0)+result(2*ipts+2*npts_tot,0))*0.5; // Sigma xy
                            sigma(2*ipts,1) = sigma(2*ipts+1,0); //Sigma xy
                            sigma(2*ipts+1,1) = weight[ipts]*E/((1.-2.*nu)*(1.+nu))*((1.-nu)*result(2*ipts+2*npts_tot+1,0)+nu*result(2*ipts,0)); // Sigma y
                      }
                      );
#else

    for (int64_t ipts=0; ipts< npts_tot; ipts++) {
        sigma(2*ipts,0) = weight[ipts]*E/((1.-2.*nu)*(1.+nu))*((1.-nu)*result(2*ipts,0)+nu*result(2*ipts+2*npts_tot+1,0)); // Sigma x
        sigma(2*ipts+1,0) = weight[ipts]*2.*E/(2.*(1.+nu))*(result(2*ipts+1,0)+result(2*ipts+2*npts_tot,0))*0.5; // Sigma xy
        sigma(2*ipts,1) = sigma(2*ipts+1,0); //Sigma xy
        sigma(2*ipts+1,1) = weight[ipts]*E/((1.-2.*nu)*(1.+nu))*((1.-nu)*result(2*ipts+2*npts_tot+1,0)+nu*result(2*ipts,0)); // Sigma y
    }
#endif

}

void TPZSolveMatrix::OrderGlobalSolution (TPZFMatrix<STATE> &global_solution, TPZFMatrix<REAL> &global_solution_x, TPZFMatrix<REAL> &global_solution_y){

    int64_t len_indexes = fIndexes.size();
    int64_t halflen = len_indexes/2;

    for (int64_t j_ind=0; j_ind<halflen; j_ind++) {

        int64_t id = fIndexes[j_ind];
        global_solution_x(j_ind,0) = global_solution(id,0);

        id = fIndexes[halflen+j_ind];
        global_solution_y(j_ind,0) = global_solution(id,0);
    }
}

/// compute the first index of each element
void TPZSolveMatrix::ComputeElementFirstIndex()
{
    int64_t nelem = fElementMatrices.size();
    fRowSize.resize(nelem);
    fColSize.resize(nelem);
    fRowFirstIndex.resize(nelem+1);
    fColFirstIndex.resize(nelem+1);
    fMatrixPosition.resize(nelem+1);
    fColFirstIndex[0] = 0;
    fRowFirstIndex[0] = 0;
    fMatrixPosition[0] = 0;

    for (MKL_INT i=0; i< fElementMatrices.size(); i++) {
        int row = fElementMatrices[i].Rows();
        int col = fElementMatrices[i].Cols();
        fColFirstIndex[i+1]= fColFirstIndex[i]+col;
        fRowFirstIndex[i+1]= fRowFirstIndex[i]+row;
        fMatrixPosition[i+1] = fMatrixPosition[i]+row*col;
        fRowSize[i] = row;
        fColSize[i] = col;
    }
}

/** @brief Multiply with the transpose matrix */
void TPZSolveMatrix::MultiplyTranspose(TPZFMatrix<STATE>  &intpoint_solution, TPZFMatrix<STATE> &nodal_forces_vec) {
    int64_t nelem = fElementMatrices.size();
    int64_t npts_tot = fRow;
    bool transpose = true;
    nodal_forces_vec.Zero();

#ifdef USING_TBB
    parallel_for(size_t(0),size_t(nelem),size_t(1),[&](size_t iel)
                      {
                            int64_t rows = fRowSize[iel];
                            int64_t cols = fColSize[iel];
                            int64_t cont_rows = fRowFirstIndex[iel];

                            TPZFMatrix<REAL> dAT = fElementMatrices[iel];
                            dAT.Transpose();

                            // Forças nodais na direção y
                            TPZFMatrix<REAL> fvx(rows,1, &intpoint_solution(cont_rows,0),rows);
                            TPZFMatrix<STATE> nodal_forcex(cols, 1, &nodal_forces_vec(cont_rows / 2, 0), cols);
                            nodal_forcex = dAT.operator*(fvx);

                            // Forças nodais na direção y
                            TPZFMatrix<REAL> fvy(rows,1, &intpoint_solution(cont_rows,1),rows);
                            TPZFMatrix<STATE> nodal_forcey(cols, 1, &nodal_forces_vec(cont_rows / 2 + npts_tot, 0), cols);
                            nodal_forcey = dAT.operator*(fvy);
                      }
                      );
#else
    for (int64_t iel = 0; iel < nelem; iel++) {
        int64_t rows = fRowSize[iel];
        int64_t cols = fColSize[iel];
        int64_t cont_rows = fRowFirstIndex[iel];

        TPZFMatrix<REAL> dAT = fElementMatrices[iel];
        dAT.Transpose();

        // Forças nodais na direção y
        TPZFMatrix<REAL> fvx(rows,1, &intpoint_solution(cont_rows,0),rows);
        TPZFMatrix<STATE> nodal_forcex(cols, 1, &nodal_forces_vec(cont_rows / 2, 0), cols);
        nodal_forcex = dAT.operator*(fvx);

        // Forças nodais na direção y
        TPZFMatrix<REAL> fvy(rows,1, &intpoint_solution(cont_rows,1),rows);
        TPZFMatrix<STATE> nodal_forcey(cols, 1, &nodal_forces_vec(cont_rows / 2 + npts_tot, 0), cols);
        nodal_forcey = dAT.operator*(fvy);
    }
#endif
}

void TPZSolveMatrix::TraditionalAssemble(TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) const
{
#ifdef USING_TBB
    parallel_for(size_t(0),size_t(2*fRow),size_t(1),[&](size_t ir)
                 {
                     nodal_forces_global(fIndexes[ir], 0) += nodal_forces_vec(ir, 0);
                 }
    );
#else
    for (int64_t ir=0; ir<2*fRow; ir++) {
        nodal_forces_global(fIndexes[ir], 0) += nodal_forces_vec(ir, 0);
    }
#endif
}

void TPZSolveMatrix::ColoredAssemble(TPZCompMesh * cmesh, TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) const
{
    int64_t nnodes_tot = cmesh->Reference()->NNodes();
    TPZManVector<REAL> nnodes_vec(nnodes_tot,0.);

    int64_t nelem_c = cmesh->NElements();
    int dim_mesh = cmesh->Reference()->Dimension();
    int64_t nelem = fElementMatrices.size();

    int cont_elem = 0;

    TPZManVector<int> nelem_cor(nelem,-1); // vetor de cores
    TPZManVector<int64_t> elem_neighbour(8,0.); // definindo que um elem pode ter no maximo 8 elem vizinhos

    for (int64_t iel1=0; iel1<nelem_c; iel1++) {
        if(!cmesh->Element(iel1)) continue;
        TPZGeoEl * gel1 = cmesh->Element(iel1)->Reference();
        if(!gel1 ||  gel1->Dimension() != dim_mesh) continue;

        TPZManVector<int64_t> nodeindices;
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
                    elem_neighbour[cont_elem_neighbour] = nelem_cor[cont_elem_cor];; // preenche o vetor de elementos vizinhos ao elemento de análise
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
            for (int64_t icor=0; icor<nelem_cor[cont_elem_cor]; icor++) {
                if (std::find(elem_neighbour.begin(), elem_neighbour.end(), icor) == elem_neighbour.end())
                    nelem_cor[cont_elem_cor] = icor;
                if (cont_elem==0)
                    nelem_cor[cont_elem_cor] = 0;
            }
            cont_elem_cor++;
        }
        cont_elem++;
    }

    // -----------------------------------------------------------------------
    // ASSEMBLAGEM POR COR
    int ncor = *std::max_element(nelem_cor.begin(), nelem_cor.end());

    int nf_tot = fCol;
    TPZManVector<int64_t> assemble_cores(nf_tot*2, 0.);
    TPZManVector<int64_t> nf_por_cor(ncor+1, 0.);
    int pos = 0;


    // VETOR CORES E NEQ
    for (int icor=0; icor<ncor+1; icor++) {
        auto it = std::find (nelem_cor.begin(), nelem_cor.end(), icor);
        while (std::find (it, nelem_cor.end(), icor) != nelem_cor.end()) {
            it = std::find (it, nelem_cor.end(), icor);
            int poscor = std::distance(nelem_cor.begin(), it);
            int cont1=0;
            int cont2=0;
            for (int iassemblecor = 0; iassemblecor<fRowSize[poscor]; iassemblecor++){
                if ((iassemblecor%2)==0) {
                    assemble_cores[iassemblecor+pos] = fIndexes[poscor*fRowSize[poscor]/2+cont1];
                    cont1++;
                }
                else {
                    assemble_cores[iassemblecor+pos] = fIndexes[poscor*fRowSize[poscor]/2+fRow+cont2];
                    cont2++;
                }

            }
            it++;
            pos += fRowSize[poscor];
            nf_por_cor[icor] += fRowSize[poscor]/2;
        }
    }

    // SE TERMINAR EM NÚMERO IMPAR DE CORES
    int64_t nf_cor = 0;
    if ( (ncor+1)%2 != 0) {

        auto it = std::find (nelem_cor.begin(), nelem_cor.end(), ncor);
        while (std::find (it, nelem_cor.end(), ncor) != nelem_cor.end()) {

            it = std::find (it, nelem_cor.end(), ncor);

            int iel = std::distance(nelem_cor.begin(), it);
            int64_t nf_el = fRowSize[iel]/2;
            nf_cor += nf_el;

            for (int64_t inf_el = 0; inf_el<nf_el*2; inf_el++) {
                int64_t id = assemble_cores[nf_tot - nf_cor + inf_el];
                if (id%2 == 0)
                    nodal_forces_global(id,0) += nodal_forces_vec(iel*fColSize[iel]/2+inf_el/2,0);
                else
                    nodal_forces_global(id,0) += nodal_forces_vec(iel*fColSize[iel]/2+nelem+inf_el/2,0);
            }
            it++;
        }
    }

    REAL ncor_to_assemble = (ncor+1)/2;
    while(ncor_to_assemble >= 0.5){
        for (int64_t cor = 2*ncor_to_assemble-1; cor>=int(ncor_to_assemble); cor--) {

            nf_cor += nf_por_cor[cor];
            auto it = std::find (nelem_cor.begin(), nelem_cor.end(), cor);
            int64_t nf_el = 0;

            while (std::find (it, nelem_cor.end(), cor) != nelem_cor.end()) {
                it = std::find (it, nelem_cor.end(), cor);
                int iel = std::distance(nelem_cor.begin(), it);

                for (int64_t inf_el = 0; inf_el<fRowSize[iel]; inf_el++) {
                    int64_t id = assemble_cores[nf_tot*2 - nf_cor*2 + nf_el*2 + inf_el];
                    if (id%2 == 0) {
                        nodal_forces_global(id, 0) += nodal_forces_vec(iel * fRowSize[iel] / 2 + inf_el / 2, 0);
                    }
                    else {
                        nodal_forces_global(id, 0) += nodal_forces_vec(iel * fRowSize[iel] / 2 + nelem * fRowSize[iel] / 2 + inf_el / 2, 0);
                    }
                }
                it++;
                nf_el += fRowSize[iel]/2;
            }
        }
        ncor_to_assemble = ncor_to_assemble/2;
    }
}