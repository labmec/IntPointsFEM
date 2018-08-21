#include "TPZSolveMatrix.h"
#include "pzmatrix.h"
#include <mkl.h>
#include <stdlib.h>

#ifdef USING_MKL
#include <mkl.h>
#endif

#ifdef USING_TBB
#include "tbb/parallel_for_each.h"
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

    if (result.Rows() != 2*n_globalsol) {
        DebugStop();
    }

    TPZVec<REAL> expand_solution(n_globalsol);

    /// gather operation
    cblas_dgthr(n_globalsol, global_solution, &expand_solution[0], &fIndexes[0]);

#ifdef USING_TBB
    using namespace tbb;
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

    result.Print(std::cout);
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
void TPZSolveMatrix::MultiplyTranspose(const TPZFMatrix<STATE>  &intpoint_solution, TPZFMatrix<STATE> &resid) const
{
    // -----------------------------------------------------------------------
    // CÁLCULO DAS FORÇAS NODAIS
    int64_t cont_cols=0;
    int64_t nelem = fColSize.size();
    TPZVec<int64_t> elem_vec_ids(nelem);

    // Vetor formado pela matriz de forças por elemento
    TPZFMatrix<REAL> nodal_forces_el(fRow,1);


    for (int64_t iel=0; iel<nelem; iel++) {

        int iel_rows = fRowSize[iel];
        cont_cols = fColFirstIndex[iel];
        // Forças nodais na direção x
        int64_t rows = fElementMatrices[iel].Cols();
        TPZFMatrix<STATE> nodal_forcex(rows,1,&nodal_forces_el(fRowFirstIndex[iel],0), rows);
        TPZFMatrix<REAL> fv(2*iel_rows,1,0.);
        for (int64_t ipts=0; ipts<iel_rows; ipts++) {
            fv(2*ipts,0) = intpoint_solution.GetVal(2*ipts+cont_cols,0); // Sigma x
            fv(2*ipts+1,0) = intpoint_solution.GetVal(2*ipts+cont_cols+1,0); // Sigma xy
        }
        bool transpose = true;

        fElementMatrices[iel].MultAdd(fv, nodal_forcex, nodal_forcex,1.,1.,transpose);
        //        nodal_forces_el[cont_elem].AddSub(0, 0, AdVec[cont_elem].operator*(fv));

        // Forças nodais na direção y
        for (int64_t ipts=0; ipts<iel_rows; ipts++) {
            fv(2*ipts,0) = intpoint_solution.GetVal(2*ipts+cont_cols+fRow/2,0); // Sigma xy
            fv(2*ipts+1,0) = intpoint_solution.GetVal(2*ipts+cont_cols+fRow/2,0); // Sigma y
        }
        TPZFMatrix<STATE> nodal_forcey(rows,1,&nodal_forces_el(fRowFirstIndex[iel]+fRow/2,0), rows);
        fElementMatrices[iel].MultAdd(fv, nodal_forcey, nodal_forcey,1.,1.,transpose);
        //        nodal_forces_el[cont_elem].AddSub(0, 1, AdVec[cont_elem].operator*(fv));

    }
    // -----------------------------------------------------------------------
    // ASSEMBLAGEM "TRADICIONAL"
    TPZManVector<REAL> nodal_forces_global(fRow, 0.);
    for (int64_t ir=0; ir<fRow; ir++) {
        resid(fIndexes[ir],0) += nodal_forces_el(ir,0);
    }
}


