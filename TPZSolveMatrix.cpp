#include "TPZSolveMatrix.h"
#include "pzmatrix.h"
#include <mkl.h>
#include <stdlib.h>

#ifdef USING_MKL
#include <mkl.h>
#endif

TPZSolveMatrix::TPZSolveMatrix()
{
    fRows = -1;
    fCols = -1;
    fElementMatrices = 1;
    fIndexes = 1;
}

TPZSolveMatrix::~TPZSolveMatrix(){
    
}

void TPZSolveMatrix::Solve(TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result)
{
    int64_t nelem = fElementMatrices.size();
    int64_t cont = 0;
    
    MKL_INT n_globalsol_eachdirection = fIndexes.size()/2;
    MKL_INT *indexes_x = (MKL_INT*)calloc(n_globalsol_eachdirection, sizeof(MKL_INT));
    MKL_INT *indexes_y = (MKL_INT*)calloc(n_globalsol_eachdirection, sizeof(MKL_INT));
    
    for (int64_t i=0; i<n_globalsol_eachdirection; i++){
        indexes_x[i] = fIndexes[i];
        indexes_y[i] = fIndexes[n_globalsol_eachdirection+i];
    }
    
    REAL *global_solution_x = (REAL*)calloc(n_globalsol_eachdirection, sizeof(REAL));
    REAL *global_solution_y = (REAL*)calloc(n_globalsol_eachdirection, sizeof(REAL));
    
    cblas_dgthr(n_globalsol_eachdirection, global_solution, global_solution_x, indexes_x);
    cblas_dgthr(n_globalsol_eachdirection, global_solution, global_solution_y, indexes_y);
    
    result.Redim(2*fRows, 2);
    
    int64_t position_in_resultmatrix = 0;
    
    for (int64_t iel=0; iel<nelem; iel++) {
        int64_t nrows_element = fElementMatrices[iel].Rows();
        int64_t ncols_element = fElementMatrices[iel].Cols();
        
        TPZFMatrix<REAL> element_solution_x(ncols_element,1,0.);
        TPZFMatrix<REAL> element_solution_y(ncols_element,1,0.);
        
        for (int64_t j_ncols_element = 0; j_ncols_element<ncols_element; j_ncols_element++) {
            element_solution_x(j_ncols_element,0) = global_solution_x[cont];
            element_solution_y(j_ncols_element,0) = global_solution_y[cont];
            cont++;
        }
        
        TPZFMatrix<REAL> sol(nrows_element,2);
        TPZFMatrix<REAL> solx(nrows_element,2);
        TPZFMatrix<REAL> soly(nrows_element,2);
        
        solx = fElementMatrices[iel].operator*(element_solution_x);
        soly = fElementMatrices[iel].operator*(element_solution_y);
        for (int64_t jrows=0; jrows<nrows_element; jrows++) {
            sol(jrows,0) = solx(jrows,0);
            sol(jrows,1) = soly(jrows,0);
        }
        
        result.AddSub(position_in_resultmatrix, 0, sol);
        position_in_resultmatrix += sol.Rows();
        
    }
    
} 

void TPZSolveMatrix::OrderGlobalSolution (TPZFMatrix<STATE> &global_solution, TPZFMatrix<REAL> &global_solution_x, TPZFMatrix<REAL> &global_solution_y){
    
    int64_t len_indexes = fIndexes.size();
    
    for (int64_t j_ind=0; j_ind<(len_indexes/2); j_ind++) {
        
        int64_t id = fIndexes[j_ind];
        global_solution_x(j_ind,0) = global_solution(id,0);
        
        id = fIndexes[(len_indexes/2)+j_ind];
        global_solution_y(j_ind,0) = global_solution(id,0);
    }
}


