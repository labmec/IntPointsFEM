/**
 * @file
 * @brief Contains the TPZSolveMatrix class which implements a solution based on a matrix procedure.
 */

#ifndef TPZSolveMatrix_h
#define TPZSolveMatrix_h

#include "pzmatrix.h"

//#ifdef USING_MKL
//#include "mkl.h"
//#endif

class  TPZSolveMatrix{
    
public:
    
    /** @brief Default constructor */
    TPZSolveMatrix();
    
    /** @brief Set solve parameters
     * @param A [in] is the input matrix
     * @param coef [in] is the coeffient's matrix
     * @param ind_x [in] is the inxedes vector associated to the solution x
     * @param ind_y [in] is the inxedes vector associated to the solution y
     */
    TPZSolveMatrix(int64_t rows, int64_t cols, TPZManVector<TPZFMatrix<REAL>> &ElementMatrices, TPZVec<int64_t> &indexes)
    {
        fRows = rows;
        fCols = cols;
        fElementMatrices = ElementMatrices;
        fIndexes = indexes;
    };
    
    /** @brief Default destructor */
    ~TPZSolveMatrix();
    
    /** @brief Set solve parameters
     * @param A [in] is the input matrix
     * @param coef [in] is the coeffient's matrix
     * @param ind_x [in] is the inxedes vector associated to the solution x
     * @param ind_y [in] is the inxedes vector associated to the solution y
     */
    void SetParameters(int64_t rows, int64_t cols, TPZManVector<TPZFMatrix<REAL>> &ElementMatrices, TPZManVector<int64_t> &indexes)
    {
        fRows = rows;
        fCols = cols;
        fElementMatrices = ElementMatrices;
        fIndexes = indexes;
    };
    
    /** @brief Solve procedure */
    void Solve(TPZFMatrix<STATE>  &global_solution, TPZFMatrix<STATE> &result);
    
private:
    /** @brief Order the coef vectors
     
     * @param A [in] is the input matrix
     * @param coef [in] is the coeffient's matrix
     * @param ind_x [in] is the inxedes vector associated to the solution x
     * @param ind_y [in] is the inxedes vector associated to the solution y
     */
    void OrderGlobalSolution (TPZFMatrix<STATE> &global_solution, TPZFMatrix<REAL> &global_solution_x, TPZFMatrix<REAL> &global_solution_y);
    
    
protected:
    
    /** @brief number of rows */
    int64_t fRows;
    
    /** @brief number of columns */
    int64_t fCols;
    
    /** @brief Matrix */
    TPZManVector<TPZFMatrix<REAL>> fElementMatrices;
    
    /** @brief Indexes vector in x direction */
    TPZManVector<int64_t> fIndexes;
    
};

#endif /* TPZSolveMatrix_h */
