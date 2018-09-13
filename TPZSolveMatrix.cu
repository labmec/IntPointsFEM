#include "TPZSolveMatrix.h"
#include "pzmatrix.h"
#include <mkl.h>
#include <stdlib.h>

#ifdef USING_MKL
#include <mkl.h>
#include <algorithm>
#endif

#include<cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>


///CUDA KERNELS
__global__
void MultiplyKernel(int nelem, const double *storage, const double *expandsolution, int *rowsizes, int *colsizes, int *matrixposition, int *rowfirstindex, int *colfirstindex, double *dres)
{
int i = blockIdx.x*blockDim.x + threadIdx.x;

cublasHandle_t handle;
cublasCreate(&handle);
double alpha = 1.0;
double beta = 0.0;

//cublasDgemv(handle, CUBLAS_OP_N, rowsizes[0], colsizes[0], &alpha, &storage[0], 32, &expandsolution[0], 1, &beta, dres, 1);
}


TPZSolveMatrix::TPZSolveMatrix() : TPZMatrix<STATE>(), fStorage(), fIndexes(), fColSizes(), fRowSizes(), fMatrixPosition()
{
}

TPZSolveMatrix::~TPZSolveMatrix()
{
}

void TPZSolveMatrix::SolveWithCUDA (const TPZFMatrix<STATE> &global_solution) const
{
    if(opt != 0) DebugStop();
    int64_t nelem = fRowSizes.size();

    MKL_INT n_globalsol = fIndexes.size();

    TPZVec<REAL> expandsolution(n_globalsol);

    /// gather operation (MUDAR PARA CUBLAS)
    cblas_dgthr(n_globalsol, global_solution, &expandsolution[0], &fIndexes[0]);

    int nstorage = fStorage.size();
    int nexpandsolution = expandsolution.size();
    int nrowsizes = fRowSizes.size();
    int ncolsizes = fColSizes.size();
    int nmatrixposition = fMatrixPosition.size();
    int nrowfirstindex = fRowFirstIndex.size();
    int ncolfirstindex = fColFirstIndex.size();

    double *dfStorage;
    double *dExpandSolution;
    int *dfRowSizes;
    int *dfColSizes;
    int *dfMatrixPosition;
    int *dfRowFirstIndex;
    int *dfColFirstIndex;
    double *dres;

    cudaMalloc(&dfStorage, nstorage*sizeof(double));
    cudaMalloc(&dExpandSolution,nexpandsolution*sizeof(double));
    cudaMalloc(&dfRowSizes, nrowsizes*sizeof(int));
    cudaMalloc(&dfColSizes, ncolsizes*sizeof(int));
    cudaMalloc(&dfMatrixPosition, nmatrixposition*sizeof(int));
    cudaMalloc(&dfRowFirstIndex, nrowfirstindex*sizeof(int));
    cudaMalloc(&dfColFirstIndex, ncolfirstindex*sizeof(int));
    cudaMalloc(&dres, nrowsizes*sizeof(double));

    cudaMemcpy(dfStorage, &fStorage[0], nstorage*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dExpandSolution, &expandsolution[0], nexpandsolution*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dfRowSizes, &fRowSizes[0], nrowsizes*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dfColSizes, &fColSizes[0], ncolsizes*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dfMatrixPosition, &fMatrixPosition[0], nmatrixposition*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dfRowFirstIndex, &fRowFirstIndex[0], nrowfirstindex*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dfColFirstIndex, &fColFirstIndex[0], ncolfirstindex*sizeof(int), cudaMemcpyHostToDevice);

//dim3 dimGrid(ceil(nelem/32.0),1,1);
//dim3 dimBlock(32,1,1);

    MultiplyKernel<<<1,1>>>(nelem, dfStorage, dExpandSolution, dfRowSizes, dfColSizes, dfMatrixPosition, dfRowFirstIndex, dfColFirstIndex, dres);

    cudaDeviceSynchronize();

    cudaFree(dfStorage);
    cudaFree(dExpandSolution);
    cudaFree(dfRowSizes);
    cudaFree(dfColSizes);
    cudaFree(dfMatrixPosition);
    cudaFree(dfRowFirstIndex);
    cudaFree(dfColFirstIndex);
    cudaFree(dres);
}

void TPZSolveMatrix::Multiply(const TPZFMatrix<STATE>  &global_solution, TPZFMatrix<STATE> &result, int transpose = 0) const
{
    DebugStop();
}

void TPZSolveMatrix::ComputeSigma(TPZStack<REAL> &weight, TPZFMatrix<REAL> &result, TPZFMatrix<REAL> &sigma)
{
    DebugStop();
}

void TPZSolveMatrix::MultiplyTranspose(TPZFMatrix<STATE>  &intpoint_solution, TPZFMatrix<STATE> &nodal_forces_vec)
{
    DebugStop();
}

void TPZSolveMatrix::TraditionalAssemble(TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) const
{
    DebugStop();
}

void TPZSolveMatrix::ColoredAssemble(TPZCompMesh * cmesh, TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) const
{
    DebugStop();
}


