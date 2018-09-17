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
#include <cusparse.h>

///CUDA KERNELS--------------------------------------
__global__ void MultiplyKernel()
{

}

__global__ void ComputeSigmaKernel(int npts_tot, double *weight, double *result, double *sigma)
{
REAL E = 200000000.;
REAL nu = 0.30;
int ipts = blockIdx.x*blockDim.x + threadIdx.x;

if (ipts < npts_tot/2) {
	sigma[2*ipts] = weight[ipts]*E/(1.-nu*nu)*(result[2*ipts]+nu*result[2*ipts+npts_tot+1]); // Sigma x
        sigma[2*ipts+1] = weight[ipts]*E/(1.-nu*nu)*(1.-nu)/2*(result[2*ipts+1]+result[2*ipts+npts_tot])*0.5; // Sigma xy
        sigma[2*ipts+npts_tot] = sigma[2*ipts+1]; //Sigma xy
        sigma[2*ipts+npts_tot+1] = weight[ipts]*E/(1.-nu*nu)*(result[2*ipts+npts_tot+1]+nu*result[2*ipts]); // Sigma y
}
}

__global__ void MultiplyTransposeKernel()
{

}

__global__ void AssembleKernel(int npts, int *indexes, double *nfvec, double *nfglob)
{
int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < npts) {
        nfglob[indexes[i]] = nfglob[indexes[i]] + nfvec[i];
    }

}
///--------------------------------------------------


void TPZSolveMatrix::HostToDevice()
{
int nstorage = fStorage.size();
int nrowsizes = fRowSizes.size();
int ncolsizes = fColSizes.size();
int nindexes = fIndexes.size();
int nmatrixposition = fMatrixPosition.size();
int nrowfirstindex = fRowFirstIndex.size();
int ncolfirstindex = fColFirstIndex.size();

cudaMalloc(&dfStorage, nstorage*sizeof(double));
cudaMalloc(&dfRowSizes, nrowsizes*sizeof(int));
cudaMalloc(&dfColSizes, ncolsizes*sizeof(int));
cudaMalloc(&dfIndexes, nindexes*sizeof(int));
cudaMalloc(&dfMatrixPosition, nmatrixposition*sizeof(int));
cudaMalloc(&dfRowFirstIndex, nrowfirstindex*sizeof(int));
cudaMalloc(&dfColFirstIndex, ncolfirstindex*sizeof(int));

cudaMemcpy(dfStorage, &fStorage[0], nstorage*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(dfRowSizes, &fRowSizes[0], nrowsizes*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(dfColSizes, &fColSizes[0], ncolsizes*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(dfIndexes, &fIndexes[0], nindexes*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(dfMatrixPosition, &fMatrixPosition[0], nmatrixposition*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(dfRowFirstIndex, &fRowFirstIndex[0], nrowfirstindex*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(dfColFirstIndex, &fColFirstIndex[0], ncolfirstindex*sizeof(int), cudaMemcpyHostToDevice);

}

void TPZSolveMatrix::FreeDeviceMemory()
{
cudaFree(dfStorage);
cudaFree(dfRowSizes);
cudaFree(dfColSizes);
cudaFree(dfIndexes);
cudaFree(dfMatrixPosition);
cudaFree(dfRowFirstIndex);
cudaFree(dfColFirstIndex);
}

void TPZSolveMatrix::SolveWithCUDA(const TPZFMatrix<STATE>  &global_solution, TPZStack<REAL> &weight, TPZFMatrix<REAL> &nodal_forces_global) const
{
int64_t nelem = fRowSizes.size();

///GATHER OPERATION------------------------------------------------
MKL_INT n_globalsol = fIndexes.size();
TPZVec<REAL> expandsolution(n_globalsol);
cblas_dgthr(n_globalsol, global_solution, &expandsolution[0], &fIndexes[0]); //USAR O METODO DA CUSPARSE
///----------------------------------------------------------------

///MULTIPLY--------------------------------------------------------
///Initialize result and expandsolution on device
double *dresult;
cudaMalloc(&dresult, 2*n_globalsol*sizeof(double));

double *dexpandsolution;
cudaMalloc(&dexpandsolution, n_globalsol*sizeof(double));
cudaMemcpy(dexpandsolution, &expandsolution[0], n_globalsol*sizeof(double), cudaMemcpyHostToDevice);

///Use CUBLAS library to do the multiplication
cublasHandle_t handle_m;
cublasCreate(&handle_m);
double alpha_m = 1.0;
double beta_m = 0.0;

for (int iel = 0; iel < nelem; iel++){
	int64_t pos = fMatrixPosition[iel];
	int64_t cols = fColSizes[iel];
	int64_t rows = fRowSizes[iel];

	int64_t cont_cols = fColFirstIndex[iel];
	int64_t cont_rows = fRowFirstIndex[iel];

	//du
	cublasDgemv(handle_m, CUBLAS_OP_N, rows, cols, &alpha_m, &dfStorage[pos], rows, &dexpandsolution[cont_cols], 1, &beta_m, &dresult[cont_rows], 1);

	//dv   
	cublasDgemv(handle_m, CUBLAS_OP_N, rows, cols, &alpha_m, &dfStorage[pos], rows, &dexpandsolution[cont_cols + fColFirstIndex[nelem]], 1, &beta_m, &dresult[cont_rows + fRowFirstIndex[nelem]], 1);

}

///Free device memory
cudaFree(dexpandsolution); 
cublasDestroy(handle_m);
///----------------------------------------------------------------

///COMPUTE SIGMA---------------------------------------------------
///Initialize sigma and weight on device
int npts_tot = fRow;
TPZVec<REAL> sigma (2*npts_tot);
double *dsigma;
cudaMalloc(&dsigma, 2*npts_tot*sizeof(double));

double *dweight;
cudaMalloc(&dweight, npts_tot/2*sizeof(double));
cudaMemcpy(dweight, &weight[0], npts_tot/2*sizeof(double), cudaMemcpyHostToDevice);

///Kernel that calculates sigma
dim3 dimGrid(ceil((npts_tot/2)/32.0),1,1);
dim3 dimBlock(32,1,1);
ComputeSigmaKernel<<<dimGrid,dimBlock>>>(npts_tot, dweight, dresult, dsigma);
cudaDeviceSynchronize();

///Free device memory
cudaFree(dresult);
cudaFree(dweight);
///----------------------------------------------------------------

///MULTIPLY TRANSPOSE----------------------------------------------
///Initialize nodal forces vector on device
TPZVec<REAL> nodal_forces_vec(npts_tot);
double *dnodal_forces_vec;
cudaMalloc(&dnodal_forces_vec, npts_tot*sizeof(double));

///Use CUBLAS to do the multiplication
cublasHandle_t handle_mt;
cublasCreate(&handle_mt);
double alpha_mt = 1.0;
double beta_mt = 0.0;

for (int64_t iel = 0; iel < nelem; iel++) {
	int64_t pos = fMatrixPosition[iel];
        int64_t rows = fRowSizes[iel];
        int64_t cols = fColSizes[iel];
       
	int64_t cont_rows = fRowFirstIndex[iel];
        int64_t cont_cols = fColFirstIndex[iel];

	//Nodal forces in x direction
        cublasDgemv(handle_mt, CUBLAS_OP_T, rows, cols, &alpha_mt, &dfStorage[pos], rows, &dsigma[cont_rows], 1, &beta_mt, &dnodal_forces_vec[cont_cols], 1);

        //Nodal forces in y direction
        cublasDgemv(handle_mt, CUBLAS_OP_T, rows, cols, &alpha_mt, &dfStorage[pos], rows, &dsigma[cont_rows + npts_tot], 1, &beta_mt, &dnodal_forces_vec[cont_cols + npts_tot/2], 1);
}

cudaMemcpy(&nodal_forces_vec[0], dnodal_forces_vec, npts_tot*sizeof(double), cudaMemcpyDeviceToHost);
for(int i = 0; i < npts_tot; i++){
std::cout << nodal_forces_vec[i] << std::endl;
}

///Free device memory
cudaFree(dsigma);
cublasDestroy(handle_mt);
///----------------------------------------------------------------

///ASSEMBLE--------------------------------------------------------
///Initialize global nodal forces vector on device------------------
int globvec = nodal_forces_global.Rows();
nodal_forces_global.Zero();
double *dnodal_forces_global;
cudaMalloc(&dnodal_forces_global, globvec*sizeof(double));

///Kernel that assemble the nodal forces vector
AssembleKernel<<<npts_tot,1>>>(npts_tot, dfIndexes,dnodal_forces_vec, dnodal_forces_global);

///Transfer global nodal forces vector the host
cudaMemcpy(&nodal_forces_global(0,0), dnodal_forces_global, globvec*sizeof(double), cudaMemcpyDeviceToHost);

///Free device memory
cudaFree(dnodal_forces_vec);
cudaFree(dnodal_forces_global);
///----------------------------------------------------------------
}

void TPZSolveMatrix::Multiply(const TPZFMatrix<STATE>  &global_solution, TPZFMatrix<REAL> &result) const
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


