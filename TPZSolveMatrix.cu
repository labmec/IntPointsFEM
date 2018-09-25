#include "TPZSolveMatrix.h"
#include "pzmatrix.h"
#include <mkl.h>
#include <stdlib.h>

#ifdef USING_MKL
#include <mkl.h>
#include <algorithm>
#endif

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

///CUDA KERNELS
__global__ void ComputeSigmaKernel(int npts_tot, double *weight, double *result, double *sigma) {
    REAL E = 200000000.;
    REAL nu = 0.30;
    int ipts = blockIdx.x * blockDim.x + threadIdx.x;

    if (ipts < npts_tot / 2) {
        sigma[2 * ipts] = weight[ipts] * E / (1. - nu * nu) * (result[2 * ipts] + nu * result[2 * ipts + npts_tot + 1]); // Sigma x
        sigma[2 * ipts + 1] = weight[ipts] * E / (1. - nu * nu) * (1. - nu) / 2 * (result[2 * ipts + 1] + result[2 * ipts + npts_tot]) * 0.5; // Sigma xy
        sigma[2 * ipts + npts_tot] = sigma[2 * ipts + 1]; //Sigma xy
        sigma[2 * ipts + npts_tot + 1] = weight[ipts] * E / (1. - nu * nu) * (result[2 * ipts + npts_tot + 1] + nu * result[2 * ipts]); // Sigma y
    }
}


__global__ void sumvecscalar( int *vector, int *out, const int scalar, int N)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < N){
	out[i] = vector[i] + scalar;
    }
 
}

void TPZSolveMatrix::HostToDevice() {
    int nstorage = fStorage.size();
    int nrowsizes = fRowSizes.size();
    int ncolsizes = fColSizes.size();
    int nindexes = fIndexes.size();
    int nmatrixposition = fMatrixPosition.size();
    int nrowfirstindex = fRowFirstIndex.size();
    int ncolfirstindex = fColFirstIndex.size();

    cudaMalloc(&dfStorage, nstorage * sizeof(double));
    cudaMalloc(&dfRowSizes, nrowsizes * sizeof(int));
    cudaMalloc(&dfColSizes, ncolsizes * sizeof(int));
    cudaMalloc(&dfIndexes, nindexes * sizeof(int));
    cudaMalloc(&dfMatrixPosition, nmatrixposition * sizeof(int));
    cudaMalloc(&dfRowFirstIndex, nrowfirstindex * sizeof(int));
    cudaMalloc(&dfColFirstIndex, ncolfirstindex * sizeof(int));

    cudaMemcpy(dfStorage, &fStorage[0], nstorage * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dfRowSizes, &fRowSizes[0], nrowsizes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dfColSizes, &fColSizes[0], ncolsizes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dfIndexes, &fIndexes[0], nindexes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dfMatrixPosition, &fMatrixPosition[0], nmatrixposition * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dfRowFirstIndex, &fRowFirstIndex[0], nrowfirstindex * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dfColFirstIndex, &fColFirstIndex[0], ncolfirstindex * sizeof(int), cudaMemcpyHostToDevice);
}

void TPZSolveMatrix::FreeDeviceMemory() {
    cudaFree(dfStorage);
    cudaFree(dfRowSizes);
    cudaFree(dfColSizes);
    cudaFree(dfIndexes);
    cudaFree(dfMatrixPosition);
    cudaFree(dfRowFirstIndex);
    cudaFree(dfColFirstIndex);
}

void TPZSolveMatrix::SolveWithCUDA(TPZCompMesh *cmesh, const TPZFMatrix<STATE> &global_solution, TPZStack<REAL> &weight, TPZFMatrix<REAL> &nodal_forces_global) const {
    int64_t nelem = fRowSizes.size();
    MKL_INT n_globalsol = fIndexes.size();

///GATHER OPERATION------------------------------------------------
///Initialize and transfer expandsolution and globalsolution to the device
    double *dexpandsolution;
    cudaMalloc(&dexpandsolution, n_globalsol * sizeof(double));

    double *dglobal_solution;
    cudaMalloc(&dglobal_solution, global_solution.Rows() * sizeof(double));
    cudaMemcpy(dglobal_solution, &global_solution[0], global_solution.Rows() * sizeof(double), cudaMemcpyHostToDevice);

    cusparseHandle_t handle_gthr;
    cusparseCreate (&handle_gthr);
    cusparseDgthr(handle_gthr, n_globalsol, dglobal_solution, &dexpandsolution[0], &dfIndexes[0], CUSPARSE_INDEX_BASE_ZERO);

///Free device memory
    cudaFree(dglobal_solution);
    cusparseDestroy(handle_gthr);
///----------------------------------------------------------------

///MULTIPLY--------------------------------------------------------
///Initialize result on device
    TPZVec<REAL> result(2 * n_globalsol);
    double *dresult;
    cudaMalloc(&dresult, 2 * n_globalsol * sizeof(double));

///Use CUBLAS library to do the multiplication
    cudaStream_t stream_m[2 * nelem];
    cublasHandle_t handle_m[2 * nelem];

    double alpha_m = 1.0;
    double beta_m = 0.0;

    for (int iel = 0; iel < nelem; iel++) {
        cudaStreamCreate(&stream_m[2 * iel]);
        cudaStreamCreate(&stream_m[2 * iel + 1]);

        cublasCreate(&handle_m[2 * iel]);
        cublasCreate(&handle_m[2 * iel + 1]);

        int64_t pos = fMatrixPosition[iel];
        int64_t cols = fColSizes[iel];
        int64_t rows = fRowSizes[iel];

        int64_t cont_cols = fColFirstIndex[iel];
        int64_t cont_rows = fRowFirstIndex[iel];

        //du
        cublasSetStream(handle_m[2 * iel], stream_m[2 * iel]);
        cublasDgemv(handle_m[2 * iel], CUBLAS_OP_N, rows, cols, &alpha_m, &dfStorage[pos], rows, &dexpandsolution[cont_cols], 1, &beta_m, &dresult[cont_rows], 1);

        //dv
        cublasSetStream(handle_m[2 * iel + 1], stream_m[2 * iel + 1]);
        cublasDgemv(handle_m[2 * iel + 1], CUBLAS_OP_N, rows, cols, &alpha_m, &dfStorage[pos], rows, &dexpandsolution[cont_cols + fColFirstIndex[nelem]], 1, &beta_m, &dresult[cont_rows + fRowFirstIndex[nelem]], 1);
    }

///Free device memory
    cudaFree(dexpandsolution);
    cublasDestroy(*handle_m);
    cudaStreamSynchronize(0);
///----------------------------------------------------------------

///COMPUTE SIGMA---------------------------------------------------
///Initialize and transfer sigma and weight to the device
    int npts_tot = fRow;
    TPZVec<REAL> sigma(2 * npts_tot);
    double *dsigma;
    cudaMalloc(&dsigma, 2 * npts_tot * sizeof(double));

    double *dweight;
    cudaMalloc(&dweight, npts_tot / 2 * sizeof(double));
    cudaMemcpy(dweight, &weight[0], npts_tot / 2 * sizeof(double), cudaMemcpyHostToDevice);

///Kernel that calculates sigma
    dim3 dimGrid(ceil((npts_tot / 2) / 32.0), 1, 1);
    dim3 dimBlock(32, 1, 1);
    ComputeSigmaKernel <<< dimGrid, dimBlock >>> (npts_tot, dweight, dresult, dsigma);
    cudaDeviceSynchronize();

///Free device memory
    cudaFree(dresult);
    cudaFree(dweight);
///----------------------------------------------------------------

///MULTIPLY TRANSPOSE----------------------------------------------
///Initialize nodal_forces_vector on device
    double *dnodal_forces_vec;
    cudaMalloc(&dnodal_forces_vec, npts_tot * sizeof(double));

///Use CUBLAS to do the multiplication
    cudaStream_t stream_mt[2 * nelem];
    cublasHandle_t handle_mt[2 * nelem];

    double alpha_mt = 1.0;
    double beta_mt = 0.0;

    for (int64_t iel = 0; iel < nelem; iel++) {
        cudaStreamCreate(&stream_mt[2 * iel]);
        cudaStreamCreate(&stream_mt[2 * iel + 1]);

        cublasCreate(&handle_mt[2 * iel]);
        cublasCreate(&handle_mt[2 * iel + 1]);

        int64_t pos = fMatrixPosition[iel];
        int64_t rows = fRowSizes[iel];
        int64_t cols = fColSizes[iel];

        int64_t cont_rows = fRowFirstIndex[iel];
        int64_t cont_cols = fColFirstIndex[iel];

        //Nodal forces in x direction
        cublasSetStream(handle_mt[2 * iel], stream_mt[2 * iel]);
        cublasDgemv(handle_mt[2 * iel], CUBLAS_OP_T, rows, cols, &alpha_mt, &dfStorage[pos], rows, &dsigma[cont_rows], 1, &beta_mt, &dnodal_forces_vec[cont_cols], 1);

        //Nodal forces in y direction
        cublasSetStream(handle_mt[2 * iel + 1], stream_mt[2 * iel + 1]);
        cublasDgemv(handle_mt[2 * iel + 1], CUBLAS_OP_T, rows, cols, &alpha_mt, &dfStorage[pos], rows, &dsigma[cont_rows + npts_tot], 1, &beta_mt, &dnodal_forces_vec[cont_cols + npts_tot / 2], 1);
    }

///Free device memory
    cudaFree(dsigma);
    cublasDestroy(*handle_mt);
    cudaStreamSynchronize(0);
///----------------------------------------------------------------

///ASSEMBLE--------------------------------------------------------
    ColoringElements(cmesh);

///Initialize fIndexesColor on device
    cudaMalloc(&dfIndexesColor, n_globalsol * sizeof(double));
    cudaMemcpy(dfIndexesColor, &fIndexesColor[0], n_globalsol * sizeof(double), cudaMemcpyHostToDevice);

///Initialize nodal_forces_global on device
    int64_t ncolor = *std::max_element(fElemColor.begin(), fElemColor.end())+1;
    int64_t sz = fIndexes.size();
    int64_t neq = nodal_forces_global.Rows();

    nodal_forces_global.Resize(ncolor * neq,1);
    double *dnodal_forces_global;
    cudaMalloc(&dnodal_forces_global, ncolor * neq * sizeof(double));

    cusparseHandle_t handle_sctr;
    cusparseCreate(&handle_sctr);
    cusparseDsctr(handle_sctr, sz, dnodal_forces_vec, &dfIndexesColor[0], &dnodal_forces_global[0], CUSPARSE_INDEX_BASE_ZERO);

    cudaMemcpy(&nodal_forces_global(0, 0), dnodal_forces_global, ncolor * neq * sizeof(double), cudaMemcpyDeviceToHost);
    nodal_forces_global.Print(std::cout);

///Free device memory
    cusparseDestroy(handle_sctr);
    cudaFree(dnodal_forces_vec);
    cudaFree(dcoloredindexes);
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

void TPZSolveMatrix::ColoredElements(TPZCompMesh * cmesh) const
{
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

void TPZSolveMatrix::ColoredAssemble(TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global)
{

}


