#include "TPZSolveMatrix.h"
#include "pzmatrix.h"
#include <mkl.h>
#include <stdlib.h>
# include <chrono>
#ifdef USING_MKL

#include <mkl.h>
#include <algorithm>

#endif

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <omp.h>
#include <chrono>

using namespace std::chrono;


///CUDA KERNELS
__global__ void ComputeSigmaKernel(int npts_tot, double *weight, double *result, double *sigma) {
    REAL E = 200000000.;
    REAL nu = 0.30;
    int ipts = blockIdx.x * blockDim.x + threadIdx.x;

    if (ipts < npts_tot / 2) {
        sigma[2 * ipts] = weight[ipts] * E / (1. - nu * nu) *
                          (result[2 * ipts] + nu * result[2 * ipts + npts_tot + 1]); // Sigma x
        sigma[2 * ipts + 1] = weight[ipts] * E / (1. - nu * nu) * (1. - nu) / 2 *
                              (result[2 * ipts + 1] + result[2 * ipts + npts_tot]) * 0.5; // Sigma xy
        sigma[2 * ipts + npts_tot] = sigma[2 * ipts + 1]; //Sigma xy
        sigma[2 * ipts + npts_tot + 1] = weight[ipts] * E / (1. - nu * nu) *
                                         (result[2 * ipts + npts_tot + 1] + nu * result[2 * ipts]); // Sigma y
    }
}

void TPZSolveMatrix::AllocateMemory(TPZCompMesh *cmesh) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
//    cudaEventRecord(start);

    int nelem = fRowSizes.size();
    int nindexes = fIndexes.size();
    int neq = cmesh->NEquations();
    int npts_tot = fRow;
    int64_t ncolor = *std::max_element(fElemColor.begin(), fElemColor.end()) + 1;

    cudaMalloc(&dglobal_solution, neq * sizeof(double));
    cudaMalloc(&dindexes, nindexes * sizeof(int));

    cudaEventRecord(start);

    cudaMalloc(&dstorage, nelem*fColSizes[0]*fRowSizes[0] * sizeof(double));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Allocate: " << milliseconds/1000 << std::endl;


    cudaMalloc(&dexpandsolution, nindexes * sizeof(double));
    cudaMalloc(&dresult, 2 * nindexes * sizeof(double));
    cudaMalloc(&dweight, npts_tot/2 * sizeof(double));
    cudaMalloc(&dsigma, 2 * npts_tot * sizeof(double));
    cudaMalloc(&dnodal_forces_vec, npts_tot * sizeof(double));
    cudaMalloc(&dindexescolor, nindexes * sizeof(int));
    cudaMalloc(&dnodal_forces_global, ncolor * neq * sizeof(double));

//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//    float milliseconds = 0;
//    cudaEventElapsedTime(&milliseconds, start, stop);
//    std::cout << "Allocate: " << milliseconds/1000 << std::endl;

}

void TPZSolveMatrix::FreeMemory() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
//    cudaEventRecord(start);

    cudaFree(dglobal_solution);
    cudaFree(dindexes);

    cudaEventRecord(start);
    cudaFree(dstorage);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Free: " << milliseconds/1000 << std::endl;
  

    cudaFree(dexpandsolution);
    cudaFree(dresult);
    cudaFree(dweight);
    cudaFree(dsigma);
    cudaFree(dnodal_forces_vec);
    cudaFree(dindexescolor);
    cudaFree(dnodal_forces_global);

    cublasDestroy(handle_cublas);
    cusparseDestroy(handle_cusparse);

//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//    float milliseconds = 0;
//    cudaEventElapsedTime(&milliseconds, start, stop);
//    std::cout << "Free: " << milliseconds/1000 << std::endl;

}

void TPZSolveMatrix::cuSparseHandle() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cusparseCreate (&handle_cusparse);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
//    std::cout << "cuSPARSE: " << milliseconds/1000 << std::endl;

}

void TPZSolveMatrix::cuBlasHandle() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cublasCreate (&handle_cublas);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
//    std::cout << "cuBLAS: " << milliseconds/1000 << std::endl;
}


void TPZSolveMatrix::MultiplyCUDA(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const {

    int64_t nelem = fRowSizes.size();
    int64_t nindexes = fIndexes.size();
    result.Resize(2 * nindexes, 1);
    result.Zero();

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMemcpy(dstorage, &fStorage[0], nelem*fColSizes[0]*fRowSizes[0] * sizeof(double), cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Copy: " << milliseconds/1000 << std::endl;

    cudaEventRecord(start);

    cudaMemcpy(dindexes, &fIndexes[0], nindexes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dglobal_solution, &global_solution[0], global_solution.Rows() * sizeof(double), cudaMemcpyHostToDevice);

  
    cusparseDgthr(handle_cusparse, nindexes, dglobal_solution, &dexpandsolution[0], &dindexes[0], CUSPARSE_INDEX_BASE_ZERO);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
//    std::cout << "Gather: " << milliseconds/1000 << std::endl;


    cudaEventRecord(start);

    double alpha = 1.0;
    double beta = 0.0;

    int64_t cols = fColSizes[0];
    int64_t rows = fRowSizes[0];

    cublasDgemmStridedBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N, rows, 1, cols, &alpha, dstorage, rows, rows*cols, &dexpandsolution[0], cols, cols*1, &beta, &dresult[0], rows, rows*1, nelem);

    cublasDgemmStridedBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N, rows, 1, cols, &alpha, dstorage, rows, rows*cols, &dexpandsolution[fColFirstIndex[nelem]], cols, cols*1, &beta,  &dresult[fRowFirstIndex[nelem]], rows, rows*1, nelem);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Multiply: " << milliseconds/1000 << std::endl;

    cudaEventRecord(start);

    cudaMemcpy(&result(0, 0), dresult, 2 * nindexes * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Copy: " << milliseconds/1000 << std::endl;

}

void TPZSolveMatrix::ComputeSigmaCUDA(TPZStack<REAL> &weight, TPZFMatrix<REAL> &result, TPZFMatrix<STATE> &sigma) {
    REAL E = 200000000.;
    REAL nu = 0.30;
    int npts_tot = fRow;
    int nindexes = fIndexes.size();
    sigma.Resize(2 * npts_tot, 1);
    sigma.Zero();

    cudaEvent_t start, stop;
    float milliseconds = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMemcpy(dweight, &weight[0], weight.size() * sizeof(double), cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
//    std::cout << "Copy: " << milliseconds/1000 << std::endl;

    cudaEventRecord(start);

    dim3 dimGrid(ceil((npts_tot / 2) / 256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);
    ComputeSigmaKernel <<< dimGrid, dimBlock >>> (npts_tot, dweight, dresult, dsigma);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
//    std::cout << "Sigma: " << milliseconds/1000 << std::endl;

    cudaEventRecord(start);

    cudaMemcpy(&sigma(0, 0), dsigma, 2 * npts_tot * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
//    std::cout << "Copy: " << milliseconds/1000 << std::endl;

}

void TPZSolveMatrix::MultiplyTransposeCUDA(TPZFMatrix<STATE> &sigma, TPZFMatrix<STATE> &nodal_forces_vec) {
    int64_t nelem = fRowSizes.size();
    int64_t npts_tot = fRow;

    nodal_forces_vec.Resize(npts_tot, 1);
    nodal_forces_vec.Zero();

    double alpha = 1.0;
    double beta = 0.;

    int64_t cols = fColSizes[0];
    int64_t rows = fRowSizes[0];

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cublasDgemmStridedBatched(handle_cublas, CUBLAS_OP_T, CUBLAS_OP_N, cols, 1, rows, &alpha, dstorage, rows, 0, &dsigma[0], rows, rows*1, &beta, &dnodal_forces_vec[0], cols, cols*1, nelem);

    cublasDgemmStridedBatched(handle_cublas, CUBLAS_OP_T, CUBLAS_OP_N, cols, 1, rows, &alpha, dstorage, rows, 0, &dsigma[npts_tot], rows, rows*1, &beta,  &dnodal_forces_vec[npts_tot / 2], cols, cols*1, nelem);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
//    std::cout << "Transpose: " << milliseconds/1000 << std::endl;

    cudaEventRecord(start);

    cudaMemcpy(&nodal_forces_vec(0, 0), dnodal_forces_vec, npts_tot * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
//    std::cout << "Copy: " << milliseconds/1000 << std::endl;
}

void TPZSolveMatrix::ColoredAssembleCUDA(TPZFMatrix<STATE> &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) {
    int64_t ncolor = *std::max_element(fElemColor.begin(), fElemColor.end()) + 1;
    int64_t nindexes = fIndexes.size();
    int64_t neq = nodal_forces_global.Rows();
    int64_t npts_tot = fRow;

    nodal_forces_global.Resize(neq * ncolor, 1);
    nodal_forces_global.Zero();

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMemcpy(dindexescolor, &fIndexesColor[0], nindexes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dnodal_forces_global, &nodal_forces_global(0, 0), ncolor * neq * sizeof(double), cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
//    std::cout << "Copy: " << milliseconds/1000 << std::endl;

    cudaEventRecord(start);

    cusparseDsctr(handle_cusparse, nindexes, dnodal_forces_vec, &dindexescolor[0], &dnodal_forces_global[0], CUSPARSE_INDEX_BASE_ZERO);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
//    std::cout << "Scatter: " << milliseconds/1000 << std::endl;

    cudaEventRecord(start);

    int64_t colorassemb = ncolor / 2;
    double alpha = 1.;
    while (colorassemb > 0) {

        int64_t firsteq = (ncolor - colorassemb) * neq;

        cublasDaxpy(handle_cublas, firsteq, &alpha, &dnodal_forces_global[firsteq], 1., &dnodal_forces_global[0], 1.);

        ncolor -= colorassemb;
        colorassemb = ncolor / 2;
    }

    nodal_forces_global.Resize(neq, 1);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
//    std::cout << "Assemble: " << milliseconds/1000 << std::endl;

    cudaEventRecord(start);

    cudaMemcpy(&nodal_forces_global(0, 0), dnodal_forces_global, neq * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
//    std::cout << "Copy: " << milliseconds/1000 << std::endl;
}


void TPZSolveMatrix::Multiply(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const {
    int64_t nelem = fRowSizes.size();

    int64_t nindexes = fIndexes.size();

    result.Resize(2 * nindexes, 1);
    result.Zero();

    TPZVec<REAL> expandsolution(nindexes);

    high_resolution_clock::time_point t1, t2;
    duration<double> time_span;

    t1 = high_resolution_clock::now();
    cblas_dgthr(nindexes, global_solution, &expandsolution[0], &fIndexes[0]);
    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
//    std::cout << "Gather: " << time_span.count() << std::endl;

    t1 = high_resolution_clock::now();
    for (int64_t iel = 0; iel < nelem; iel++) {
        int64_t pos = fMatrixPosition[iel];
        int64_t cols = fColSizes[iel];
        int64_t rows = fRowSizes[iel];
        TPZFMatrix<REAL> elmatrix(rows, cols, &fStorage[0], rows * cols);

        int64_t cont_cols = fColFirstIndex[iel];
        int64_t cont_rows = fRowFirstIndex[iel];

        TPZFMatrix<REAL> element_solution_x(cols, 1, &expandsolution[cont_cols], cols);
        TPZFMatrix<REAL> element_solution_y(cols, 1, &expandsolution[cont_cols + fColFirstIndex[nelem]], cols);

        TPZFMatrix<REAL> solx(rows, 1, &result(cont_rows, 0), rows); //du
        TPZFMatrix<REAL> soly(rows, 1, &result(cont_rows + fRowFirstIndex[nelem], 0), rows); //dv

        elmatrix.Multiply(element_solution_x, solx);
        elmatrix.Multiply(element_solution_y, soly);
    }
    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "Multiply: " << time_span.count() << std::endl;

}

void TPZSolveMatrix::ComputeSigma(TPZStack<REAL> &weight, TPZFMatrix<REAL> &result, TPZFMatrix<STATE> &sigma) {
    high_resolution_clock::time_point t1, t2;
    duration<double> time_span;

    t1 = high_resolution_clock::now();

    REAL E = 200000000.;
    REAL nu = 0.30;
    int npts_tot = fRow;
    sigma.Resize(2 * npts_tot, 1);

    for (int64_t ipts = 0; ipts < npts_tot / 2; ipts++) {
        sigma(2 * ipts, 0) = weight[ipts] * E / (1. - nu * nu) *
                             (result(2 * ipts, 0) + nu * result(2 * ipts + npts_tot + 1, 0)); // Sigma x
        sigma(2 * ipts + 1, 0) = weight[ipts] * E / (1. - nu * nu) * (1. - nu) / 2 *
                                 (result(2 * ipts + 1, 0) + result(2 * ipts + npts_tot, 0)) * 0.5; // Sigma xy
        sigma(2 * ipts + npts_tot, 0) = sigma(2 * ipts + 1, 0); //Sigma xy
        sigma(2 * ipts + npts_tot + 1, 0) = weight[ipts] * E / (1. - nu * nu) *
                                            (result(2 * ipts + npts_tot + 1, 0) + nu * result(2 * ipts, 0)); // Sigma y
    }
    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
//    std::cout << "Sigma: " << time_span.count() << std::endl;


}

void TPZSolveMatrix::MultiplyTranspose(TPZFMatrix<STATE> &intpoint_solution, TPZFMatrix<STATE> &nodal_forces_vec) {
    high_resolution_clock::time_point t1, t2;
    duration<double> time_span;

    t1 = high_resolution_clock::now();

    int64_t nelem = fRowSizes.size();
    int64_t npts_tot = fRow;
    nodal_forces_vec.Resize(npts_tot, 1);
    nodal_forces_vec.Zero();


    for (int64_t iel = 0; iel < nelem; iel++) {
        int64_t pos = fMatrixPosition[iel];
        int64_t rows = fRowSizes[iel];
        int64_t cols = fColSizes[iel];
        int64_t cont_rows = fRowFirstIndex[iel];
        int64_t cont_cols = fColFirstIndex[iel];
        TPZFMatrix<REAL> elmatrix(rows, cols, &fStorage[pos], rows * cols);

        // Nodal forces in x direction
        TPZFMatrix<REAL> fvx(rows, 1, &intpoint_solution(cont_rows, 0), rows);
        TPZFMatrix<STATE> nodal_forcex(cols, 1, &nodal_forces_vec(cont_cols, 0), cols);
        elmatrix.MultAdd(fvx, nodal_forcex, nodal_forcex, 1, 0, 1);

        // Nodal forces in y direction
        TPZFMatrix<REAL> fvy(rows, 1, &intpoint_solution(cont_rows + npts_tot, 0), rows);
        TPZFMatrix<STATE> nodal_forcey(cols, 1, &nodal_forces_vec(cont_cols + npts_tot / 2, 0), cols);
        elmatrix.MultAdd(fvy, nodal_forcey, nodal_forcey, 1, 0, 1);
    }
    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
//    std::cout << "Transpose: " << time_span.count() << std::endl;


}

void TPZSolveMatrix::ColoredAssemble(TPZFMatrix<STATE> &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) {
    high_resolution_clock::time_point t1, t2;
    duration<double> time_span;

    t1 = high_resolution_clock::now();


    int64_t ncolor = *std::max_element(fElemColor.begin(), fElemColor.end()) + 1;
    int64_t sz = fIndexes.size();
    int64_t neq = nodal_forces_global.Rows();
    nodal_forces_global.Resize(neq * ncolor, 1);

    cblas_dsctr(sz, nodal_forces_vec, &fIndexesColor[0], &nodal_forces_global(0, 0));

    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
//    std::cout << "Scatter: " << time_span.count() << std::endl;


    t1 = high_resolution_clock::now();

    int64_t colorassemb = ncolor / 2.;
    while (colorassemb > 0) {

        int64_t firsteq = (ncolor - colorassemb) * neq;

        cblas_daxpy(firsteq, 1., &nodal_forces_global(firsteq, 0), 1., &nodal_forces_global(0, 0), 1.);

        ncolor -= colorassemb;
        colorassemb = ncolor / 2;
    }
    nodal_forces_global.Resize(neq, 1);
    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
//    std::cout << "Assemble: " << time_span.count() << std::endl;


}

void TPZSolveMatrix::TraditionalAssemble(TPZFMatrix<STATE> &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) const {
    for (int64_t ir = 0; ir < fRow; ir++) {
        nodal_forces_global(fIndexes[ir], 0) += nodal_forces_vec(ir, 0);
    }
}

void TPZSolveMatrix::ColoringElements(TPZCompMesh *cmesh) const {
    int64_t nelem_c = cmesh->NElements();
    int64_t nconnects = cmesh->NConnects();
    TPZVec<int64_t> connects_vec(nconnects, 0);

    int64_t contcolor = 0;
    bool needstocontinue = true;

    while (needstocontinue) {
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
            fIndexesColor[cont_cols + icols] = fIndexes[cont_cols + icols] + fElemColor[iel] * neq;
            fIndexesColor[cont_cols + fRow / 2 + icols] =
                    fIndexes[cont_cols + fRow / 2 + icols] + fElemColor[iel] * neq;
        }
    }
}
