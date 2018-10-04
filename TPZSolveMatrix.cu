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
#include <omp.h>

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

void TPZSolveMatrix::SolveWithCUDA(TPZCompMesh *cmesh, const TPZFMatrix<STATE> &global_solution, TPZStack<REAL> &weight,
                                   TPZFMatrix<REAL> &nodal_forces_global) const {
    int64_t nelem = fRowSizes.size();
    int64_t nindexes = fIndexes.size();
    int64_t npts_tot = fRow;
    int64_t ncolor = *std::max_element(fElemColor.begin(), fElemColor.end()) + 1;
    int64_t neq = nodal_forces_global.Rows();

    nodal_forces_global.Resize(ncolor * neq, 1);
    nodal_forces_global.Zero();

    ///Create handles for CUBLAS and CUSPARSE
    cusparseHandle_t handle_cusparse;
    cusparseCreate(&handle_cusparse);

    cublasHandle_t handle_cublas;
    cublasCreate(&handle_cublas);

    ///Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float timing = 0;

    ///Allocation and Transfer memory
    cudaEventRecord(start); //start gather timing

    double *dexpandsolution;
    cudaMalloc(&dexpandsolution, nindexes * sizeof(double));

    double *dglobal_solution;
    cudaMalloc(&dglobal_solution, global_solution.Rows() * sizeof(double));
    cudaMemcpy(dglobal_solution, &global_solution[0], global_solution.Rows() * sizeof(double), cudaMemcpyHostToDevice);

    int *dindexes;
    cudaMalloc(&dindexes, fIndexes.size() * sizeof(int));
    cudaMemcpy(dindexes, &fIndexes[0], fIndexes.size() * sizeof(int), cudaMemcpyHostToDevice);

    double *dresult;
    cudaMalloc(&dresult, 2 * nindexes * sizeof(double));

    double *dstorage;
    cudaMalloc(&dstorage, fStorage.size() * sizeof(double));
    cudaMemcpy(dstorage, &fStorage[0], fStorage.size() * sizeof(double), cudaMemcpyHostToDevice);

    double *dsigma;
    cudaMalloc(&dsigma, 2 * npts_tot * sizeof(double));

    double *dweight;
    cudaMalloc(&dweight, weight.size() * sizeof(double));
    cudaMemcpy(dweight, &weight[0], weight.size() * sizeof(double), cudaMemcpyHostToDevice);

    double *dnodal_forces_vec;
    cudaMalloc(&dnodal_forces_vec, npts_tot * sizeof(double));

    int *dindexescolor;
    cudaMalloc(&dindexescolor, nindexes * sizeof(int));
    cudaMemcpy(dindexescolor, &fIndexesColor[0], nindexes * sizeof(int), cudaMemcpyHostToDevice);

    double *dnodal_forces_global;
    cudaMalloc(&dnodal_forces_global, ncolor * neq * sizeof(double));
    cudaMemcpy(dnodal_forces_global, &nodal_forces_global(0, 0), ncolor * neq * sizeof(double), cudaMemcpyHostToDevice);

    cudaEventRecord(stop); //stop gather timing

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timing, start, stop);
    std::cout << "Allocation and Transfer of memory: " << timing / 1000 << std::endl;
    ///GATHER OPERATION------------------------------------------------
    cusparseDgthr(handle_cusparse, nindexes, dglobal_solution, &dexpandsolution[0], &dindexes[0],
                  CUSPARSE_INDEX_BASE_ZERO);
    ///----------------------------------------------------------------
    ///MULTIPLY--------------------------------------------------------
    cudaEventRecord(start); //start multiply timing
    double alpha_m = 1.0;
    double beta_m = 0.0;

//    cublasDgemmBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N, fRowSizes[0], 1, fColSizes[0], &alpha_m, (const double**) dstorage, fRowSizes[0], (const double**) &dexpandsolution[0], fColSizes[0], &beta_m,(double**) &dresult[0], fRowSizes[0], nelem);

//    cublasDgemmBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N, fRowSizes[0], 1, fColSizes[0], &alpha_m, (const double**) dstorage, fRowSizes[0], (const double**) &dexpandsolution[fColFirstIndex[nelem]], fColSizes[0], &beta_m, (double**) &dresult[fRowFirstIndex[nelem]], fRowSizes[0], nelem);
    for (int iel = 0; iel < nelem; iel++) {
        int64_t pos = fMatrixPosition[iel];
        int64_t cols = fColSizes[iel];
        int64_t rows = fRowSizes[iel];
        int64_t cont_cols = fColFirstIndex[iel];
        int64_t cont_rows = fRowFirstIndex[iel];

        //du
        cublasDgemv(handle_cublas, CUBLAS_OP_N, rows, cols, &alpha_m, &dstorage[pos], rows, &dexpandsolution[cont_cols],
                    1, &beta_m, &dresult[cont_rows], 1);
        //dv
        cublasDgemv(handle_cublas, CUBLAS_OP_N, rows, cols, &alpha_m, &dstorage[pos], rows,
                    &dexpandsolution[cont_cols + fColFirstIndex[nelem]], 1, &beta_m,
                    &dresult[cont_rows + fRowFirstIndex[nelem]], 1);
    }
    cudaEventRecord(stop); //stop multiply timing

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timing, start, stop);
    std::cout << "Multiply matrices: " << timing / 1000 << std::endl;

    ///----------------------------------------------------------------
    TPZFMatrix<REAL> vec(2 * nindexes, 1);
    cudaMemcpy(&vec(0, 0), dresult, 2 * nindexes * sizeof(double), cudaMemcpyDeviceToHost);
    vec.Print(std::cout);
    ///COMPUTE SIGMA---------------------------------------------------
    cudaEventRecord(start); //start sigma timing
    dim3 dimGrid(ceil((npts_tot / 2) / 32.0), 1, 1);
    dim3 dimBlock(32, 1, 1);
    ComputeSigmaKernel << < dimGrid, dimBlock >> > (npts_tot, dweight, dresult, dsigma);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); //stop sigma timing

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timing, start, stop);
    std::cout << "Sigma: " << timing / 1000 << std::endl;
    ///----------------------------------------------------------------

    ///MULTIPLY TRANSPOSE----------------------------------------------
    cudaEventRecord(start); //start transpose timing
    double alpha_mt = 1.0;
    double beta_mt = 0.;

    for (int64_t iel = 0; iel < nelem; iel++) {
        int64_t pos = fMatrixPosition[iel];
        int64_t rows = fRowSizes[iel];
        int64_t cols = fColSizes[iel];
        int64_t cont_rows = fRowFirstIndex[iel];
        int64_t cont_cols = fColFirstIndex[iel];

        //Nodal forces in x direction
        cublasDgemv(handle_cublas, CUBLAS_OP_T, rows, cols, &alpha_mt, &dstorage[pos], rows, &dsigma[cont_rows], 1,
                    &beta_mt, &dnodal_forces_vec[cont_cols], 1);

        //Nodal forces in y direction
        cublasDgemv(handle_cublas, CUBLAS_OP_T, rows, cols, &alpha_mt, &dstorage[pos], rows,
                    &dsigma[cont_rows + npts_tot], 1, &beta_mt, &dnodal_forces_vec[cont_cols + npts_tot / 2], 1);
    }
    cudaEventRecord(stop); //stop transpose timing

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timing, start, stop);
    std::cout << "Multiply Transpose: " << timing / 1000 << std::endl;
    ///----------------------------------------------------------------

    ///ASSEMBLE--------------------------------------------------------
    cudaEventRecord(start); //start assemble timing
    cusparseDsctr(handle_cusparse, nindexes, dnodal_forces_vec, &dindexescolor[0], &dnodal_forces_global[0],
                  CUSPARSE_INDEX_BASE_ZERO);

    int64_t colorassemb = ncolor / 2;
    double alpha = 1.;
    while (colorassemb > 0) {

        int64_t firsteq = (ncolor - colorassemb) * neq;

        cublasDaxpy(handle_cublas, firsteq, &alpha, &dnodal_forces_global[firsteq], 1., &dnodal_forces_global[0], 1.);

        ncolor -= colorassemb;
        colorassemb = ncolor / 2;
    }

    nodal_forces_global.Resize(neq, 1);
    cudaEventRecord(stop); //stop assemble timing

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timing, start, stop);
    std::cout << "Assemble: " << timing / 1000 << std::endl;
    ///----------------------------------------------------------------

    ///Transfer global forces vector from device to host
    cudaMemcpy(&nodal_forces_global(0, 0), dnodal_forces_global, neq * sizeof(double), cudaMemcpyDeviceToHost);

    ///Free device memory
    cusparseDestroy(handle_cusparse);
    cublasDestroy(handle_cublas);
    cudaFree(dglobal_solution);
    cudaFree(dindexes);
    cudaFree(dexpandsolution);
    cudaFree(dresult);
    cudaFree(dweight);
    cudaFree(dsigma);
    cudaFree(dstorage);
    cudaFree(dnodal_forces_vec);
    cudaFree(dindexescolor);
    cudaFree(dnodal_forces_global);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void TPZSolveMatrix::Multiply(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const {
    int64_t nelem = fRowSizes.size();

    int64_t nindexes = fIndexes.size();

    result.Resize(2 * nindexes, 1);
    result.Zero();

    TPZVec<REAL> expandsolution(nindexes);

/// gather operation
    cblas_dgthr(nindexes, global_solution, &expandsolution[0], &fIndexes[0]);


    std::clock_t start2 = clock();
    for (int64_t iel = 0; iel < nelem; iel++) {
        int64_t pos = fMatrixPosition[iel];
        int64_t cols = fColSizes[iel];
        int64_t rows = fRowSizes[iel];
        TPZFMatrix<REAL> elmatrix(rows, cols, &fStorage[pos], rows * cols);

        int64_t cont_cols = fColFirstIndex[iel];
        int64_t cont_rows = fRowFirstIndex[iel];

        TPZFMatrix<REAL> element_solution_x(cols, 1, &expandsolution[cont_cols], cols);
        TPZFMatrix<REAL> element_solution_y(cols, 1, &expandsolution[cont_cols + fColFirstIndex[nelem]], cols);

        TPZFMatrix<REAL> solx(rows, 1, &result(cont_rows, 0), rows); //du
        TPZFMatrix<REAL> soly(rows, 1, &result(cont_rows + fRowFirstIndex[nelem], 0), rows); //dv

        elmatrix.Multiply(element_solution_x, solx);
        elmatrix.Multiply(element_solution_y, soly);
    }
    std::clock_t stop2 = clock();
    REAL time2 = REAL(stop2 - start2) / CLOCKS_PER_SEC;
    std::cout << "Multiply matrices: " << std::setprecision(5) << std::fixed << time2 << std::endl;

}

void TPZSolveMatrix::ComputeSigma(TPZStack<REAL> &weight, TPZFMatrix<REAL> &result, TPZFMatrix<STATE> &sigma) {
    REAL E = 200000000.;
    REAL nu = 0.30;
    int npts_tot = fRow;
    sigma.Resize(2 * npts_tot, 1);

    std::clock_t start = clock();
    for (int64_t ipts = 0; ipts < npts_tot / 2; ipts++) {
        sigma(2 * ipts, 0) = weight[ipts] * E / (1. - nu * nu) *
                             (result(2 * ipts, 0) + nu * result(2 * ipts + npts_tot + 1, 0)); // Sigma x
        sigma(2 * ipts + 1, 0) = weight[ipts] * E / (1. - nu * nu) * (1. - nu) / 2 *
                                 (result(2 * ipts + 1, 0) + result(2 * ipts + npts_tot, 0)) * 0.5; // Sigma xy
        sigma(2 * ipts + npts_tot, 0) = sigma(2 * ipts + 1, 0); //Sigma xy
        sigma(2 * ipts + npts_tot + 1, 0) = weight[ipts] * E / (1. - nu * nu) *
                                            (result(2 * ipts + npts_tot + 1, 0) + nu * result(2 * ipts, 0)); // Sigma y
    }
    std::clock_t stop = clock();
    REAL time = REAL(stop - start) / CLOCKS_PER_SEC;
    std::cout << "Sigma: " << std::setprecision(5) << std::fixed << time << std::endl;
}

void TPZSolveMatrix::MultiplyTranspose(TPZFMatrix<STATE> &intpoint_solution, TPZFMatrix<STATE> &nodal_forces_vec) {
    int64_t nelem = fRowSizes.size();
    int64_t npts_tot = fRow;
    nodal_forces_vec.Resize(npts_tot, 1);
    nodal_forces_vec.Zero();

    std::clock_t start = clock();

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
    std::clock_t stop = clock();
    REAL time = REAL(stop - start) / CLOCKS_PER_SEC;
    std::cout << "Multiply Transpose: " << std::setprecision(5) << std::fixed << time << std::endl;

}

void
TPZSolveMatrix::TraditionalAssemble(TPZFMatrix<STATE> &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) const {
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

void TPZSolveMatrix::ColoredAssemble(TPZFMatrix<STATE> &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) {
    int64_t ncolor = *std::max_element(fElemColor.begin(), fElemColor.end()) + 1;
    int64_t sz = fIndexes.size();
    int64_t neq = nodal_forces_global.Rows();
    nodal_forces_global.Resize(neq * ncolor, 1);

    std::clock_t start = clock();
    cblas_dsctr(sz, nodal_forces_vec, &fIndexesColor[0], &nodal_forces_global(0, 0));

    int64_t colorassemb = ncolor / 2.;
    while (colorassemb > 0) {

        int64_t firsteq = (ncolor - colorassemb) * neq;

        cblas_daxpy(firsteq, 1., &nodal_forces_global(firsteq, 0), 1., &nodal_forces_global(0, 0), 1.);

        ncolor -= colorassemb;
        colorassemb = ncolor / 2;
    }
    nodal_forces_global.Resize(neq, 1);
    std::clock_t stop = clock();
    REAL time = REAL(stop - start) / CLOCKS_PER_SEC;
    std::cout << "Sigma: " << std::setprecision(5) << std::fixed << time << std::endl;
}
