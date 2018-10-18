#include "TPZSolveVector.h"
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
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    cudaEventRecord(start);

    int nelem = fRowSizes.size();
    int nindexes = fIndexes.size()/2; //numero real de indices(sem duplicar)
    int neq = cmesh->NEquations();
    int npts_tot = fRow;
    int64_t ncolor = *std::max_element(fElemColor.begin(), fElemColor.end()) + 1;

    cudaMalloc(&dglobal_solution, neq * sizeof(double));
    cudaMalloc(&dindexes, 2*nindexes * sizeof(int)); //2* pq esta duplicado
    cudaMalloc(&dstoragevec, nelem*fColSizes[0]*fRowSizes[0] * sizeof(double));
    cudaMalloc(&dexpandsolution, 2*nindexes * sizeof(double)); //sol duplicada
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
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    cudaEventRecord(start);

    cudaFree(dglobal_solution);
    cudaFree(dindexes);
    cudaFree(dstorage);
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
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    cudaEventRecord(start);

    cusparseCreate (&handle_cusparse);

//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//    float milliseconds = 0;
//    cudaEventElapsedTime(&milliseconds, start, stop);
//    std::cout << "cuSPARSE: " << milliseconds/1000 << std::endl;

}

void TPZSolveMatrix::cuBlasHandle() {
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    cudaEventRecord(start);

    cublasCreate (&handle_cublas);

//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//    float milliseconds = 0;
//    cudaEventElapsedTime(&milliseconds, start, stop);
//    std::cout << "cuBLAS: " << milliseconds/1000 << std::endl;
}


void TPZSolveMatrix::MultiplyCUDA(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const {

//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    cudaEventRecord(start);

    int64_t nelem = fRowSizes.size();
    int64_t nindexes = fIndexes.size();

    cudaMemcpy(dindexes, &fIndexes[0], nindexes * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(dglobal_solution, &global_solution[0], global_solution.Rows() * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(dstorage, &fStorage[0], fColSizes[0]*fRowSizes[0] * sizeof(double), cudaMemcpyHostToDevice);

    result.Resize(2 * nindexes, 1);
    result.Zero();

    cusparseDgthr(handle_cusparse, nindexes, dglobal_solution, &dexpandsolution[0], &dindexes[0], CUSPARSE_INDEX_BASE_ZERO);

//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//    float milliseconds = 0;
//    cudaEventElapsedTime(&milliseconds, start, stop);
//    std::cout << "Gather: " << milliseconds/1000 << std::endl;


//    cudaEventRecord(start);

    double alpha = 1.0;
    double beta = 0.0;

    int64_t cols = fColSizes[0];
    int64_t rows = fRowSizes[0];

    cublasDgemmStridedBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N, rows, 1, cols, &alpha, dstorage, rows, 0, &dexpandsolution[0], cols, cols*1, &beta, &dresult[0], rows, rows*1, nelem);

    cublasDgemmStridedBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N, rows, 1, cols, &alpha, dstorage, rows, 0, &dexpandsolution[fColFirstIndex[nelem]], cols, cols*1, &beta,  &dresult[fRowFirstIndex[nelem]], rows, rows*1, nelem);

    cudaMemcpy(&result(0, 0), dresult, 2 * nindexes * sizeof(double), cudaMemcpyDeviceToHost);

//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//    milliseconds = 0;
//    cudaEventElapsedTime(&milliseconds, start, stop);
//    std::cout << "Multiply: " << milliseconds/1000 << std::endl;

}

void TPZSolveMatrix::ComputeSigmaCUDA(TPZStack<REAL> &weight, TPZFMatrix<REAL> &result, TPZFMatrix<STATE> &sigma) {
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    cudaEventRecord(start);

    REAL E = 200000000.;
    REAL nu = 0.30;
    int npts_tot = fRow;
    int nindexes = fIndexes.size();

    cudaMemcpy(dweight, &weight[0], weight.size() * sizeof(double), cudaMemcpyHostToDevice);

    sigma.Resize(2 * npts_tot, 1);
    sigma.Zero();

    dim3 dimGrid(ceil((npts_tot / 2) / 256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);
    ComputeSigmaKernel <<< dimGrid, dimBlock >>> (npts_tot, dweight, dresult, dsigma);
    cudaDeviceSynchronize();

    cudaMemcpy(&sigma(0, 0), dsigma, 2 * npts_tot * sizeof(double), cudaMemcpyDeviceToHost);

//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//    float milliseconds = 0;
//    cudaEventElapsedTime(&milliseconds, start, stop);
//    std::cout << "Sigma: " << milliseconds/1000 << std::endl;

}

void TPZSolveMatrix::MultiplyTransposeCUDA(TPZFMatrix<STATE> &sigma, TPZFMatrix<STATE> &nodal_forces_vec) {
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    cudaEventRecord(start);

    int64_t nelem = fRowSizes.size();
    int64_t npts_tot = fRow;

    nodal_forces_vec.Resize(npts_tot, 1);
    nodal_forces_vec.Zero();

    double alpha = 1.0;
    double beta = 0.;

    int64_t cols = fColSizes[0];
    int64_t rows = fRowSizes[0];

    cublasDgemmStridedBatched(handle_cublas, CUBLAS_OP_T, CUBLAS_OP_N, cols, 1, rows, &alpha, dstorage, rows, 0, &dsigma[0], rows, rows*1, &beta, &dnodal_forces_vec[0], cols, cols*1, nelem);

    cublasDgemmStridedBatched(handle_cublas, CUBLAS_OP_T, CUBLAS_OP_N, cols, 1, rows, &alpha, dstorage, rows, 0, &dsigma[npts_tot], rows, rows*1, &beta,  &dnodal_forces_vec[npts_tot / 2], cols, cols*1, nelem);

    cudaMemcpy(&nodal_forces_vec(0, 0), dnodal_forces_vec, npts_tot * sizeof(double), cudaMemcpyDeviceToHost);

//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//    float milliseconds = 0;
//    cudaEventElapsedTime(&milliseconds, start, stop);
//    std::cout << "Transpose: " << milliseconds/1000 << std::endl;


}

void TPZSolveMatrix::ColoredAssembleCUDA(TPZFMatrix<STATE> &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) {

//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    cudaEventRecord(start);

    int64_t ncolor = *std::max_element(fElemColor.begin(), fElemColor.end()) + 1;
    int64_t nindexes = fIndexes.size();
    int64_t neq = nodal_forces_global.Rows();
    int64_t npts_tot = fRow;

    nodal_forces_global.Resize(neq * ncolor, 1);
    nodal_forces_global.Zero();

    cudaMemcpy(dindexescolor, &fIndexesColor[0], nindexes * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(dnodal_forces_global, &nodal_forces_global(0, 0), ncolor * neq * sizeof(double), cudaMemcpyHostToDevice);

    cusparseDsctr(handle_cusparse, nindexes, dnodal_forces_vec, &dindexescolor[0], &dnodal_forces_global[0], CUSPARSE_INDEX_BASE_ZERO);

//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//    float milliseconds = 0;
//    cudaEventElapsedTime(&milliseconds, start, stop);
//    std::cout << "Scatter: " << milliseconds/1000 << std::endl;

//    cudaEventRecord(start);

    int64_t colorassemb = ncolor / 2;
    double alpha = 1.;
    while (colorassemb > 0) {

        int64_t firsteq = (ncolor - colorassemb) * neq;

        cublasDaxpy(handle_cublas, firsteq, &alpha, &dnodal_forces_global[firsteq], 1., &dnodal_forces_global[0], 1.);

        ncolor -= colorassemb;
        colorassemb = ncolor / 2;
    }

    nodal_forces_global.Resize(neq, 1);
    cudaMemcpy(&nodal_forces_global(0, 0), dnodal_forces_global, neq * sizeof(double), cudaMemcpyDeviceToHost);

//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//    milliseconds = 0;
//    cudaEventElapsedTime(&milliseconds, start, stop);
//    std::cout << "Assemble: " << milliseconds/1000 << std::endl;


}


void TPZSolveMatrix::Multiply(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const {
    int64_t nelem = fRowSizes.size();

    int64_t nindexes = fIndexes.size();

    result.Resize(2 * nindexes, 1);
    result.Zero();

    TPZVec<REAL> expandsolution(nindexes);

    clock_t start, stop;

    start = clock();
    cblas_dgthr(nindexes, global_solution, &expandsolution[0], &fIndexes[0]);
    stop = clock();
    std::cout << "Gather: " << REAL(stop - start) / CLOCKS_PER_SEC << std::endl;

    start = clock();
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
    stop = clock();
    std::cout << "Multiply: " << REAL(stop - start) / CLOCKS_PER_SEC << std::endl;

}

void TPZSolveMatrix::ComputeSigma(TPZStack<REAL> &weight, TPZFMatrix<REAL> &result, TPZFMatrix<STATE> &sigma) {
    clock_t start, stop;

    start = clock();

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
    stop = clock();
    std::cout << "Sigma: " << REAL(stop - start) / CLOCKS_PER_SEC << std::endl;

}

void TPZSolveMatrix::MultiplyTranspose(TPZFMatrix<STATE> &intpoint_solution, TPZFMatrix<STATE> &nodal_forces_vec) {
    clock_t start, stop;

    start = clock();

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
    stop = clock();
    std::cout << "Transpose: " << REAL(stop - start) / CLOCKS_PER_SEC << std::endl;

}

void TPZSolveMatrix::ColoredAssemble(TPZFMatrix<STATE> &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) {
    clock_t start, stop;

    start = clock();


    int64_t ncolor = *std::max_element(fElemColor.begin(), fElemColor.end()) + 1;
    int64_t sz = fIndexes.size();
    int64_t neq = nodal_forces_global.Rows();
    nodal_forces_global.Resize(neq * ncolor, 1);

    cblas_dsctr(sz, nodal_forces_vec, &fIndexesColor[0], &nodal_forces_global(0, 0));

    stop = clock();
    std::cout << "Scatter: " << REAL(stop - start) / CLOCKS_PER_SEC << std::endl;


    start = clock();

    int64_t colorassemb = ncolor / 2.;
    while (colorassemb > 0) {

        int64_t firsteq = (ncolor - colorassemb) * neq;

        cblas_daxpy(firsteq, 1., &nodal_forces_global(firsteq, 0), 1., &nodal_forces_global(0, 0), 1.);

        ncolor -= colorassemb;
        colorassemb = ncolor / 2;
    }
    nodal_forces_global.Resize(neq, 1);
    stop = clock();
    std::cout << "Assemble: " << REAL(stop - start) / CLOCKS_PER_SEC << std::endl;

}


void TPZSolveMatrix::MultiplyVectors(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const {
    int64_t n_globalsol = fIndexes.size()/2; //o vetor de indices esta duplicado
    int64_t nelem = fRowSizes.size();
    int rows = fRowSizes[0];
    int cols = fColSizes[0];

    TPZFMatrix<REAL> expandsolution(2*n_globalsol,1); //vetor solucao duplicado

    cblas_dgthr(2*n_globalsol, global_solution, &expandsolution(0,0), &fIndexes[0]);

    result.Resize(2*n_globalsol,1);
    result.Zero();

    for (int i = 0; i < cols; i++) {
        cblas_dsbmv(CblasColMajor, CblasUpper, nelem * rows / 2, 0, 1., &fStorageVec[i * nelem * rows], 1, &expandsolution(i * nelem, 0), 1, 1., &result(0,0), 1);
        cblas_dsbmv(CblasColMajor, CblasUpper, nelem * rows / 2, 0, 1., &fStorageVec[i * nelem * rows + nelem * rows / 2], 1, &expandsolution(i * nelem, 0), 1, 1., &result(nelem * rows / 2,0), 1);

        cblas_dsbmv(CblasColMajor, CblasUpper, nelem * rows / 2, 0, 1., &fStorageVec[i * nelem * rows], 1, &expandsolution(i * nelem + n_globalsol, 0), 1, 1., &result(n_globalsol,0), 1);
        cblas_dsbmv(CblasColMajor, CblasUpper, nelem * rows / 2, 0, 1., &fStorageVec[i * nelem * rows + nelem * rows / 2], 1, &expandsolution(i * nelem + n_globalsol, 0), 1, 1., &result(n_globalsol + nelem * rows / 2,0), 1);

    }
}

void TPZSolveMatrix::MultiplyVectorsCUDA(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const{
    int64_t n_globalsol = fIndexes.size()/2; //o vetor de indices esta duplicado
    int64_t nelem = fRowSizes.size();
    int rows = fRowSizes[0];
    int cols = fColSizes[0];

    cudaMemcpy(dindexes, &fIndexes[0], 2*n_globalsol * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dglobal_solution, &global_solution[0], global_solution.Rows() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dstoragevec, &fStorageVec[0], nelem*fColSizes[0]*fRowSizes[0] * sizeof(double), cudaMemcpyHostToDevice);

    cusparseDgthr(handle_cusparse, 2*n_globalsol, dglobal_solution, &dexpandsolution[0], &dindexes[0], CUSPARSE_INDEX_BASE_ZERO);

    result.Resize(2*n_globalsol,1);
    result.Zero();

    cudaMemcpy(dresult, &result(0,0), 2*n_globalsol * sizeof(double), cudaMemcpyHostToDevice);

    double alpha = 1.;
    double beta = 1.;

    for (int i = 0; i < cols; i++) {
        cublasDsbmv(handle_cublas, CUBLAS_FILL_MODE_LOWER, nelem * rows / 2, 0, &alpha, &dstoragevec[i * nelem * rows], 1, &dexpandsolution[i * nelem], 1, &beta, &dresult[0], 1);
        cublasDsbmv(handle_cublas, CUBLAS_FILL_MODE_LOWER, nelem * rows / 2, 0, &alpha, &dstoragevec[i * nelem * rows + nelem * rows / 2], 1, &dexpandsolution[i * nelem], 1, &beta, &dresult[nelem * rows / 2], 1);

        cublasDsbmv(handle_cublas, CUBLAS_FILL_MODE_LOWER, nelem * rows / 2, 0, &alpha, &dstoragevec[i * nelem * rows], 1, &dexpandsolution[i * nelem + n_globalsol], 1, &beta, &dresult[n_globalsol], 1);
        cublasDsbmv(handle_cublas, CUBLAS_FILL_MODE_LOWER, nelem * rows / 2, 0, &alpha, &dstoragevec[i * nelem * rows + nelem * rows / 2], 1, &dexpandsolution[i * nelem + n_globalsol], 1, &beta, &dresult[n_globalsol + nelem * rows / 2], 1);
    }

    cudaMemcpy(&result(0, 0), dresult, 2 * n_globalsol * sizeof(double), cudaMemcpyDeviceToHost);
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
