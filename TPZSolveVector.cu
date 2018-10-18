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


void TPZSolveVector::AllocateMemory(TPZCompMesh *cmesh) {
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

void TPZSolveVector::FreeMemory() {
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    cudaEventRecord(start);

    cudaFree(dglobal_solution);
    cudaFree(dindexes);
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

void TPZSolveVector::cuSparseHandle() {
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

void TPZSolveVector::cuBlasHandle() {
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


void TPZSolveVector::Multiply(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const {
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

void TPZSolveVector::MultiplyCUDA(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const{
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

void TPZSolveVector::TraditionalAssemble(TPZFMatrix<STATE> &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) const {
    for (int64_t ir = 0; ir < fRow; ir++) {
        nodal_forces_global(fIndexes[ir], 0) += nodal_forces_vec(ir, 0);
    }
}

void TPZSolveVector::ColoringElements(TPZCompMesh *cmesh) const {
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
