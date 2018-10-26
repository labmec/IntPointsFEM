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
#include <chrono>
using namespace std::chrono;

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/transform.h>

__global__ void multvec(int n, double *a, double *b, double *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] += a[i]*b[i];
    }
}

void multvec_cpu(int n, double *a, double *b, double *c){
    for (int i = 0; i < n; i++) {
        c[i] += a[i]*b[i];
    }

}

struct saxpy_functor : public thrust::binary_function<double,double,double>
{
    const double a;

    saxpy_functor(double _a) : a(_a) {}

    __host__ __device__
        double operator()(const double& x, const double& y) const { 
            return a * x + y;
        }
};


void TPZSolveVector::AllocateMemory(TPZCompMesh *cmesh) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
//    cudaEventRecord(start);

    int nelem = fRowSizes.size();
    int nindexes = fIndexes.size()/2; //numero real de indices(sem duplicar)
    int neq = cmesh->NEquations();
    int npts_tot = fRow;
    int64_t ncolor = *std::max_element(fElemColor.begin(), fElemColor.end()) + 1;

    cudaMalloc(&dglobal_solution, neq * sizeof(double));
    cudaMalloc(&dindexes, 2*nindexes * sizeof(int)); //2* pq esta duplicado

    cudaEventRecord(start);
    cudaMalloc(&dstoragevec, nelem*fColSizes[0]*fRowSizes[0] * sizeof(double));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Allocate: " << milliseconds/1000 << std::endl;


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
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
//    cudaEventRecord(start);

    cudaFree(dglobal_solution);
    cudaFree(dindexes);

    cudaEventRecord(start);
    cudaFree(dstoragevec);
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
//   std::cout << "Free: " << milliseconds/1000 << std::endl;

}

void TPZSolveVector::cuSparseHandle() {
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

void TPZSolveVector::cuBlasHandle() {
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


void TPZSolveVector::Multiply(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const {
    int64_t n_globalsol = fIndexes.size()/2; //o vetor de indices esta duplicado
    int64_t nelem = fRowSizes.size();
    int rows = fRowSizes[0];
    int cols = fColSizes[0];

    TPZFMatrix<REAL> expandsolution(2*n_globalsol,1); //vetor solucao duplicado

    cblas_dgthr(2*n_globalsol, global_solution, &expandsolution(0,0), &fIndexes[0]);

    result.Resize(2*n_globalsol,1);
    result.Zero();

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    //Usando o metodo criado
        for (int i = 0; i < cols; i++) {
            multvec_cpu(nelem * rows / 2, &fStorageVec[i * nelem * rows], &expandsolution(i * nelem, 0), &result(0,0));
            multvec_cpu(nelem * rows / 2, &fStorageVec[i * nelem * rows + nelem * rows / 2], &expandsolution(i * nelem, 0), &result(nelem * rows / 2,0));

            multvec_cpu(nelem * rows / 2, &fStorageVec[i * nelem * rows], &expandsolution(i * nelem + n_globalsol, 0), &result(n_globalsol,0));
            multvec_cpu(nelem * rows / 2, &fStorageVec[i * nelem * rows + nelem * rows / 2], &expandsolution(i * nelem + n_globalsol, 0), &result(n_globalsol + nelem * rows / 2,0));
    }

    //Usando dsbmv (multiplicacao matriz-vetor com matriz de banda 0)
//    for (int i = 0; i < cols; i++) {
//        cblas_dsbmv(CblasColMajor, CblasUpper, nelem * rows / 2, 0, 1., &fStorageVec[i * nelem * rows], 1, &expandsolution(i * nelem, 0), 1, 1., &result(0,0), 1);
//        cblas_dsbmv(CblasColMajor, CblasUpper, nelem * rows / 2, 0, 1., &fStorageVec[i * nelem * rows + nelem * rows / 2], 1, &expandsolution(i * nelem, 0), 1, 1., &result(nelem * rows / 2,0), 1);
//
//        cblas_dsbmv(CblasColMajor, CblasUpper, nelem * rows / 2, 0, 1., &fStorageVec[i * nelem * rows], 1, &expandsolution(i * nelem + n_globalsol, 0), 1, 1., &result(n_globalsol,0), 1);
//        cblas_dsbmv(CblasColMajor, CblasUpper, nelem * rows / 2, 0, 1., &fStorageVec[i * nelem * rows + nelem * rows / 2], 1, &expandsolution(i * nelem + n_globalsol, 0), 1, 1., &result(n_globalsol + nelem * rows / 2,0), 1);
//
//    }

//    TPZVec<int64_t> solpos(rows*cols/2);
//    for (int i = 0; i < cols; i++) {
//        for (int j = 0; j < cols; j++) {
//            solpos[i*cols + j] = nelem * ((j + i) % cols);
//        }
//    }
//
    //Usando daxpy (multiplicacao escalar-vetor) obs: apenas 1 matriz para todos os elementos
//    for (int i = 0; i < rows * cols / 2; i++) {
//        cblas_daxpy(nelem, fStorageVec[i], &expandsolution(solpos[i],0), 1, &result((i%cols)*nelem,0), 1);
//        cblas_daxpy(nelem, fStorageVec[i + rows*cols/2], &expandsolution(solpos[i],0), 1, &result((i%cols)*nelem + nelem*rows/2,0), 1);
//
//        cblas_daxpy(nelem, fStorageVec[i], &expandsolution(solpos[i] + n_globalsol,0), 1, &result((i%cols)*nelem + n_globalsol,0), 1);
//        cblas_daxpy(nelem, fStorageVec[i + rows*cols/2], &expandsolution(solpos[i] + n_globalsol,0), 1, &result((i%cols)*nelem + n_globalsol + nelem*rows/2,0), 1);
//    }

  high_resolution_clock::time_point t2 = high_resolution_clock::now();

  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

  std::cout << "Multiply: " << time_span.count() << std::endl;

}

void TPZSolveVector::MultiplyCUDA(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int64_t n_globalsol = fIndexes.size()/2; //o vetor de indices esta duplicado
    int64_t nelem = fRowSizes.size();
    int rows = fRowSizes[0];
    int cols = fColSizes[0];

    result.Resize(2*n_globalsol,1);
    result.Zero();

    cudaMemcpy(dstoragevec, &fStorageVec[0], nelem*fColSizes[0]*fRowSizes[0] * sizeof(double), cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Copy: " << milliseconds/1000 << std::endl;

    cudaMemcpy(dindexes, &fIndexes[0], 2*n_globalsol * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dglobal_solution, &global_solution[0], global_solution.Rows() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dresult, &result(0,0), 2*n_globalsol * sizeof(double), cudaMemcpyHostToDevice);

    cusparseDgthr(handle_cusparse, 2*n_globalsol, dglobal_solution, &dexpandsolution[0], &dindexes[0], CUSPARSE_INDEX_BASE_ZERO);

    double alpha = 1.;
    double beta = 1.;

    cudaEventRecord(start);

    //Multiplicacao vetor-vetor:
    //Usando dsbmv (multiplicacao matriz-vetor com matriz de banda 0)
//    for (int i = 0; i < cols; i++) {
//        //du
//        cublasDsbmv(handle_cublas, CUBLAS_FILL_MODE_LOWER, nelem * rows / 2, 0, &alpha, &dstoragevec[i * nelem * rows], 1, &dexpandsolution[i * nelem], 1, &beta, &dresult[0], 1);
//        cublasDsbmv(handle_cublas, CUBLAS_FILL_MODE_LOWER, nelem * rows / 2, 0, &alpha, &dstoragevec[i * nelem * rows + nelem * rows / 2], 1, &dexpandsolution[i * nelem], 1, &beta, &dresult[nelem * rows / 2], 1);
//
//        //dv
//        cublasDsbmv(handle_cublas, CUBLAS_FILL_MODE_LOWER, nelem * rows / 2, 0, &alpha, &dstoragevec[i * nelem * rows], 1, &dexpandsolution[i * nelem + n_globalsol], 1, &beta, &dresult[n_globalsol], 1);
//        cublasDsbmv(handle_cublas, CUBLAS_FILL_MODE_LOWER, nelem * rows / 2, 0, &alpha, &dstoragevec[i * nelem * rows + nelem * rows / 2], 1, &dexpandsolution[i * nelem + n_globalsol], 1, &beta, &dresult[n_globalsol + nelem * rows / 2], 1);
//    }

    //Usando o metodo criado
//    dim3 dimGrid(ceil((nelem * rows / 2) / 128.0), 1, 1);
//    dim3 dimBlock(128, 1, 1);
//    for (int i = 0; i < cols; i++) {
//        //du
//        multvec<<<dimGrid, dimBlock>>>(nelem * rows / 2, &dstoragevec[i * nelem * rows], &dexpandsolution[i * nelem], &dresult[0]);
//        multvec<<<dimGrid, dimBlock>>>(nelem * rows / 2, &dstoragevec[i * nelem * rows + nelem * rows / 2], &dexpandsolution[i * nelem], &dresult[nelem * rows / 2]);
//
//        //dv
//        multvec<<<dimGrid, dimBlock>>>(nelem * rows / 2, &dstoragevec[i * nelem * rows], &dexpandsolution[i * nelem + n_globalsol], &dresult[n_globalsol]);
//        multvec<<<dimGrid, dimBlock>>>(nelem * rows / 2, &dstoragevec[i * nelem * rows + nelem * rows / 2], &dexpandsolution[i * nelem + n_globalsol], &dresult[n_globalsol + nelem * rows / 2]);
//    }
//    cudaDeviceSynchronize();

//    TPZVec<int64_t> solpos(rows*cols/2);
//    for (int i = 0; i < cols; i++) {
//        for (int j = 0; j < cols; j++) {
//            solpos[i*cols + j] = nelem * ((j + i) % cols);
//        }
//    }
//
//    //Usando daxpy (multiplicacao escalar-vetor) obs: apenas 1 matriz para todos os elementos
//    for(int i = 0; i < rows*cols/2; i++){
//        double al1 = fStorageVec[i];
//        double al2 = fStorageVec[i+rows*cols/2];
//        //du
//        cublasDaxpy(handle_cublas, nelem, &al1, &dexpandsolution[solpos[i]], 1., &dresult[(i%cols)*nelem], 1.);
//        cublasDaxpy(handle_cublas, nelem, &al2, &dexpandsolution[solpos[i]], 1., &dresult[(i%cols)*nelem + nelem*rows/2], 1.);
//
//        //dv
//        cublasDaxpy(handle_cublas, nelem, &al1, &dexpandsolution[solpos[i] + n_globalsol], 1., &dresult[(i%cols)*nelem + n_globalsol], 1.);
//        cublasDaxpy(handle_cublas, nelem, &al2, &dexpandsolution[solpos[i] + n_globalsol], 1., &dresult[(i%cols)*nelem + n_globalsol + nelem*rows/2], 1.);
//    }

//    double *dsolx;
//    double *dsoly;
//    cudaMalloc(&dsolx, rows*cols*nelem * sizeof(double));
//    cudaMalloc(&dsoly, rows*cols*nelem * sizeof(double));
//
//    //Usando Ddgmm (multiplicacao de matriz diagonal-matriz)
//    for (int i = 0; i < cols; i++) {
//        //du
//        cublasDdgmm(handle_cublas, CUBLAS_SIDE_LEFT, nelem*rows/2, 1, &dstoragevec[i * nelem * rows], nelem*rows/2, &dexpandsolution[i * nelem], 1, &dsolx[i * nelem * rows], nelem*rows/2);
//    	cublasDdgmm(handle_cublas, CUBLAS_SIDE_LEFT, nelem*rows/2, 1, &dstoragevec[i * nelem * rows + nelem*rows/2], nelem*rows/2, &dexpandsolution[i * nelem], 1, &dsolx[i * nelem * rows + nelem*rows/2], nelem*rows/2);
//
//        cublasDaxpy(handle_cublas, nelem * rows, &alpha, &dsolx[i * nelem * rows], 1., &dresult[0], 1.);
//
//        //dv
//    	cublasDdgmm(handle_cublas, CUBLAS_SIDE_LEFT, nelem*rows/2, 1, &dstoragevec[i * nelem * rows], nelem*rows/2, &dexpandsolution[i * nelem + n_globalsol], 1, &dsoly[i * nelem * rows], nelem*rows/2);
//    	cublasDdgmm(handle_cublas, CUBLAS_SIDE_LEFT, nelem*rows/2, 1, &dstoragevec[i * nelem * rows + nelem*rows/2], nelem*rows/2, &dexpandsolution[i * nelem + n_globalsol], 1, &dsoly[i * nelem * rows + nelem*rows/2], nelem*rows/2);
//
//        cublasDaxpy(handle_cublas, nelem * rows, &alpha, &dsoly[i * nelem * rows], 1., &dresult[n_globalsol], 1.);
//    }

    thrust::device_vector <double> exp(dexpandsolution, dexpandsolution + 2*n_globalsol);
    thrust::device_vector <double> stor(dstoragevec, dstoragevec + nelem*rows*cols);
    thrust::device_vector <double> solx(rows*cols*nelem);
    thrust::device_vector <double> soly(rows*cols*nelem);
    thrust::device_vector <double> res(dresult, dresult + 2*n_globalsol);

    //Usando thrust (multiplicacao vetor-vetor)
    for (int i = 0; i < cols; i++) {
        //du
		transform(&stor[i * nelem * rows], &stor[i * nelem * rows + nelem*rows/2], &exp[i*nelem], &solx[i*nelem*rows], thrust::multiplies<double>());
        transform(&stor[i * nelem * rows + nelem*rows/2], &stor[i * nelem * rows + nelem*rows ], &exp[i*nelem], &solx[i*nelem*rows + nelem*rows/2], thrust::multiplies<double>());

		transform(&solx[i * nelem * rows] , &solx[i * nelem * rows + nelem * rows], &res[0], &res[0], saxpy_functor(1));

        //dv
        transform(&stor[i * nelem * rows], &stor[i * nelem * rows + nelem*rows/2], &exp[i*nelem + n_globalsol], &soly[i*nelem*rows], thrust::multiplies<double>());
        transform(&stor[i * nelem * rows + nelem*rows/2], &stor[i * nelem * rows + nelem*rows], &exp[i*nelem + n_globalsol], &soly[i*nelem*rows + nelem*rows/2], thrust::multiplies<double>());

        transform(&soly[i * nelem * rows] , &soly[i * nelem * rows + nelem * rows], &res[n_globalsol], &res[n_globalsol], saxpy_functor(1));
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Multiply: " << milliseconds/1000 << std::endl;
   
    cudaEventRecord(start);

    cudaMemcpy(&result(0, 0), dresult, 2 * n_globalsol * sizeof(double), cudaMemcpyDeviceToHost);
    thrust::copy(res.begin(), res.end(), &result(0,0));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Copy: " << milliseconds/1000 << std::endl;

}

void TPZSolveVector::ComputeSigma( TPZVec<REAL> &weight, TPZFMatrix<REAL> &result, TPZFMatrix<STATE> &sigma) {
    return;
}

void TPZSolveVector::MultiplyTranspose(TPZFMatrix<STATE>  &sigma, TPZFMatrix<STATE> &nodal_forces_vec) {
    return;
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
