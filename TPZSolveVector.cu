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
    cudaEventRecord(start);

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

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Allocate: " << milliseconds/1000 << std::endl;

}

void TPZSolveVector::FreeMemory() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaFree(dglobal_solution);
    cudaFree(dindexes);
    cudaFree(dstoragevec);
    cudaFree(dexpandsolution);
    cudaFree(dresult);
    cudaFree(dweight);
    cudaFree(dsigma);
    cudaFree(dnodal_forces_vec);
    cudaFree(dindexescolor);
    cudaFree(dnodal_forces_global);

    cublasDestroy(handle_cublas);
    cusparseDestroy(handle_cusparse);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Free: " << milliseconds/1000 << std::endl;

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
    std::cout << "cuSPARSE: " << milliseconds/1000 << std::endl;

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
    std::cout << "cuBLAS: " << milliseconds/1000 << std::endl;
}


void TPZSolveVector::Multiply(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const {
    high_resolution_clock::time_point t1, t2;
    duration<double> time_span;

    int64_t n_globalsol = fIndexes.size()/2; //o vetor de indices esta duplicado
    int64_t nelem = fRowSizes.size();
    int rows = fRowSizes[0];
    int cols = fColSizes[0];
    TPZFMatrix<REAL> expandsolution(2*n_globalsol,1); //vetor solucao duplicado
    result.Resize(2*n_globalsol,1);
    result.Zero();

    t1 = high_resolution_clock::now();
    cblas_dgthr(2*n_globalsol, global_solution, &expandsolution(0,0), &fIndexes[0]);

    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "Gather: " << time_span.count() << std::endl;

    t1 = high_resolution_clock::now();

    //Usando o metodo criado
        for (int i = 0; i < cols; i++) {
            multvec_cpu(nelem * rows / 2, &fStorageVec[i * nelem * rows], &expandsolution(i * nelem, 0), &result(0,0));
            multvec_cpu(nelem * rows / 2, &fStorageVec[i * nelem * rows + nelem * rows / 2], &expandsolution(i * nelem, 0), &result(nelem * rows / 2,0));

            multvec_cpu(nelem * rows / 2, &fStorageVec[i * nelem * rows], &expandsolution(i * nelem + n_globalsol, 0), &result(n_globalsol,0));
            multvec_cpu(nelem * rows / 2, &fStorageVec[i * nelem * rows + nelem * rows / 2], &expandsolution(i * nelem + n_globalsol, 0), &result(n_globalsol + nelem * rows / 2,0));
    }

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  std::cout << "Multiply: " << time_span.count() << std::endl;
}

void TPZSolveVector::MultiplyCUDA(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int64_t n_globalsol = fIndexes.size()/2; //o vetor de indices esta duplicado
    int64_t nelem = fRowSizes.size();
    int rows = fRowSizes[0];
    int cols = fColSizes[0];
    result.Resize(2*n_globalsol,1);
    result.Zero();

    cudaEventRecord(start);

    cudaMemcpy(dindexes, &fIndexes[0], 2*n_globalsol * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dglobal_solution, &global_solution[0], global_solution.Rows() * sizeof(double), cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Copy: " << milliseconds/1000 << std::endl;

    cudaEventRecord(start);

    cusparseDgthr(handle_cusparse, 2*n_globalsol, dglobal_solution, &dexpandsolution[0], &dindexes[0], CUSPARSE_INDEX_BASE_ZERO);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Gather: " << milliseconds/1000 << std::endl;

    cudaEventRecord(start);

    cudaMemcpy(dstoragevec, &fStorageVec[0], nelem*fColSizes[0]*fRowSizes[0] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dresult, &result(0,0), 2*n_globalsol * sizeof(double), cudaMemcpyHostToDevice);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Copy: " << milliseconds/1000 << std::endl;

    double alpha = 1.;
    double beta = 1.;

    cudaEventRecord(start);

    //Usando o metodo criado
    dim3 dimGrid(ceil((nelem * rows / 2) / 128.0), 1, 1);
    dim3 dimBlock(128, 1, 1);
    for (int i = 0; i < cols; i++) {
        //du
        multvec<<<dimGrid, dimBlock>>>(nelem * rows / 2, &dstoragevec[i * nelem * rows], &dexpandsolution[i * nelem], &dresult[0]);
        multvec<<<dimGrid, dimBlock>>>(nelem * rows / 2, &dstoragevec[i * nelem * rows + nelem * rows / 2], &dexpandsolution[i * nelem], &dresult[nelem * rows / 2]);

        //dv
        multvec<<<dimGrid, dimBlock>>>(nelem * rows / 2, &dstoragevec[i * nelem * rows], &dexpandsolution[i * nelem + n_globalsol], &dresult[n_globalsol]);
        multvec<<<dimGrid, dimBlock>>>(nelem * rows / 2, &dstoragevec[i * nelem * rows + nelem * rows / 2], &dexpandsolution[i * nelem + n_globalsol], &dresult[n_globalsol + nelem * rows / 2]);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Multiply: " << milliseconds/1000 << std::endl;
   
    cudaEventRecord(start);

    cudaMemcpy(&result(0, 0), dresult, 2 * n_globalsol * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Copy: " << milliseconds/1000 << std::endl;

}

void TPZSolveVector::ComputeSigma( TPZVec<REAL> &weight, TPZFMatrix<REAL> &result, TPZFMatrix<STATE> &sigma) {
    high_resolution_clock::time_point t1, t2;
    duration<double> time_span;

    REAL E = 200000000.;
    REAL nu =0.30;
    int nelem = fRowSizes.size();
    int npts_el = fRow/nelem;
    TPZFMatrix<REAL> aux(nelem,1,0);
    sigma.Resize(2*fRow,1);
    sigma.Zero();

    t1 = high_resolution_clock::now();

    for (int64_t ipts=0; ipts< npts_el/2; ipts++) {
        //sigma xx
        cblas_daxpy(nelem, nu, &result((2*ipts + npts_el + 1)*nelem ,0), 1, &aux(0,0), 1);
        cblas_daxpy(nelem, 1, &result(2*ipts*nelem ,0), 1, &aux(0,0), 1);
        multvec_cpu(nelem, &weight[ipts*nelem], &aux(0,0), &sigma(2*ipts*nelem,0));
        cblas_dscal (nelem, E/(1.-nu*nu), &sigma(2*ipts*nelem,0), 1);
        aux.Zero();

        //sigma yy
        cblas_daxpy(nelem, nu, &result(2*ipts*nelem,0), 1, &aux(0,0), 1);
        cblas_daxpy(nelem, 1, &result((2*ipts + npts_el + 1)*nelem,0), 1, &aux(0,0), 1);
        multvec_cpu(nelem, &weight[ipts*nelem], &aux(0,0), &sigma((2*ipts+npts_el+1)*nelem,0));
        cblas_dscal (nelem, E/(1.-nu*nu), &sigma((2*ipts+npts_el+1)*nelem,0), 1);
        aux.Zero();

        //sigma xy
        cblas_daxpy(nelem, 1, &result((2*ipts + 1)*nelem,0), 1, &aux(0,0), 1);
        cblas_daxpy(nelem, 1, &result((2*ipts + npts_el)*nelem,0), 1, &aux(0,0), 1);
        multvec_cpu(nelem, &weight[ipts*nelem], &aux(0,0), &sigma((2*ipts+1)*nelem,0));
        cblas_dscal (nelem, E/(1.-nu*nu)*(1.-nu), &sigma((2*ipts+1)*nelem,0), 1);
        aux.Zero();

        cblas_daxpy(nelem, 1, &sigma((2*ipts+1)*nelem,0), 1, &sigma((2*ipts+npts_el)*nelem,0), 1);
    }
    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "Sigma: " << time_span.count() << std::endl;
}

void TPZSolveVector::ComputeSigmaCUDA( TPZVec<REAL> &weight, TPZFMatrix<REAL> &result, TPZFMatrix<STATE> &sigma) {
    REAL E = 200000000.;
    REAL nu =0.30;
    double alpha = 1.;
    double coef = E/(1.-nu*nu);
    int nelem = fRowSizes.size();
    int npts_el = fRow/nelem;
    TPZFMatrix<REAL> aux(nelem,1,0.);
    double *daux;
    cudaMalloc(&daux, nelem*sizeof(double));
    sigma.Resize(2*fRow,1);
    sigma.Zero();

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMemcpy(dweight, &weight[0], weight.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(daux, &aux(0,0), nelem * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dsigma, &sigma(0,0), 2*fRow * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dresult, &result(0,0), result.Rows() * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Copy: " << milliseconds/1000 << std::endl;

    cudaEventRecord(start);
    int rows = fRowSizes[0];
    dim3 dimGrid(ceil(nelem / 128.0), 1, 1);
    dim3 dimBlock(128, 1, 1);
    for (int64_t ipts=0; ipts< npts_el/2; ipts++) {
        //sigma xx
        cublasDaxpy(handle_cublas, nelem, &nu, &dresult[(2*ipts + npts_el + 1)*nelem], 1., &daux[0], 1);
        cublasDaxpy(handle_cublas, nelem, &alpha, &dresult[2*ipts*nelem], 1, &daux[0], 1);
        multvec<<<dimGrid, dimBlock>>>(nelem, &dweight[ipts*nelem], &daux[0], &dsigma[2*ipts*nelem]);
	    cublasDscal(handle_cublas, nelem, &coef, &dsigma[2*ipts*nelem], 1);
        aux.Zero();
        cudaMemcpy(daux, &aux(0,0), nelem * sizeof(double), cudaMemcpyHostToDevice);


        //sigma yy
        cublasDaxpy(handle_cublas, nelem, &nu, &dresult[2*ipts*nelem], 1., &daux[0], 1);
        cublasDaxpy(handle_cublas, nelem, &alpha, &dresult[(2*ipts + npts_el + 1)*nelem], 1, &daux[0], 1);
        multvec<<<dimGrid, dimBlock>>>(nelem, &dweight[ipts*nelem], &daux[0], &dsigma[(2*ipts+npts_el+1)*nelem]);
        cublasDscal(handle_cublas, nelem,&coef, &dsigma[(2*ipts+npts_el+1)*nelem], 1);
        aux.Zero();
        cudaMemcpy(daux, &aux(0,0), nelem * sizeof(double), cudaMemcpyHostToDevice);

        //sigma xy
        cublasDaxpy(handle_cublas, nelem, &alpha, &dresult[(2*ipts + 1)*nelem], 1., &daux[0], 1);
        cublasDaxpy(handle_cublas, nelem, &alpha, &dresult[(2*ipts + npts_el)*nelem], 1, &daux[0], 1);
        multvec<<<dimGrid, dimBlock>>>(nelem, &dweight[ipts*nelem], &daux[0], &dsigma[(2*ipts+1)*nelem]);
        cublasDscal(handle_cublas, nelem,&coef, &dsigma[(2*ipts+1)*nelem], 1);
        aux.Zero();
        cudaMemcpy(daux, &aux(0,0), nelem * sizeof(double), cudaMemcpyHostToDevice);

        cublasDaxpy(handle_cublas, nelem, &alpha, &dsigma[(2*ipts+1)*nelem], 1., &dsigma[(2*ipts+npts_el)*nelem], 1);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Sigma: " << milliseconds/1000 << std::endl;

    cudaEventRecord(start);

    cudaMemcpy(&sigma(0, 0), dsigma, 2 * fRow * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Copy: " << milliseconds/1000 << std::endl;
}

void TPZSolveVector::MultiplyTranspose(TPZFMatrix<STATE>  &sigma, TPZFMatrix<STATE> &nodal_forces_vec) {
    high_resolution_clock::time_point t1, t2;
    duration<double> time_span;

    int64_t npts_tot = fRow;
    nodal_forces_vec.Resize(npts_tot,1);
    nodal_forces_vec.Zero();

    int cols = fColSizes[0];
    int rows = fRowSizes[0];
    int  nelem = fRowSizes.size();
    t1 = high_resolution_clock::now();

    for (int i = 0; i < cols; i++) {
        //Fx
        multvec_cpu((cols-i)*nelem, &fStorageVec[(((cols-i)%cols)*rows + i)*nelem], &sigma(i * nelem, 0), &nodal_forces_vec(0, 0));
        multvec_cpu(i*nelem, &fStorageVec[((cols-i)%cols)*rows*nelem], &sigma(0, 0), &nodal_forces_vec(nelem*((cols - i)%cols), 0));

        multvec_cpu((cols-i)*nelem, &fStorageVec[(((cols-i)%cols)*rows + i)*nelem + cols*nelem], &sigma(i * nelem + cols*nelem, 0), &nodal_forces_vec(0, 0));
        multvec_cpu(i*nelem, &fStorageVec[((cols-i)%cols)*rows*nelem + cols*nelem], &sigma(cols*nelem, 0), &nodal_forces_vec(nelem*((cols - i)%cols), 0));

        //Fy
        multvec_cpu((cols-i)*nelem, &fStorageVec[(((cols-i)%cols)*rows + i)*nelem], &sigma(i * nelem + fRow, 0),  &nodal_forces_vec(npts_tot/2, 0));
        multvec_cpu(i*nelem, &fStorageVec[((cols-i)%cols)*rows*nelem], &sigma(fRow, 0), &nodal_forces_vec(npts_tot/2 + nelem*((cols - i)%cols), 0));

        multvec_cpu((cols-i)*nelem, &fStorageVec[(((cols-i)%cols)*rows + i)*nelem + cols*nelem], &sigma(i * nelem + cols*nelem + fRow, 0), &nodal_forces_vec(npts_tot/2, 0));
        multvec_cpu(i*nelem, &fStorageVec[((cols-i)%cols)*rows*nelem + cols*nelem], &sigma(cols*nelem + fRow, 0), &nodal_forces_vec(npts_tot/2 + nelem*((cols - i)%cols), 0));
    }
    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "Transpose: " << time_span.count() << std::endl;
}

void TPZSolveVector::MultiplyTransposeCUDA(TPZFMatrix<STATE>  &sigma, TPZFMatrix<STATE> &nodal_forces_vec) {
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int64_t npts_tot = fRow;
    nodal_forces_vec.Resize(npts_tot,1);
    nodal_forces_vec.Zero();

    int cols = fColSizes[0];
    int rows = fRowSizes[0];
    int  nelem = fRowSizes.size();

    dim3 dimBlock(128, 1, 1);

    cudaEventRecord(start);
    cudaMemcpy(dnodal_forces_vec, &nodal_forces_vec(0,0), npts_tot * sizeof(double), cudaMemcpyHostToDevice);

    for (int i = 0; i < cols; i++) {
        //Fx
        multvec<<<ceil(((cols-i)*nelem)/ 128.0), dimBlock>>>((cols-i)*nelem, &dstoragevec[(((cols-i)%cols)*rows + i)*nelem], &dsigma[i * nelem], &dnodal_forces_vec[0]);
        multvec<<<ceil((i*nelem) / 128.0), dimBlock>>>(i*nelem, &dstoragevec[((cols-i)%cols)*rows*nelem], &dsigma[0], &dnodal_forces_vec[nelem*((cols - i)%cols)]);

        multvec<<<ceil(((cols-i)*nelem)/ 128.0), dimBlock>>>((cols-i)*nelem, &dstoragevec[(((cols-i)%cols)*rows + i)*nelem + cols*nelem], &dsigma[i * nelem + cols*nelem], &dnodal_forces_vec[0]);
        multvec<<<ceil((i*nelem) / 128.0), dimBlock>>>(i*nelem, &dstoragevec[((cols-i)%cols)*rows*nelem + cols*nelem], &dsigma[cols*nelem], &dnodal_forces_vec[nelem*((cols - i)%cols)]);

        //Fy
        multvec<<<ceil(((cols-i)*nelem)/ 128.0), dimBlock>>>((cols-i)*nelem, &dstoragevec[(((cols-i)%cols)*rows + i)*nelem], &dsigma[i * nelem + fRow],  &dnodal_forces_vec[npts_tot/2]);
        multvec<<<ceil((i*nelem) / 128.0), dimBlock>>>(i*nelem, &dstoragevec[((cols-i)%cols)*rows*nelem], &dsigma[fRow], &dnodal_forces_vec[npts_tot/2 + nelem*((cols - i)%cols)]);

        multvec<<<ceil(((cols-i)*nelem)/ 128.0), dimBlock>>>((cols-i)*nelem, &dstoragevec[(((cols-i)%cols)*rows + i)*nelem + cols*nelem], &dsigma[i * nelem + cols*nelem + fRow], &dnodal_forces_vec[npts_tot/2]);
        multvec<<<ceil((i*nelem) / 128.0), dimBlock>>>(i*nelem, &dstoragevec[((cols-i)%cols)*rows*nelem + cols*nelem], &dsigma[cols*nelem + fRow], &dnodal_forces_vec[npts_tot/2 + nelem*((cols - i)%cols)]);
    }
    cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Transpose: " << milliseconds/1000 << std::endl;

    cudaEventRecord(start);
    cudaMemcpy(&nodal_forces_vec(0, 0), dnodal_forces_vec, npts_tot * sizeof(double), cudaMemcpyDeviceToHost);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Copy: " << milliseconds/1000 << std::endl;
}

void TPZSolveVector::ColoredAssembleCUDA(TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) {
    int64_t ncolor = *std::max_element(fElemColor.begin(), fElemColor.end()) + 1;
    int64_t nindexes = fIndexesColor.size();
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
    std::cout << "Copy: " << milliseconds/1000 << std::endl;

    cudaEventRecord(start);

    cusparseDsctr(handle_cusparse, nindexes, dnodal_forces_vec, &dindexescolor[0], &dnodal_forces_global[0], CUSPARSE_INDEX_BASE_ZERO);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Scatter: " << milliseconds/1000 << std::endl;

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
    std::cout << "Assemble: " << milliseconds/1000 << std::endl;

    cudaEventRecord(start);

    cudaMemcpy(&nodal_forces_global(0, 0), dnodal_forces_global, neq * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Copy: " << milliseconds/1000 << std::endl;

}


void TPZSolveVector::ColoredAssemble(TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) {
    high_resolution_clock::time_point t1, t2;
    duration<double> time_span;

    int64_t ncolor = *std::max_element(fElemColor.begin(), fElemColor.end())+1;
    int64_t sz = fIndexesColor.size();
    int64_t neq = nodal_forces_global.Rows();
    nodal_forces_global.Resize(neq*ncolor,1);

    t1 = high_resolution_clock::now();

    cblas_dsctr(sz, nodal_forces_vec, &fIndexesColor[0], &nodal_forces_global(0,0));

    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "Scatter: " << time_span.count() << std::endl;

    t1 = high_resolution_clock::now();
    int64_t colorassemb = ncolor / 2.;
    while (colorassemb > 0) {

        int64_t firsteq = (ncolor - colorassemb) * neq;
        cblas_daxpy(firsteq, 1., &nodal_forces_global(firsteq, 0), 1., &nodal_forces_global(0, 0), 1.);

        ncolor -= colorassemb;
        colorassemb = ncolor/2;
    }
    nodal_forces_global.Resize(neq, 1);

    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "Scatter: " << time_span.count() << std::endl;
}

void TPZSolveVector::TraditionalAssemble(TPZFMatrix<STATE> &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) const {
    for (int64_t ir=0; ir<fRow/2; ir++) {
        nodal_forces_global(fIndexes[ir], 0) += nodal_forces_vec(ir, 0);
        nodal_forces_global(fIndexes[ir+fRow], 0) += nodal_forces_vec(ir+fRow/2, 0);
    }
}

void TPZSolveVector::ColoringElements(TPZCompMesh *cmesh) const {
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
            //cel->Reference()->SetMaterialId(contcolor);

            for (icon = 0; icon < ncon; icon++) {
                connects_vec[connectlist[icon]] = 1;
            }
        }
        contcolor++;
        connects_vec.Fill(0);
    }
    int64_t ind = fIndexes.size()/2;
    TPZVec<REAL> indexes(ind);

    for (int i = 0; i < ind/2; i++) {
        indexes[ind/2 + i] = fIndexes[ind+i];
        indexes[i] = fIndexes[i];
    }

    int64_t nelem = fRowSizes.size();
    int64_t neq = cmesh->NEquations();
    for (int64_t iel = 0; iel < nelem; iel++) {
        int64_t cols = fColSizes[iel];

        for (int64_t icols = 0; icols < cols; icols++) {
            fIndexesColor[iel + icols*nelem] = indexes[iel + icols*nelem] + fElemColor[iel]*neq;
            fIndexesColor[iel + icols*nelem + fRow/2] = indexes[iel + icols*nelem + fRow/2] + fElemColor[iel]*neq;
        }
    }
}
