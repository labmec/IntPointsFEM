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
#include <thrust/transform.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>
using namespace thrust::placeholders;

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

void sigmaxx(int nelem, double *weight, double *dudx, double *dvdy, double *sigxx){
    REAL E = 200000000.;
    REAL nu =0.30;
    for (int i = 0; i < nelem; i++) {
        sigxx[i] = weight[i] * (dudx[i] * E * (1. - nu) / ((1. - 2 * nu) * (1. + nu))  + dvdy[i] * E * nu / ((1. - 2 * nu) * (1. + nu))); //plane strain
//        sigxx[i] = weight[i]*E/(1.-nu*nu)*(dudx[i] + nu*dvdy[i]);
    }
}

void sigmayy(int nelem, double *weight, double *dudx, double *dvdy, double *sigyy){
    REAL E = 200000000.;
    REAL nu =0.30;
    for (int i = 0; i < nelem; i++) {
        sigyy[i] = weight[i] * (dudx[i] * E * nu / ((1. - 2 * nu) * (1. + nu)) + dvdy[i] * E * (1. - nu) / ((1. - 2 * nu) * (1. + nu))); //plane strain
//        sigyy[i] = weight[i]*E/(1.-nu*nu)*(nu*dudx[i] + dvdy[i]);
    }
}

void sigmaxy(int nelem, double *weight, double *dvdx, double *dudy, double *sigxy, double *sigyx){
    REAL E = 200000000.;
    REAL nu =0.30;
    for (int i = 0; i < nelem; i++) {
        sigxy[i] = weight[i] * E / (2 * (1. + nu)) * (dvdx[i] + dudy[i]); //plane strain
        //        sigxy[i] = weight[i]*E/(1.+nu)*(dvdx[i] + dudy[i]);
        sigyx[i] = sigxy[i];
    }
}

__global__ void sigmaxxkernel(int nelem, double *weight, double *dudx, double *dvdy, double *sigxx){
    REAL E = 200000000.;
    REAL nu =0.30;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nelem) {
        sigxx[i] = weight[i] * (dudx[i] * E * (1. - nu) / ((1. - 2 * nu) * (1. + nu))  + dvdy[i] * E * nu / ((1. - 2 * nu) * (1. + nu))); //plane strain
//        sigxx[i] = weight[i]*E/(1.-nu*nu)*(dudx[i] + nu*dvdy[i]);
    }
}

__global__ void sigmayykernel(int nelem, double *weight, double *dudx, double *dvdy, double *sigyy){
    REAL E = 200000000.;
    REAL nu =0.30;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nelem) {
        sigyy[i] = weight[i] * (dudx[i] * E * nu / ((1. - 2 * nu) * (1. + nu)) + dvdy[i] * E * (1. - nu) / ((1. - 2 * nu) * (1. + nu))); //plane strain
//        sigyy[i] = weight[i]*E/(1.-nu*nu)*(nu*dudx[i] + dvdy[i]);
    }
}

__global__ void sigmaxykernel(int nelem, double *weight, double *dvdx, double *dudy, double *sigxy, double *sigyx){
    REAL E = 200000000.;
    REAL nu =0.30;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nelem) {
        sigxy[i] = weight[i] * E / (2 * (1. + nu)) * (dvdx[i] + dudy[i]); //plane strain
//        sigxy[i] = weight[i]*E/(1.+nu)*(dvdx[i] + dudy[i]);
        sigyx[i] = sigxy[i];
    }
}


void TPZSolveVector::AllocateMemory(TPZCompMesh *cmesh) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int nelem = fRowSizes.size();
    int nindexes = fIndexes.size(); //numero real de indices(sem duplicar)
    int neq = cmesh->NEquations();
    int npts_tot = fRow;
    int64_t ncolor = *std::max_element(fElemColor.begin(), fElemColor.end()) + 1;

    cudaMalloc(&dglobal_solution, neq * sizeof(double));
    cudaMalloc(&dindexes, nindexes * sizeof(int));
    cudaMalloc(&dstoragevec, nelem*fColSizes[0]*fRowSizes[0] * sizeof(double));
    cudaMalloc(&dexpandsolution, nindexes * sizeof(double)); //sol duplicada
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

void TPZSolveVector::MultiplyCUDA(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const{
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int64_t n_globalsol = fIndexes.size();
    int64_t nelem = fRowSizes.size();
    int rows = fRowSizes[0];
    int cols = fColSizes[0];
    result.Resize(2*n_globalsol,1);
    result.Zero();

    cudaEventRecord(start);

    cudaMemcpy(dindexes, &fIndexes[0], n_globalsol * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dglobal_solution, &global_solution[0], global_solution.Rows() * sizeof(double), cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Copy: " << milliseconds/1000 << std::endl;

    cudaEventRecord(start);
    thrust::device_ptr<int> tindexes = thrust::device_pointer_cast(dindexes);
    thrust::device_ptr<double> tglobal_solution = thrust::device_pointer_cast(dglobal_solution);
    thrust::device_ptr<double> texpandsolution = thrust::device_pointer_cast(dexpandsolution);

    thrust::gather(thrust::device, &tindexes[0], &tindexes[n_globalsol], &tglobal_solution[0], &texpandsolution[0]);
//    cusparseDgthr(handle_cusparse, n_globalsol, dglobal_solution, &dexpandsolution[0], &dindexes[0], CUSPARSE_INDEX_BASE_ZERO);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Gather: " << milliseconds/1000 << std::endl;

    TPZFMatrix<REAL> expandsolution(n_globalsol,1);
    cudaMemcpy(&expandsolution(0,0), dexpandsolution, n_globalsol * sizeof(double), cudaMemcpyDeviceToHost);


    TPZFMatrix<REAL> expand_x(n_globalsol/2,1,&expandsolution(0,0),n_globalsol/2);
    expand_x.Resize(n_globalsol,1);
    expand_x.AddSub(n_globalsol/2,0,expand_x);

    TPZFMatrix<REAL> expand_y(n_globalsol/2,1,&expandsolution(n_globalsol/2,0),n_globalsol/2);
    expand_y.Resize(n_globalsol,1);
    expand_y.AddSub(n_globalsol/2,0,expand_y);

    cudaEventRecord(start);

    cudaMemcpy(dstoragevec, &fStorageVec[0], nelem*fColSizes[0]*fRowSizes[0] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dresult, &result(0,0), 2*n_globalsol * sizeof(double), cudaMemcpyHostToDevice);
    double *dexpand_x;
    double *dexpand_y;
    cudaMalloc(&dexpand_x, n_globalsol*sizeof(double));
    cudaMalloc(&dexpand_y, n_globalsol*sizeof(double));
    cudaMemcpy(dexpand_x, &expand_x(0,0), n_globalsol * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dexpand_y, &expand_y(0,0), n_globalsol * sizeof(double), cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Copy: " << milliseconds/1000 << std::endl;

    cudaEventRecord(start);
    double alpha = 1.;

    thrust::device_ptr<double> tstorage = thrust::device_pointer_cast(dstoragevec);
    thrust::device_ptr<double> texpand_x = thrust::device_pointer_cast(dexpand_x);
    thrust::device_ptr<double> texpand_y = thrust::device_pointer_cast(dexpand_y);
    thrust::device_ptr<double> tres = thrust::device_pointer_cast(dresult);

    double *dsol;
    cudaMalloc(&dsol, nelem * rows*sizeof(double));
    thrust::device_ptr<double> tsol = thrust::device_pointer_cast(dsol);

    for (int i = 0; i < cols; i++) {
        //du
	thrust::transform(&tstorage[i * nelem * rows], &tstorage[i * nelem * rows + nelem * rows / 2], &texpand_x[i * nelem], &tsol[0], thrust::multiplies<double>());
	thrust::transform(&tstorage[i * nelem * rows + nelem * rows / 2], &tstorage[i * nelem * rows + nelem * rows / 2 + nelem * rows / 2], &texpand_x[i * nelem], &tsol[nelem * rows / 2], thrust::multiplies<double>());
	thrust::transform(&tsol[0], &tsol[nelem * rows], &tres[0], &tres[0], thrust::plus<double>());

        //dv
        thrust::transform(&tstorage[i * nelem * rows], &tstorage[i * nelem * rows + nelem * rows / 2], &texpand_y[i * nelem], &tsol[0], thrust::multiplies<double>());
        thrust::transform(&tstorage[i * nelem * rows + nelem * rows / 2], &tstorage[i * nelem * rows + nelem * rows / 2 + nelem * rows / 2], &texpand_y[i * nelem], &tsol[nelem * rows / 2], thrust::multiplies<double>());
        thrust::transform(&tsol[0], &tsol[nelem * rows], &tres[n_globalsol], &tres[n_globalsol], thrust::plus<double>());
    }

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
    cudaFree(dsol);
    cudaFree(dexpand_x);
    cudaFree(dexpand_y);

}

void TPZSolveVector::ComputeSigmaCUDA( TPZVec<REAL> &weight, TPZFMatrix<REAL> &result, TPZFMatrix<STATE> &sigma) {
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int nelem = fRowSizes.size();
    int npts_el = fRow/nelem;
    sigma.Resize(2*fRow,1);
    sigma.Zero();

    cudaEventRecord(start);

    cudaMemcpy(dweight, &weight[0], weight.size() * sizeof(double), cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Copy: " << milliseconds/1000 << std::endl;

    cudaEventRecord(start);
    dim3 dimGrid(ceil(nelem/512.0), 1, 1);
    dim3 dimBlock(512, 1, 1);

    for (int64_t ipts=0; ipts< npts_el/2; ipts++) {
        sigmaxxkernel<<<dimGrid, dimBlock>>>(nelem, &dweight[ipts*nelem], &dresult[2*ipts*nelem], &dresult[(2*ipts + npts_el + 1)*nelem], &dsigma[2*ipts*nelem]);
        sigmayykernel<<<dimGrid, dimBlock>>>(nelem, &dweight[ipts*nelem], &dresult[(2*ipts + npts_el + 1)*nelem], &dresult[2*ipts*nelem], &dsigma[(2*ipts+npts_el+1)*nelem]);
        sigmaxykernel<<<dimGrid, dimBlock>>>(nelem, &dweight[ipts*nelem], &dresult[(2*ipts + npts_el)*nelem], &dresult[(2*ipts + 1)*nelem], &dsigma[(2*ipts+1)*nelem], &dsigma[(2*ipts+npts_el)*nelem]);
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


    cudaEventRecord(start);
    cudaMemcpy(dnodal_forces_vec, &nodal_forces_vec(0,0), npts_tot * sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Copy: " << milliseconds/1000 << std::endl;


    cudaEventRecord(start);

    thrust::device_ptr<double> tstoragevec = thrust::device_pointer_cast(dstoragevec);
    thrust::device_ptr<double> tsigma = thrust::device_pointer_cast(dsigma);
    thrust::device_ptr<double> tnodal_forces_vec = thrust::device_pointer_cast(dnodal_forces_vec);

    double alpha =1.;

    double *dforce; //se for usar cuBLAS
    cudaMalloc(&dforce, nelem*cols*sizeof(double));
    thrust::device_ptr<double> tforce = thrust::device_pointer_cast(dforce);

    for (int i = 0; i < cols; i++) {
        //Fx
	thrust::transform(&tstoragevec[(((cols-i)%cols)*rows + i)*nelem], &tstoragevec[(((cols-i)%cols)*rows + i)*nelem + (cols-i)*nelem], &tsigma[i * nelem], &tforce[0], thrust::multiplies<double>());
        thrust::transform(&tstoragevec[((cols-i)%cols)*rows*nelem], &tstoragevec[((cols-i)%cols)*rows*nelem + i*nelem], &tsigma[0], &tforce[nelem*((cols - i)%cols)], thrust::multiplies<double>());
        thrust::transform(&tforce[0], &tforce[nelem * cols], &tnodal_forces_vec[0], &tnodal_forces_vec[0], thrust::plus<double>());

        thrust::transform(&tstoragevec[(((cols-i)%cols)*rows + i)*nelem + cols*nelem], &tstoragevec[(((cols-i)%cols)*rows + i)*nelem + cols*nelem + (cols-i)*nelem], &tsigma[i * nelem + cols*nelem], &tforce[0], thrust::multiplies<double>());
        thrust::transform(&tstoragevec[((cols-i)%cols)*rows*nelem + cols*nelem], &tstoragevec[((cols-i)%cols)*rows*nelem + cols*nelem + i*nelem], &tsigma[cols*nelem], &tforce[nelem*((cols - i)%cols)], thrust::multiplies<double>());
        thrust::transform(&tforce[0], &tforce[nelem * cols], &tnodal_forces_vec[0], &tnodal_forces_vec[0], thrust::plus<double>());


        //Fy
  	thrust::transform(&tstoragevec[(((cols-i)%cols)*rows + i)*nelem], &tstoragevec[(((cols-i)%cols)*rows + i)*nelem + (cols-i)*nelem], &tsigma[i * nelem + fRow], &tforce[0], thrust::multiplies<double>());
        thrust::transform(&tstoragevec[((cols-i)%cols)*rows*nelem], &tstoragevec[((cols-i)%cols)*rows*nelem + i*nelem], &tsigma[fRow], &tforce[nelem*((cols - i)%cols)], thrust::multiplies<double>());
        thrust::transform(&tforce[0], &tforce[nelem * cols], &tnodal_forces_vec[npts_tot/2], &tnodal_forces_vec[npts_tot/2], thrust::plus<double>());

        thrust::transform(&tstoragevec[(((cols-i)%cols)*rows + i)*nelem + cols*nelem], &tstoragevec[(((cols-i)%cols)*rows + i)*nelem + cols*nelem + (cols-i)*nelem], &tsigma[i * nelem + cols*nelem + fRow], &tforce[0], thrust::multiplies<double>());
        thrust::transform(&tstoragevec[((cols-i)%cols)*rows*nelem + cols*nelem], &tstoragevec[((cols-i)%cols)*rows*nelem + cols*nelem + i*nelem], &tsigma[cols*nelem + fRow], &tforce[nelem*((cols - i)%cols)], thrust::multiplies<double>());
        thrust::transform(&tforce[0], &tforce[nelem * cols], &tnodal_forces_vec[npts_tot/2], &tnodal_forces_vec[npts_tot/2], thrust::plus<double>());

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
cudaFree(dforce);

}

void TPZSolveVector::ColoredAssembleCUDA(TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) {
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int64_t ncolor = *std::max_element(fElemColor.begin(), fElemColor.end()) + 1;
    int64_t nindexes = fIndexes.size();
    int64_t neq = nodal_forces_global.Rows();
    int64_t npts_tot = fRow;

    nodal_forces_global.Resize(neq * ncolor, 1);
    nodal_forces_global.Zero();

    cudaEventRecord(start);

    cudaMemcpy(dindexescolor, &fIndexesColor[0], nindexes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dnodal_forces_global, &nodal_forces_global(0, 0), ncolor * neq * sizeof(double), cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Copy: " << milliseconds/1000 << std::endl;

    cudaEventRecord(start);
    thrust::device_ptr<double> tnodal_forces_vec = thrust::device_pointer_cast(dnodal_forces_vec);
    thrust::device_ptr<int> tindexescolor = thrust::device_pointer_cast(dindexescolor);
    thrust::device_ptr<double> tnodal_forces_global = thrust::device_pointer_cast(dnodal_forces_global);


    thrust::scatter(thrust::device, &dnodal_forces_vec[0], &dnodal_forces_vec[nindexes], dindexescolor, dnodal_forces_global);
//    cusparseDsctr(handle_cusparse, nindexes, dnodal_forces_vec, &dindexescolor[0], &dnodal_forces_global[0], CUSPARSE_INDEX_BASE_ZERO);

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

        thrust::transform(&tnodal_forces_global[firsteq], &tnodal_forces_global[firsteq + firsteq], &tnodal_forces_global[0], &tnodal_forces_global[0], thrust::plus<double>());

//        cublasDaxpy(handle_cublas, firsteq, &alpha, &dnodal_forces_global[firsteq], 1., &dnodal_forces_global[0], 1.);

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

void TPZSolveVector::Multiply(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const {
    high_resolution_clock::time_point t1, t2;
    duration<double> time_span;

    int64_t n_globalsol = fIndexes.size(); 
    int64_t nelem = fRowSizes.size();
    int rows = fRowSizes[0];
    int cols = fColSizes[0];
    TPZFMatrix<REAL> expandsolution(n_globalsol,1); //vetor solucao duplicado
    result.Resize(2*n_globalsol,1);
    result.Zero();

    t1 = high_resolution_clock::now();
    cblas_dgthr(n_globalsol, global_solution, &expandsolution(0,0), &fIndexes[0]);

    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "Gather: " << time_span.count() << std::endl;

    TPZFMatrix<REAL> expand_x(n_globalsol/2,1,&expandsolution(0,0),n_globalsol/2);
    expand_x.Resize(n_globalsol,1);
    expand_x.AddSub(n_globalsol/2,0,expand_x);

    TPZFMatrix<REAL> expand_y(n_globalsol/2,1,&expandsolution(n_globalsol/2,0),n_globalsol/2);
    expand_y.Resize(n_globalsol,1);
    expand_y.AddSub(n_globalsol/2,0,expand_y);

    t1 = high_resolution_clock::now();

TPZVec<REAL> sol(nelem * rows,0);

        for (int i = 0; i < cols; i++) {
        std::transform(&fStorageVec[i * nelem * rows], &fStorageVec[i * nelem * rows + nelem * rows / 2], &expand_x(i * nelem,0), &sol[0], std::multiplies<double>());
        std::transform(&fStorageVec[i * nelem * rows + nelem * rows / 2], &fStorageVec[i * nelem * rows + nelem * rows / 2 + nelem * rows / 2], &expand_x[i * nelem], &sol[nelem * rows / 2], std::multiplies<double>());
        std::transform(&sol[0], &sol[nelem * rows], &result(0, 0), &result(0, 0), std::plus<double>());

        //dv
        std::transform(&fStorageVec[i * nelem * rows], &fStorageVec[i * nelem * rows + nelem * rows / 2], &expand_y[i * nelem], &sol[0], std::multiplies<double>());
        std::transform(&fStorageVec[i * nelem * rows + nelem * rows / 2], &fStorageVec[i * nelem * rows + nelem * rows / 2 + nelem * rows / 2], &expand_y[i * nelem], &sol[nelem * rows / 2], std::multiplies<double>());
        std::transform(&sol[0], &sol[nelem * rows], &result(n_globalsol, 0), &result(n_globalsol, 0), std::plus<double>());
    }
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  std::cout << "Multiply: " << time_span.count() << std::endl;
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
//        cblas_daxpy(nelem, 1, &result(2*ipts*nelem ,0), 1, &aux(0,0), 1);
        std::transform(&result(2*ipts*nelem ,0), &result(2*ipts*nelem + nelem,0),  &aux(0,0),  &aux(0,0), std::plus<double>());
        std::transform(&weight[ipts*nelem], &weight[ipts*nelem + nelem], &aux(0,0), &sigma(2*ipts*nelem,0), std::multiplies<double>());  
        cblas_dscal (nelem, E/(1.-nu*nu), &sigma(2*ipts*nelem,0), 1);
        aux.Zero();

        //sigma yy
        cblas_daxpy(nelem, nu, &result(2*ipts*nelem,0), 1, &aux(0,0), 1);
        cblas_daxpy(nelem, 1, &result((2*ipts + npts_el + 1)*nelem,0), 1, &aux(0,0), 1);
//        std::transform(&result((2*ipts + npts_el + 1)*nelem,0), &result((2*ipts + npts_el + 1)*nelem + nelem,0), &aux(0,0), &aux(0,0), std::plus<double>());
        std::transform(&weight[ipts*nelem], &weight[ipts*nelem + nelem], &aux(0,0), &sigma((2*ipts+npts_el+1)*nelem,0), std::multiplies<double>());        
        cblas_dscal (nelem, E/(1.-nu*nu), &sigma((2*ipts+npts_el+1)*nelem,0), 1);
        aux.Zero();

        //sigma xy
        cblas_daxpy(nelem, 1, &result((2*ipts + 1)*nelem,0), 1, &aux(0,0), 1);
//        cblas_daxpy(nelem, 1, &result((2*ipts + npts_el)*nelem,0), 1, &aux(0,0), 1);
        std::transform(&result((2*ipts + npts_el)*nelem,0), &result((2*ipts + npts_el)*nelem + nelem,0),  &aux(0,0),  &aux(0,0), std::plus<double>());
        std::transform(&weight[ipts*nelem], &weight[ipts*nelem + nelem], &aux(0,0), &sigma((2*ipts+1)*nelem,0), std::multiplies<double>());
        cblas_dscal (nelem, E/(1.-nu*nu)*(1.-nu), &sigma((2*ipts+1)*nelem,0), 1);
        aux.Zero();

        cblas_daxpy(nelem, 1, &sigma((2*ipts+1)*nelem,0), 1, &sigma((2*ipts+npts_el)*nelem,0), 1);
    }
    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "Sigma: " << time_span.count() << std::endl;
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

TPZVec<REAL> force(cols*nelem);

    for (int i = 0; i < cols; i++) {
        //Fx
	std::transform(&fStorageVec[(((cols-i)%cols)*rows + i)*nelem], &fStorageVec[(((cols-i)%cols)*rows + i)*nelem + (cols-i)*nelem], &sigma(i * nelem,0), &force[0], std::multiplies<double>());
        std::transform(&fStorageVec[((cols-i)%cols)*rows*nelem], &fStorageVec[((cols-i)%cols)*rows*nelem + i*nelem], &sigma(0,0), &force[nelem*((cols - i)%cols)], std::multiplies<double>());
        std::transform(&force[0], &force[cols*nelem],  &nodal_forces_vec(0,0), &nodal_forces_vec(0,0), std::plus<double>());
//        cblas_daxpy(cols*nelem, 1, &force[0], 1, &nodal_forces_vec(0,0), 1);

        std::transform(&fStorageVec[(((cols-i)%cols)*rows + i)*nelem + cols*nelem], &fStorageVec[(((cols-i)%cols)*rows + i)*nelem + cols*nelem + (cols-i)*nelem], &sigma(i * nelem + cols*nelem,0), &force[0], std::multiplies<double>());
        std::transform(&fStorageVec[((cols-i)%cols)*rows*nelem + cols*nelem], &fStorageVec[((cols-i)%cols)*rows*nelem + cols*nelem + i*nelem], &sigma(cols*nelem,0), &force[nelem*((cols - i)%cols)], std::multiplies<double>());
        std::transform(&force[0], &force[cols*nelem],  &nodal_forces_vec(0,0), &nodal_forces_vec(0,0), std::plus<double>());
//        cblas_daxpy(cols*nelem, 1, &force[0], 1, &nodal_forces_vec(0,0), 1);

        //Fy
  	std::transform(&fStorageVec[(((cols-i)%cols)*rows + i)*nelem], &fStorageVec[(((cols-i)%cols)*rows + i)*nelem + (cols-i)*nelem], &sigma(i * nelem + fRow,0), &force[0], std::multiplies<double>());
        std::transform(&fStorageVec[((cols-i)%cols)*rows*nelem], &fStorageVec[((cols-i)%cols)*rows*nelem + i*nelem], &sigma(fRow,0), &force[nelem*((cols - i)%cols)], std::multiplies<double>());
        std::transform(&force[0], &force[cols*nelem],  &nodal_forces_vec(npts_tot/2,0), &nodal_forces_vec(npts_tot/2,0), std::plus<double>());
//        cblas_daxpy(cols*nelem, 1, &force[0], 1, &nodal_forces_vec(npts_tot/2,0), 1);

        std::transform(&fStorageVec[(((cols-i)%cols)*rows + i)*nelem + cols*nelem], &fStorageVec[(((cols-i)%cols)*rows + i)*nelem + cols*nelem + (cols-i)*nelem], &sigma(i * nelem + cols*nelem + fRow,0), &force[0], std::multiplies<double>());
        std::transform(&fStorageVec[((cols-i)%cols)*rows*nelem + cols*nelem], &fStorageVec[((cols-i)%cols)*rows*nelem + cols*nelem + i*nelem], &sigma(cols*nelem + fRow,0), &force[nelem*((cols - i)%cols)], std::multiplies<double>());
        std::transform(&force[0], &force[cols*nelem],  &nodal_forces_vec(npts_tot/2,0), &nodal_forces_vec(npts_tot/2,0), std::plus<double>());
//        cblas_daxpy(cols*nelem, 1, &force[0], 1, &nodal_forces_vec(npts_tot/2,0), 1);
    }

    t2 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "Transpose: " << time_span.count() << std::endl;
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
    std::cout << "Assemble: " << time_span.count() << std::endl;
}

void TPZSolveVector::TraditionalAssemble(TPZFMatrix<STATE> &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) const {
    for (int64_t ir=0; ir<fRow/2; ir++) {
        nodal_forces_global(fIndexes[ir], 0) += nodal_forces_vec(ir, 0);
        nodal_forces_global(fIndexes[ir+fRow/2], 0) += nodal_forces_vec(ir+fRow/2, 0);
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

    int64_t nelem = fRowSizes.size();
    int64_t neq = cmesh->NEquations();
    for (int64_t iel = 0; iel < nelem; iel++) {
        int64_t cols = fColSizes[iel];

        for (int64_t icols = 0; icols < cols; icols++) {
            fIndexesColor[iel + icols*nelem] = fIndexes[iel + icols*nelem] + fElemColor[iel]*neq;
            fIndexesColor[iel + icols*nelem + fRow/2] = fIndexes[iel + icols*nelem + fRow/2] + fElemColor[iel]*neq;
        }
    }
}
