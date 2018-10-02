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
        sigma[2 * ipts] = weight[ipts] * E / (1. - nu * nu) * (result[2 * ipts] + nu * result[2 * ipts + npts_tot + 1]); // Sigma x
        sigma[2 * ipts + 1] = weight[ipts] * E / (1. - nu * nu) * (1. - nu) / 2 * (result[2 * ipts + 1] + result[2 * ipts + npts_tot]) * 0.5; // Sigma xy
        sigma[2 * ipts + npts_tot] = sigma[2 * ipts + 1]; //Sigma xy
        sigma[2 * ipts + npts_tot + 1] = weight[ipts] * E / (1. - nu * nu) * (result[2 * ipts + npts_tot + 1] + nu * result[2 * ipts]); // Sigma y
    }
}

void TPZSolveMatrix::SolveWithCUDA(TPZCompMesh *cmesh, const TPZFMatrix<STATE> &global_solution, TPZStack<REAL> &weight, TPZFMatrix<REAL> &nodal_forces_global) const {
    int64_t nelem = fRowSizes.size();
    int64_t n_globalsol = fIndexes.size();

///GATHER OPERATION------------------------------------------------
///Initialize and transfer findexes, expandsolution and globalsolution to the device
    double *dexpandsolution;
    cudaMalloc(&dexpandsolution, n_globalsol * sizeof(double));

    double *dglobal_solution;
    cudaMalloc(&dglobal_solution, global_solution.Rows() * sizeof(double));
    cudaMemcpy(dglobal_solution, &global_solution[0], global_solution.Rows() * sizeof(double), cudaMemcpyHostToDevice);

    int *dindexes;
    cudaMalloc(&dindexes, n_globalsol * sizeof(int));
    cudaMemcpy(dindexes, &fIndexes[0], n_globalsol * sizeof(int), cudaMemcpyHostToDevice);

/// Gather global solution
    cusparseHandle_t handle_gthr;
    cusparseCreate (&handle_gthr);
    cusparseDgthr(handle_gthr, n_globalsol, dglobal_solution, &dexpandsolution[0], &dindexes[0], CUSPARSE_INDEX_BASE_ZERO);

///Free device memory
    cudaFree(dglobal_solution);
    cudaFree(dindexes);
    cusparseDestroy(handle_gthr);
///----------------------------------------------------------------

///MULTIPLY--------------------------------------------------------
///Initialize result on device
    double *dresult;
    cudaMalloc(&dresult, 2 * n_globalsol * sizeof(double));

    int nstorage = fStorage.size();
    double *dstorage;
    cudaMalloc(&dstorage, nstorage * sizeof(double));
    cudaMemcpy(dstorage, &fStorage[0], nstorage * sizeof(double), cudaMemcpyHostToDevice);


///Use CUBLAS library to do the multiplication
    cudaStream_t *stream_m = (cudaStream_t *)malloc(nelem*sizeof(cudaStream_t));
    cublasHandle_t handle_m;
    cublasCreate(&handle_m);

    for (int iel = 0; iel < nelem; iel++) {
        cudaStreamCreate(&(stream_m[iel]));
    }

    double alpha_m = 1.0;
    double beta_m = 0.0;
    #pragma omp parallel for
    for (int iel = 0; iel < nelem; iel++) {
        int64_t pos = fMatrixPosition[iel];
        int64_t cols = fColSizes[iel];
        int64_t rows = fRowSizes[iel];

        int64_t cont_cols = fColFirstIndex[iel];
        int64_t cont_rows = fRowFirstIndex[iel];


        //du
        cublasSetStream(handle_m, stream_m[iel]);
        cublasDgemv(handle_m, CUBLAS_OP_N, rows, cols, &alpha_m, &dstorage[pos], rows, &dexpandsolution[cont_cols], 1, &beta_m, &dresult[cont_rows], 1);

        //dv
        cublasSetStream(handle_m, stream_m[iel]);
        cublasDgemv(handle_m, CUBLAS_OP_N, rows, cols, &alpha_m, dstorage, rows, &dexpandsolution[cont_cols + fColFirstIndex[nelem]], 1, &beta_m, &dresult[cont_rows + fRowFirstIndex[nelem]], 1);

//	cudaFree(dstorage);
    }

///Free device memory
  //  cudaFree(dstorage);
    cudaFree(dexpandsolution);
    cudaStreamSynchronize(0);
///----------------------------------------------------------------

///COMPUTE SIGMA---------------------------------------------------
///Initialize and transfer sigma and weight to the device
    int npts_tot = fRow;
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
///Initialize nodal_forces_vec on device
    double *dnodal_forces_vec;
    cudaMalloc(&dnodal_forces_vec, npts_tot * sizeof(double));
  
///Use CUBLAS to do the multiplication
    double alpha_mt = 1.0;
    double beta_mt = 0.0;
    #pragma omp parallel for
    for (int64_t iel = 0; iel < nelem; iel++) {
        int64_t pos = fMatrixPosition[iel];
        int64_t rows = fRowSizes[iel];
        int64_t cols = fColSizes[iel];

        int64_t cont_rows = fRowFirstIndex[iel];
        int64_t cont_cols = fColFirstIndex[iel];

        //Nodal forces in x direction
        cublasSetStream(handle_m, stream_m[iel]);
        cublasDgemv(handle_m, CUBLAS_OP_T, rows, cols, &alpha_mt, dstorage, rows, &dsigma[cont_rows], 1, &beta_mt, &dnodal_forces_vec[cont_cols], 1);

        //Nodal forces in y direction
        cublasSetStream(handle_m, stream_m[iel]);
        cublasDgemv(handle_m, CUBLAS_OP_T, rows, cols, &alpha_mt, dstorage, rows, &dsigma[cont_rows + npts_tot], 1, &beta_mt, &dnodal_forces_vec[cont_cols + npts_tot / 2], 1);

    }
//Free device memory
    cudaFree(dsigma);
    cublasDestroy(handle_m);
    cudaStreamSynchronize(0); 
    cudaFree(dstorage);
///----------------------------------------------------------------

///ASSEMBLE--------------------------------------------------------
    ColoringElements(cmesh);

    int64_t ncolor = *std::max_element(fElemColor.begin(), fElemColor.end())+1;
    int64_t sz = fIndexes.size();
    int64_t neq = nodal_forces_global.Rows();

    nodal_forces_global.Resize(ncolor * neq,1);
    nodal_forces_global.Zero();

///Initialize fIndexesColor and nodal_forces_global on device
    int *dindexescolor;
    cudaMalloc(&dindexescolor, sz * sizeof(int));
    cudaMemcpy(dindexescolor, &fIndexesColor[0], sz* sizeof(int), cudaMemcpyHostToDevice);

    double *dnodal_forces_global;
    cudaMalloc(&dnodal_forces_global, ncolor * neq * sizeof(double));
    cudaMemcpy(dnodal_forces_global, &nodal_forces_global(0,0), ncolor*neq*sizeof(double), cudaMemcpyHostToDevice);

///Scatter operation
    cusparseHandle_t handle_sctr;
    cusparseCreate(&handle_sctr);
    cusparseDsctr(handle_sctr, sz, dnodal_forces_vec, &dindexescolor[0], &dnodal_forces_global[0], CUSPARSE_INDEX_BASE_ZERO);

    int64_t colorassemb = ncolor / 2.;
    while (colorassemb > 0) {

        int64_t firsteq = (ncolor - colorassemb) * neq;

        cublasHandle_t handle_daxpy;
	cublasCreate(&handle_daxpy);
	double alpha = 1.;
	cublasDaxpy(handle_daxpy, neq*ncolor, &alpha, &dnodal_forces_global[firsteq], 1., &dnodal_forces_global[0], 1.);

        ncolor -= colorassemb;
        colorassemb = ncolor/2;

    cudaMemcpy(&nodal_forces_global(0, 0), dnodal_forces_global, neq * sizeof(double), cudaMemcpyDeviceToHost);
    }
    nodal_forces_global.Resize(neq, 1);

    cudaMemcpy(&nodal_forces_global(0, 0), dnodal_forces_global, neq * sizeof(double), cudaMemcpyDeviceToHost);

///Free device memory
    cusparseDestroy(handle_sctr);
    cudaFree(dnodal_forces_vec);
    cudaFree(dindexescolor);
    cudaFree(dnodal_forces_global);
}

void TPZSolveMatrix::Multiply(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const
{
    int64_t nelem = fRowSizes.size();

    int64_t n_globalsol = fIndexes.size();

    result.Resize(2*n_globalsol,1);
    result.Zero();

    TPZVec<REAL> expandsolution(n_globalsol);

/// gather operation
    cblas_dgthr(n_globalsol, global_solution, &expandsolution[0], &fIndexes[0]);

#ifdef USING_TBB
    parallel_for(size_t(0),size_t(nelem),size_t(1),[&](size_t iel)
             {
                int64_t pos = fMatrixPosition[iel];
                int64_t cols = fColSizes[iel];
                int64_t rows = fRowSizes[iel];
                TPZFMatrix<REAL> elmatrix(rows,cols,&fStorage[pos],rows*cols);

                int64_t cont_cols = fColFirstIndex[iel];
                int64_t cont_rows = fRowFirstIndex[iel];

                 TPZFMatrix<REAL> element_solution_x(cols,1,&expandsolution[cont_cols],cols);
                 TPZFMatrix<REAL> element_solution_y(cols,1,&expandsolution[cont_cols+fColFirstIndex[nelem]],cols);

                TPZFMatrix<REAL> solx(rows,1,&result(cont_rows,0),rows);
                TPZFMatrix<REAL> soly(rows,1,&result(cont_rows+fRowFirstIndex[nelem],0),rows);

                elmatrix.Multiply(element_solution_x,solx);
                elmatrix.Multiply(element_solution_y,soly);
             }
             );

#else
    for (int64_t iel=0; iel<nelem; iel++) {
        int64_t pos = fMatrixPosition[iel];
        int64_t cols = fColSizes[iel];
        int64_t rows = fRowSizes[iel];
        TPZFMatrix<REAL> elmatrix(rows,cols,&fStorage[pos],rows*cols);

        int64_t cont_cols = fColFirstIndex[iel];
        int64_t cont_rows = fRowFirstIndex[iel];

        TPZFMatrix<REAL> element_solution_x(cols,1,&expandsolution[cont_cols],cols);
        TPZFMatrix<REAL> element_solution_y(cols,1,&expandsolution[cont_cols+fColFirstIndex[nelem]],cols);

        TPZFMatrix<REAL> solx(rows,1,&result(cont_rows,0),rows); //du
        TPZFMatrix<REAL> soly(rows,1,&result(cont_rows+fRowFirstIndex[nelem],0),rows); //dv

        elmatrix.Multiply(element_solution_x,solx);
        elmatrix.Multiply(element_solution_y,soly);
    }
#endif
}

void TPZSolveMatrix::ComputeSigma( TPZStack<REAL> &weight, TPZFMatrix<REAL> &result, TPZFMatrix<STATE> &sigma)
{
    REAL E = 200000000.;
    REAL nu =0.30;
    int npts_tot = fRow;
    sigma.Resize(2*npts_tot,1);

#ifdef USING_TBB
    parallel_for(size_t(0),size_t(npts_tot/2),size_t(1),[&](size_t ipts)
                      {
                            sigma(2*ipts,0) = weight[ipts]*E/(1.-nu*nu)*(result(2*ipts,0)+nu*result(2*ipts+npts_tot+1,0)); // Sigma x
                            sigma(2*ipts+1,0) = weight[ipts]*E/(1.-nu*nu)*(1.-nu)/2*(result(2*ipts+1,0)+result(2*ipts+npts_tot,0))*0.5; // Sigma xy
                            sigma(2*ipts+npts_tot,0) = sigma(2*ipts+1,0); //Sigma xy
                            sigma(2*ipts+npts_tot+1,0) = weight[ipts]*E/(1.-nu*nu)*(result(2*ipts+npts_tot+1,0)+nu*result(2*ipts,0)); // Sigma y
                      }
                      );
#else

    for (int64_t ipts=0; ipts< npts_tot/2; ipts++) {
        sigma(2*ipts,0) = weight[ipts]*E/(1.-nu*nu)*(result(2*ipts,0)+nu*result(2*ipts+npts_tot+1,0)); // Sigma x
        sigma(2*ipts+1,0) = weight[ipts]*E/(1.-nu*nu)*(1.-nu)/2*(result(2*ipts+1,0)+result(2*ipts+npts_tot,0))*0.5; // Sigma xy
        sigma(2*ipts+npts_tot,0) = sigma(2*ipts+1,0); //Sigma xy
        sigma(2*ipts+npts_tot+1,0) = weight[ipts]*E/(1.-nu*nu)*(result(2*ipts+npts_tot+1,0)+nu*result(2*ipts,0)); // Sigma y
    }
#endif
}

void TPZSolveMatrix::MultiplyTranspose(TPZFMatrix<STATE>  &intpoint_solution, TPZFMatrix<STATE> &nodal_forces_vec)
{
    int64_t nelem = fRowSizes.size();
    int64_t npts_tot = fRow;
    nodal_forces_vec.Resize(npts_tot,1);
    nodal_forces_vec.Zero();

#ifdef USING_TBB
    parallel_for(size_t(0),size_t(nelem),size_t(1),[&](size_t iel)
                  {
                        int64_t pos = fMatrixPosition[iel];
                        int64_t rows = fRowSizes[iel];
                        int64_t cols = fColSizes[iel];
                        int64_t cont_rows = fRowFirstIndex[iel];
                        int64_t cont_cols = fColFirstIndex[iel];
                        TPZFMatrix<REAL> elmatrix(rows,cols,&fStorage[pos],rows*cols);

                        // Forças nodais na direção x
                        TPZFMatrix<REAL> fvx(rows,1, &intpoint_solution(cont_rows,0),rows);
                        TPZFMatrix<STATE> nodal_forcex(cols, 1, &nodal_forces_vec(cont_cols, 0), cols);
                        elmatrix.MultAdd(fvx,nodal_forcex,nodal_forcex,1,0,1);

                        // Forças nodais na direção y
                        TPZFMatrix<REAL> fvy(rows,1, &intpoint_solution(cont_rows+npts_tot,0),rows);
                        TPZFMatrix<STATE> nodal_forcey(cols, 1, &nodal_forces_vec(cont_cols + npts_tot/2, 0), cols);
                        elmatrix.MultAdd(fvy,nodal_forcey,nodal_forcey,1,0,1);
                  }
                  );
#else
    for (int64_t iel = 0; iel < nelem; iel++) {
        int64_t pos = fMatrixPosition[iel];
        int64_t rows = fRowSizes[iel];
        int64_t cols = fColSizes[iel];
        int64_t cont_rows = fRowFirstIndex[iel];
        int64_t cont_cols = fColFirstIndex[iel];
        TPZFMatrix<REAL> elmatrix(rows,cols,&fStorage[pos],rows*cols);

        // Nodal forces in x direction
        TPZFMatrix<REAL> fvx(rows,1, &intpoint_solution(cont_rows,0),rows);
        TPZFMatrix<STATE> nodal_forcex(cols, 1, &nodal_forces_vec(cont_cols, 0), cols);
        elmatrix.MultAdd(fvx,nodal_forcex,nodal_forcex,1,0,1);

        // Nodal forces in y direction
        TPZFMatrix<REAL> fvy(rows,1, &intpoint_solution(cont_rows+npts_tot,0),rows);
        TPZFMatrix<STATE> nodal_forcey(cols, 1, &nodal_forces_vec(cont_cols + npts_tot/2, 0), cols);
        elmatrix.MultAdd(fvy,nodal_forcey,nodal_forcey,1,0,1);
    }
#endif
}

void TPZSolveMatrix::TraditionalAssemble(TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) const
{
#ifdef USING_TBB
    parallel_for(size_t(0),size_t(fRow),size_t(1),[&](size_t ir)
             {
                 nodal_forces_global(fIndexes[ir], 0) += nodal_forces_vec(ir, 0);
             }
);
#else
    for (int64_t ir=0; ir<fRow; ir++) {
        nodal_forces_global(fIndexes[ir], 0) += nodal_forces_vec(ir, 0);
    }
#endif
}

void TPZSolveMatrix::ColoringElements(TPZCompMesh * cmesh) const
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
    int64_t ncolor = *std::max_element(fElemColor.begin(), fElemColor.end())+1;
    int64_t sz = fIndexes.size();
    int64_t neq = nodal_forces_global.Rows();
    nodal_forces_global.Resize(neq*ncolor,1);


    cblas_dsctr(sz, nodal_forces_vec, &fIndexesColor[0], &nodal_forces_global(0,0));

    int64_t colorassemb = ncolor / 2.;
    while (colorassemb > 0) {

        int64_t firsteq = (ncolor - colorassemb) * neq;

        cblas_daxpy(neq*ncolor, 1., &nodal_forces_global(firsteq, 0), 1., &nodal_forces_global(0, 0), 1.);

        ncolor -= colorassemb;
        colorassemb = ncolor/2;
    }
    nodal_forces_global.Resize(neq, 1);
}