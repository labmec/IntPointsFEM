#include "TPZCudaCalls.h"
#include "pzreal.h"
#include "pzvec.h"

// #include "MatMulKernels.h"
#include "ComputeSigmaKernel.h"
#include "MatrixAssembleKernel.h"


#define NT          256

// #if __CUDA_ARCH__ >= 200
//     #define MY_KERNEL_MAX_THREADS  (2 * NT)
//     #define MY_KERNEL_MIN_BLOCKS   3
// #else
//     #define MY_KERNEL_MAX_THREADS  NT
//     #define MY_KERNEL_MIN_BLOCKS   2
// #endif

TPZCudaCalls::TPZCudaCalls() {
	cusparse_h = false;
	cublas_h = false;
	heap_q = false;
}

TPZCudaCalls::~TPZCudaCalls() {
	if(cublas_h == true) {
		cublasDestroy(handle_cublas);
	}
	if(cusparse_h == true) {
		cusparseDestroy(handle_cusparse);			
	}
}

void TPZCudaCalls::Multiply(bool trans, int *m, int *n, int *k, REAL *A, int *strideA, 
	REAL *B, int *strideB,  REAL *C, int *strideC, REAL alpha, int nmatrices) {
	int numBlocks = (nmatrices + NT - 1) / NT;
	MatrixMultiplicationKernel<<<numBlocks,NT>>> (trans, m, n, k, A, strideA, B, strideB, C, strideC, alpha, nmatrices);
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		std::string error_string = cudaGetErrorString(error);
		std::string error_message = "failed to perform MatrixMultiplicationKernel: " + error_string;
		throw std::runtime_error(error_message);      
	}

}

void TPZCudaCalls::GatherOperation(int n, REAL *x, REAL *y, int *id) {
	if(cusparse_h == false) {
		cusparse_h = true;
		cusparseStatus_t result = cusparseCreate(&handle_cusparse);
		if (result != CUSPARSE_STATUS_SUCCESS) {
			throw std::runtime_error("failed to initialize cuSparse");      
		}			
	}
	cusparseStatus_t result = cusparseDgthr(handle_cusparse, n, x, y, id, CUSPARSE_INDEX_BASE_ZERO);
	if (result != CUSPARSE_STATUS_SUCCESS) {
		throw std::runtime_error("failed to perform cusparseDgthr");      
	}	
}

void TPZCudaCalls::ScatterOperation(int n, REAL *x, REAL *y, int *id) {
	if(cusparse_h == false) {
		cusparse_h = true;
		cusparseStatus_t result = cusparseCreate(&handle_cusparse);
		if (result != CUSPARSE_STATUS_SUCCESS) {
			throw std::runtime_error("failed to initialize cuSparse");      
		}			
	}
	cusparseStatus_t result = cusparseDsctr(handle_cusparse, n, x, id, y, CUSPARSE_INDEX_BASE_ZERO);
	if (result != CUSPARSE_STATUS_SUCCESS) {
		throw std::runtime_error("failed to perform cusparseDsctr");      
	}	
}

void TPZCudaCalls::DaxpyOperation(int n, double alpha, double *x, double *y) {
	if(cublas_h == false) {
		cublas_h = true;
		cublasStatus_t result = cublasCreate(&handle_cublas);
		if (result != CUBLAS_STATUS_SUCCESS) {
			throw std::runtime_error("failed to initialize cuBLAS");      
		}			
	}
	cublasStatus_t result = cublasDaxpy(handle_cublas, n, &alpha, x, 1., y, 1.);
	if (result != CUBLAS_STATUS_SUCCESS) {
		throw std::runtime_error("failed to perform cublasDaxpy");      
	}	
}

void TPZCudaCalls::SpMV(int opt, int sym, int m, int k, int nnz, REAL alpha, REAL *csrVal, int *csrRowPtr, int *csrColInd, REAL *B, REAL *C) {
	if(cusparse_h == false) {
		cusparse_h = true;
		cusparseStatus_t result = cusparseCreate(&handle_cusparse);
		if (result != CUSPARSE_STATUS_SUCCESS) {
			throw std::runtime_error("failed to initialize cuSparse");      
		}			
	}
	cusparseMatDescr_t descr;
	cusparseCreateMatDescr(&descr);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    if(sym == 0) {
	   cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);        
    } 
    else {
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_SYMMETRIC); 
    }
    cusparseOperation_t op;
    if(opt == 0) { 
        op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    } else {
        op = CUSPARSE_OPERATION_TRANSPOSE;
    }

	REAL beta = 0.;
	cusparseStatus_t result = cusparseDcsrmv(handle_cusparse, op, m, k, nnz, &alpha, descr, csrVal, csrRowPtr, csrColInd, B, &beta, C);
	if (result != CUSPARSE_STATUS_SUCCESS) {
		throw std::runtime_error("failed to perform cusparseDcsrmv");      
	}	
}

void TPZCudaCalls::SpMSpM(int opt, int sym, int m, int n, int k, int nnzA, REAL *csrValA, int *csrRowPtrA, int *csrColIndA, 
	int nnzB, REAL *csrValB, int *csrRowPtrB, int *csrColIndB, 
	int nnzC, REAL *csrValC, int *csrRowPtrC) {
	if(cusparse_h == false) {
		cusparse_h = true;
		cusparseStatus_t result = cusparseCreate(&handle_cusparse);
		if (result != CUSPARSE_STATUS_SUCCESS) {
			throw std::runtime_error("failed to initialize cuSparse");      
		}			
	}

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    if(sym == 0) {
       cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);        
    } 
    else {
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_SYMMETRIC); 
    }

    cusparseOperation_t op;
    if(opt == 0) { 
        op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    } else {
        op = CUSPARSE_OPERATION_TRANSPOSE;
    }

	int *csrColIndC;
	cudaMalloc((void**)&csrColIndC, sizeof(int)*nnzC);

	cusparseStatus_t result = cusparseDcsrgemm(handle_cusparse, op, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, k, 
		descr, nnzA, csrValA, csrRowPtrA, csrColIndA, 
		descr, nnzB, csrValB, csrRowPtrB, csrColIndB,
		descr, csrValC, csrRowPtrC, csrColIndC);
	if (result != CUSPARSE_STATUS_SUCCESS) {
		throw std::runtime_error("failed to perform cusparseDcsrgemm");      
	}	
}

void TPZCudaCalls::ComputeSigma(bool update_mem, int npts, REAL *glob_delta_strain, REAL *glob_sigma, REAL lambda, REAL mu, REAL mc_phi, REAL mc_psi, REAL mc_cohesion, REAL *dPlasticStrain,  
	REAL *dMType, REAL *dAlpha, REAL *dSigma, REAL *dStrain, REAL *weight) {
	
	int numBlocks = (npts + 256 - 1) / 256;
	ComputeSigmaKernel<<<numBlocks,256>>> (update_mem, npts, glob_delta_strain, glob_sigma, lambda, mu, mc_phi, mc_psi, mc_cohesion, dPlasticStrain, dMType, dAlpha, dSigma, dStrain, weight);
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		std::string error_string = cudaGetErrorString(error);
		std::string error_message = "failed to perform ComputeSigmaKernel: " + error_string;
		throw std::runtime_error(error_message);      
	}
}

void TPZCudaCalls::MatrixAssemble(int nnz, REAL *Kg, int first_el, int last_el, int64_t *el_color_index, REAL *weight, int *dof_indexes,
	REAL *storage, int *rowsizes, int *colsizes, int *rowfirstindex, int *colfirstindex, int *matrixposition, int *ia_to_sequence, int *ja_to_sequence) {
	int nel = last_el - first_el;
	int numBlocks = (nel + NT - 1) / NT;

	MatrixAssembleKernel<<<numBlocks,NT>>> (nel, nnz, Kg, first_el, el_color_index, weight, dof_indexes, storage, rowsizes, colsizes, rowfirstindex, colfirstindex, matrixposition, 
		ia_to_sequence, ja_to_sequence);
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		std::string error_string = cudaGetErrorString(error);
		std::string error_message = "failed to perform MatrixAssembleKernel: " + error_string;
		throw std::runtime_error(error_message);      
	}
}

void TPZCudaCalls::MatrixAssemble(REAL *K, int nnz, REAL *Kg, int first_el, int last_el, int64_t *el_color_index, int *dof_indexes,
	int *colsizes, int *colfirstindex, int *ia_to_sequence, int *ja_to_sequence) {
	int nel = last_el - first_el;
	int numBlocks = (nel + NT - 1) / NT;

	MatrixAssembleKernel<<<nel,1>>> (K, nel, nnz, Kg, el_color_index, dof_indexes, colsizes, colfirstindex, 
    ia_to_sequence, ja_to_sequence);
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		std::string error_string = cudaGetErrorString(error);
		std::string error_message = "failed to perform MatrixAssembleKernel: " + error_string;
		throw std::runtime_error(error_message);      
	}
}

void TPZCudaCalls::SolveCG(int n, int nnzA, REAL *csrValA, int *csrRowPtrA, int *csrColIndA, REAL *r, REAL *x) {
    if(cusparse_h == false) {
        cusparse_h = true;
        cusparseStatus_t result = cusparseCreate(&handle_cusparse);
        if (result != CUSPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("failed to initialize cuSparse");      
        }           
    }

    if(cublas_h == false) {
        cublas_h = true;
        cublasStatus_t result = cublasCreate(&handle_cublas);
        if (result != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("failed to initialize cuBLAS");      
        }           
    }

    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
    cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    REAL alpha = 1.0;
    REAL alpham1 = -1.0;
    REAL beta = 0.0;
    REAL r0 = 0.;
    REAL b;
    REAL r1;
    REAL dot;
    REAL a;
    REAL na;

    REAL *d_Ax;
    REAL *d_p;
    cudaMalloc((void **)&d_Ax, n*sizeof(REAL));
    cudaMalloc((void **)&d_p, n*sizeof(REAL));

    cusparseDcsrmv(handle_cusparse,CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnzA, &alpha, descr, csrValA, csrRowPtrA, csrColIndA, x, &beta, d_Ax);
    cublasDaxpy(handle_cublas, n, &alpham1, d_Ax, 1, r, 1);


    cublasDdot(handle_cublas, n, r, 1, r, 1, &r1);

    const REAL tol = 1.e-5;
    const int max_iter = 10000;
    int k;

    k = 1;

    while (r1 > tol*tol && k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;
            cublasDscal(handle_cublas, n, &b, d_p, 1);
            cublasDaxpy(handle_cublas, n, &alpha, r, 1, d_p, 1);
        }
        else
        {
            cublasDcopy(handle_cublas, n, r, 1, d_p, 1);
        }
        cusparseDcsrmv(handle_cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnzA, &alpha, descr, csrValA, csrRowPtrA, csrColIndA, d_p, &beta, d_Ax);
        cublasDdot(handle_cublas, n, d_p, 1, d_Ax, 1, &dot);
        a = r1 / dot;

        cublasDaxpy(handle_cublas, n, &a, d_p, 1, x, 1);
        na = -a;
        cublasDaxpy(handle_cublas, n, &na, d_Ax, 1, r, 1);

        r0 = r1;
        cublasDdot(handle_cublas, n, r, 1, r, 1, &r1);
        cudaThreadSynchronize();
        k++;
    }
    cudaFree(d_p);
    cudaFree(d_Ax);
}

// void assemble_poisson_matrix_coo(std::vector<float>& vals, std::vector<int>& row, std::vector<int>& col,
//                      std::vector<float>& rhs, int Nrows, int Ncols) {

//         //nnz: 5 entries per row (node) for nodes in the interior
//     // 1 entry per row (node) for nodes on the boundary, since we set them explicitly to 1.
//     int nnz = 5*Nrows*Ncols - (2*(Ncols-1) + 2*(Nrows-1))*4;
//     vals.resize(nnz);
//     row.resize(nnz);
//     col.resize(nnz);
//     rhs.resize(Nrows*Ncols);

//     int counter = 0;
//     for(int i = 0; i < Nrows; ++i) {
//         for (int j = 0; j < Ncols; ++j) {
//             int idx = j + Ncols*i;
//             if (i == 0 || j == 0 || j == Ncols-1 || i == Nrows-1) {
//                 vals[counter] = 1.;
//                 row[counter] = idx;
//                 col[counter] = idx;
//                 counter++;
//                 rhs[idx] = 1.;
// //                if (i == 0) {
// //                    rhs[idx] = 3.;
// //                }
//             } else { // -laplace stencil
//                 // above
//                 vals[counter] = -1.;
//                 row[counter] = idx;
//                 col[counter] = idx-Ncols;
//                 counter++;
//                 // left
//                 vals[counter] = -1.;
//                 row[counter] = idx;
//                 col[counter] = idx-1;
//                 counter++;
//                 // center
//                 vals[counter] = 4.;
//                 row[counter] = idx;
//                 col[counter] = idx;
//                 counter++;
//                 // right
//                 vals[counter] = -1.;
//                 row[counter] = idx;
//                 col[counter] = idx+1;
//                 counter++;
//                 // below
//                 vals[counter] = -1.;
//                 row[counter] = idx;
//                 col[counter] = idx+Ncols;
//                 counter++;

//                 rhs[idx] = 0;
//             }
//         }
//     }
// }

// void TPZCudaCalls::Teste() {
// 	    // --- create library handles:
//     cusolverSpHandle_t cusolver_handle;
//     cusolverStatus_t cusolver_status;
//     cusolver_status = cusolverSpCreate(&cusolver_handle);
//     std::cout << "status create cusolver handle: " << cusolver_status << std::endl;

//     cusparseHandle_t cusparse_handle;
//     cusparseStatus_t cusparse_status;
//     cusparse_status = cusparseCreate(&cusparse_handle);
//     std::cout << "status create cusparse handle: " << cusparse_status << std::endl;

//     // --- prepare matrix:
//     int Nrows = 4;
//     int Ncols = 4;
//     std::vector<float> csrVal;
//     std::vector<int> cooRow;
//     std::vector<int> csrColInd;
//     std::vector<float> b;

//     assemble_poisson_matrix_coo(csrVal, cooRow, csrColInd, b, Nrows, Ncols);

//     int nnz = csrVal.size();
//     int m = Nrows * Ncols;
//     std::vector<int> csrRowPtr(m+1);

//     // --- prepare solving and copy to GPU:
//     std::vector<float> x(m);
//     float tol = 1e-5;
//     int reorder = 0;
//     int singularity = 0;

//     float *db, *dcsrVal, *dx;
//     int *dcsrColInd, *dcsrRowPtr, *dcooRow;
//     cudaMalloc((void**)&db, m*sizeof(float));
//     cudaMalloc((void**)&dx, m*sizeof(float));
//     cudaMalloc((void**)&dcsrVal, nnz*sizeof(float));
//     cudaMalloc((void**)&dcsrColInd, nnz*sizeof(int));
//     cudaMalloc((void**)&dcsrRowPtr, (m+1)*sizeof(int));
//     cudaMalloc((void**)&dcooRow, nnz*sizeof(int));

//     cudaMemcpy(db, b.data(), b.size()*sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(dcsrVal, csrVal.data(), csrVal.size()*sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(dcsrColInd, csrColInd.data(), csrColInd.size()*sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(dcooRow, cooRow.data(), cooRow.size()*sizeof(int), cudaMemcpyHostToDevice);

//     cusparse_status = cusparseXcoo2csr(cusparse_handle, dcooRow, nnz, m,
//                                        dcsrRowPtr, CUSPARSE_INDEX_BASE_ZERO);
//     std::cout << "status cusparse coo2csr conversion: " << cusparse_status << std::endl;

//     cudaDeviceSynchronize(); // matrix format conversion has to be finished!

//     // --- everything ready for computation:

//     cusparseMatDescr_t descrA;

//     cusparse_status = cusparseCreateMatDescr(&descrA);
//     std::cout << "status cusparse createMatDescr: " << cusparse_status << std::endl;

//     // optional: print dense matrix that has been allocated on GPU

//     std::vector<float> A(m*m, 0);
//     float *dA;
//     cudaMalloc((void**)&dA, A.size()*sizeof(float));

//     cusparseScsr2dense(cusparse_handle, m, m, descrA, dcsrVal,
//                        dcsrRowPtr, dcsrColInd, dA, m);

//     cudaMemcpy(A.data(), dA, A.size()*sizeof(float), cudaMemcpyDeviceToHost);
//     std::cout << "A: \n";
//     for (int i = 0; i < m; ++i) {
//         for (int j = 0; j < m; ++j) {
//             std::cout << A[i*m + j] << " ";
//         }
//         std::cout << std::endl;
//     }

//     cudaFree(dA);

//     std::cout << "b: \n";
//     cudaMemcpy(b.data(), db, (m)*sizeof(int), cudaMemcpyDeviceToHost);
//     for (auto a : b) {
//         std::cout << a << ",";
//     }
//     std::cout << std::endl;


//     // --- solving!!!!

// // // does not work:
// //    cusolver_status = cusolverSpScsrlsvchol(cusolver_handle, m, nnz, descrA, dcsrVal,
// //                       dcsrRowPtr, dcsrColInd, db, tol, reorder, dx,
// //                       &singularity);

//      cusolver_status = cusolverSpScsrlsvqr(cusolver_handle, m, nnz, descrA, dcsrVal,
//                         dcsrRowPtr, dcsrColInd, db, tol, reorder, dx,
//                         &singularity);

//     cudaDeviceSynchronize();

//     std::cout << "singularity (should be -1): " << singularity << std::endl;

//     std::cout << "status cusolver solving (!): " << cusolver_status << std::endl;

//     cudaMemcpy(x.data(), dx, m*sizeof(float), cudaMemcpyDeviceToHost);

//     cusparse_status = cusparseDestroy(cusparse_handle);
//     std::cout << "status destroy cusparse handle: " << cusparse_status << std::endl;

//     cusolver_status = cusolverSpDestroy(cusolver_handle);
//     std::cout << "status destroy cusolver handle: " << cusolver_status << std::endl;

//     for (auto a : x) {
//         std::cout << a << " ";
//     }
//     std::cout << std::endl;


//     cudaFree(db);
//     cudaFree(dx);
//     cudaFree(dcsrVal);
//     cudaFree(dcsrColInd);
//     cudaFree(dcsrRowPtr);
//     cudaFree(dcooRow);


// }