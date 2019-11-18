#include "pzreal.h"

#ifdef O_LINEAR
#define ndof 8
#define NT_sm 256
#elif O_QUADRATIC
#define ndof 18
#define NT_sm 110
#elif O_CUBIC
#define ndof 32
#define NT_sm 64
#endif

__constant__ REAL De[3 * 3];

__device__ void ComputeTangentMatrixDevice(REAL *dep, int el_npts, int el_dofs, REAL *storage, REAL *K){

    REAL DeBip[3 * ndof];
    for(int i = 0; i < 3 * el_dofs; i++) DeBip[i] = 0.0;
    MultAddDevice(false, 3, el_dofs, 3, dep, storage, DeBip, 1., 0.);
    MultAddDevice(true, el_dofs, el_dofs, 3, storage, DeBip, K, 1, 1.);
}

__global__
void MatrixAssembleKernel(int nel, REAL *Kc, REAL *dep, int *el_color_index, REAL *storage, int *rowsizes, int *colsizes, int *rowfirstindex, int *colfirstindex, int *matrixposition, int *matrixstride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < nel) {
        int iel = el_color_index[tid];

        int el_npts = rowsizes[iel]/3;
        int el_dofs = colsizes[iel];
        int colpos = colfirstindex[iel];
        int first_el_ip = rowfirstindex[iel]/3;
        int matpos = matrixposition[iel];

        REAL K[ndof * ndof];
        for(int i = 0; i < el_dofs * el_dofs; i++) K[i] = 0;
        #ifdef USE_SHARED
             __shared__ REAL s_storage[NT_sm * ndof * 3]; // max allowed word is 48k
            for (int ip = 0; ip < el_npts; ip++) {
                for(int i = 0; i < ndof * 3; i++) {
                   s_storage[i + threadIdx.x * ndof * 3] = storage[matpos + i + ip * el_dofs * 3];
                }
                __syncthreads(); 
                ComputeTangentMatrixDevice(&dep[first_el_ip * 3 * 3 + ip * 3 * 3], el_npts, el_dofs, &s_storage[threadIdx.x * ndof * 3], K);
            }
        #else
            for (int ip = 0; ip < el_npts; ip++) {
                ComputeTangentMatrixDevice(&dep[first_el_ip * 3 * 3 + ip * 3 * 3], el_npts, el_dofs, &storage[matpos + ip * el_dofs * 3], K);
            }
        #endif

        int stride = matrixstride[tid];
        int c = stride;
        for(int i_dof = 0 ; i_dof < el_dofs; i_dof++){
            for(int j_dof = i_dof; j_dof < el_dofs; j_dof++){
                Kc[c] += K[i_dof * el_dofs + j_dof];
                c++;
            }
        }	
     }
}

