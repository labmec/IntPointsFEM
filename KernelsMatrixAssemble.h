#include "pzreal.h"

#ifdef O_LINEAR
#define ndof 8
#define NPTS 12
#define NT_sm 256
#define TILE_WIDTH 4

#elif O_QUADRATIC
#define ndof 18
#define NPTS 27
#define NT_sm 110
#define TILE_WIDTH 3

#elif O_CUBIC
#define ndof 32
#define NPTS 48
#define NT_sm 64
#define TILE_WIDTH 8

#endif

__constant__ REAL De[NPTS * NPTS];

__device__ REAL DeBip[4 * NPTS * ndof];
__device__ REAL transpose[4 * NPTS * ndof];

__device__ __forceinline__ void ComputeTangentMatrixDevice(int el_npts, int el_dofs, REAL *storage, REAL weight, REAL *K){   

    REAL DeBip[3 * ndof];
    for(int i = 0; i < 3 * el_dofs; i++) DeBip[i] = 0.0;
    REAL omega = weight;
    MultAddDevice(false, 3, el_dofs, 3, De, storage, DeBip, 1., 0.);
    MultAddDevice(true, el_dofs, el_dofs, 3, storage, DeBip, K, omega, 1.);
}


__device__ int64_t me(int *ia_to_sequence, int *ja_to_sequence, int64_t & i_dest, int64_t & j_dest) {
    int64_t row(i_dest),col(j_dest);
    if (i_dest > j_dest) {
        int64_t temp = i_dest;
        row = col;
        col = temp;
    }
    for(int ic=ia_to_sequence[row] ; ic < ia_to_sequence[row+1]; ic++ ) {
        if ( ja_to_sequence[ic] == col )
        {
            return ic;
        }
    }
    return 0; 
}

__global__ 
void MatrixAssembleKernel(int nel, REAL *Kg, int64_t *el_color_index, REAL *weight, int *dof_indexes, 
    REAL *storage, int *rowsizes, int *colsizes, int *rowfirstindex, int *colfirstindex, int *matrixposition, int *ia_to_sequence, int *ja_to_sequence) {

    // int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // if(tid < nel) {
    //     int iel = el_color_index[tid];

    //     int el_npts = rowsizes[iel]/3;
    //     int el_dofs = colsizes[iel];
    //     int colpos = colfirstindex[iel];
    //     int first_el_ip = rowfirstindex[iel]/3;
    //     int matpos = matrixposition[iel];

    //     int64_t dest[ndof];
    //     REAL K[ndof * ndof];
    //     for(int i = 0; i < el_dofs; i++) dest[i] = dof_indexes[colpos + i];
    //     for(int i = 0; i < el_dofs * el_dofs; i++) K[i] = 0;
    //     __shared__ REAL s_storage[NT_sm * ndof * 3]; // max allowed word is 48k
    //     for (int ip = 0; ip < el_npts; ip++) {
    //         for(int i = 0; i < ndof * 3; i++) {
    //            s_storage[i + threadIdx.x * ndof * 3] = storage[matpos + i + ip * ndof * 3];
    //         }
    //         __syncthreads();     
    //         ComputeTangentMatrixDevice(el_npts, el_dofs, &s_storage[threadIdx.x * ndof * 3], weight[first_el_ip + ip], K);   
    //     }
    //     for (int i_dof = 0; i_dof < el_dofs; i_dof++) {
    //         int64_t i_dest = dest[i_dof];
    //         for (int j_dof = i_dof; j_dof < el_dofs; j_dof++) {
    //             int64_t j_dest = dest[j_dof];
    //             int64_t index = me(ia_to_sequence, ja_to_sequence, i_dest, j_dest);
    //             Kg[index] += K[i_dof * el_dofs + j_dof];
    //         }
    //     }           
    //  }
}

__global__ void MatMul(int opt, REAL* A, REAL* B, REAL* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) {
    REAL CValue = 0;
    int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;

    __shared__ REAL As[TILE_WIDTH][TILE_WIDTH];
    __shared__ REAL Bs[TILE_WIDTH][TILE_WIDTH];

    int x = ACols;

    for (int k = 0; k < (TILE_WIDTH + x - 1)/TILE_WIDTH; k++) {
        if (k*TILE_WIDTH + threadIdx.x < ACols && Row < ARows) {
            As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_WIDTH + threadIdx.x];
        }
        else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (k*TILE_WIDTH + threadIdx.y < BRows && Col < BCols) {
            Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_WIDTH + threadIdx.y)*BCols + Col];
        } 
        else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        } 

        __syncthreads();
            for (int n = 0; n < TILE_WIDTH; ++n) {
                CValue += Bs[threadIdx.x][n] * As[threadIdx.y][n]* Bs[n][threadIdx.x];
            }             

        __syncthreads();
    }

    if (Row < CRows && Col < CCols) {
        C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;
    } 
}


__global__ void transposeNoBankConflicts(REAL *odata, REAL *idata, int width, int height, int nreps)
{
    __shared__ REAL tile[TILE_WIDTH][TILE_WIDTH+1];

    int xIndex = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int yIndex = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    xIndex = blockIdx.y * TILE_WIDTH + threadIdx.x;
    yIndex = blockIdx.x * TILE_WIDTH + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

    for (int r=0; r < nreps; r++)
    {
        for (int i=0; i<TILE_WIDTH; i+=TILE_WIDTH)
        {
            tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
        }

        __syncthreads();

        for (int i=0; i<TILE_WIDTH; i+=TILE_WIDTH)
        {
            odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
        }
    }
}


__global__ 
void MatrixAssembleKernelGS(int nel, REAL * __restrict__ Kc, const int64_t* __restrict__ el_color_index, const REAL* __restrict__ weight, 
	REAL* __restrict__ storage, const int* __restrict__ rowsizes, const int* __restrict__ colsizes, const int* __restrict__ rowfirstindex, 
    const int* __restrict__ colfirstindex, const int* __restrict__ matrixposition) {

	int tid = blockIdx.x;
    
    if(tid < nel) {
        int iel = el_color_index[tid];

        int el_npts = rowsizes[iel];
        int el_dofs = colsizes[iel];
        int colpos = colfirstindex[iel];
        int first_el_ip = rowfirstindex[iel]/3;
        int matpos = matrixposition[iel];


        // int dimGrid_x1 = ceilf(ndof/TILE_WIDTH);
        // int dimGrid_y1 = ceilf(el_npts/TILE_WIDTH);
        int dimGrid_x1 = (TILE_WIDTH + el_dofs - 1)/TILE_WIDTH;
        int dimGrid_y1 = (TILE_WIDTH + el_npts - 1)/TILE_WIDTH;
        int dimGrid_z1 = 1;

        // int dimGrid_x2 = ceilf(el_dofs/TILE_WIDTH);
        // int dimGrid_y2 = ceilf(el_dofs/TILE_WIDTH);
        int dimGrid_x2 = (TILE_WIDTH + el_dofs - 1)/TILE_WIDTH;
        int dimGrid_y2 = (TILE_WIDTH + el_dofs - 1)/TILE_WIDTH;

        int dimGrid_z2 = 1;

        int dimBlock_x = TILE_WIDTH;
        int dimBlock_y = TILE_WIDTH;
        int dimBlock_z = 1;

        dim3 dimGrid1(dimGrid_x1, dimGrid_y1, dimGrid_z1);  
        dim3 dimGrid2(dimGrid_x2, dimGrid_y2, dimGrid_z2);  
        dim3 dimBlock(dimBlock_x, dimBlock_y, dimBlock_z); 

        dim3 grid((TILE_WIDTH + el_dofs - 1)/TILE_WIDTH, (TILE_WIDTH + el_npts - 1)/TILE_WIDTH), threads(TILE_WIDTH,TILE_WIDTH);

        for(int i = 0; i < NPTS * ndof; i++) transpose[i] = 0;
        for(int i = 0; i < NPTS * NPTS; i++) DeBip[i] = 0;

        MatMul<<<dimGrid1,dimBlock>>>(0, De, &storage[matpos], &DeBip[tid * el_dofs * el_npts], el_npts, el_npts, el_npts, el_dofs, el_npts, el_dofs); 
        // transposeNoBankConflicts<<<grid, threads>>>(&transpose[tid * el_dofs * el_npts], &storage[matpos], el_dofs, el_npts, 1);
        // MatMul<<<dimGrid2,dimBlock>>>(0, &transpose[tid * el_dofs * el_npts], &DeBip[tid * el_dofs * el_npts], &Kc[tid * el_dofs * el_dofs], el_dofs, el_npts, el_npts, el_dofs, el_dofs, el_dofs);

       
        // REAL K[ndof * ndof];
        // for(int i = 0; i < el_dofs * el_dofs; i++) K[i] = 0;
        // __shared__ REAL s_storage[NT_sm * ndof * 3]; // max allowed word is 48k
        // for (int ip = 0; ip < el_npts; ip++) {
        //     // for(int i = 0; i < ndof * 3; i++) {
        //     //    s_storage[i + threadIdx.x * ndof * 3] = storage[matpos + i + ip * ndof * 3];
        //     // }
        //     // __syncthreads(); 
        //     // ComputeTangentMatrixDevice(el_npts, el_dofs, &s_storage[threadIdx.x * ndof * 3], weight[first_el_ip + ip], K);   
        //     ComputeTangentMatrixDevice(el_npts, el_dofs, &storage[matpos + ip * ndof * 3], weight[first_el_ip + ip], K);   
        // }

        // int c = tid*(el_dofs * el_dofs + el_dofs)/2;
        // for(int i_dof = 0 ; i_dof < el_dofs; i_dof++){
        //     for(int j_dof = i_dof; j_dof < el_dofs; j_dof++){
        //         Kc[c] += K[i_dof * el_dofs + j_dof];
        //         c++;
        //     }
        // }	
     }
}

