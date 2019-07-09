#include "pzreal.h"
#include "MatMulKernels.h"

__device__ void ComputeConstitutiveMatrixDevice(int64_t point_index, REAL *De){
    REAL lambda = 555.555555555556;
    REAL mu = 833.333333333333;
    
    De[0] = lambda + 2.0*mu;
    De[1] = 0.;
    De[2] = lambda;
    De[3] = 0.;
    De[4]= mu;
    De[5]= 0.;
    De[6] = lambda;
    De[7] = 0.;
    De[8]= lambda + 2.0*mu;
}

__device__ void ComputeTangentMatrixDevice(int el_npts, int el_dofs, REAL *storage, REAL *weight, REAL *K){   

    int n_sigma_comps = 3;

    // REAL De[3 * 3];
    // REAL DeBip[3 * 18];
    REAL *De = (REAL*)malloc(n_sigma_comps * n_sigma_comps * sizeof(REAL));
    REAL *DeBip = (REAL*)malloc(n_sigma_comps * el_dofs * sizeof(REAL));
    for(int i = 0; i < n_sigma_comps * el_dofs; i++) DeBip[i] = 0;

        int c = 0;
#pragma unroll
    for (int ip = 0; ip < el_npts; ip++) {
    	ComputeConstitutiveMatrixDevice(ip,De);

        REAL omega = weight[ip];

        MultAddDevice(false, n_sigma_comps, el_dofs, n_sigma_comps, De, &storage[c], DeBip, 1., 0.);
        MultAddDevice(true, el_dofs, el_dofs, n_sigma_comps, &storage[c], DeBip, K, omega, 1.);

        c += n_sigma_comps * el_dofs;
    }
    free(De);
    free(DeBip);
}

__device__ void ComputeTangentMatrixDevice(int ip, int el_npts, int el_dofs, REAL *storage, REAL *weight, REAL *K){       
    int n_sigma_comps = 3;

    REAL De[3 * 3];
    REAL DeBip[3 * 18];
    // REAL *De = (REAL*)malloc(n_sigma_comps * n_sigma_comps * sizeof(REAL));
    // REAL *DeBip = (REAL*)malloc(n_sigma_comps * el_dofs * sizeof(REAL));
    for(int i = 0; i < n_sigma_comps * el_dofs; i++) DeBip[i] = 0;

    ComputeConstitutiveMatrixDevice(ip,De);

    REAL omega = weight[ip];

    MultAddDevice(false, n_sigma_comps, el_dofs, n_sigma_comps, De, &storage[n_sigma_comps * el_dofs * ip], DeBip, 1., 0.);
    MultAddDevice(true, el_dofs, el_dofs, n_sigma_comps, &storage[n_sigma_comps * el_dofs * ip], DeBip, K, omega, 1.);


    // free(De);
    // free(DeBip);
}

__device__ int64_t me(int *ia_to_sequence, int *ja_to_sequence, int64_t & i_dest, int64_t & j_dest) {

    // Get the matrix entry at (row,col) without bound checking
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

// #define ColorbyIp_Q

__global__ 
__launch_bounds__(2*256, 3)
void MatrixAssembleKernel(REAL *K, int nel, int nnz, REAL *Kg, int64_t *el_color_index, int *dof_indexes, int *colsizes, int *colfirstindex, 
    int *ia_to_sequence, int *ja_to_sequence, int *ia_to_sequence_linear, int *ja_to_sequence_linear, REAL *KgLinear, int64_t *ip_color_indexes) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int n_sigma_comps;
    n_sigma_comps = 3;

    if(tid < nel) {
        int iel = el_color_index[tid];
#ifdef ColorbyIp_Q       

        int ip = ip_color_indexes[tid];
#endif 

        int el_dofs = colsizes[iel];
        int colpos = colfirstindex[iel];

        for (int i_dof = 0; i_dof < el_dofs; i_dof++) {

        int64_t i_dest = dof_indexes[colpos + i_dof];

            for (int j_dof = 0; j_dof < el_dofs; j_dof++) {

                int64_t j_dest = dof_indexes[colpos + j_dof];
                // STATE val = K[i_dof * el_dofs + j_dof];
                STATE val = K[i_dof + j_dof * el_dofs + el_dofs * el_dofs * tid];
                
                if (i_dest <= j_dest) {
                    int index = me(ia_to_sequence, ja_to_sequence, i_dest, j_dest);
                    int64_t  index_linear = me(ia_to_sequence_linear, ja_to_sequence_linear, i_dest, j_dest);

#ifdef ColorbyIp_Q            
                    if(ip == 0) {
                     STATE val_linear = KgLinear[index_linear];
                       Kg[index+nnz*ip] += val + val_linear;
                   }
                   else  {
                    Kg[index+nnz*ip] += val;

                    }
#else  
                    STATE val_linear = KgLinear[index_linear];
                    Kg[index] += val + val_linear;
#endif  

                }
            }
        }           
    }
}


__global__ 
__launch_bounds__(2*256, 3)
void MatrixAssembleKernel(int nel, int nnz, REAL *Kg, int first_el, int64_t *el_color_index, REAL *weight, int *dof_indexes, 
	REAL *storage, int *rowsizes, int *colsizes, int *rowfirstindex, int *colfirstindex, int *matrixposition, int *ia_to_sequence, int *ja_to_sequence,
    int *ia_to_sequence_linear, int *ja_to_sequence_linear, REAL *KgLinear, int64_t *ip_color_indexes) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int n_sigma_comps;
	n_sigma_comps = 3;

    // printf("nnz = %d\n", nnz);

	if(tid < nel) {
        int iel = el_color_index[tid];
#ifdef ColorbyIp_Q       

        int ip = ip_color_indexes[tid];
#endif 

        // printf("iel = %d\n", iel);
        // printf("nnz = %d\n", nnz);
        // printf("ip = %d\n", ip);

        int el_npts = rowsizes[iel]/n_sigma_comps;
        int el_dofs = colsizes[iel];
        int colpos = colfirstindex[iel];
        int first_el_ip = rowfirstindex[iel]/n_sigma_comps;
        int matpos = matrixposition[iel];

            // REAL K[18 * 18];
        REAL *K = (REAL*)malloc(el_dofs * el_dofs * sizeof(REAL));
        for(int i = 0; i < el_dofs * el_dofs; i++) K[i] = 0;
#ifdef ColorbyIp_Q            
        ComputeTangentMatrixDevice(ip, el_npts, el_dofs, &storage[matpos], &weight[first_el_ip], K);
#else  
        ComputeTangentMatrixDevice(el_npts, el_dofs, &storage[matpos], &weight[first_el_ip], K);
#endif  

       // for(int k=0; k < el_dofs*el_dofs; k++ ) {
       //      printf("k[%d] = %g\n",k, K[k]);
       //  }

        for (int i_dof = 0; i_dof < el_dofs; i_dof++) {

        int64_t i_dest = dof_indexes[colpos + i_dof];

            for (int j_dof = 0; j_dof < el_dofs; j_dof++) {

                int64_t j_dest = dof_indexes[colpos + j_dof];
                STATE val = K[i_dof * el_dofs + j_dof];
                
                if (i_dest <= j_dest) {
                    int index = me(ia_to_sequence, ja_to_sequence, i_dest, j_dest);
                    int64_t  index_linear = me(ia_to_sequence_linear, ja_to_sequence_linear, i_dest, j_dest);
                    // printf("index = %d\n", index);
                    // printf("%d\n", nnz*ip);
#ifdef ColorbyIp_Q            
                    if(ip == 0) {
                     STATE val_linear = KgLinear[index_linear];
                       Kg[index+nnz*ip] += val + val_linear;
                   }
                   else  {
                    Kg[index+nnz*ip] += val;

                    }
#else  
                    STATE val_linear = KgLinear[index_linear];
                    Kg[index] += val + val_linear;
#endif  

                }
            }
        }			
    free(K);
    }
}

