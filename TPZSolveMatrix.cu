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


///CUDA KERNELS
__global__ void ComputeSigmaKernel(int npts_tot, double *weight, double *result, double *sigma)
{
REAL E = 200000000.;
REAL nu = 0.30;
int ipts = blockIdx.x*blockDim.x + threadIdx.x;

if (ipts < npts_tot/2) {
	    sigma[2*ipts] = weight[ipts]*E/(1.-nu*nu)*(result[2*ipts]+nu*result[2*ipts+npts_tot+1]); // Sigma x
        sigma[2*ipts+1] = weight[ipts]*E/(1.-nu*nu)*(1.-nu)/2*(result[2*ipts+1]+result[2*ipts+npts_tot])*0.5; // Sigma xy
        sigma[2*ipts+npts_tot] = sigma[2*ipts+1]; //Sigma xy
        sigma[2*ipts+npts_tot+1] = weight[ipts]*E/(1.-nu*nu)*(result[2*ipts+npts_tot+1]+nu*result[2*ipts]); // Sigma y
}
}

__global__ void AssembleKernel(int npts, int *indexes, double *nfvec, double *nfglob)
{
//int i = blockIdx.x*blockDim.x + threadIdx.x;
//
//    if (i < npts) {
//	nfglob[indexes[i]] += nfvec[i];
//    }
}

void TPZSolveMatrix::HostToDevice()
{
int nstorage = fStorage.size();
int nrowsizes = fRowSizes.size();
int ncolsizes = fColSizes.size();
int nindexes = fIndexes.size();
int nmatrixposition = fMatrixPosition.size();
int nrowfirstindex = fRowFirstIndex.size();
int ncolfirstindex = fColFirstIndex.size();

cudaMalloc(&dfStorage, nstorage*sizeof(double));
cudaMalloc(&dfRowSizes, nrowsizes*sizeof(int));
cudaMalloc(&dfColSizes, ncolsizes*sizeof(int));
cudaMalloc(&dfIndexes, nindexes*sizeof(int));
cudaMalloc(&dfMatrixPosition, nmatrixposition*sizeof(int));
cudaMalloc(&dfRowFirstIndex, nrowfirstindex*sizeof(int));
cudaMalloc(&dfColFirstIndex, ncolfirstindex*sizeof(int));

cudaMemcpy(dfStorage, &fStorage[0], nstorage*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(dfRowSizes, &fRowSizes[0], nrowsizes*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(dfColSizes, &fColSizes[0], ncolsizes*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(dfIndexes, &fIndexes[0], nindexes*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(dfMatrixPosition, &fMatrixPosition[0], nmatrixposition*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(dfRowFirstIndex, &fRowFirstIndex[0], nrowfirstindex*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(dfColFirstIndex, &fColFirstIndex[0], ncolfirstindex*sizeof(int), cudaMemcpyHostToDevice);

}

void TPZSolveMatrix::FreeDeviceMemory()
{
cudaFree(dfStorage);
cudaFree(dfRowSizes);
cudaFree(dfColSizes);
cudaFree(dfIndexes);
cudaFree(dfMatrixPosition);
cudaFree(dfRowFirstIndex);
cudaFree(dfColFirstIndex);
}

void  TPZSolveMatrix::SolveWithCUDA(const TPZFMatrix<STATE>  &global_solution, TPZStack<REAL> &weight, TPZFMatrix<REAL> &nodal_forces_global) const
{
int64_t nelem = fRowSizes.size();

///GATHER OPERATION------------------------------------------------
MKL_INT n_globalsol = fIndexes.size();
TPZVec<REAL> expandsolution(n_globalsol);
cblas_dgthr(n_globalsol, global_solution, &expandsolution[0], &fIndexes[0]); //USAR O METODO DA CUSPARSE
///----------------------------------------------------------------

///MULTIPLY--------------------------------------------------------
///Initialize result and expandsolution on device
TPZVec<REAL> result(2*n_globalsol);
double *dresult;
cudaMalloc(&dresult, 2*n_globalsol*sizeof(double));

double *dexpandsolution;
cudaMalloc(&dexpandsolution, n_globalsol*sizeof(double));
cudaMemcpy(dexpandsolution, &expandsolution[0], n_globalsol*sizeof(double), cudaMemcpyHostToDevice);

//Use CUBLAS library to do the multiplication
cudaStream_t stream_m[2*nelem];
cublasHandle_t handle_m[2*nelem];

double alpha_m = 1.0;
double beta_m = 0.0;

for (int iel = 0; iel < nelem; iel++){
	cudaStreamCreate(&stream_m[2*iel]);
	cudaStreamCreate(&stream_m[2*iel+1]);

	cublasCreate(&handle_m[2*iel]);
	cublasCreate(&handle_m[2*iel+1]);

	int64_t pos = fMatrixPosition[iel];
	int64_t cols = fColSizes[iel];
	int64_t rows = fRowSizes[iel];

	int64_t cont_cols = fColFirstIndex[iel];
	int64_t cont_rows = fRowFirstIndex[iel];

	//du
	cublasSetStream(handle_m[2*iel],stream_m[2*iel]);
	cublasDgemv(handle_m[2*iel], CUBLAS_OP_N, rows, cols, &alpha_m, &dfStorage[pos], rows, &dexpandsolution[cont_cols], 1, &beta_m, &dresult[cont_rows], 1);

	//dv   
	cublasSetStream(handle_m[2*iel+1],stream_m[2*iel+1]);
	cublasDgemv(handle_m[2*iel+1], CUBLAS_OP_N, rows, cols, &alpha_m, &dfStorage[pos], rows, &dexpandsolution[cont_cols + fColFirstIndex[nelem]], 1, &beta_m, &dresult[cont_rows + fRowFirstIndex[nelem]], 1);

}

///Free device memory
cudaFree(dexpandsolution); 
cublasDestroy(*handle_m);
cudaStreamSynchronize(0);
///----------------------------------------------------------------

///COMPUTE SIGMA---------------------------------------------------
///Initialize sigma and weight on device
int npts_tot = fRow;
TPZVec<REAL> sigma (2*npts_tot);
double *dsigma;
cudaMalloc(&dsigma, 2*npts_tot*sizeof(double));

double *dweight;
cudaMalloc(&dweight, npts_tot/2*sizeof(double));
cudaMemcpy(dweight, &weight[0], npts_tot/2*sizeof(double), cudaMemcpyHostToDevice);

///Kernel that calculates sigma
dim3 dimGrid(ceil((npts_tot/2)/32.0),1,1);
dim3 dimBlock(32,1,1);
ComputeSigmaKernel<<<dimGrid,dimBlock>>>(npts_tot, dweight, dresult, dsigma);
cudaDeviceSynchronize();

///Free device memory
cudaFree(dresult);
cudaFree(dweight);
///----------------------------------------------------------------

///MULTIPLY TRANSPOSE----------------------------------------------
///Initialize nodal forces vector on device
TPZVec<REAL> nodal_forces_vec(npts_tot);
double *dnodal_forces_vec;
cudaMalloc(&dnodal_forces_vec, npts_tot*sizeof(double));

///Use CUBLAS to do the multiplication
cudaStream_t stream_mt[2*nelem];
cublasHandle_t handle_mt[2*nelem];

double alpha_mt = 1.0;
double beta_mt = 0.0;

for (int64_t iel = 0; iel < nelem; iel++) {
	cudaStreamCreate(&stream_mt[2*iel]);
	cudaStreamCreate(&stream_mt[2*iel+1]);

	cublasCreate(&handle_mt[2*iel]);
	cublasCreate(&handle_mt[2*iel+1]);

	int64_t pos = fMatrixPosition[iel];
        int64_t rows = fRowSizes[iel];
        int64_t cols = fColSizes[iel];
       
	int64_t cont_rows = fRowFirstIndex[iel];
        int64_t cont_cols = fColFirstIndex[iel];

	//Nodal forces in x direction
        cublasSetStream(handle_mt[2*iel],stream_mt[2*iel]);
        cublasDgemv(handle_mt[2*iel], CUBLAS_OP_T, rows, cols, &alpha_mt, &dfStorage[pos], rows, &dsigma[cont_rows], 1, &beta_mt, &dnodal_forces_vec[cont_cols], 1);

        //Nodal forces in y direction
        cublasSetStream(handle_mt[2*iel+1],stream_mt[2*iel+1]);
        cublasDgemv(handle_mt[2*iel+1], CUBLAS_OP_T, rows, cols, &alpha_mt, &dfStorage[pos], rows, &dsigma[cont_rows + npts_tot], 1, &beta_mt, &dnodal_forces_vec[cont_cols + npts_tot/2], 1);
}

cudaMemcpy(&nodal_forces_vec[0], dnodal_forces_vec, npts_tot*sizeof(double), cudaMemcpyDeviceToHost);
std::cout << "Nodal forces vector:" << std::endl;
for(int i = 0; i < npts_tot; i++){
std::cout << nodal_forces_vec[i] << std::endl;
}

///Free device memory
cudaFree(dsigma);
cublasDestroy(*handle_mt);
///----------------------------------------------------------------

///ASSEMBLE--------------------------------------------------------
///Initialize global nodal forces vector on device------------------
int globvec = nodal_forces_global.Rows();
nodal_forces_global.Zero();
double *dnodal_forces_global;
cudaMalloc(&dnodal_forces_global, globvec*sizeof(double));

///Kernel that assemble the nodal forces vectot
dim3 dimGrid_assemb(ceil(npts_tot/32.0),1,1);
dim3 dimBlock_assemb(32,1,1);
AssembleKernel<<<dimGrid,dimBlock>>>(npts_tot, dfIndexes,dnodal_forces_vec, dnodal_forces_global);

///Transfer global nodal forces vector the host
cudaMemcpy(&nodal_forces_global(0,0), dnodal_forces_global, globvec*sizeof(double), cudaMemcpyDeviceToHost);
    
//std::cout << "Assemble:\n" << std::endl;
//nodal_forces_global.Print(std::cout);

///Free device memory
cudaFree(dnodal_forces_vec);
cudaFree(dnodal_forces_global);
///----------------------------------------------------------------
}

void TPZSolveMatrix::Multiply(const TPZFMatrix<STATE>  &global_solution, TPZFMatrix<REAL> &result) const
{
    DebugStop();
}

void TPZSolveMatrix::ComputeSigma(TPZStack<REAL> &weight, TPZFMatrix<REAL> &result, TPZFMatrix<REAL> &sigma)
{
    DebugStop();
}

void TPZSolveMatrix::MultiplyTranspose(TPZFMatrix<STATE>  &intpoint_solution, TPZFMatrix<STATE> &nodal_forces_vec)
{
    DebugStop();
}

void TPZSolveMatrix::TraditionalAssemble(TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) const
{
    DebugStop();
}

void TPZSolveMatrix::ColoredElements(TPZCompMesh * cmesh, TPZVec<int> &nelem_color) const
{
	int nelem = fRowSizes.size();
	int nelem_c = cmesh->NElements();
	int dim_mesh = cmesh->Dimension();

	// INÍCIO DA ASSEMBLAGEM
	int64_t nnodes_tot = cmesh->Reference()->NNodes();
	TPZVec<int> nnodes_vec(nnodes_tot,0.);

	int cont_elem = 0;

	for (int i = 0; i < nelem; i++) {
		nelem_color[i] = -1;
	}

//    TPZVec<int> nelem_color(nelem,-1); // vetor de cores
	TPZVec<int> elem_neighbour(100,0.);

	for (int64_t iel1=0; iel1<nelem_c; iel1++) {
		if(!cmesh->Element(iel1)) continue;
		TPZGeoEl * gel1 = cmesh->Element(iel1)->Reference();
		if(!gel1 ||  gel1->Dimension() != dim_mesh) continue;

		TPZVec<int64_t> nodeindices;
		gel1->GetNodeIndices(nodeindices); // Armazena os nós do elemento finito

		// ** Início da verificação de qual coord é repetida:
		TPZGeoEl * gel2;
		// contadores
		int64_t cont_elem_cor = 0;
		int64_t cont_elem_neighbour = 0;

		// inicializa com nnodes_vec nulo, e preenche com 1 os nós repetidos
		nnodes_vec.Fill(0);
		for (int64_t iel2=0; iel2<nelem_c; iel2++) {
			if(!cmesh->Element(iel2)) continue;
			gel2 = cmesh->Element(iel2)->Reference();
			if(!gel2 ||  gel2->Dimension() != dim_mesh) continue;

			for (int64_t inode=0; inode<gel2->NNodes(); inode++) {
				if(std::find (nodeindices.begin(), nodeindices.end(), gel2->NodeIndex(inode)) != nodeindices.end()){
					nnodes_vec[gel2->NodeIndex(inode)] = 1; // preenchendo nnodes_vec
					elem_neighbour[cont_elem_neighbour] = nelem_color[cont_elem_cor]; // preenche o vetor de elementos vizinhos ao elemento de análise
					cont_elem_neighbour++;
				}
			}
			cont_elem_cor++;
		}
		// ** fim da verificação

		// Preenche a cor
		for (int64_t inodes_tot=0; inodes_tot<nnodes_tot; inodes_tot++) {
			cont_elem_cor = cont_elem;
			if (nnodes_vec[inodes_tot] == 1){
				for (int64_t iel2=iel1; iel2<nelem_c; iel2++) {
					if(!cmesh->Element(iel2)) continue;
					gel2 = cmesh->Element(iel2)->Reference();
					if(!gel2 ||  gel2->Dimension() != dim_mesh) continue;

					gel2->GetNodeIndices(nodeindices);
					if (std::find(nodeindices.begin(), nodeindices.end(), inodes_tot) != nodeindices.end()){
						nelem_color[cont_elem_cor] = 1+nelem_color[cont_elem];
					}
				}
			}

			// Verifica se pode ser uma cor menor
			for (int64_t icolor=0; icolor<nelem_color[cont_elem_cor]; icolor++) {
				if (std::find(elem_neighbour.begin(), elem_neighbour.end(), icolor) == elem_neighbour.end())
					nelem_color[cont_elem_cor] = icolor;
				if (cont_elem==0)
					nelem_color[cont_elem_cor] = 0;
			}
			cont_elem_cor++;
		}
		cont_elem++;
	}
}

void TPZSolveMatrix::ColoredAssemble(TPZVec<int> &nelem_color, TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global)
{
	int nf_tot = fCol;
	int neq = nodal_forces_global.Rows();
	int nelem = nelem_color.size();
	int ncolor = *std::max_element(nelem_color.begin(), nelem_color.end())+1;

	nodal_forces_global.Resize(ncolor*neq,1);
	int pos = 0;
	for (int icolor=0; icolor < ncolor; icolor++) {
		for (int iel = 0; iel < nelem; iel++) {
			if (icolor == nelem_color[iel]) {
				for (int ipts = 0; ipts < fRowSizes[iel]/2; ipts++) {
					int idx = fIndexes[iel * fRowSizes[iel]/2 + ipts];
					int idy = fIndexes[iel * fRowSizes[iel]/2 + ipts + fRow / 2];
					nodal_forces_global(idx + pos, 0) += nodal_forces_vec(iel * fRowSizes[iel] / 2 + ipts, 0);
					nodal_forces_global(idy + pos, 0) += nodal_forces_vec(iel * fRowSizes[iel] / 2 + ipts + fRow / 2, 0);
				}
			}
		}
		pos += neq;
	}

	int colorid;

	if(ncolor%2 != 0 && ncolor > 1){ //se o numero de cores da malha eh impar, adiciona a ultima cor nas posicoes 0 a neq
		colorid = ncolor - 1;
		TPZFMatrix<REAL> assemblecolorid(neq, 1, &nodal_forces_global(colorid * neq, 0), neq);
		nodal_forces_global.AddSub(0, 0, assemblecolorid);
		ncolor -= 1; //menos uma cor para assemblar
	}

	double colorassemb = ncolor/2.;
	while (colorassemb >= 1) {
		for (int icolor = 0; icolor < colorassemb; icolor++) {
			colorid = icolor + colorassemb;
			TPZFMatrix<REAL> assemblecolorid(neq, 1, &nodal_forces_global(colorid * neq, 0), neq);
			nodal_forces_global.AddSub(icolor * neq, 0, assemblecolorid);
		}
		colorassemb = colorassemb / 2;
		if (std::fmod(colorassemb,2) != 0 && colorassemb > 1) { //se depois de "dobrar" o vetor, restar um numero impar de cores, adiciona ultima cor nas posicoes 0 a neq
			colorid = 2 * colorassemb - 1;
			TPZFMatrix<REAL> assemblecolorid(neq, 1, &nodal_forces_global(colorid * neq, 0), neq);
			nodal_forces_global.AddSub(0, 0, assemblecolorid);
			colorassemb = ceil(colorassemb)/2;
		}
	}
	nodal_forces_global.Resize(neq,1);
}


