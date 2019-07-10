#include "pzreal.h"
#include "TPZTensor.h"

#include "SpectralDecompKernels.h"
#include "SigmaProjectionKernels.h"

__device__ void TranslateStrainDevice(REAL *delta_strain, REAL *full_delta_strain) {
    int dim = 2;
    int n_sigma_comps = 3;

    if (dim == 2) {
            full_delta_strain[_XX_] = delta_strain[0];
            full_delta_strain[_XY_] = delta_strain[1];
            full_delta_strain[_XZ_] = 0;
            full_delta_strain[_YY_] = delta_strain[2];
            full_delta_strain[_YZ_] = 0;
            full_delta_strain[_ZZ_] = 0;
    }else{
        full_delta_strain = delta_strain;
    }
}

__device__ void ElasticStrainDevice(REAL *plastic_strain, REAL *strain, REAL *elastic_strain) {
    for(int i = 0; i < 6; i++) {
        elastic_strain[i] = strain[i] - plastic_strain[i];
    }
}

__device__ void PlasticStrainDevice(REAL *strain, REAL *elastic_strain, REAL *plastic_strain) {
    for(int i = 0; i < 6; i++) {
        plastic_strain[i] = strain[i] - elastic_strain[i];
    }
}

__device__ void ComputeTrialStressDevice(REAL *elastic_strain, REAL *sigma_trial, REAL mu, REAL lambda) {
    sigma_trial[_XX_] = (lambda + 2 * mu) * elastic_strain[_XX_] + lambda * elastic_strain[_YY_] + lambda * elastic_strain[_ZZ_];
    sigma_trial[_YY_] = lambda * elastic_strain[_XX_] + (lambda + 2 * mu) * elastic_strain[_YY_] + lambda * elastic_strain[_ZZ_];
    sigma_trial[_ZZ_] = lambda * elastic_strain[_XX_] + lambda * elastic_strain[_YY_] + (lambda + 2 * mu) * elastic_strain[_ZZ_];

    sigma_trial[_XY_] = mu * elastic_strain[_XY_];
    sigma_trial[_XZ_] = mu * elastic_strain[_XZ_];
    sigma_trial[_YZ_] = mu * elastic_strain[_YZ_];
}

__device__ void ReconstructStressTensorDevice(REAL *sigma_projected, REAL *eigenvectors, REAL *sigma) {
    sigma[_XX_] = (sigma_projected[0]*eigenvectors[0]*eigenvectors[0] + sigma_projected[1]*eigenvectors[3]*eigenvectors[3] + sigma_projected[2]*eigenvectors[6]*eigenvectors[6]);
    sigma[_YY_] = (sigma_projected[0]*eigenvectors[1]*eigenvectors[1] + sigma_projected[1]*eigenvectors[4]*eigenvectors[4] + sigma_projected[2]*eigenvectors[7]*eigenvectors[7]);
    sigma[_ZZ_] = (sigma_projected[0]*eigenvectors[2]*eigenvectors[2] + sigma_projected[1]*eigenvectors[5]*eigenvectors[5] + sigma_projected[2]*eigenvectors[8]*eigenvectors[8]);

    sigma[_XY_] = (sigma_projected[0]*eigenvectors[0]*eigenvectors[1] + sigma_projected[1]*eigenvectors[3]*eigenvectors[4] + sigma_projected[2]*eigenvectors[6]*eigenvectors[7]);
    sigma[_XZ_] = (sigma_projected[0]*eigenvectors[0]*eigenvectors[2] + sigma_projected[1]*eigenvectors[3]*eigenvectors[5] + sigma_projected[2]*eigenvectors[6]*eigenvectors[8]);
    sigma[_YZ_] = (sigma_projected[0]*eigenvectors[1]*eigenvectors[2] + sigma_projected[1]*eigenvectors[4]*eigenvectors[5] + sigma_projected[2]*eigenvectors[7]*eigenvectors[8]);
}

__device__ void ComputeStrainDevice(REAL *sigma, REAL *elastic_strain, REAL mu, REAL lambda) {
    elastic_strain[_XX_] = (lambda + mu) / (3 * lambda * mu + 2 * mu * mu) * sigma[_XX_] - lambda / (6 * lambda * mu + 4 * mu * mu) * (sigma[_YY_] + sigma[_ZZ_]);
    elastic_strain[_YY_] = (lambda + mu) / (3 * lambda * mu + 2 * mu * mu) * sigma[_YY_] - lambda / (6 * lambda * mu + 4 * mu * mu) * (sigma[_XX_] + sigma[_ZZ_]);
    elastic_strain[_ZZ_] = (lambda + mu) / (3 * lambda * mu + 2 * mu * mu) * sigma[_ZZ_] - lambda / (6 * lambda * mu + 4 * mu * mu) * (sigma[_YY_] + sigma[_XX_]);

    elastic_strain[_XY_] = 1 / mu * sigma[_XY_];
    elastic_strain[_XZ_] = 1 / mu * sigma[_XZ_];
    elastic_strain[_YZ_] = 1 / mu * sigma[_YZ_];
}

__device__ void TranslateStressDevice(REAL *full_stress, REAL *stress){    
    int dim = 2;
    int n_sigma_comps = 3;
    
    if (dim == 2) {
            stress[0] =  full_stress[_XX_];
            stress[1] =  full_stress[_XY_];
            stress[2] =  full_stress[_YY_];
    }else{
        stress = full_stress;
    }
    
}


__global__ void ComputeSigmaKernel(bool update_mem, int npts, REAL *glob_delta_strain, REAL *glob_sigma, REAL lambda, REAL mu, REAL mc_phi, REAL mc_psi, REAL mc_cohesion, 
    REAL *dPlasticStrain,  REAL *dMType, REAL *dAlpha, REAL *dSigma, REAL *dStrain, REAL *weight) {
 int ipts = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ REAL K;
    __shared__ REAL G;

    K = lambda + 2 * mu/3;
    G = mu;

    REAL strain[6];
    REAL elastic_strain[6];
    REAL sigma[6];
    REAL plastic_strain[6];
    REAL aux_tensor[3];

    REAL eigenvectors[9];
    REAL sigma_projected[3];

    REAL el_alpha;
    int el_mtype;

    if(ipts < npts) {
        for(int i = 0; i < 3; i++) {
            aux_tensor[i] = glob_delta_strain[3*ipts + i];
        }
        for(int i = 0; i < 6; i++) {
            strain[i] = dStrain[6*ipts + i];
        }        

        // Compute sigma
        TranslateStrainDevice(aux_tensor, strain);

        for(int i = 0; i < 6; i++) {
            plastic_strain[i] = dPlasticStrain[6*ipts + i];
        }

        ElasticStrainDevice(plastic_strain, strain, elastic_strain);
        ComputeTrialStressDevice(elastic_strain, sigma, mu, lambda);
        SpectralDecompositionDevice(sigma, sigma_projected, eigenvectors);
        ProjectSigmaDevice(sigma_projected, sigma_projected, el_mtype, el_alpha, mc_phi, mc_psi, mc_cohesion, K, G);
        ReconstructStressTensorDevice(sigma_projected, eigenvectors, sigma);

        // Update plastic strain
        ComputeStrainDevice(sigma, elastic_strain, mu, lambda);
        PlasticStrainDevice(strain, elastic_strain, plastic_strain);

        //Copy to stress vector
        TranslateStressDevice(sigma, aux_tensor);
        for(int i = 0; i < 3; i++) {
            glob_sigma[3*ipts + i] = weight[ipts] * aux_tensor[i];
        }

        if(update_mem) {
            //Update mem
            for(int i = 0; i < 6; i++) {
                dSigma[6*ipts + i] = sigma[i];
                dStrain[6*ipts + i] = strain[i];
                dPlasticStrain[6*ipts + i] = plastic_strain[i];
            }
            dMType[ipts] = el_mtype;
            dAlpha[ipts] = el_alpha;

        }

    }
}


