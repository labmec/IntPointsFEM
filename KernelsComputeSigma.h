#include "pzreal.h"
#include "TPZTensor.h"

#include "FunctionsStressStrain.h"
#include "FunctionsTangentOperator.h"

__global__ void ComputeSigmaKernel(bool update_mem, int npts, REAL *glob_delta_strain, REAL *glob_sigma, REAL lambda, REAL mu, REAL mc_phi, REAL mc_psi, REAL mc_cohesion, 
    REAL *dPlasticStrain,  REAL *dMType, REAL *dAlpha, REAL *dSigma, REAL *dStrain, REAL *weight) {
    int ipts = blockIdx.x * blockDim.x + threadIdx.x;

    REAL K;
    REAL G;

    K = lambda + 2 * mu/3;
    G = mu;

    REAL strain[6];
    REAL elastic_strain[6];
    REAL sigma[6];
    REAL plastic_strain[6];
    REAL aux_tensor[3];

    REAL eigenvectors[9];
    REAL sigma_projected[3];
    REAL gradient[9];

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
        ComposeStrain(aux_tensor, strain);

        for(int i = 0; i < 6; i++) {
            plastic_strain[i] = dPlasticStrain[6*ipts + i];
        }

        ElasticStrain(plastic_strain, strain, elastic_strain);
        ComputeTrialStress(elastic_strain, sigma, mu, lambda);
        SpectralDecomposition(sigma, sigma_projected, eigenvectors);
        ProjectSigma(sigma_projected, sigma_projected, el_mtype, el_alpha, mc_phi, mc_psi, mc_cohesion, K, G, gradient, false);
        ReconstructStressTensor(sigma_projected, eigenvectors, sigma);

        // Update plastic strain
        ComputeStrain(sigma, elastic_strain, mu, lambda);
        PlasticStrain(strain, elastic_strain, plastic_strain);

        //Copy to stress vector
        TranslateStress(sigma, aux_tensor);
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

__global__ void ComputeSigmaDepKernel(bool update_mem, int npts, REAL *glob_delta_strain, REAL *glob_sigma, REAL *glob_dep, REAL lambda, REAL mu, REAL mc_phi, REAL mc_psi, REAL mc_cohesion,
                                   REAL *dPlasticStrain,  REAL *dMType, REAL *dAlpha, REAL *dSigma, REAL *dStrain, REAL *weight) {
    int ipts = blockIdx.x * blockDim.x + threadIdx.x;

    REAL K;
    REAL G;

    K = lambda + 2 * mu/3;
    G = mu;

    REAL strain[6];
    REAL elastic_strain[6];
    REAL sigma[6];
    REAL plastic_strain[6];
    REAL aux_tensor[3];

    REAL stress_eigenvalues[3];
    REAL stress_eigenvectors[9];
    REAL gradient[9];
    REAL dep[36];

    REAL sigma_projected[3];

    REAL strain_eigenvalues[3];
    REAL strain_eigenvectors[9];

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
        ComposeStrain(aux_tensor, strain);

        for(int i = 0; i < 6; i++) {
            plastic_strain[i] = dPlasticStrain[6*ipts + i];
        }

        ElasticStrain(plastic_strain, strain, elastic_strain);
        ComputeTrialStress(elastic_strain, sigma, mu, lambda);
        SpectralDecomposition(sigma, stress_eigenvalues, stress_eigenvectors);

        elastic_strain[1] = elastic_strain[1] / 2;
        SpectralDecomposition(elastic_strain, strain_eigenvalues, strain_eigenvectors);
        ProjectSigma(stress_eigenvalues, sigma_projected, el_mtype, el_alpha, mc_phi, mc_psi, mc_cohesion, K, G, gradient, true);
        TangentOperator(gradient, stress_eigenvalues, strain_eigenvalues, strain_eigenvectors, dep, G, lambda);

        ReconstructStressTensor(sigma_projected, stress_eigenvectors, sigma);


        // Update plastic strain
        ComputeStrain(sigma, elastic_strain, mu, lambda);
        PlasticStrain(strain, elastic_strain, plastic_strain);

        //Copy to stress vector
        TranslateStress(sigma, aux_tensor);
        for(int i = 0; i < 3; i++) {
            glob_sigma[3*ipts + i] = weight[ipts] * aux_tensor[i];
        }

        TranslateDep(dep,  gradient);
        for(int i = 0; i < 9; i++) {
            glob_dep[9*ipts + i] = weight[ipts] * gradient[i];
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
