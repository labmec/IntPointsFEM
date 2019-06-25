//
// Created by natalia on 17/05/19.
//

#include "TPZConstitutiveLawProcessor.h"
#include "SpectralDecomp.h"
#include "SigmaProjection.h"

#ifdef USING_TBB
#include "tbb/parallel_for.h"
#include "tbb/tick_count.h"
#endif

#include "Timer.h"




TPZConstitutiveLawProcessor::TPZConstitutiveLawProcessor() : fNpts(-1), fWeight(0), fMaterial(), fPlasticStrain(0,0), fMType(0,0), fAlpha(0,0) {
#ifdef USING_CUDA
    fCudaCalls = new TPZCudaCalls();
    dWeight.resize(0);
#endif
}

TPZConstitutiveLawProcessor::TPZConstitutiveLawProcessor(int npts, TPZVec<REAL> weight, TPZMaterial *material) : fNpts(-1), fWeight(0), fMaterial(), fPlasticStrain(0,0), fMType(0,0), fAlpha(0,0) {
    SetIntPoints(npts);
    SetWeightVector(weight);
    SetMaterial(material);
}

TPZConstitutiveLawProcessor::~TPZConstitutiveLawProcessor() {

}

TPZConstitutiveLawProcessor::TPZConstitutiveLawProcessor(const TPZConstitutiveLawProcessor &copy) {
    fNpts = copy.fNpts;
    fWeight = copy.fWeight;
    fMaterial = copy.fMaterial;
    fPlasticStrain = copy.fPlasticStrain;
    fMType = copy.fMType;
    fAlpha = copy.fAlpha;
}

TPZConstitutiveLawProcessor &TPZConstitutiveLawProcessor::operator=(const TPZConstitutiveLawProcessor &copy) {
    if(&copy == this){
        return *this;
    }

    fNpts = copy.fNpts;
    fWeight = copy.fWeight;
    fMaterial = copy.fMaterial;
    fPlasticStrain = copy.fPlasticStrain;
    fMType = copy.fMType;
    fAlpha = copy.fAlpha;

    return *this;
}

void TPZConstitutiveLawProcessor::SetIntPoints(int64_t npts) {
    fNpts = npts;
}

void TPZConstitutiveLawProcessor::SetWeightVector(TPZVec<REAL> weight) {
    fWeight = weight;
}

void TPZConstitutiveLawProcessor::SetMaterial(TPZMaterial *material) {
    fMaterial = dynamic_cast<TPZMatElastoPlastic2D<TPZPlasticStepPV<TPZYCMohrCoulombPV,TPZElasticResponse> , TPZElastoPlasticMem> *>(material);
}

void TPZConstitutiveLawProcessor::ElasticStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &elastic_strain) {
    elastic_strain = delta_strain - fPlasticStrain;
}

void TPZConstitutiveLawProcessor::PlasticStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &elastic_strain) {
    fPlasticStrain = delta_strain - elastic_strain;
}

void TPZConstitutiveLawProcessor::ComputeStress(TPZFMatrix<REAL> &elastic_strain, TPZFMatrix<REAL> &sigma) {
    REAL lambda = fMaterial->GetPlasticModel().fER.Lambda();
    REAL mu =  fMaterial->GetPlasticModel().fER.Mu();

    int dim = 2;

    int ipts;
#ifdef USING_TBB
    #ifdef GET_TIME
    tbb::tick_count t0 = tbb::tick_count::now();
    #endif
    tbb::parallel_for(size_t(0),size_t(fNpts),size_t(1),[&](size_t ipts)
#else
    #ifdef GET_TIME
    Timer timer;
    timer.Start();
    #endif
    for (ipts = 0; ipts < fNpts; ipts++)
#endif
    {
        //plane strain
        sigma(4 * ipts, 0) = elastic_strain(2 * ipts, 0) * (lambda + 2. * mu) + elastic_strain(2 * ipts + dim * fNpts + 1, 0) * lambda; // Sigma xx
        sigma(4 * ipts + 1, 0) = elastic_strain(2 * ipts + dim * fNpts + 1, 0) * (lambda + 2. * mu) + elastic_strain(2 * ipts, 0) * lambda; // Sigma yy
        sigma(4 * ipts + 2, 0) = lambda * (elastic_strain(2 * ipts, 0) + elastic_strain(2 * ipts + dim * fNpts + 1, 0)); // Sigma zz
        sigma(4 * ipts + 3, 0) = mu * (elastic_strain(2 * ipts + 1, 0) + elastic_strain(2 * ipts + dim * fNpts, 0)); // Sigma xy
    }
#ifdef USING_TBB
    );
    #ifdef GET_TIME
    tbb::tick_count t1 = tbb::tick_count::now();
    std::cout << "Elapsed time (ComputeStress): " << (t1-t0).seconds() * 1000 << "ms" << std::endl;
    #endif
#else
    #ifdef GET_TIME
    timer.Stop();
    std::cout << "Elapsed time (ComputeStress): " << timer.ElapsedTime() << timer.Unit() << std::endl;
    #endif
#endif
}

void TPZConstitutiveLawProcessor::ComputeStrain(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &elastic_strain) {
    REAL E = fMaterial->GetPlasticModel().fER.E();
    REAL nu = fMaterial->GetPlasticModel().fER.Poisson();

    int dim = 2;

    int ipts;
#ifdef USING_TBB
    tbb::parallel_for(size_t(0),size_t(fNpts),size_t(1),[&](size_t ipts)
#else
    for (ipts = 0; ipts < fNpts; ipts++)
#endif
    {
        elastic_strain(2 * ipts + 0, 0) = 1 / fWeight[ipts] * (1. / E * (sigma(2 * ipts, 0) * (1. - nu * nu) - sigma(2 * ipts + dim * fNpts + 1, 0) * (nu + nu * nu))); //exx
        elastic_strain(2 * ipts + 1, 0) = 1 / fWeight[ipts] * ((1. + nu) / E * sigma(2 * ipts + 1, 0)); //exy
        elastic_strain(2 * ipts + dim * fNpts + 0, 0) = elastic_strain(2 * ipts + 1, 0); //exy
        elastic_strain(2 * ipts + dim * fNpts + 1, 0) = 1 / fWeight[ipts] * (1. / E * (sigma(2 * ipts + dim * fNpts + 1, 0) * (1. - nu * nu) - sigma(2 * ipts, 0) * (nu + nu * nu))); //eyy
    }
#ifdef USING_TBB
    );
#endif
}

void TPZConstitutiveLawProcessor::SpectralDecomposition(TPZFMatrix<REAL> &sigma_trial, TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &eigenvectors) {
    REAL maxel;
    TPZVec<REAL> interval(2* fNpts);

    int ipts;
#ifdef USING_TBB
    tbb::parallel_for(size_t(0),size_t(fNpts),size_t(1),[&](size_t ipts)
#else
    for (ipts = 0; ipts < fNpts; ipts++)
#endif
    {
        REAL maxel;
        Normalize(&sigma_trial(4*ipts, 0), maxel);
        Interval(&sigma_trial(4*ipts, 0), &interval[2*ipts]);
        NewtonIterations(&interval[2*ipts], &sigma_trial(4*ipts, 0), &eigenvalues(3*ipts, 0), maxel);
        Eigenvectors(&sigma_trial(4*ipts, 0), &eigenvalues(3*ipts, 0), &eigenvectors(9*ipts,0),maxel);
    }
#ifdef USING_TBB
    );
#endif
}

void TPZConstitutiveLawProcessor::ProjectSigma(TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &sigma_projected) {
    REAL mc_psi = fMaterial->GetPlasticModel().fYC.Psi();
    REAL mc_phi = fMaterial->GetPlasticModel().fYC.Phi();
    REAL mc_cohesion = fMaterial->GetPlasticModel().fYC.Cohesion();
    REAL K = fMaterial->GetPlasticModel().fER.K();
    REAL G = fMaterial->GetPlasticModel().fER.G();

    int ipts;
#ifdef USING_TBB
    tbb::parallel_for(size_t(0),size_t(fNpts),size_t(1),[&](size_t ipts)
#else
    for (ipts = 0; ipts < fNpts; ipts++)
#endif
    {
        bool check = false;
        fMType(ipts,0) = 0;
        check = PhiPlane(&eigenvalues(3*ipts, 0), &sigma_projected(3*ipts, 0), mc_phi, mc_cohesion); //elastic domain
        if (!check) { //plastic domain
            fMType(ipts,0) = 1;
            check = ReturnMappingMainPlane(&eigenvalues(3*ipts, 0), &sigma_projected(3*ipts, 0), fAlpha(ipts,0), mc_phi, mc_psi, mc_cohesion, K, G); //main plane
            if (!check) { //edges or apex
                if  (((1 - sin(mc_psi)) * eigenvalues(0 + 3*ipts, 0) - 2. * eigenvalues(1 + 3*ipts, 0) + (1 + sin(mc_psi)) * eigenvalues(2 + 3*ipts, 0)) > 0) { // right edge
                    check = ReturnMappingRightEdge(&eigenvalues(3*ipts, 0), &sigma_projected(3*ipts, 0), fAlpha(ipts,0), mc_phi, mc_psi, mc_cohesion, K, G);
                } else { //left edge
                    check = ReturnMappingLeftEdge(&eigenvalues(3*ipts, 0), &sigma_projected(3*ipts, 0), fAlpha(ipts,0), mc_phi, mc_psi, mc_cohesion, K, G);
                }
                if (!check) { //apex
                    fMType(ipts,0) = -1;
                    ReturnMappingApex(&eigenvalues(3*ipts, 0), &sigma_projected(3*ipts, 0), fAlpha(ipts,0), mc_phi, mc_psi, mc_cohesion, K);
                }
            }
        }
    }
#ifdef USING_TBB
    );
#endif
}

void TPZConstitutiveLawProcessor::StressCompleteTensor(TPZFMatrix<REAL> &sigma_projected, TPZFMatrix<REAL> &eigenvectors, TPZFMatrix<REAL> &sigma){
    TPZVec<REAL> weight;

    int dim = 2;
    int ipts;
#ifdef USING_TBB
    tbb::parallel_for(size_t(0),size_t(fNpts),size_t(1),[&](size_t ipts)
#else
    for (ipts = 0; ipts < fNpts; ipts++)
#endif
    {
        sigma(2*ipts + 0,0) = fWeight[ipts]*(sigma_projected(3*ipts + 0,0)*eigenvectors(9*ipts + 0,0)*eigenvectors(9*ipts + 0,0) + sigma_projected(3*ipts + 1,0)*eigenvectors(9*ipts + 3,0)*eigenvectors(9*ipts + 3,0) + sigma_projected(3*ipts + 2,0)*eigenvectors(9*ipts + 6,0)*eigenvectors(9*ipts + 6,0));
        sigma(2*ipts + 1,0) = fWeight[ipts]*(sigma_projected(3*ipts + 0,0)*eigenvectors(9*ipts + 0,0)*eigenvectors(9*ipts + 1,0) + sigma_projected(3*ipts + 1,0)*eigenvectors(9*ipts + 3,0)*eigenvectors(9*ipts + 4,0) + sigma_projected(3*ipts + 2,0)*eigenvectors(9*ipts + 6,0)*eigenvectors(9*ipts + 7,0));
        sigma(2*ipts + dim * fNpts,0) = sigma(2*ipts + 1,0);
        sigma(2*ipts + dim * fNpts + 1,0) = fWeight[ipts]*(sigma_projected(3*ipts + 0,0)*eigenvectors(9*ipts + 1,0)*eigenvectors(9*ipts + 1,0) + sigma_projected(3*ipts + 1,0)*eigenvectors(9*ipts + 4,0)*eigenvectors(9*ipts + 4,0) + sigma_projected(3*ipts + 2,0)*eigenvectors(9*ipts + 7,0)*eigenvectors(9*ipts + 7,0));
    }
#ifdef USING_TBB
    );
#endif
}

void TPZConstitutiveLawProcessor::ComputeSigma(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &sigma) {
    int dim = 2;

    sigma.Resize(dim * dim * fNpts, 1);

    fPlasticStrain.Resize(dim * dim * fNpts, 1);
    fPlasticStrain.Zero();

    fMType.Resize(fNpts, 1);
    fMType.Zero();

    fAlpha.Resize(fNpts, 1);
    fAlpha.Zero();

    TPZFMatrix<REAL> elastic_strain(dim * dim * fNpts, 1, 0.);
    TPZFMatrix<REAL> sigma_trial(dim * dim * fNpts, 1, 0.);
    TPZFMatrix<REAL> eigenvalues(3 * fNpts, 1, 0.);
    TPZFMatrix<REAL> eigenvectors(9 * fNpts, 1, 0.);
    TPZFMatrix<REAL> sigma_projected(3 * fNpts, 1, 0.);

    // Compute sigma
    ElasticStrain(delta_strain, elastic_strain);
    ComputeStress(elastic_strain, sigma_trial);
    SpectralDecomposition(sigma_trial, eigenvalues, eigenvectors);
    ProjectSigma(eigenvalues, sigma_projected);
    StressCompleteTensor(sigma_projected, eigenvectors, sigma);


    // Update plastic strain
    ComputeStrain(sigma, elastic_strain);
    PlasticStrain(delta_strain, elastic_strain);
}

#ifdef USING_CUDA
void TPZConstitutiveLawProcessor::ComputeSigma(TPZVecGPU<REAL> &delta_strain, TPZVecGPU<REAL> &sigma) {
    // REAL lambda = fMaterial->GetPlasticModel().fER.Lambda();
    // REAL mu =  fMaterial->GetPlasticModel().fER.Mu();
    // REAL mc_phi = fMaterial->GetPlasticModel().fYC.Phi();
    // REAL mc_psi = fMaterial->GetPlasticModel().fYC.Psi();
    // REAL mc_cohesion = fMaterial->GetPlasticModel().fYC.Cohesion();
    // REAL K = fMaterial->GetPlasticModel().fER.K();
    // REAL G = fMaterial->GetPlasticModel().fER.G();

    // int dim = 2;

    // sigma.resize(dim * dim * fNpts);
    // sigma.Zero();

    // TPZVecGPU<REAL> eigenvalues(3 * fNpts);
    // TPZVecGPU<REAL> eigenvectors(9 * fNpts);
    // eigenvalues.Zero();
    // eigenvectors.Zero();
    // TPZVecGPU<REAL> elastic_strain(dim * dim * fNpts);
    // TPZVecGPU<REAL> sigma_trial(dim * dim * fNpts);

    // TPZVecGPU<REAL> sigma_projected(3 * fNpts);

    // fCudaCalls->ElasticStrain(delta_strain.getData(), elastic_strain.getData(), dim * dim * fNpts);
    // fCudaCalls->ComputeStress(elastic_strain.getData(), sigma_trial.getData(), fNpts, mu, lambda); 
    // fCudaCalls->SpectralDecomposition(sigma_trial.getData(), eigenvalues.getData(), eigenvectors.getData(), fNpts); 
    // fCudaCalls->ProjectSigma(eigenvalues.getData(), sigma_projected.getData(), fNpts, mc_phi, mc_psi, mc_cohesion, K, G);
    // ERRADO



    // TPZFMatrix<REAL> sigma_projected(3 * fNpts, 1, 0.);

    // // Compute sigma
    // ElasticStrain(delta_strain, elastic_strain);
    // ComputeStress(elastic_strain, sigma_trial);
    // SpectralDecomposition(sigma_trial, eigenvalues, eigenvectors);
    // ProjectSigma(eigenvalues, sigma_projected);
    // StressCompleteTensor(sigma_projected, eigenvectors, sigma);

    // // Update plastic strain
    // ComputeStrain(sigma, elastic_strain);
    // PlasticStrain(delta_strain, elastic_strain);
}
#endif
