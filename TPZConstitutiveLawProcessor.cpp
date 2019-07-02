//
// Created by natalia on 17/05/19.
//

#include "TPZConstitutiveLawProcessor.h"
#include "TPZMatWithMem.h"
#include "SpectralDecomp.h"
#include "SigmaProjection.h"
#include <functional>

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

void TPZConstitutiveLawProcessor::ElasticStrain(TPZFMatrix<REAL> &plastic_strain, TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &elastic_strain) {
    elastic_strain = delta_strain - plastic_strain;
}

void TPZConstitutiveLawProcessor::PlasticStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &elastic_strain, TPZFMatrix<REAL> &plastic_strain) {
    plastic_strain = delta_strain - elastic_strain;
}

void TPZConstitutiveLawProcessor::De(TPZFMatrix<REAL> & De){
    
    REAL lambda = fMaterial->GetPlasticModel().fER.Lambda();
    REAL mu = fMaterial->GetPlasticModel().fER.G();
    
    De.Zero();
    
    De(_XX_, _XX_) += lambda;
    De(_XX_, _YY_) += lambda;
    De(_XX_, _ZZ_) += lambda;
    De(_YY_, _XX_) += lambda;
    De(_YY_, _YY_) += lambda;
    De(_YY_, _ZZ_) += lambda;
    De(_ZZ_, _XX_) += lambda;
    De(_ZZ_, _YY_) += lambda;
    De(_ZZ_, _ZZ_) += lambda;
    
    int i;
    for (i = 0; i < 6; i++) De(i, i) += mu;
    
    De(_XX_, _XX_) += mu;
    De(_YY_, _YY_) += mu;
    De(_ZZ_, _ZZ_) += mu;
}

void TPZConstitutiveLawProcessor::ComputeTrialStress(TPZFMatrix<REAL> &elastic_strain, TPZFMatrix<REAL> &sigma_trial) {
    TPZFNMatrix<36,REAL> De(6,6,0.0);
    this->De(De);

    TPZFNMatrix<6,REAL> el_delta_strain, el_stress;
    De.Multiply(elastic_strain, sigma_trial);
}

void TPZConstitutiveLawProcessor::ComputeStrain(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &elastic_strain) {
    TPZFNMatrix<36,REAL> De(6,6,0.0);
    TPZFNMatrix<36,REAL> DeInv(6,6,0.0);
    this->De(De);

    De.Inverse(DeInv,ENoDecompose);
    DeInv.Multiply(sigma,elastic_strain);
}

void TPZConstitutiveLawProcessor::SpectralDecomposition(TPZFMatrix<REAL> &sigma_trial, TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &eigenvectors) {
    TPZVec<REAL> interval(2);
    REAL maxel;
    Normalize(&sigma_trial(0, 0), maxel);
    Interval(&sigma_trial(0, 0), &interval[0]);
    NewtonIterations(&interval[0], &sigma_trial(0, 0), &eigenvalues(0, 0), maxel);
    Eigenvectors(&sigma_trial(0, 0), &eigenvalues(0, 0), &eigenvectors(0,0),maxel);
}

void TPZConstitutiveLawProcessor::ProjectSigma(TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &sigma_projected, REAL &alpha, int &mtype) {
    REAL mc_psi = fMaterial->GetPlasticModel().fYC.Psi();
    REAL mc_phi = fMaterial->GetPlasticModel().fYC.Phi();
    REAL mc_cohesion = fMaterial->GetPlasticModel().fYC.Cohesion();
    REAL K = fMaterial->GetPlasticModel().fER.K();
    REAL G = fMaterial->GetPlasticModel().fER.G();

    bool check = false;
    mtype = 0;
    check = PhiPlane(&eigenvalues(0, 0), &sigma_projected(0, 0), mc_phi, mc_cohesion); //elastic domain
    if (!check) { //plastic domain
        mtype = 1;
        check = ReturnMappingMainPlane(&eigenvalues(0, 0), &sigma_projected(0, 0), alpha, mc_phi, mc_psi, mc_cohesion, K, G); //main plane
        if (!check) { //edges or apex
            if  (((1 - sin(mc_psi)) * eigenvalues(0, 0) - 2. * eigenvalues(1, 0) + (1 + sin(mc_psi)) * eigenvalues(2, 0)) > 0) { // right edge
                check = ReturnMappingRightEdge(&eigenvalues(0, 0), &sigma_projected(0, 0), alpha, mc_phi, mc_psi, mc_cohesion, K, G);
            } else { //left edge
                check = ReturnMappingLeftEdge(&eigenvalues(0, 0), &sigma_projected(0, 0), alpha, mc_phi, mc_psi, mc_cohesion, K, G);
            }
            if (!check) { //apex
                mtype = -1;
                ReturnMappingApex(&eigenvalues(0, 0), &sigma_projected(0, 0), alpha, mc_phi, mc_psi, mc_cohesion, K);
            }
        }
    }

}

void TPZConstitutiveLawProcessor::ReconstructStressTensor(TPZFMatrix<REAL> &sigma_projected, TPZFMatrix<REAL> &eigenvectors, TPZFMatrix<REAL> &sigma){
    
    sigma(_XX_, 0) = (sigma_projected(0,0)*eigenvectors(0,0)*eigenvectors(0,0) + sigma_projected(1,0)*eigenvectors(3,0)*eigenvectors(3,0) + sigma_projected(2,0)*eigenvectors(6,0)*eigenvectors(6,0));
    sigma(_YY_, 0) = (sigma_projected(0,0)*eigenvectors(1,0)*eigenvectors(1,0) + sigma_projected(1,0)*eigenvectors(4,0)*eigenvectors(4,0) + sigma_projected(2,0)*eigenvectors(7,0)*eigenvectors(7,0));
    sigma(_ZZ_, 0) = (sigma_projected(0,0)*eigenvectors(2,0)*eigenvectors(2,0) + sigma_projected(1,0)*eigenvectors(5,0)*eigenvectors(5,0) + sigma_projected(2,0)*eigenvectors(8,0)*eigenvectors(8,0));

    sigma(_XY_, 0) = (sigma_projected(0,0)*eigenvectors(0,0)*eigenvectors(1,0) + sigma_projected(1,0)*eigenvectors(3,0)*eigenvectors(4,0) + sigma_projected(2,0)*eigenvectors(6,0)*eigenvectors(7,0));
    sigma(_XZ_, 0) = (sigma_projected(0,0)*eigenvectors(0,0)*eigenvectors(2,0) + sigma_projected(1,0)*eigenvectors(3,0)*eigenvectors(5,0) + sigma_projected(2,0)*eigenvectors(6,0)*eigenvectors(8,0));
    sigma(_YZ_, 0) = (sigma_projected(0,0)*eigenvectors(1,0)*eigenvectors(2,0) + sigma_projected(1,0)*eigenvectors(4,0)*eigenvectors(5,0) + sigma_projected(2,0)*eigenvectors(7,0)*eigenvectors(8,0));

}

void TPZConstitutiveLawProcessor::ComputeSigma(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &sigma) {
    
    int64_t rows = delta_strain.Rows();
    int64_t cols = delta_strain.Cols();
    sigma.Resize(rows,cols);

    fPlasticStrain.Resize(6 * fNpts, 1);
    fPlasticStrain.Zero();

    fMType.Resize(1 * fNpts, 1);
    fMType.Zero();

    fAlpha.Resize(1 * fNpts, 1);
    fAlpha.Zero();


//    { /// Example of lambda expression.
//        int int_var = 42;
//        auto lambda_func = [& int_var](){cout <<
//            "This lambda has a copy of int_var when created: " << int_var << endl;};
//        for(int i = 0; i < 3; i++) {
//            int_var++;
//            lambda_func();
//        }
//    }

    
    TPZMatWithMem<TPZElastoPlasticMem> * mat = dynamic_cast<TPZMatWithMem<TPZElastoPlasticMem> *>(fMaterial);
    auto EvaluateFlux = [this] (int ipts, REAL & alpha, int & mtype, TPZFMatrix<REAL> & delta_eps, TPZFMatrix<REAL> & sigma, TPZMatWithMem<TPZElastoPlasticMem> * mat)
    {
        TPZFMatrix<REAL> el_delta_strain(3, 1, 0.);
        TPZFMatrix<REAL> elastic_strain(6, 1, 0.);
        TPZFMatrix<REAL> sigma_trial(6, 1, 0.);
        TPZFMatrix<REAL> eigenvalues(3, 1, 0.);
        TPZFMatrix<REAL> eigenvectors(9, 1, 0.);
        TPZFMatrix<REAL> sigma_projected(3, 1, 0.);
        TPZFMatrix<REAL> el_plastic_strain(6, 1, 0.);
        
        // Return Mapping components
        ElasticStrain(el_plastic_strain, delta_eps, elastic_strain);
        ComputeTrialStress(elastic_strain, sigma_trial);
        SpectralDecomposition(sigma_trial, eigenvalues, eigenvectors);
        ProjectSigma(eigenvalues, sigma_projected, alpha, mtype);
        ReconstructStressTensor(sigma_projected, eigenvectors, sigma);
        
        // Update plastic strain
        ComputeStrain(sigma, elastic_strain);
        PlasticStrain(delta_eps, elastic_strain, el_plastic_strain);
        
    };
    
    // The constitutive law is computing assuming full tensors
    int ipts;
#ifdef USING_TBB
    tbb::parallel_for(size_t(0),size_t(fNpts),size_t(1),[&](size_t ipts)
#else
    for (ipts = 0; ipts < fNpts; ipts++)
#endif    
    {
        
        
        TPZFMatrix<REAL> full_delta_strain(6, 1, 0.);
        TPZFMatrix<REAL> full_sigma(6, 1, 0.);
        TPZFMatrix<REAL> el_sigma(3, 1, 0.);

        REAL alpha;
        int mtype;

        TPZFMatrix<REAL> el_delta_strain(3, 1, 0.);
        delta_strain.GetSub(3 * ipts, 0, 3, 1, el_delta_strain);
        // Compute sigma
        TranslateStrain(el_delta_strain, full_delta_strain);
        
        EvaluateFlux(ipts,alpha, mtype, full_delta_strain,full_sigma,mat);

        //Copy to stress vector
        TranslateStress(full_sigma, el_sigma);
        
        el_sigma(0,0) *= fWeight[ipts];
        el_sigma(1,0) *= fWeight[ipts];
        el_sigma(2,0) *= fWeight[ipts];
        sigma.AddSub(3 * ipts, 0, el_sigma);

//        //Copy to PlasticStrain vector
//        fPlasticStrain.AddSub(6 * ipts, 0, el_plastic_strain);

        //Copy to MType and Alpha vectors
        fAlpha(ipts,0) = alpha;
        fMType(ipts,0) = mtype;
    }
#ifdef USING_TBB
);
#endif

}

void TPZConstitutiveLawProcessor::TranslateStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &full_delta_strain){
    
    int dim = 2;
    int n_sigma_comps = 3;
    
    if (dim == 2) {
            full_delta_strain.Resize(6,1);
            full_delta_strain.Zero();
        	full_delta_strain(_XX_,0) = delta_strain(0,0);
            full_delta_strain(_XY_,0) = delta_strain(1,0);
            full_delta_strain(_YY_,0) = delta_strain(2,0);
    }else{
        full_delta_strain = delta_strain;
    }
    
}

void TPZConstitutiveLawProcessor::TranslateStress(TPZFMatrix<REAL> &full_stress, TPZFMatrix<REAL> &stress){
    
    int dim = 2;
    int n_sigma_comps = 3;
    
    if (dim == 2) {
            stress(0,0) =  full_stress(_XX_,0);
            stress(1,0) =  full_stress(_XY_,0);
            stress(2,0) =  full_stress(_YY_,0);
    }else{
        stress = full_stress;
    }
    
}

#ifdef USING_CUDA
void TPZConstitutiveLawProcessor::ComputeSigma(TPZVecGPU<REAL> &delta_strain, TPZVecGPU<REAL> &sigma) {
    REAL lambda = fMaterial->GetPlasticModel().fER.Lambda();
    REAL mu =  fMaterial->GetPlasticModel().fER.Mu();
    REAL mc_phi = fMaterial->GetPlasticModel().fYC.Phi();
    REAL mc_psi = fMaterial->GetPlasticModel().fYC.Psi();
    REAL mc_cohesion = fMaterial->GetPlasticModel().fYC.Cohesion();

    int64_t rows = delta_strain.getSize();
    sigma.resize(rows);

    dPlasticStrain.resize(6 * fNpts);
    dPlasticStrain.Zero();

    dMType.resize(1 * fNpts);
    dMType.Zero();

    dAlpha.resize(1 * fNpts);
    dAlpha.Zero();

    fCudaCalls->ComputeSigma(fNpts, delta_strain.getData(), sigma.getData(), lambda, mu, mc_phi, mc_psi, mc_cohesion, dPlasticStrain.getData(),  dMType.getData(), dAlpha.getData(), dWeight.getData());

}
#endif

#ifdef USING_CUDA
void TPZConstitutiveLawProcessor::TransferDataToGPU() {
    dWeight.resize(fWeight.size());
    dWeight.set(&fWeight[0], fWeight.size());
}
#endif
