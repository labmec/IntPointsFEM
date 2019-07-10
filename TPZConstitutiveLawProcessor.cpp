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
    dSigma.resize(0);
    dStrain.resize(0);
    dPlasticStrain.resize(0);
    dMType.resize(0);
    dAlpha.resize(0);
#endif
}

TPZConstitutiveLawProcessor::TPZConstitutiveLawProcessor(int npts, TPZVec<REAL> weight, TPZMaterial *material) : fNpts(-1), fWeight(0), fMaterial(), fPlasticStrain(0,0), fMType(0,0), fAlpha(0,0) {
    SetUpDataByIntPoints(npts);
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

#ifdef USING_CUDA
    dWeight = copy.dWeight;
    dSigma = copy.dSigma;
    dStrain = copy.dStrain;
    dPlasticStrain = copy.dPlasticStrain;
    dMType = copy.dMType;
    dAlpha = copy.dAlpha;
#endif
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

#ifdef USING_CUDA
    dWeight = copy.dWeight;
    dSigma = copy.dSigma;
    dStrain = copy.dStrain;
    dPlasticStrain = copy.dPlasticStrain;
    dMType = copy.dMType;
    dAlpha = copy.dAlpha;
#endif

    return *this;
}

void TPZConstitutiveLawProcessor::SetUpDataByIntPoints(int64_t npts) {
    fNpts = npts;
    
    fSigma.Resize(6 * fNpts, 1);
    fSigma.Zero();
    
    fStrain.Resize(6 * fNpts, 1);
    fStrain.Zero();
    
    fPlasticStrain.Resize(6 * fNpts, 1);
    fPlasticStrain.Zero();
    
    fMType.Resize(1 * fNpts, 1);
    fMType.Zero();
    
    fAlpha.Resize(1 * fNpts, 1);
    fAlpha.Zero();

    #ifdef USING_CUDA
    dSigma.resize(6 * fNpts);
    dSigma.Zero();
    
    dStrain.resize(6 * fNpts);
    dStrain.Zero();

    dPlasticStrain.resize(6 * fNpts);
    dPlasticStrain.Zero();

    dMType.resize(1 * fNpts);
    dMType.Zero();

    dAlpha.resize(1 * fNpts);
    dAlpha.Zero();

    #endif
}

void TPZConstitutiveLawProcessor::SetWeightVector(TPZVec<REAL> &weight) {
    fWeight = weight;
}

void TPZConstitutiveLawProcessor::SetMaterial(TPZMaterial *material) {
    fMaterial = dynamic_cast<TPZMatElastoPlastic2D<TPZPlasticStepPV<TPZYCMohrCoulombPV,TPZElasticResponse> , TPZElastoPlasticMem> *>(material);
}

void TPZConstitutiveLawProcessor::ElasticStrain(TPZFMatrix<REAL> &plastic_strain, TPZFMatrix<REAL> & strain ,TPZFMatrix<REAL> &elastic_strain) {
    elastic_strain = strain - plastic_strain;
}

void TPZConstitutiveLawProcessor::PlasticStrain(TPZFMatrix<REAL> &strain, TPZFMatrix<REAL> &elastic_strain, TPZFMatrix<REAL> &plastic_strain) {
    plastic_strain = strain - elastic_strain;
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

void TPZConstitutiveLawProcessor::ComputeSigma(TPZFMatrix<REAL> & glob_delta_strain, TPZFMatrix<REAL> & glob_sigma) {
    
    int64_t rows = glob_delta_strain.Rows();
    int64_t cols = glob_delta_strain.Cols();
    glob_sigma.Resize(rows,cols);

    // Variables required for lambda's capture block
    TPZMatWithMem<TPZElastoPlasticMem> * mat = dynamic_cast<TPZMatWithMem<TPZElastoPlasticMem> *>(fMaterial);

    
    // The constitutive law is computing assuming full tensors
#ifdef USING_TBB
    
    tbb::parallel_for(size_t(0), size_t(fNpts), size_t(1) , [this, mat, & glob_delta_strain, & glob_sigma] (size_t & ipts) {
        
        TPZFNMatrix<6,REAL> strain(6, 1, 0.);
        TPZFNMatrix<6,REAL> elastic_strain(6, 1, 0.);
        TPZFNMatrix<6,REAL> sigma(6, 1, 0.);
        TPZFNMatrix<6,REAL> plastic_strain(6, 1, 0.);
        TPZFNMatrix<3,REAL> aux_tensor(3, 1, 0.);
        
        REAL alpha;
        int mtype;
        
        //Get from delta strain vector
        glob_delta_strain.GetSub(3 * ipts, 0, 3, 1, aux_tensor);
        
        //Get last strain vector
        fStrain.GetSub(6 * ipts, 0, 6, 1, strain);
        
        // Translate and
        ComposeStrain(aux_tensor, strain);
        
        //Get from plastic strain vector
        fPlasticStrain.GetSub(6 * ipts, 0, 6, 1, plastic_strain);
        
        TPZFMatrix<REAL> eigenvectors(9, 1, 0.);
        TPZFMatrix<REAL> sigma_projected(3, 1, 0.);
        
        // Return Mapping components
        ElasticStrain(plastic_strain, strain, elastic_strain);
        ComputeTrialStress(elastic_strain, sigma);
        SpectralDecomposition(sigma, sigma_projected, eigenvectors);
        ProjectSigma(sigma_projected, sigma_projected, alpha, mtype);
        ReconstructStressTensor(sigma_projected, eigenvectors, sigma);
        
        // Update plastic strain
        ComputeStrain(sigma, elastic_strain);
        PlasticStrain(strain, elastic_strain, plastic_strain);
        
        //Copy to stress vector
        TranslateStress(sigma, aux_tensor);
        
        aux_tensor *= fWeight[ipts];
        glob_sigma.PutSub(3 * ipts, 0, aux_tensor);
        
        if (mat->GetUpdateMem()) {
            //Accumulate to strain vector
            fStrain.AddSub(6 * ipts, 0, strain);
            
            //Copy to plastic strain vector
            fPlasticStrain.PutSub(6 * ipts, 0, plastic_strain);
            
            //Copy to MType and Alpha vectors
            fAlpha(ipts,0) = alpha;
            fMType(ipts,0) = mtype;
        }
        
    }
);
#else
    
    TPZFNMatrix<6,REAL> strain(6, 1, 0.);
    TPZFNMatrix<6,REAL> elastic_strain(6, 1, 0.);
    TPZFNMatrix<6,REAL> sigma(6, 1, 0.);
    TPZFNMatrix<6,REAL> plastic_strain(6, 1, 0.);
    TPZFNMatrix<3,REAL> aux_tensor(3, 1, 0.);
    
    // Lambda for evaluate flux, this is supposed to be implemented in GPU
    auto EvaluateFlux = [this, mat, & glob_delta_strain, & glob_sigma, & strain, & elastic_strain, & plastic_strain, & sigma, & aux_tensor] (int & ipts)
    {
        
    
        REAL alpha;
        int mtype;
        
        //Get from delta strain vector
        glob_delta_strain.GetSub(3 * ipts, 0, 3, 1, aux_tensor);
        
        //Get last strain vector
        fStrain.GetSub(6 * ipts, 0, 6, 1, strain);
        
        // Translate and
        ComposeStrain(aux_tensor, strain);
        
        //Get from plastic strain vector
        fPlasticStrain.GetSub(6 * ipts, 0, 6, 1, plastic_strain);
        
        TPZFMatrix<REAL> eigenvectors(9, 1, 0.);
        TPZFMatrix<REAL> sigma_projected(3, 1, 0.);
        
        // Return Mapping components
        ElasticStrain(plastic_strain, strain, elastic_strain);
        ComputeTrialStress(elastic_strain, sigma);
        SpectralDecomposition(sigma, sigma_projected, eigenvectors);
        ProjectSigma(sigma_projected, sigma_projected, alpha, mtype);
        ReconstructStressTensor(sigma_projected, eigenvectors, sigma);
        
        // Update plastic strain
        ComputeStrain(sigma, elastic_strain);
        PlasticStrain(strain, elastic_strain, plastic_strain);
        
        //Copy to stress vector
        TranslateStress(sigma, aux_tensor);
        
        aux_tensor *= fWeight[ipts];
        glob_sigma.PutSub(3 * ipts, 0, aux_tensor);
        
        if (mat->GetUpdateMem()) {
            
            fSigma.AddSub(6 * ipts, 0, sigma);
            
            //Accumulate to strain vector
            fStrain.AddSub(6 * ipts, 0, strain);
            
            //Copy to plastic strain vector
            fPlasticStrain.PutSub(6 * ipts, 0, plastic_strain);
            
            //Copy to MType and Alpha vectors
            fAlpha(ipts,0) = alpha;
            fMType(ipts,0) = mtype;
        }
        
    };
    
    for (int ipts = 0; ipts < fNpts; ipts++)
    {
        EvaluateFlux(ipts);
    }
#endif
    
    
    if (mat->GetUpdateMem()) {
        
#ifdef USING_TBB
        tbb::parallel_for(size_t(0),size_t(fNpts),size_t(1), [this, mat] (size_t & ipts){
            
            TPZFNMatrix<6,REAL> strain(6, 1, 0.);
            TPZFNMatrix<6,REAL> plastic_strain(6, 1, 0.);
            TPZFNMatrix<6,REAL> sigma(6, 1, 0.);
            
            TPZElastoPlasticMem & memory = mat->MemItem(ipts);
            fSigma.GetSub(6 * ipts, 0, 6, 1, sigma);
            fStrain.GetSub(6 * ipts, 0, 6, 1, strain);
            fPlasticStrain.GetSub(6 * ipts, 0, 6, 1, plastic_strain);
            memory.m_sigma.CopyFrom(sigma);
            memory.m_elastoplastic_state.m_eps_t.CopyFrom(strain);
            memory.m_elastoplastic_state.m_eps_p.CopyFrom(plastic_strain);
            memory.m_elastoplastic_state.m_hardening = fAlpha(ipts,0);
            memory.m_elastoplastic_state.m_m_type = fMType(ipts,0);
            
        }
                          );
#else
        // Lambda expression to update memory
        auto UpdateElastoPlasticState = [this, mat, & strain, & plastic_strain, & sigma] (int ipts)
        {
            
            TPZElastoPlasticMem & memory = mat->MemItem(ipts);
            fSigma.GetSub(6 * ipts, 0, 6, 1, sigma);
            fStrain.GetSub(6 * ipts, 0, 6, 1, strain);
            fPlasticStrain.GetSub(6 * ipts, 0, 6, 1, plastic_strain);
            memory.m_sigma.CopyFrom(sigma);
            memory.m_elastoplastic_state.m_eps_t.CopyFrom(strain);
            memory.m_elastoplastic_state.m_eps_p.CopyFrom(plastic_strain);
            memory.m_elastoplastic_state.m_hardening = fAlpha(ipts,0);
            memory.m_elastoplastic_state.m_m_type = fMType(ipts,0);
            
        };
        
        for (int ipts = 0; ipts < fNpts; ipts++)
        {
            UpdateElastoPlasticState(ipts);
            
        }
        
#endif
    }

}

void TPZConstitutiveLawProcessor::ComposeStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> & strain){
    
    int dim = 2;
    if (dim == 2) {
        	strain(_XX_,0) += delta_strain(0,0);
            strain(_XY_,0) += delta_strain(1,0);
            strain(_YY_,0) += delta_strain(2,0);
    }else{
        strain += delta_strain;
    }
    
}

void TPZConstitutiveLawProcessor::TranslateStress(TPZFMatrix<REAL> &full_stress, TPZFMatrix<REAL> &stress){
    
    int dim = 2;
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



    fCudaCalls->ComputeSigma(fNpts, delta_strain.getData(), sigma.getData(), lambda, mu, mc_phi, mc_psi, mc_cohesion, dPlasticStrain.getData(),  dMType.getData(), dAlpha.getData(), dWeight.getData());

}
#endif

#ifdef USING_CUDA
void TPZConstitutiveLawProcessor::TransferDataToGPU() {
    dWeight.resize(fWeight.size());
    dWeight.set(&fWeight[0], fWeight.size());
}
#endif
