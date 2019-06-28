//
// Created by natalia on 17/05/19.
//
#include "TPZPlasticStepPV.h"
#include "TPZYCMohrCoulombPV.h"
#include "TPZElastoPlasticMem.h"
#include "TPZMatElastoPlastic2D.h"

#ifdef USING_CUDA
#include "TPZVecGPU.h"
#include "TPZCudaCalls.h"
#endif

#ifndef TPZConstitutiveLawProcessor_h
#define TPZConstitutiveLawProcessor_h

class TPZConstitutiveLawProcessor {

public:
    
    TPZConstitutiveLawProcessor();

    TPZConstitutiveLawProcessor(int npts, TPZVec<REAL> weight, TPZMaterial *material);

    ~TPZConstitutiveLawProcessor();

    TPZConstitutiveLawProcessor(const TPZConstitutiveLawProcessor &copy);

    TPZConstitutiveLawProcessor &operator=(const TPZConstitutiveLawProcessor &copy);

    void SetIntPoints(int64_t npts);

    void SetWeightVector(TPZVec<REAL> weight);

    void SetMaterial(TPZMaterial *material);

    void ElasticStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &elastic_strain);

    void TranslateStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &full_delta_strain);
    
    void TranslateStress(TPZFMatrix<REAL> &full_stress, TPZFMatrix<REAL> &stress);
    
    void ComputeTrialStress(TPZFMatrix<REAL> &elastic_strain, TPZFMatrix<REAL> &sigma_trial);

    void SpectralDecomposition(TPZFMatrix<REAL> &sigma_trial, TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &eigenvectors);

    void ProjectSigma(TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &sigma_projected);

    void StressCompleteTensor(TPZFMatrix<REAL> &sigma_projected, TPZFMatrix<REAL> &eigenvectors, TPZFMatrix<REAL> &sigma);

    void ComputeStrain(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &elastic_strain);

    void PlasticStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &elastic_strainS);

    void ComputeSigma(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &sigma);
    
    void De(TPZFMatrix<REAL> & De);

#ifdef USING_CUDA
    void ComputeSigma(TPZVecGPU<REAL> &delta_strain, TPZVecGPU<REAL> &sigma);
#endif

    TPZVec<REAL> fWeight;
    
private:
    
    int64_t fNpts;


    TPZMatElastoPlastic2D<TPZPlasticStepPV<TPZYCMohrCoulombPV,TPZElasticResponse>, TPZElastoPlasticMem> *fMaterial;

    TPZFMatrix<REAL> fPlasticStrain;

    TPZFMatrix<REAL> fMType;

    TPZFMatrix<REAL> fAlpha;

#ifdef USING_CUDA
    TPZCudaCalls *fCudaCalls;
    TPZVecGPU<REAL> dWeight;
#endif

};


#endif /* TPZConstitutiveLawProcessor_h */
