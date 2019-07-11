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

    void SetUpDataByIntPoints(int64_t npts);

    void SetWeightVector(TPZVec<REAL> &weight);

    TPZVec<REAL> &WeightVector() {
        return fWeight;
    }

#ifdef USING_CUDA
    TPZVecGPU<REAL> &WeightVectorDev() {
        return dWeight;
    }
#endif

    void SetMaterial(TPZMaterial *material);

    void ElasticStrain(TPZFMatrix<REAL> &plastic_strain, TPZFMatrix<REAL> & strain ,TPZFMatrix<REAL> &elastic_strain);

    void ComposeStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &full_delta_strain);
    
    void TranslateStress(TPZFMatrix<REAL> &full_stress, TPZFMatrix<REAL> &stress);
    
    void ComputeTrialStress(TPZFMatrix<REAL> &elastic_strain, TPZFMatrix<REAL> &sigma_trial);

    void SpectralDecomposition(TPZFMatrix<REAL> &sigma_trial, TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &eigenvectors);

    void ProjectSigma(TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &sigma_projected, REAL &alpha, int &mtype);

    void ReconstructStressTensor(TPZFMatrix<REAL> &sigma_projected, TPZFMatrix<REAL> &eigenvectors, TPZFMatrix<REAL> &sigma);

    void ComputeStrain(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &elastic_strain);

    void PlasticStrain(TPZFMatrix<REAL> &strain, TPZFMatrix<REAL> &elastic_strain, TPZFMatrix<REAL> &plastic_strain);

    void ComputeSigma(TPZFMatrix<REAL> & glob_delta_strain, TPZFMatrix<REAL> & glob_sigma);
    
    void De(TPZFMatrix<REAL> & De);

#ifdef USING_CUDA
    void De();
    
    void ComputeSigma(TPZVecGPU<REAL> &delta_strain, TPZVecGPU<REAL> &sigma);

    void TransferDataToGPU();
#endif

    TPZVec<REAL> fWeight;
    
private:
    
    int64_t fNpts;

    TPZMatElastoPlastic2D<TPZPlasticStepPV<TPZYCMohrCoulombPV,TPZElasticResponse>, TPZElastoPlasticMem> *fMaterial;

    TPZFMatrix<REAL> fSigma;
    
    TPZFMatrix<REAL> fStrain;
    
    TPZFMatrix<REAL> fPlasticStrain;

    TPZFMatrix<REAL> fMType;

    TPZFMatrix<REAL> fAlpha;

#ifdef USING_CUDA
    TPZCudaCalls *fCudaCalls;

    TPZVecGPU<REAL> dWeight;

    TPZVecGPU<REAL> dSigma;

    TPZVecGPU<REAL> dStrain;

    TPZVecGPU<REAL> dPlasticStrain;

    TPZVecGPU<REAL> dMType;
    
    TPZVecGPU<REAL> dAlpha;
#endif

};


#endif /* TPZConstitutiveLawProcessor_h */
