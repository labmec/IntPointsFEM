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

    ~TPZConstitutiveLawProcessor();

    TPZConstitutiveLawProcessor(const TPZConstitutiveLawProcessor &copy);

    TPZConstitutiveLawProcessor &operator=(const TPZConstitutiveLawProcessor &copy);

    void SetUpDataByIntPoints(int64_t npts);

    void SetWeightVector(TPZVec<REAL> &weight);

    void SetMaterial(TPZMaterial *material);

    void ComputeSigma(TPZFMatrix<REAL> & glob_delta_strain, TPZFMatrix<REAL> & glob_sigma);

    void ComputeSigmaDep(TPZFMatrix<REAL> & glob_delta_strain, TPZFMatrix<REAL> & glob_sigma, TPZFMatrix<REAL> & glob_dep);

#ifdef USING_CUDA
    void ComputeSigma(TPZVecGPU<REAL> &delta_strain, TPZVecGPU<REAL> &sigma);

    void ComputeSigmaDep(TPZVecGPU<REAL> &delta_strain, TPZVecGPU<REAL> &sigma, TPZVecGPU<REAL> &glob_dep);

    void TransferDataToGPU();
#endif

private:
    
    int64_t fNpts;

    TPZMatElastoPlastic2D<TPZPlasticStepPV<TPZYCMohrCoulombPV,TPZElasticResponse>, TPZElastoPlasticMem> *fMaterial;

    TPZVec<REAL> fWeight;

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
