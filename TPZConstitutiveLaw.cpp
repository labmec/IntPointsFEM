//
//  TPZConstitutiveLaw.cpp
//  IntPointsFEM
//
//  Created by Omar Durán on 6/20/19.
//

#include "TPZConstitutiveLaw.h"

template <class TPZElastoPlasticMem>
void TPZConstitutiveLaw<TPZElastoPlasticMem>::ComputeFlux(TPZMatWithMem<TPZElastoPlasticMem> * mat, int int_point_index, TPZFMatrix<REAL> & delta_eps, TPZFMatrix<REAL> & sigma){


    
}

template <class TPZElastoPlasticMem>
void TPZConstitutiveLaw<TPZElastoPlasticMem>::ComputeFluxAndTanget(TPZMatWithMem<TPZElastoPlasticMem> * mat, int int_point_index, TPZFMatrix<REAL> & delta_eps, TPZFMatrix<REAL> & sigma, TPZFMatrix<STATE> & dep){
    
    DebugStop();
    
}

template <class TPZElastoPlasticMem>
void TPZConstitutiveLaw<TPZElastoPlasticMem>::ElasticStrain(TPZFMatrix<REAL> &plastic_strain, TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &elastic_strain){
    elastic_strain = delta_strain - plastic_strain;
}

template <class TPZElastoPlasticMem>
void TPZConstitutiveLaw<TPZElastoPlasticMem>::ComputeTrialStress(TPZFMatrix<REAL> &elastic_strain, TPZFMatrix<REAL> &sigma_trial){
    
}

template <class TPZElastoPlasticMem>
void TPZConstitutiveLaw<TPZElastoPlasticMem>::SpectralDecomposition(TPZFMatrix<REAL> &sigma_trial, TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &eigenvectors){
    
}

template <class TPZElastoPlasticMem>
void TPZConstitutiveLaw<TPZElastoPlasticMem>::ProjectSigma(TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &sigma_projected, REAL &alpha, int &mtype){
    
}

template <class TPZElastoPlasticMem>
void TPZConstitutiveLaw<TPZElastoPlasticMem>::ReconstructStressTensor(TPZFMatrix<REAL> &sigma_projected, TPZFMatrix<REAL> &eigenvectors, TPZFMatrix<REAL> &sigma){
    
}

template <class TPZElastoPlasticMem>
void TPZConstitutiveLaw<TPZElastoPlasticMem>::ComputeStrain(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &elastic_strain){
    
}

template <class TPZElastoPlasticMem>
void TPZConstitutiveLaw<TPZElastoPlasticMem>::PlasticStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &elastic_strain, TPZFMatrix<REAL> &plastic_strain){
    
}

template <class TPZElastoPlasticMem>
void TPZConstitutiveLaw<TPZElastoPlasticMem>::De(TPZFMatrix<REAL> & De, TPZElasticResponse & ER){
    
    REAL lambda = ER.Lambda();
    REAL mu = ER.G();
    
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
