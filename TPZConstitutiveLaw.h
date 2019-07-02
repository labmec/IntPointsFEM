//
//  TPZConstitutiveLaw.h
//  IntPointsFEM
//
//  Created by Omar Durán on 6/20/19.
//

#ifndef TPZConstitutiveLaw_h
#define TPZConstitutiveLaw_h

#include <stdio.h>
#include "TPZMatWithMem.h"
#include "TPZElastoPlasticMem.h"
#include "pzfmatrix.h"

template <class TPZElastoPlasticMem>
class TPZConstitutiveLaw {
    
public:
    
    /// Evaluates flux from potential gradient
    static void ComputeFlux(TPZMatWithMem<TPZElastoPlasticMem> * mat, int int_point_index, TPZFMatrix<REAL> & delta_eps, TPZFMatrix<REAL> & sigma);
    
    /// Evaluates flux from potential gradient
    static void ComputeFluxAndTanget(TPZMatWithMem<TPZElastoPlasticMem> * mat, int int_point_index, TPZFMatrix<REAL> & delta_eps, TPZFMatrix<REAL> & sigma, TPZFMatrix<STATE> & dep);
    
    static void  ElasticStrain(TPZFMatrix<REAL> &plastic_strain, TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &elastic_strain);
    
    static void  ComputeTrialStress(TPZFMatrix<REAL> &elastic_strain, TPZFMatrix<REAL> &sigma_trial);
    
    static void  SpectralDecomposition(TPZFMatrix<REAL> &sigma_trial, TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &eigenvectors);
    
    static void  ProjectSigma(TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &sigma_projected, REAL &alpha, int &mtype);
    
    static void  ReconstructStressTensor(TPZFMatrix<REAL> &sigma_projected, TPZFMatrix<REAL> &eigenvectors, TPZFMatrix<REAL> &sigma);
    
    static void  ComputeStrain(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &elastic_strain);
    
    static void  PlasticStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &elastic_strain, TPZFMatrix<REAL> &plastic_strain);
    
    static void  De(TPZFMatrix<REAL> & De, TPZElasticResponse & ER);
    
    
};

#endif /* TPZConstitutiveLaw_h */
