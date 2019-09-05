//
//  TPBrConstitutiveLawProcessor.cpp
//  IntPointsFEM
//
//  Created by Omar Durán on 9/5/19.
//

#include "TPBrConstitutiveLawProcessor_impl.h"

template class TPBrConstitutiveLawProcessor<TPZPlasticStepPV<TPZYCMohrCoulombPV,TPZElasticResponse>, TPZElastoPlasticMem >;
