//
// Created by natalia on 05/09/2019.
//

#include "TPZPlasticStepPV.h"
#include "TPZYCMohrCoulombPV.h"
#include "TPBrIntPointsStructMatrix_impl.h"


template class TPBrIntPointsStructMatrix<TPZPlasticStepPV<TPZYCMohrCoulombPV,TPZElasticResponse>, TPZElastoPlasticMem >;
