//
// Created by natalia on 05/09/2019.
//

#include "TPZPlasticStepPV.h"
#include "TPZYCMohrCoulombPV.h"
#include "TPZIntPointsStructMatrix_impl.h"

//template class TPZIntPointsStructMatrix<TPZPlasticStepPV<TPZYCMohrCoulombPV,TPZElasticResponse>>;

template class TPZIntPointsStructMatrix<TPZMatElastoPlastic2D<TPZPlasticStepPV<TPZYCMohrCoulombPV,TPZElasticResponse>, TPZElastoPlasticMem>>;