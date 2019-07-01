//
//  TPZConstitutiveLaw.cpp
//  IntPointsFEM
//
//  Created by Omar Durán on 6/20/19.
//

#include "TPZConstitutiveLaw.h"

template <class TPZElastoPlasticMem>
void TPZConstitutiveLaw<TPZElastoPlasticMem>::ComputeFlux(TPZMatWithMem<TPZElastoPlasticMem> * mat, int int_point_index, const TPZVec<STATE> & delta_epsilon, TPZVec<STATE> & sigma){

    
    
    
}

template <class TPZElastoPlasticMem>
void TPZConstitutiveLaw<TPZElastoPlasticMem>::ComputeFluxAndTanget(TPZMatWithMem<TPZElastoPlasticMem> * mat, int int_point_index, const TPZVec<STATE> & delta_epsilon, TPZVec<STATE> & sigma, TPZFMatrix<STATE> & dep){
    
    
    
}
