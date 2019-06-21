//
//  TPZConstitutiveLaw.cpp
//  IntPointsFEM
//
//  Created by Omar Durán on 6/20/19.
//

#include "TPZConstitutiveLaw.h"

template <class TMEM>
void TPZConstitutiveLaw<TMEM>::ComputeFlux(TPZMatWithMem<TMEM> * mat, int int_point_index, const TPZVec<STATE> & delta_epsilon, TPZVec<STATE> & sigma){

}

template <class TMEM>
void TPZConstitutiveLaw<TMEM>::ComputeFluxAndTanget(TPZMatWithMem<TMEM> * mat, int int_point_index, const TPZVec<STATE> & delta_epsilon, TPZVec<STATE> & sigma, TPZFMatrix<STATE> & dep){
    
    
}
