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
    static void ComputeFlux(TPZMatWithMem<TPZElastoPlasticMem> * mat, int int_point_index, const TPZVec<STATE> & delta_epsilon, TPZVec<STATE> & sigma);
    
    /// Evaluates flux from potential gradient
    static void ComputeFluxAndTanget(TPZMatWithMem<TPZElastoPlasticMem> * mat, int int_point_index, const TPZVec<STATE> & delta_epsilon, TPZVec<STATE> & sigma, TPZFMatrix<STATE> & dep);
    
};

#endif /* TPZConstitutiveLaw_h */
