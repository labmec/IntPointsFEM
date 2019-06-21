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
#include "pzfmatrix.h"

template <class TMEM>
class TPZConstitutiveLaw {
    
public:
    
    /// Evaluates flux from potential gradient
    static void ComputeFlux(TPZMatWithMem<TMEM> * mat, int int_point_index, const TPZVec<STATE> & delta_epsilon, TPZVec<STATE> & sigma);
    
    /// Evaluates flux from potential gradient
    static void ComputeFluxAndTanget(TPZMatWithMem<TMEM> * mat, int int_point_index, const TPZVec<STATE> & delta_epsilon, TPZVec<STATE> & sigma, TPZFMatrix<STATE> & dep);
    
};

#endif /* TPZConstitutiveLaw_h */
