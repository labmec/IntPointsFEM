//
//  TPBrConstitutiveLawProcessor.hpp
//  IntPointsFEM
//
//  Created by Omar Durán on 9/5/19.
//

#ifndef TPBrConstitutiveLawProcessor_h
#define TPBrConstitutiveLawProcessor_h

#include "TPZPlasticStepPV.h"
#include "TPZYCMohrCoulombPV.h"
#include "TPZElastoPlasticMem.h"
#include "TPZMatElastoPlastic2D.h"
#include "TPZPlasticState.h"

template <class T, class MEM = TPZElastoPlasticMem>
class TPBrConstitutiveLawProcessor {

public:

    TPBrConstitutiveLawProcessor();

    TPBrConstitutiveLawProcessor(int npts, TPZVec <REAL> & weight);

    ~TPBrConstitutiveLawProcessor();

    TPBrConstitutiveLawProcessor(const TPBrConstitutiveLawProcessor &copy);

    TPBrConstitutiveLawProcessor &operator=(const TPBrConstitutiveLawProcessor &copy);

    void SetWeightVector(TPZVec<REAL> &weight);

    void SetUpDataByIntPoints(int npts);

    TPZVec<REAL> &WeightVector();

    void ComputeSigma(TPZFMatrix<REAL> & glob_delta_strain, TPZFMatrix<REAL> & glob_sigma);

    T & GetPlasticModel() {
        return fPlasticModel;
    }

    void SetPlasticModel(T & model) {
        fPlasticModel = model;
    }

private:

    TPZVec<REAL> fWeight;

    int64_t fNpts;

    TPZVec<TPZPlasticState<STATE>> fStateVec;

    T fPlasticModel;
};
#endif /* TPBrConstitutiveLawProcessor_h */
