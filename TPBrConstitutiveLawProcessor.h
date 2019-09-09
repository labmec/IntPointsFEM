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

    void SetUpDataByIntPoints(int npts);

    void ComputeSigma(TPZFMatrix<REAL> & glob_delta_strain, TPZFMatrix<REAL> & glob_sigma);
    
    void ComputeSigma(TPZFMatrix<REAL> & glob_delta_strain, TPZFMatrix<REAL> & glob_sigma, TPZFMatrix<REAL> & glob_Dep);

    void SetWeightVector(TPZVec<REAL> &weight) {
        fWeight = weight;
    }

    TPZVec<REAL> &WeightVector() {
        return fWeight;
    }

    void SetNumIntPoints(int npts) {
        fNpts = npts;
    }

    int NumIntPoints() {
        return fNpts;
    }

    void SetStateVector(TPZVec<TPZPlasticState<STATE>> & state) {
        fStateVec = state;
    }

    TPZVec<TPZPlasticState<STATE>> & StateVector() {
        return fStateVec;
    }

    void SetPlasticModel(T & model) {
        for (int i = 0; i < fNpts; ++i) {
            fPlasticModel[i] = model;
        }
    }

    TPZVec<T> & GetPlasticModel() {
        return fPlasticModel;
    }

    void SetMemory(TPZMatWithMem<MEM> * memory) {
        fMatMem = memory;
    }

    TPZMatWithMem<MEM> * GetMemory() {
        return fMatMem;
    }

private:

    TPZVec<REAL> fWeight;

    int64_t fNpts;

    TPZVec<TPZPlasticState<STATE>> fStateVec;

    TPZVec<T> fPlasticModel;

    TPZMatWithMem<MEM> *fMatMem;
};
#endif /* TPBrConstitutiveLawProcessor_h */
