//
//  TPBrConstitutiveLawProcessor.hpp
//  IntPointsFEM
//
//  Created by Omar Durán on 9/5/19.
//
#include "TPBrConstitutiveLawProcessor.h"
#include "TPZMatWithMem.h"

#ifdef USING_TBB
#include "tbb/parallel_for.h"
#endif

template<class T, class MEM>
TPBrConstitutiveLawProcessor<T, MEM>::TPBrConstitutiveLawProcessor() : fPlasticModel(0), fMatMem() {
    fNpts = -1;
    fWeight.Resize(0);
    fStateVec.resize(0);
}

template<class T, class MEM>
TPBrConstitutiveLawProcessor<T, MEM>::TPBrConstitutiveLawProcessor(int npts, TPZVec<REAL> &weight) {
    SetUpDataByIntPoints(npts);
    SetWeightVector(weight);
}

template<class T, class MEM>
TPBrConstitutiveLawProcessor<T, MEM>::~TPBrConstitutiveLawProcessor() {

}

template<class T, class MEM>
TPBrConstitutiveLawProcessor<T, MEM>::TPBrConstitutiveLawProcessor(const TPBrConstitutiveLawProcessor &copy) {
    fNpts = copy.fNpts;
    fWeight = copy.fWeight;
    fPlasticModel = copy.fPlasticModel;
    fStateVec = copy.fStateVec;
    fMatMem = copy.fMatMem;
}

template<class T, class MEM>
TPBrConstitutiveLawProcessor<T, MEM> &
TPBrConstitutiveLawProcessor<T, MEM>::operator=(const TPBrConstitutiveLawProcessor &copy) {
    if (&copy == this) {
        return *this;
    }

    fNpts = copy.fNpts;
    fWeight = copy.fWeight;
    fPlasticModel = copy.fPlasticModel;
    fStateVec = copy.fStateVec;
    fMatMem = copy.fMatMem;

    return *this;
}

template<class T, class MEM>
void TPBrConstitutiveLawProcessor<T, MEM>::SetUpDataByIntPoints(int npts) {
    fNpts = npts;
    fStateVec.resize(fNpts);
    fPlasticModel.resize(fNpts);
}

template<class T, class MEM>
void
TPBrConstitutiveLawProcessor<T, MEM>::ComputeSigma(TPZFMatrix<REAL> &glob_delta_strain, TPZFMatrix<REAL> &glob_sigma) {

    glob_sigma.Resize(3 * fNpts, 1);

#ifdef USING_TBB
    tbb::parallel_for(size_t(0),size_t(fNpts),size_t(1),[&](size_t ipts)
#else
    for (int ipts = 0; ipts < fNpts; ipts++)
#endif
    {
        TPZTensor<REAL> epsTotal;
        TPZTensor<REAL> sigma;

        epsTotal[_XX_] = glob_delta_strain(3 * ipts + 0, 0);
        epsTotal[_XY_] = glob_delta_strain(3 * ipts + 1, 0)/2;
        epsTotal[_XZ_] = 0;
        epsTotal[_YY_] = glob_delta_strain(3 * ipts + 2, 0);
        epsTotal[_YZ_] = 0;
        epsTotal[_ZZ_] = 0;

        fPlasticModel[ipts].SetState(fStateVec[ipts]);

        epsTotal.Add(fStateVec[ipts].m_eps_t, 1.);

        fPlasticModel[ipts].ApplyStrainComputeSigma(epsTotal, sigma);

        glob_sigma(3 * ipts + 0, 0) = sigma[_XX_]*fWeight[ipts];
        glob_sigma(3 * ipts + 1, 0) = sigma[_XY_]*fWeight[ipts];
        glob_sigma(3 * ipts + 2, 0) = sigma[_YY_]*fWeight[ipts];

        if (fMatMem->GetUpdateMem()) {
            fMatMem->MemItem(ipts).m_sigma        = sigma;
            fMatMem->MemItem(ipts).m_elastoplastic_state = fPlasticModel[ipts].GetState();
            fMatMem->MemItem(ipts).m_plastic_steps = fPlasticModel[ipts].IntegrationSteps();
            fMatMem->MemItem(ipts).m_ER = fPlasticModel[ipts].GetElasticResponse();
        }
    }
#ifdef USING_TBB
    );
#endif
}


template<class T, class MEM>
void
TPBrConstitutiveLawProcessor<T, MEM>::ComputeSigma(TPZFMatrix<REAL> & glob_delta_strain, TPZFMatrix<REAL> & glob_sigma, TPZFMatrix<REAL> & glob_Dep){
    
    glob_sigma.Resize(3 * fNpts, 1);
    glob_Dep.Resize(3 * 3 * fNpts, 1);
    
#ifdef USING_TBB
    tbb::parallel_for(size_t(0),size_t(fNpts),size_t(1),[&](size_t ipts)
#else
                      for (int ipts = 0; ipts < fNpts; ipts++)
#endif
                      {
                          TPZTensor<REAL> epsTotal;
                          TPZTensor<REAL> sigma;
                          
                          epsTotal[_XX_] = glob_delta_strain(3 * ipts + 0, 0);
                          epsTotal[_XY_] = glob_delta_strain(3 * ipts + 1, 0)/2;
                          epsTotal[_XZ_] = 0;
                          epsTotal[_YY_] = glob_delta_strain(3 * ipts + 2, 0);
                          epsTotal[_YZ_] = 0;
                          epsTotal[_ZZ_] = 0;
                          
                          fPlasticModel[ipts].SetState(fStateVec[ipts]);
                          
                          epsTotal.Add(fStateVec[ipts].m_eps_t, 1.);
                          
                          TPZFNMatrix<36,REAL> De_full(6,6);
                          De_full.Zero();
                          fPlasticModel[ipts].ApplyStrainComputeSigma(epsTotal, sigma, &De_full);
                          
                          glob_Dep(3 * 3 * ipts + 0, 0) = De_full(_XX_,_XX_);
                          glob_Dep(3 * 3 * ipts + 1, 0) = De_full(_XX_,_XY_)/2;
                          glob_Dep(3 * 3 * ipts + 2, 0) = De_full(_XX_,_YY_);
                  
                          glob_Dep(3 * 3 * ipts + 3, 0) = De_full(_XY_,_XX_);
                          glob_Dep(3 * 3 * ipts + 4, 0) = De_full(_XY_,_XY_)/2;
                          glob_Dep(3 * 3 * ipts + 5, 0) = De_full(_XY_,_YY_);
                  
                          glob_Dep(3 * 3 * ipts + 6, 0) = De_full(_YY_,_XX_);
                          glob_Dep(3 * 3 * ipts + 7, 0) = De_full(_YY_,_XY_)/2;
                          glob_Dep(3 * 3 * ipts + 8, 0) = De_full(_YY_,_YY_);
                          
                          glob_sigma(3 * ipts + 0, 0) = sigma[_XX_]*fWeight[ipts];
                          glob_sigma(3 * ipts + 1, 0) = sigma[_XY_]*fWeight[ipts];
                          glob_sigma(3 * ipts + 2, 0) = sigma[_YY_]*fWeight[ipts];
                          
                          if (fMatMem->GetUpdateMem()) {
                              fMatMem->MemItem(ipts).m_sigma        = sigma;
                              fMatMem->MemItem(ipts).m_elastoplastic_state = fPlasticModel[ipts].GetState();
                              fMatMem->MemItem(ipts).m_plastic_steps = fPlasticModel[ipts].IntegrationSteps();
                              fMatMem->MemItem(ipts).m_ER = fPlasticModel[ipts].GetElasticResponse();
                          }
                      }
#ifdef USING_TBB
                      );
#endif
    
}
