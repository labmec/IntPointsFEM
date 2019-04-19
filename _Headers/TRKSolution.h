//
// Created by natalia on 28/03/19.
//

#ifndef TRKSolution_h
#define TRKSolution_h

#include "TElastoPlasticData.h"
#include "TPZYCMohrCoulombPV.h"
#include "TPZPlasticStepPV.h"
#include "TPZElastoPlasticMem.h"
#include "TPZMatElastoPlastic2D.h"
#include "TPZTensor.h"


class TRKSolution {
    
protected:
    
    TPZMatElastoPlastic2D < TPZPlasticStepPV<TPZYCMohrCoulombPV, TPZElasticResponse>, TPZElastoPlasticMem > *m_material;
    
    TPZPlasticStepPV<TPZYCMohrCoulombPV, TPZElasticResponse> m_elastoplastic_model;

    REAL m_re = -1;

    REAL m_rw = -1;

    TPZTensor<REAL> m_sigma;

    REAL m_pw = -1;

    REAL m_sigma0 = -1;

    REAL m_theta = -1;

    int m_n_points = 0;
    
    TPZManVector<TPZElastoPlasticMem,10> m_memory_vector;

public:
    
    /// Default constructor
    TRKSolution();

    /// Copy constructor
    TRKSolution(const TRKSolution &  other);

    /// Assignmet constructor
    TRKSolution & operator=(const TRKSolution &  other);

    /// Default destructor
    ~TRKSolution();

    void SetMaterial(TPZMatElastoPlastic2D < TPZPlasticStepPV<TPZYCMohrCoulombPV, TPZElasticResponse>, TPZElastoPlasticMem > *material);

    TPZMatElastoPlastic2D < TPZPlasticStepPV<TPZYCMohrCoulombPV, TPZElasticResponse>, TPZElastoPlasticMem > * Material();
    
   void SetElastoPlasticModel(TPZPlasticStepPV<TPZYCMohrCoulombPV, TPZElasticResponse> & model);

    void SetExternalRadius(REAL re);

    REAL ExternalRadius();

    void SetWellboreRadius(REAL rw);

    REAL WellboreRadius();

    void SetStressXYZ(TPZTensor<REAL> &sigma, REAL m_theta);

    TPZTensor<REAL> StressXYZ();

    REAL Theta();

    void SetWellborePressure(REAL pw);

    REAL WellborePressure();

    void SetInitialStress(REAL sigma0);

    REAL InitialStress();
    
    void SetNumberOfPoints(int n_points);
    
    int GetNumberOfPoints();

    void CreateMaterial();
    
    void FillPointsMemory();

    void F (REAL r, REAL ur, REAL sigma_r, REAL &d_ur, REAL &d_sigmar, REAL & lambda, REAL & G);
    
    void F_II (REAL r, REAL ur, REAL sigma_r, REAL &d_ur, REAL &d_sigmar, REAL & lambda, REAL & G);

    void ParametersAtRe (TPZFNMatrix<3,REAL> &sigma, REAL &u_re);

    void RKProcess(std::ostream &out, bool euler);
    
    void RKProcessII(std::ostream &out, bool euler);
    
    void ReconstructEpsilon(TPZTensor<REAL> & sigma, TPZTensor<REAL> & eps_t, REAL & lambda, REAL & G);

};


#endif //TRKSolution_h
