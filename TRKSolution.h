//
// Created by natalia on 28/03/19.
//

#ifndef INTEGRATIONPOINTEXPERIMENTS_TRKSOLUTION_H
#define INTEGRATIONPOINTEXPERIMENTS_TRKSOLUTION_H

#include "TElastoPlasticData.h"
#include "TPZYCMohrCoulombPV.h"
#include "TPZPlasticStepPV.h"
#include "TPZElastoPlasticMem.h"
#include "TPZMatElastoPlastic2D.h"


class TRKSolution {
protected:
    TPZMatElastoPlastic2D < TPZPlasticStepPV<TPZYCMohrCoulombPV, TPZElasticResponse>, TPZElastoPlasticMem > *m_material;

    REAL m_re;

    REAL m_rw;

    TPZTensor<REAL> m_sigma;

    REAL m_pw;

    REAL m_sigma0;

    REAL m_theta;

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

    void SetExternRadius(REAL re);

    REAL ExternRadius();

    void SetWellboreRadius(REAL rw);

    REAL WellboreRadius();

    void SetStressXYZ(TPZTensor<REAL> &sigma, REAL m_theta);

    TPZTensor<REAL> StressXYZ();

    REAL Theta();

    void SetWellborePressure(REAL pw);

    REAL WellborePressure();

    void SetInitialStress(REAL sigma0);

    REAL InitialStress();

    void CreateMaterial();

    void F (REAL r, REAL ur, REAL sigma_r, REAL &d_ur, REAL &d_sigmar);

    void ParametersAtRe (TPZFNMatrix<3,REAL> &sigma, REAL &u_re);

    void RKProcess(int np, std::ostream &out, bool euler);








};


#endif //INTEGRATIONPOINTEXPERIMENTS_TRKSOLUTION_H
