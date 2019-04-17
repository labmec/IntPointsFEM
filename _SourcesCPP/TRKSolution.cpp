//
// Created by natalia on 28/03/19.
//

#include "TRKSolution.h"

TRKSolution::TRKSolution() {

}

TRKSolution::TRKSolution(const TRKSolution &  other) {
    m_material = other.m_material;
    m_re = other.m_re;
    m_rw = other.m_rw;
    m_sigma = other.m_sigma;
    m_pw = other.m_pw;
    m_sigma0 = other.m_sigma0;
    m_theta = other.m_theta;
    m_n_points = other.m_n_points;
    m_memory_vector.resize(0);
}

TRKSolution & TRKSolution::operator=(const TRKSolution &  other) {
    /// check for self-assignment
    if(&other == this){
        return *this;
    }
    m_material = other.m_material;
    m_re = other.m_re;
    m_rw = other.m_rw;
    m_sigma = other.m_sigma;
    m_pw = other.m_pw;
    m_sigma0 = other.m_sigma0;
    m_theta = other.m_theta;
    m_n_points = other.m_n_points;
    m_memory_vector = other.m_memory_vector;

    return *this;
}

TRKSolution::~TRKSolution() {

}

void TRKSolution::SetMaterial(TPZMatElastoPlastic2D < TPZPlasticStepPV<TPZYCMohrCoulombPV, TPZElasticResponse>, TPZElastoPlasticMem > *material) {
    m_material = material;
}

TPZMatElastoPlastic2D < TPZPlasticStepPV<TPZYCMohrCoulombPV, TPZElasticResponse>, TPZElastoPlasticMem > * TRKSolution::Material() {
    return m_material;
}

void TRKSolution::SetElastoPlasticModel(TPZPlasticStepPV<TPZYCMohrCoulombPV, TPZElasticResponse> & model){
    m_elastoplastic_model = model;
}

void TRKSolution::SetExternalRadius(REAL re) {
    m_re = re;
}

REAL TRKSolution::ExternalRadius() {
    return m_re;
}

void TRKSolution::SetWellboreRadius(REAL rw) {
    m_rw = rw;
}

REAL TRKSolution::WellboreRadius() {
    return m_rw;
}

void TRKSolution::SetStressXYZ(TPZTensor<REAL> &sigma, REAL theta) {
    m_sigma = sigma;
    m_theta = theta;
}

TPZTensor<REAL> TRKSolution::StressXYZ() {
    return m_sigma;
}

REAL TRKSolution::Theta() {
    return m_theta;
}

void TRKSolution::SetWellborePressure(REAL pw) {
    m_pw = pw;
}

REAL TRKSolution::WellborePressure() {
    return m_pw;
}

void TRKSolution::SetInitialStress(REAL sigma0) {
    m_sigma0 = sigma0;
}

REAL TRKSolution::InitialStress() {
    return m_sigma0;
}

void TRKSolution::SetNumberOfPoints(int n_points) {
    m_n_points = n_points;
}

int TRKSolution::GetNumberOfPoints() {
    return m_n_points;
}

void TRKSolution::FillPointsMemory(){
    m_memory_vector.Resize(m_n_points+1);
    for (auto & item : m_memory_vector) {
        item = m_material->GetDefaultMemory();
    }
}

void TRKSolution::F (REAL r, REAL ur, REAL sigma_r, REAL &d_ur, REAL &d_sigmar, REAL & lambda, REAL & G) {
    
    d_ur = (r*sigma_r-lambda*ur)/(r*lambda+2*G*r);
    d_sigmar = (-sigma_r + (2*G*ur/r + lambda*(ur/r + (r*sigma_r-lambda*ur)/(r*(lambda + 2*G)))))/r;
}

void TRKSolution::ParametersAtRe(TPZFNMatrix<3,REAL> &sigma, REAL &u_re) {
    
    int p_index = 0; /// Parameters are Re are associated with index 0
    REAL nu = m_memory_vector[p_index].m_ER.Poisson();
    REAL G = m_memory_vector[p_index].m_ER.G();

    sigma.Resize(1,3);

    // Stress at re
    sigma(0,0) = (m_sigma.XX() + m_sigma.YY()) / 2 * (1 - pow(m_rw / m_re, 2)) + (1 - 4 * pow(m_rw / m_re, 2) + 3 * pow(m_rw / m_re, 4)) * (m_sigma.XX() - m_sigma.YY()) / 2 * cos(2 * m_theta) +
                        m_sigma.XY() * (1 - 4 * pow(m_rw / m_re, 2) + 3 * pow(m_rw / m_re, 4)) * sin(2 * m_theta) + m_pw * pow(m_rw / m_re, 2) - m_sigma0;
    sigma(0,1) = (m_sigma.XX() + m_sigma.YY()) / 2 * (1 + pow(m_rw / m_re, 2)) - (1 + 3 * pow(m_rw / m_re, 4)) * (m_sigma.XX() - m_sigma.YY()) / 2 * cos(2 * m_theta) -
                        m_sigma.XY() * (1 + 3 * pow(m_rw / m_re, 4)) * sin(2 * m_theta) - m_pw * pow(m_rw / m_re, 2) - m_sigma0;
    sigma(0,2) = m_sigma.ZZ() - nu * (2 * (m_sigma.XX() - m_sigma.YY()) * (m_rw / m_re) * (m_rw / m_re) * cos(2 * m_theta) + 4 * m_sigma.XY() * (m_rw / m_re) * (m_rw / m_re) * sin(2 * m_theta)) - m_sigma0;

    // Displacement at re
    u_re = -(m_re * sigma(0,2) - m_re * sigma(0,1)) / (2 * G);
    
    
}

void TRKSolution::RKProcess(std::ostream &out, bool euler) {
    
    if (m_n_points <= 0) {
        DebugStop();
    }
    
    REAL h = (m_rw - m_re) / m_n_points;
    TPZVec<REAL> r(m_n_points+1);
    TPZVec<REAL> u(m_n_points+1);
    TPZFNMatrix<3,REAL> sigma(m_n_points+1,3,0.);

    // Displacement and stress at re
    TPZFNMatrix<3,REAL> sigma_re;
    REAL u_re;
    ParametersAtRe(sigma_re, u_re);

    r[0] = m_re;
    u[0] = u_re;
    sigma(0,0) = sigma_re(0,0);
    sigma(0,1) = sigma_re(0,1);
    sigma(0,2) = sigma_re(0,2);

    for (int i = 0; i < m_n_points; i++) {
        REAL du_k1;
        REAL dsigma_rr_k1;
        
        REAL lambda = m_memory_vector[i].m_ER.Lambda();
        REAL G = m_memory_vector[i].m_ER.G();
        
        //k1
        F(r[i], u[i], sigma(i,0), du_k1, dsigma_rr_k1, lambda, G);

        if (euler == false) {
            REAL du_k2, du_k3, du_k4;
            REAL dsigma_rr_k2, dsigma_rr_k3, dsigma_rr_k4;
            //k2
            F(r[i] + h / 2., u[i] + h * du_k1 / 2., sigma(i, 0) + h * dsigma_rr_k1 / 2., du_k2, dsigma_rr_k2, lambda, G);

            //k3
            F(r[i] + h / 2., u[i] + h * du_k2 / 2., sigma(i, 0) + h * dsigma_rr_k2 / 2., du_k3, dsigma_rr_k3, lambda, G);

            //k4
            F(r[i] + h, u[i] + h * du_k3, sigma(i, 0) + h * dsigma_rr_k3, du_k4, dsigma_rr_k4, lambda, G);

            
            //u_ip1, sigma_ip1
            u[i + 1] = u[i] + 1. / 6. * h * (du_k1 + 2. * du_k2 + 2. * du_k3 + du_k4);
            sigma(i+1,0) = sigma(i,0) + 1. / 6. * h * (dsigma_rr_k1 + 2. * dsigma_rr_k2 + 2. * dsigma_rr_k3 + dsigma_rr_k4);

        } else if (euler == true) {
            //u_ip1, sigma_ip1
            u[i + 1] = u[i] + h * du_k1;
            sigma(i+1,0) = sigma(i,0) + h * dsigma_rr_k1;
        }

        r[i + 1] = r[i] + h;
        sigma(i+1,1) = 2 * G * u[i + 1] / r[i + 1] + lambda * (u[i + 1] / r[i + 1] + (r[i + 1] * sigma(i+1,0) - lambda * u[i + 1]) / (r[i + 1] * (lambda + 2 * G)));
        sigma(i+1,2) = (-lambda*(lambda* (sigma(i+1,0) - 2* sigma(i+1,0) - sigma(i+1,1)) - 2 *G *(sigma(i+1,0) + sigma(i+1,1))))/(2*(lambda + G)*(lambda + 2*G));
    }

    out << "radius" << "  " << "u" << "   " << "sigma_rr" << "   " << "sigma_tt" << "   " << "sigma_zz" << std::endl;
    for (int i = 0; i < m_n_points; i++) {
        out << r[i] << "  " << u[i] << "   " << sigma(i,0) << "   " << sigma(i,1) << "   " << sigma(i,2) << std::endl;
    }
}

void TRKSolution::RKProcessII(std::ostream &out, bool euler) {

    /// Working with tensors ...
    
    if (m_n_points <= 0) {
        DebugStop();
    }
    
    REAL h = (m_rw - m_re) / m_n_points;
    TPZVec<REAL> r(m_n_points+1);
    TPZVec<REAL> u(m_n_points+1);
    TPZFNMatrix<3,REAL> sigma(m_n_points+1,3,0.);
    
    // Displacement and stress at re
    TPZFNMatrix<3,REAL> sigma_re;
    REAL u_re;
    ParametersAtRe(sigma_re, u_re);
    
    r[0] = m_re;
    u[0] = u_re;
    sigma(0,0) = sigma_re(0,0);
    sigma(0,1) = sigma_re(0,1);
    sigma(0,2) = sigma_re(0,2);
    
    m_memory_vector[0].m_sigma.XX() = sigma_re(0,0);
    m_memory_vector[0].m_sigma.YY() = sigma_re(0,1);
    m_memory_vector[0].m_sigma.ZZ() = sigma_re(0,2);
    REAL lambda = m_memory_vector[0].m_ER.Lambda();
    REAL G = m_memory_vector[0].m_ER.G();
    
    for (int i = 0; i < m_n_points; i++) {
        REAL du_k1;
        REAL dsigma_rr_k1;
        
        /// Assuming that Lamé parameters suffer small change between two RK points
        /// http://www.ecs.umass.edu/~arwade/courses/str-mech/polar.pdf
        
        TPZElastoPlasticMem & memory = m_memory_vector[i];
        
        //k1
        F_II(r[i], u[i], sigma(i,0), du_k1, dsigma_rr_k1, memory, lambda, G);
        
//        if (euler == false) {
//            REAL du_k2, du_k3, du_k4;
//            REAL dsigma_rr_k2, dsigma_rr_k3, dsigma_rr_k4;
//            //k2
//            F(r[i] + h / 2., u[i] + h * du_k1 / 2., sigma(i, 0) + h * dsigma_rr_k1 / 2., du_k2, dsigma_rr_k2, lambda, G);
//
//            //k3
//            F(r[i] + h / 2., u[i] + h * du_k2 / 2., sigma(i, 0) + h * dsigma_rr_k2 / 2., du_k3, dsigma_rr_k3, lambda, G);
//
//            //k4
//            F(r[i] + h, u[i] + h * du_k3, sigma(i, 0) + h * dsigma_rr_k3, du_k4, dsigma_rr_k4, lambda, G);
//
//            //u_ip1, sigma_ip1
//            u[i + 1] = u[i] + 1. / 6. * h * (du_k1 + 2. * du_k2 + 2. * du_k3 + du_k4);
//            sigma(i+1,0) = sigma(i,0) + 1. / 6. * h * (dsigma_rr_k1 + 2. * dsigma_rr_k2 + 2. * dsigma_rr_k3 + dsigma_rr_k4);
//
//        } else if (euler == true) {
            //u_ip1, sigma_ip1
            r[i + 1] = r[i] + h;
            u[i + 1] = u[i] + h * du_k1;
        
            /// update elastoplastic state
            {
                REAL eps_t = u[i + 1]/r[i + 1];
                REAL sigma_r = sigma(i,0) + h * dsigma_rr_k1;
                REAL eps_r = (r[i + 1]*sigma_r-lambda*u[i + 1])/(r[i + 1]*lambda+2*G*r[i + 1]);
                
                TPZTensor<REAL> eps_total,sigma;
                eps_total.Zero();
                eps_total.XX() = eps_r;
                eps_total.YY() = eps_t;
                TPZFMatrix<REAL> Dep(6,6,0.0);
                m_elastoplastic_model.SetState(m_memory_vector[i].m_elastoplastic_state);
                m_elastoplastic_model.ApplyStrainComputeSigma(eps_total, sigma, &Dep);
//                sigma.Print(std::cout);
//                Dep.Print("Dep = ",std::cout,EMathematicaInput);
//                std::cout << "l = " << Dep(0,5) << std::endl;
//                std::cout << "G = " << Dep(4,4) << std::endl;
                m_memory_vector[i+1].m_elastoplastic_state = m_elastoplastic_model.GetState();
                m_memory_vector[i+1].m_sigma = sigma;
                lambda = Dep(0,5);
                G = Dep(4,4);
                
            }
        
        // XX stands for radial direction
        // YY stands for azimuthal direction
        sigma(i+1,0) = m_memory_vector[i+1].m_sigma.XX();
        sigma(i+1,1) = m_memory_vector[i+1].m_sigma.YY();
        sigma(i+1,2) = m_memory_vector[i+1].m_sigma.ZZ();
        
//        }
        
//        r[i + 1] = r[i] + h;
//        sigma(i+1,1) = 2 * G * u[i + 1] / r[i + 1] + lambda * (u[i + 1] / r[i + 1] + (r[i + 1] * sigma(i+1,0) - lambda * u[i + 1]) / (r[i + 1] * (lambda + 2 * G)));
//        sigma(i+1,2) = (-lambda*(lambda* (sigma(i+1,0) - 2* sigma(i+1,0) - sigma(i+1,1)) - 2 *G *(sigma(i+1,0) + sigma(i+1,1))))/(2*(lambda + G)*(lambda + 2*G));
    }
    
    out << "radius" << "  " << "u" << "   " << "sigma_rr" << "   " << "sigma_tt" << "   " << "sigma_zz" << std::endl;
    for (int i = 0; i < m_n_points; i++) {
        out << r[i] << "  " << u[i] << "   " << sigma(i,0) << "   " << sigma(i,1) << "   " << sigma(i,2) << std::endl;
    }
}

void TRKSolution::F_II (REAL r, REAL ur, REAL sigma_r, REAL &d_ur, REAL &d_sigmar, TPZElastoPlasticMem memory, REAL & lambda, REAL & G){
    
    REAL eps_t = ur/r;
    REAL sigma_t = (4*eps_t*G*(G + lambda) + lambda*sigma_r)/(2*G + lambda);
    d_ur = (r*sigma_r-lambda*ur)/(r*lambda+2*G*r);
    d_sigmar = (-sigma_r + sigma_t)/r;
}

void TRKSolution::ReconstructEpsilon(TPZTensor<REAL> & sigma, TPZTensor<REAL> & eps_t, REAL & lambda, REAL & G){
    DebugStop();
}
