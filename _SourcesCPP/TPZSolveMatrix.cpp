#include "TPZSolveMatrix.h"
#include "pzmatrix.h"
#include <mkl.h>
#include <stdlib.h>
#include "TPZTensor.h"

#ifdef USING_MKL
#include <mkl.h>
#include <algorithm>
#endif

#ifdef USING_TBB
#include "tbb/parallel_for_each.h"
using namespace tbb;
#endif

//MATERIAL PROPERTIES (organizar isso aqui!)
#define mc_cohesion         10.0
#define mc_phi              20.0*M_PI/180
#define mc_psi              20.0*M_PI/180
#define G                   400*mc_cohesion
#define nu                  0.3
#define E                   2.0*G*(1+nu)
#define K                   nu * E / ((1. +nu)*(1. - 2. * nu)) + 2. * E / (2. * (1. + nu)) / 3.

//PrincipalStress
void Normalize(double *sigma, double &maxel) {
    maxel = sigma[0];
    for (int i = 1; i < 4; i++) {
        if (fabs(sigma[i]) > fabs(maxel)) {
            maxel = sigma[i];
        }
    }
    for (int i = 0; i < 4; i++) {
        sigma[i] /= maxel;
    }
}

void TPZSolveMatrix::Interval(double *sigma, double *interval) {
    TPZVec<REAL> lower_vec(3);
    TPZVec<REAL> upper_vec(3);

    //row 1 |sigma_xx sigma_xy 0|
    lower_vec[0] = sigma[0] - fabs(sigma[3]);
    upper_vec[0] = sigma[0] + fabs(sigma[3]);

    //row 2 |sigma_xy sigma_yy 0|
    lower_vec[1] = sigma[1] - fabs(sigma[3]);
    upper_vec[1] = sigma[1] + fabs(sigma[3]);

    //row 3 |0 0 sigma_zz|
    lower_vec[2] = sigma[2];
    upper_vec[2] = sigma[2];

    interval[0] = upper_vec[0];
    interval[1] = lower_vec[0];

    for (int i = 1; i < 3; i++) {
        if (upper_vec[i] > interval[0]) { //upper interval
            interval[0] = upper_vec[i];
        }

        if (lower_vec[i] < interval[1]) { //lower interval
            interval[1] = lower_vec[i];
        }
    }
}

void TPZSolveMatrix::NewtonIterations(double *interval, double *sigma, double *eigenvalues, double &maxel) {
    int numiterations = 20;
    REAL tol = 10e-8;

    REAL res, f, df, x;
    int it;

    for (int i = 0; i < 2; i++) {
        x = interval[i];
        it = 0;

        f = sigma[0] * sigma[1] - x * (sigma[0] + sigma[1]) + x * x - sigma[3] * sigma[3];
        res = abs(f);

        while (it < numiterations && res > tol) {
            df = -sigma[0] - sigma[1] + 2 * x;

            x -= f / df;
            f = sigma[0] * sigma[1] - x * (sigma[0] + sigma[1]) + x * x - sigma[3] * sigma[3];
            res = abs(f);
            it++;
        }
        eigenvalues[i] = x;

    }
    eigenvalues[2] = sigma[0] + sigma[1] + sigma[2] - eigenvalues[0] - eigenvalues[1];

    eigenvalues[0] *= maxel;
    eigenvalues[1] *= maxel;
    eigenvalues[2] *= maxel;

    std::sort(eigenvalues, eigenvalues+3, [](int i, int j) { return abs(i) > abs(j); }); //store eigenvalues in descending order (absolute value)
}

//ProjectSigma
bool PhiPlane(double *eigenvalues, double *sigma_projected) {
    const REAL sinphi = sin(mc_phi);
    const REAL cosphi = cos(mc_phi);

    REAL phi = eigenvalues[0] - eigenvalues[2] + (eigenvalues[0] + eigenvalues[2]) * sinphi - 2. * mc_cohesion *cosphi;

    sigma_projected[0] = eigenvalues[0];
    sigma_projected[1] = eigenvalues[1];
    sigma_projected[2] = eigenvalues[2];

    bool check_validity = (fabs(phi) < 1.e-12) || (phi < 0.0);
    return check_validity;
}

bool ReturnMappingMainPlane(double *eigenvalues, double *sigma_projected, double &m_hardening) {

    const REAL sinphi = sin(mc_phi);
    const REAL sinpsi = sin(mc_psi);
    const REAL cosphi = cos(mc_phi);
    const REAL sinphi2 = sinphi*sinphi;
    const REAL cosphi2 = 1. - sinphi2;
    const REAL constA = 4. * G *(1. + sinphi * sinpsi / 3.) + 4. * K * sinphi*sinpsi;

    REAL phi = eigenvalues[0] - eigenvalues[2]+(eigenvalues[0] + eigenvalues[2]) * sinphi - 2. * mc_cohesion*cosphi;

    REAL gamma = 0;
    int n_iterations = 30;
    for (int i = 0; i < n_iterations; i++) {
        double jac = -constA - 4. * cosphi2 * 0; // H=0
        double delta_gamma = - phi / jac;
        gamma += delta_gamma;
        phi = eigenvalues[0] - eigenvalues[2]+(eigenvalues[0] + eigenvalues[2]) * sinphi - 2. * mc_cohesion * cosphi - constA * gamma;
        if (fabs(phi) < 1.e-12) {
            break;
        }
    }

    sigma_projected[0] -= (2. * G *(1 + sinpsi / 3.) + 2. * K * sinpsi) * gamma;
    sigma_projected[1] += (4. * G / 3. - K * 2.) * sinpsi * gamma;
    sigma_projected[2] += (2. * G * (1 - sinpsi / 3.) - 2. * K * sinpsi) * gamma;

    m_hardening += gamma * 2. * cosphi;

    bool check_validity = (eigenvalues[0] > eigenvalues[1] || fabs(eigenvalues[0]-eigenvalues[1]) < 1.e-12) && (eigenvalues[1] > eigenvalues[2] || fabs(eigenvalues[1]-eigenvalues[2]) < 1.e-12);
    return check_validity;
}

bool ReturnMappingRightEdge(double *eigenvalues, double *sigma_projected, double &m_hardening) {
    const REAL sinphi = sin(mc_phi);
    const REAL sinpsi = sin(mc_psi);
    const REAL cosphi = cos(mc_phi);
    const REAL sinphi2 = sinphi*sinphi;
    const REAL cosphi2 = 1. - sinphi2;

    TPZVec<REAL> gamma(2, 0.), phi(2, 0.), sigma_bar(2, 0.), ab(2, 0.);

    TPZVec<TPZVec<REAL>> jac(2), jac_inv(2);
    for (int i = 0; i < 2; i++) {
        jac[i].Resize(2, 0.);
        jac_inv[i].Resize(2, 0.);
    }

    sigma_bar[0] = eigenvalues[0] - eigenvalues[2]+(eigenvalues[0] + eigenvalues[2]) * sinphi;
    sigma_bar[1] = eigenvalues[0] - eigenvalues[1] + (eigenvalues[0] + eigenvalues[1]) * sinphi;

    phi[0] = sigma_bar[0] - 2. * cosphi * mc_cohesion;
    phi[1] = sigma_bar[1] - 2. * cosphi * mc_cohesion;

    ab[0] = 4. * G * (1 + sinphi * sinpsi / 3.) + 4. * K * sinphi * sinpsi;
    ab[1] = 2. * G * (1. + sinphi + sinpsi - sinphi * sinpsi / 3.) + 4. * K * sinphi * sinpsi;

    int n_iterations = 30;
    for (int i = 0; i < n_iterations; i++) {

        jac[0][0] = -ab[0];
        jac[1][0] = -ab[1];
        jac[0][1] = -ab[1];
        jac[1][1] = -ab[0];

        double det_jac = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];

        jac_inv[0][0] = jac[1][1] / det_jac;
        jac_inv[1][0] = -jac[1][0] / det_jac;
        jac_inv[0][1] = -jac[0][1] / det_jac;
        jac_inv[1][1] = jac[0][0] / det_jac;

        gamma[0] -= (jac_inv[0][0] * phi[0] + jac_inv[0][1] * phi[1]);
        gamma[1] -= (jac_inv[1][0] * phi[0] + jac_inv[1][1] * phi[1]);

        phi[0] = sigma_bar[0] - ab[0] * gamma[0] - ab[1] * gamma[1] - 2. * cosphi * mc_cohesion;
        phi[1] = sigma_bar[1] - ab[1] * gamma[0] - ab[0] * gamma[1] - 2. * cosphi * mc_cohesion;

        double res = (fabs(phi[0]) + fabs(phi[1]));

        if (fabs(res) < 1.e-12) {
            break;
        }
    }

    sigma_projected[0] -= (2. * G * (1 + sinpsi / 3.) + 2. * K * sinpsi) * (gamma[0] + gamma[1]);
    sigma_projected[1] += ((4. * G / 3. - K * 2.) * sinpsi) * gamma[0] + (2. * G * (1. - sinpsi / 3.) - 2. * K * sinpsi) * gamma[1];
    sigma_projected[2] += (2. * G * (1 - sinpsi / 3.) - 2. * K * sinpsi) * gamma[0] + ((4. * G / 3. - 2. * K) * sinpsi) * gamma[1];

    m_hardening += (gamma[0] + gamma[1]) * 2. * cosphi;

    bool check_validity = (eigenvalues[0] > eigenvalues[1] || fabs(eigenvalues[0]-eigenvalues[1]) < 1.e-12) && (eigenvalues[1] > eigenvalues[2] || fabs(eigenvalues[1]-eigenvalues[2]) < 1.e-12);
    return check_validity;
}

bool ReturnMappingLeftEdge(double *eigenvalues, double *sigma_projected, double &m_hardening) {
    const REAL sinphi = sin(mc_phi);
    const REAL sinpsi = sin(mc_psi);
    const REAL cosphi = cos(mc_phi);
    const REAL sinphi2 = sinphi*sinphi;
    const REAL cosphi2 = 1. - sinphi2;

    TPZVec<REAL> gamma(2, 0.), phi(2, 0.), sigma_bar(2, 0.), ab(2, 0.);

    TPZVec<TPZVec<REAL>> jac(2), jac_inv(2);
    for (int i = 0; i < 2; i++) {
        jac[i].Resize(2, 0.);
        jac_inv[i].Resize(2, 0.);
    }

    sigma_bar[0] = eigenvalues[0] - eigenvalues[2]+(eigenvalues[0] + eigenvalues[2]) * sinphi;
    sigma_bar[1] = eigenvalues[1] - eigenvalues[2]+(eigenvalues[1] + eigenvalues[2]) * sinphi;

    ab[0] = 4. * G * (1 + sinphi * sinpsi / 3.) + 4. * K * sinphi * sinpsi;
    ab[1] = 2. * G * (1. - sinphi - sinpsi - sinphi * sinpsi / 3.) + 4. * K * sinphi * sinpsi;

    phi[0] = sigma_bar[0] - 2. * cosphi * mc_cohesion;
    phi[1] = sigma_bar[1] - 2. * cosphi * mc_cohesion;

    int n_iterations = 30;
    for (int i = 0; i < n_iterations; i++) {

        jac[0][0] = -ab[0] - 4. * cosphi2 * 0;
        jac[1][0] = -ab[1] - 4. * cosphi2 * 0;
        jac[0][1] = -ab[1] - 4. * cosphi2 * 0;
        jac[1][1] = -ab[0] - 4. * cosphi2 * 0;

        REAL det_jac = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];

        jac_inv[0][0] = jac[1][1] / det_jac;
        jac_inv[1][0] = -jac[1][0] / det_jac;
        jac_inv[0][1] = -jac[0][1] / det_jac;
        jac_inv[1][1] = jac[0][0] / det_jac;

        gamma[0] -= (jac_inv[0][0] * phi[0] + jac_inv[0][1] * phi[1]);
        gamma[1] -= (jac_inv[1][0] * phi[0] + jac_inv[1][1] * phi[1]);

        phi[0] = sigma_bar[0] - ab[0] * gamma[0] - ab[1] * gamma[1] - 2. * cosphi * mc_cohesion;
        phi[1] = sigma_bar[1] - ab[1] * gamma[0] - ab[0] * gamma[1] - 2. * cosphi * mc_cohesion;

        REAL res = (fabs(phi[0]) + fabs(phi[1]));

        if (fabs(res) < 1.e-12) {
            break;
        }
    }

    sigma_projected[0] += -(2. * G * (1 + sinpsi / 3.) + 2. * K * sinpsi) * gamma[0] + ((4. * G / 3. - 2. * K) * sinpsi) * gamma[1];
    sigma_projected[1] += ((4. * G / 3. - K * 2.) * sinpsi) * gamma[0] - (2. * G * (1. + sinpsi / 3.) + 2. * K * sinpsi) * gamma[1];
    sigma_projected[2] += (2. * G * (1 - sinpsi / 3.) - 2. * K * sinpsi) * (gamma[0] + gamma[1]);

    m_hardening += (gamma[0] + gamma[1]) * 2. * cosphi;

    bool check_validity = (eigenvalues[0] > eigenvalues[1] || fabs(eigenvalues[0]-eigenvalues[1]) < 1.e-12) && (eigenvalues[1] > eigenvalues[2] || fabs(eigenvalues[1]-eigenvalues[2]) < 1.e-12);
    return check_validity;
}

void ReturnMappingApex(double *eigenvalues, double *sigma_projected, double &m_hardening) {
    const REAL sinpsi = sin(mc_psi);
    const REAL cosphi = cos(mc_phi);
    const REAL cotphi = 1. / tan(mc_phi);

    REAL ptrnp1 = 0.;
    for (int i = 0; i < 3; i++) {
        ptrnp1 += eigenvalues[i];
    }
    ptrnp1 /= 3.;

    REAL DEpsPV = 0.;
    REAL alpha = cos(mc_phi) / sin(mc_psi);
    REAL res = mc_cohesion * cotphi - ptrnp1;
    REAL pnp1;

    int n_iterations = 30;
    for (int i = 0; i < n_iterations; i++) {
        const REAL jac = K; //H = 0
        DEpsPV -= res / jac;

        pnp1 = ptrnp1 - K * DEpsPV;
        res = mc_cohesion * cotphi - pnp1;

        if (fabs(res) < 1.e-12) {
            break;
        }
    }

    m_hardening += alpha * DEpsPV;
    for (int i = 0; i < 3; i++) {
        sigma_projected[i] = pnp1;
    }
}

void TPZSolveMatrix::DeltaStrain(TPZFMatrix<REAL> &global_solution, TPZFMatrix<REAL> &delta_strain) {
    int64_t nelem = fRowSizes.size();
    int64_t n_globalsol = fIndexes.size();

    delta_strain.Resize(2*n_globalsol,1);
    delta_strain.Zero();

    TPZVec<REAL> expandsolution(n_globalsol);

/// gather operation
    cblas_dgthr(n_globalsol, global_solution, &expandsolution[0], &fIndexes[0]);

    for (int64_t iel = 0; iel < nelem; iel++) {
        for (int i = 0; i < fRowSizes[iel]; i++) {
            for (int j = 0; j < 1; j++) {
                for (int k = 0; k < fColSizes[iel]; k++) {
                    delta_strain(j * fRowSizes[iel] + i + fRowFirstIndex[iel], 0) += fStorage[k * fRowSizes[iel] + i + fMatrixPosition[iel]] * expandsolution[j * fColSizes[iel] + k + fColFirstIndex[iel]];
                    delta_strain(j * fRowSizes[iel] + i + fRowFirstIndex[iel] + n_globalsol, 0) += fStorage[k * fRowSizes[iel] + i + fMatrixPosition[iel]] * expandsolution[j * fColSizes[iel] + k + fColFirstIndex[iel] + n_globalsol/2];
                }
            }
        }
    }
}

void TPZSolveMatrix::ElasticStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &total_strain, TPZFMatrix<REAL> &plastic_strain, TPZFMatrix<REAL> &elastic_strain) {
    elastic_strain = total_strain + delta_strain - plastic_strain;
}

void TPZSolveMatrix::SigmaTrial(TPZFMatrix<REAL> &elastic_strain, TPZFMatrix<REAL> &sigma_trial) {
//REAL E = 200000000.;
//REAL E = 20000000.;
//    REAL nu =0.30;
//    REAL E = 2.0*400*20.0*(1+nu);
    int64_t npts = fRow/fDim;
    sigma_trial.Resize(4*npts,1);

#ifdef USING_TBB
    parallel_for(size_t(0),size_t(npts_tot/2),size_t(1),[&](size_t ipts)
                      {
                            sigma(2*ipts,0) = weight[ipts]*E/(1.-nu*nu)*(result(2*ipts,0)+nu*result(2*ipts+npts_tot+1,0)); // Sigma x
                            sigma(2*ipts+1,0) = weight[ipts]*E/(1.-nu*nu)*(1.-nu)/2*(result(2*ipts+1,0)+result(2*ipts+npts_tot,0))*0.5; // Sigma xy
                            sigma(2*ipts+npts_tot,0) = sigma(2*ipts+1,0); //Sigma xy
                            sigma(2*ipts+npts_tot+1,0) = weight[ipts]*E/(1.-nu*nu)*(result(2*ipts+npts_tot+1,0)+nu*result(2*ipts,0)); // Sigma y
                      }
                      );
#else

    for (int64_t ipts=0; ipts < npts; ipts++) {
        //plane strain
        sigma_trial(4*ipts,0) = fWeight[ipts]*(elastic_strain(2*ipts,0)*E*(1.-nu)/((1.-2*nu)*(1.+nu)) + elastic_strain(2*ipts+2*npts+1,0)*E*nu/((1.-2*nu)*(1.+nu))); // Sigma xx
        sigma_trial(4*ipts+1,0) = fWeight[ipts]*(elastic_strain(2*ipts+2*npts+1,0)*E*(1.-nu)/((1.-2*nu)*(1.+nu)) + elastic_strain(2*ipts,0)*E*nu/((1.-2*nu)*(1.+nu))); // Sigma yy
        sigma_trial(4*ipts+2,0) = fWeight[ipts]*(E*nu/((1.+nu)*(1.-2*nu))*(elastic_strain(2*ipts,0) + elastic_strain(2*ipts+2*npts+1,0))); // Sigma zz
        sigma_trial(4*ipts+3,0) = fWeight[ipts]*E/(2*(1.+nu))*(elastic_strain(2*ipts+1,0)+elastic_strain(2*ipts+2*npts,0)); // Sigma xy

//    sigma(2*ipts,0) = weight[ipts]*E/(1.-nu*nu)*(result(2*ipts,0)+nu*result(2*ipts+npts_tot+1,0)); // Sigma x
//    sigma(2*ipts+1,0) = weight[ipts]*E/(1.-nu*nu)*(1.-nu)/2*(result(2*ipts+1,0)+result(2*ipts+npts_tot,0))*0.5; // Sigma xy
//    sigma(2*ipts+npts_tot,0) = sigma(2*ipts+1,0); //Sigma xy
//    sigma(2*ipts+npts_tot+1,0) = weight[ipts]*E/(1.-nu*nu)*(result(2*ipts+npts_tot+1,0)+nu*result(2*ipts,0)); // Sigma y
    }
#endif
}

void TPZSolveMatrix::PrincipalStress(TPZFMatrix<REAL> &sigma_trial, TPZFMatrix<REAL> &eigenvalues) {
    int64_t npts = fRow/fDim;

    REAL maxel;
    TPZVec<REAL> interval(2);
    eigenvalues.Resize(3*npts,1);

    for (int64_t ipts = 0; ipts < npts; ipts++) {
        Normalize(&sigma_trial(4*ipts, 0), maxel);
        Interval(&sigma_trial(4*ipts, 0), &interval[0]);
        NewtonIterations(&interval[0], &sigma_trial(4*ipts, 0), &eigenvalues(3*ipts, 0), maxel);
    }
}

void TPZSolveMatrix::ProjectSigma(TPZFMatrix<REAL> &total_strain, TPZFMatrix<REAL> &plastic_strain, TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &sigma_projected) {
    int npts = fRow/fDim;

    sigma_projected.Resize(3*npts,1);
    sigma_projected.Zero();
    TPZFMatrix<REAL> elastic_strain_np1(fDim*fRow);

    TPZVec<int> m_type(npts,0);
    TPZVec<REAL> m_hardening(npts, 0.);
    bool check = false;

    for (int ipts = 0; ipts < npts; ipts++) {
        m_type[ipts] = 0;
        check = PhiPlane(&eigenvalues(3*ipts, 0), &sigma_projected(3*ipts, 0)); //elastic domain
        if (!check) { //plastic domain
            m_type[ipts] = 1;
            check = ReturnMappingMainPlane(&eigenvalues(3*ipts, 0), &sigma_projected(3*ipts, 0), m_hardening[ipts]); //main plane
            if (!check) { //edges or apex
                if  (((1 - sin(mc_psi)) * eigenvalues(0 + 3*ipts, 0) - 2. * eigenvalues(1 + 3*ipts, 0) + (1 + sin(mc_psi)) * eigenvalues(2 + 3*ipts, 0)) > 0) { // right edge
                    check = ReturnMappingRightEdge(&eigenvalues(3*ipts, 0), &sigma_projected(3*ipts, 0), m_hardening[ipts]);
                } else { //left edge
                    check = ReturnMappingLeftEdge(&eigenvalues(3*ipts, 0), &sigma_projected(3*ipts, 0), m_hardening[ipts]);
                }
                if (!check) { //apex
                    m_type[ipts] = -1;
                    ReturnMappingApex(&eigenvalues(3*ipts, 0), &sigma_projected(3*ipts, 0), m_hardening[ipts]);
                }
            }
        }
    }
}

void TPZSolveMatrix::Multiply(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const {
int64_t nelem = fRowSizes.size();
int64_t n_globalsol = fIndexes.size();

result.Resize(2*n_globalsol,1);
result.Zero();

TPZVec<REAL> expandsolution(n_globalsol);

/// gather operation
cblas_dgthr(n_globalsol, global_solution, &expandsolution[0], &fIndexes[0]);

#ifdef USING_TBB
parallel_for(size_t(0),size_t(nelem),size_t(1),[&](size_t iel)
             {
                int64_t pos = fMatrixPosition[iel];
                int64_t cols = fColSizes[iel];
                int64_t rows = fRowSizes[iel];
                TPZFMatrix<REAL> elmatrix(rows,cols,&fStorage[pos],rows*cols);

                int64_t cont_cols = fColFirstIndex[iel];
                int64_t cont_rows = fRowFirstIndex[iel];

                 TPZFMatrix<REAL> element_solution_x(cols,1,&expandsolution[cont_cols],cols);
                 TPZFMatrix<REAL> element_solution_y(cols,1,&expandsolution[cont_cols+fColFirstIndex[nelem]],cols);

                TPZFMatrix<REAL> solx(rows,1,&result(cont_rows,0),rows);
                TPZFMatrix<REAL> soly(rows,1,&result(cont_rows+fRowFirstIndex[nelem],0),rows);

                elmatrix.Multiply(element_solution_x,solx);
                elmatrix.Multiply(element_solution_y,soly);
             }
             );

#else
for (int64_t iel=0; iel<nelem; iel++) {
    int64_t pos = fMatrixPosition[iel];
    int64_t cols = fColSizes[iel];
    int64_t rows = fRowSizes[iel];
    TPZFMatrix<REAL> elmatrix(rows,cols,&fStorage[pos],rows*cols);

    int64_t cont_cols = fColFirstIndex[iel];
    int64_t cont_rows = fRowFirstIndex[iel];

    TPZFMatrix<REAL> element_solution_x(cols,1,&expandsolution[cont_cols],cols);
    TPZFMatrix<REAL> element_solution_y(cols,1,&expandsolution[cont_cols+fColFirstIndex[nelem]],cols);

    TPZFMatrix<REAL> solx(rows,1,&result(cont_rows,0),rows); //du
    TPZFMatrix<REAL> soly(rows,1,&result(cont_rows+fRowFirstIndex[nelem],0),rows); //dv

    elmatrix.Multiply(element_solution_x,solx);
    elmatrix.Multiply(element_solution_y,soly);
}
#endif
}

void TPZSolveMatrix::MultiplyTranspose(TPZFMatrix<STATE>  &intpoint_solution, TPZFMatrix<STATE> &nodal_forces_vec) {
int64_t nelem = fRowSizes.size();
int64_t npts_tot = fRow;
nodal_forces_vec.Resize(npts_tot,1);
nodal_forces_vec.Zero();

#ifdef USING_TBB
parallel_for(size_t(0),size_t(nelem),size_t(1),[&](size_t iel)
                  {
                        int64_t pos = fMatrixPosition[iel];
                        int64_t rows = fRowSizes[iel];
                        int64_t cols = fColSizes[iel];
                        int64_t cont_rows = fRowFirstIndex[iel];
                        int64_t cont_cols = fColFirstIndex[iel];
                        TPZFMatrix<REAL> elmatrix(rows,cols,&fStorage[pos],rows*cols);

                        // Forças nodais na direção x
                        TPZFMatrix<REAL> fvx(rows,1, &intpoint_solution(cont_rows,0),rows);
                        TPZFMatrix<STATE> nodal_forcex(cols, 1, &nodal_forces_vec(cont_cols, 0), cols);
                        elmatrix.MultAdd(fvx,nodal_forcex,nodal_forcex,1,0,1);

                        // Forças nodais na direção y
                        TPZFMatrix<REAL> fvy(rows,1, &intpoint_solution(cont_rows+npts_tot,0),rows);
                        TPZFMatrix<STATE> nodal_forcey(cols, 1, &nodal_forces_vec(cont_cols + npts_tot/2, 0), cols);
                        elmatrix.MultAdd(fvy,nodal_forcey,nodal_forcey,1,0,1);
                  }
                  );
#else
for (int64_t iel = 0; iel < nelem; iel++) {
    int64_t pos = fMatrixPosition[iel];
    int64_t rows = fRowSizes[iel];
    int64_t cols = fColSizes[iel];
    int64_t cont_rows = fRowFirstIndex[iel];
    int64_t cont_cols = fColFirstIndex[iel];
    TPZFMatrix<REAL> elmatrix(rows,cols,&fStorage[pos],rows*cols);

    // Nodal forces in x direction
    TPZFMatrix<REAL> fvx(rows,1, &intpoint_solution(cont_rows,0),rows);
    TPZFMatrix<STATE> nodal_forcex(cols, 1, &nodal_forces_vec(cont_cols, 0), cols);
    elmatrix.MultAdd(fvx,nodal_forcex,nodal_forcex,1,0,1);

    // Nodal forces in y direction
    TPZFMatrix<REAL> fvy(rows,1, &intpoint_solution(cont_rows+npts_tot,0),rows);
    TPZFMatrix<STATE> nodal_forcey(cols, 1, &nodal_forces_vec(cont_cols + npts_tot/2, 0), cols);
    elmatrix.MultAdd(fvy,nodal_forcey,nodal_forcey,1,0,1);
}
#endif
}

void TPZSolveMatrix::ColoredAssemble(TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) {
    int64_t ncolor = *std::max_element(fElemColor.begin(), fElemColor.end())+1;
    int64_t sz = fIndexes.size();
    int64_t neq = nodal_forces_global.Rows();
    nodal_forces_global.Resize(neq*ncolor,1);


    cblas_dsctr(sz, nodal_forces_vec, &fIndexesColor[0], &nodal_forces_global(0,0));

    int64_t colorassemb = ncolor / 2.;
    while (colorassemb > 0) {

        int64_t firsteq = (ncolor - colorassemb) * neq;
        cblas_daxpy(firsteq, 1., &nodal_forces_global(firsteq, 0), 1., &nodal_forces_global(0, 0), 1.);

        ncolor -= colorassemb;
        colorassemb = ncolor/2;
    }
    nodal_forces_global.Resize(neq, 1);
}

void TPZSolveMatrix::TraditionalAssemble(TPZFMatrix<STATE>  &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global) const {
#ifdef USING_TBB
    parallel_for(size_t(0),size_t(fRow),size_t(1),[&](size_t ir)
             {
                 nodal_forces_global(fIndexes[ir], 0) += nodal_forces_vec(ir, 0);
             }
);
#else
    for (int64_t ir=0; ir<fRow; ir++) {
        nodal_forces_global(fIndexes[ir], 0) += nodal_forces_vec(ir, 0);
    }
#endif
}

void TPZSolveMatrix::ColoringElements(TPZCompMesh * cmesh) const {
    int64_t nelem_c = cmesh->NElements();
    int64_t nconnects = cmesh->NConnects();
    TPZVec<int64_t> connects_vec(nconnects,0);

    int64_t contcolor = 0;
    bool needstocontinue = true;

    while (needstocontinue)
    {
        int it = 0;
        needstocontinue = false;
        for (int64_t iel = 0; iel < nelem_c; iel++) {
            TPZCompEl *cel = cmesh->Element(iel);
            if (!cel || cel->Dimension() != cmesh->Dimension()) continue;

            it++;
            if (fElemColor[it-1] != -1) continue;

            TPZStack<int64_t> connectlist;
            cmesh->Element(iel)->BuildConnectList(connectlist);
            int64_t ncon = connectlist.size();

            int64_t icon;
            for (icon = 0; icon < ncon; icon++) {
                if (connects_vec[connectlist[icon]] != 0) break;
            }
            if (icon != ncon) {
                needstocontinue = true;
                continue;
            }
            fElemColor[it-1] = contcolor;
//            cel->Reference()->SetMaterialId(contcolor);

            for (icon = 0; icon < ncon; icon++) {
                connects_vec[connectlist[icon]] = 1;
            }
        }
        contcolor++;
        connects_vec.Fill(0);
    }

    int64_t nelem = fRowSizes.size();
    int64_t neq = cmesh->NEquations();
    for (int64_t iel = 0; iel < nelem; iel++) {
        int64_t cols = fColSizes[iel];
        int64_t cont_cols = fColFirstIndex[iel];

        for (int64_t icols = 0; icols < cols; icols++) {
            fIndexesColor[cont_cols + icols] = fIndexes[cont_cols + icols] + fElemColor[iel]*neq;
            fIndexesColor[cont_cols+fRow/2 + icols] = fIndexes[cont_cols + fRow/2 + icols] + fElemColor[iel]*neq;
        }
    }
}


void TPZSolveMatrix::AllocateMemory(TPZCompMesh *cmesh){
    return;
}

void TPZSolveMatrix::FreeMemory(){
    return;
}

void TPZSolveMatrix::cuSparseHandle(){
    return;
}

void TPZSolveMatrix::cuBlasHandle(){
    return;
}

void TPZSolveMatrix::MultiplyCUDA(const TPZFMatrix<STATE> &global_solution, TPZFMatrix<STATE> &result) const{
    return;
}

void TPZSolveMatrix::ComputeSigmaCUDA(TPZStack<REAL> &weight, TPZFMatrix<REAL> &result, TPZFMatrix<REAL> &sigma){
    return;
}

void TPZSolveMatrix::MultiplyTransposeCUDA(TPZFMatrix<STATE> &intpoint_solution, TPZFMatrix<STATE> &nodal_forces_vec){
    return;
}

void TPZSolveMatrix::ColoredAssembleCUDA(TPZFMatrix<STATE> &nodal_forces_vec, TPZFMatrix<STATE> &nodal_forces_global){
    return;
}


