#include "TPZSolveMatrix.h"
#include "pzmatrix.h"
#include <mkl.h>
#include <stdlib.h>
#include "TPZTensor.h"
#include "TPZVTKGeoMesh.h"

#ifdef USING_MKL
#include <mkl.h>
#include <algorithm>
#endif

#ifdef USING_TBB
#include "tbb/parallel_for_each.h"
using namespace tbb;
#endif

//Spectral decomposition
void TPZSolveMatrix::Eigenvectors(double *sigma, double *eigenvalues, double *eigenvectors) {
    TPZVec<int> multiplicity(3);
    if (eigenvalues[0] == eigenvalues[1] && eigenvalues[1] == eigenvalues[2] || sigma[3] < 1e-8) {
        eigenvectors[0] = 1;
        eigenvectors[1] = 0;
        eigenvectors[2] = 0;

        eigenvectors[3] = 0;
        eigenvectors[4] = 1;
        eigenvectors[5] = 0;

        eigenvectors[6] = 0;
        eigenvectors[7] = 0;
        eigenvectors[8] = 1;
    }
    else {
        if (eigenvalues[0] != eigenvalues[1] && eigenvalues[0] != eigenvalues[2]) {
            multiplicity[0] = 1;
        } else if (eigenvalues[0] == eigenvalues[1]) {
            multiplicity[0] = 2;
            multiplicity[1] = 2;
        } else if (eigenvalues[0] == eigenvalues[2]) {
            multiplicity[0] = 2;
            multiplicity[2] = 2;
        }
        if (eigenvalues[1] != eigenvalues[0] && eigenvalues[1] != eigenvalues[2]) {
            multiplicity[1] = 1;
        } else if (eigenvalues[1] == eigenvalues[2]) {
            multiplicity[1] = 2;
            multiplicity[2] = 2;
        }
        if (eigenvalues[2] != eigenvalues[0] && eigenvalues[2] != eigenvalues[1]) {
            multiplicity[2] = 1;
        }
    }

    for (int i = 0; i < 3; i++) {
        if (multiplicity[i] == 1) {
            TPZVec<REAL> det(3);
            det[0] = (sigma[0] - eigenvalues[i])*(sigma[1] - eigenvalues[i]) - sigma[3]*sigma[3];
            det[1] = (sigma[0] - eigenvalues[i])*(sigma[2] - eigenvalues[i]);
            det[2] = (sigma[1] - eigenvalues[i])*(sigma[2] - eigenvalues[i]);

            REAL maxdet = fabs(det[0]);
            for (int i = 1; i < 3; i++) {
                if (fabs(det[i]) > fabs(maxdet)) {
                    maxdet = fabs(det[i]);
                }
            }
            TPZVec<REAL> v(3);
            if (maxdet == fabs(det[0])) {
                v[0] = 0;
                v[1] = 0;
                v[2] = 1;

            }
            else if (maxdet == fabs(det[1])) {
                v[0] = 1/det[1]*(-(sigma[2] - eigenvalues[i])*sigma[3]);
                v[1] = 1;
                v[2] = 0;

            }
            else {
                v[0] = 1;
                v[1] = 1/det[2]*(-(sigma[2] - eigenvalues[i])*sigma[3]);
                v[2] = 0;
            }
            REAL norm = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
            eigenvectors[3*i + 0] = v[0]/norm;
            eigenvectors[3*i + 1] = v[1]/norm;
            eigenvectors[3*i + 2] = v[2]/norm;
        }
        if (multiplicity[i] == 2) {
            TPZVec<REAL> x(3);
            x[0] = sigma[0] - eigenvalues[i];
            x[1] = sigma[1] - eigenvalues[i];
            x[2] = sigma[2] - eigenvalues[i];

            REAL maxx = fabs(x[0]);
            for (int i = 1; i < 3; i++) {
                if (fabs(x[i]) > fabs(maxx)) {
                    maxx = fabs(x[i]);
                }
            }

            TPZVec<REAL> v1(3);
            TPZVec<REAL> v2(3);

            if (maxx == fabs(x[0])) {
                v1[0] = -sigma[3]/x[0];
                v1[1] = 1;
                v1[2] = 0;

                v2[0] = 0;
                v2[1] = 0;
                v2[2] = 1;

            }
            else if (maxx == fabs(x[1])) {
                v1[0] = 1;
                v1[1] = -sigma[3]/x[1];
                v1[2] = 0;

                v2[0] = 0;
                v2[1] = 0;
                v2[2] = 1;

            }
            else {
                v1[0] = 1;
                v1[1] = 0;
                v1[2] = 0;

                v2[0] = 0;
                v2[1] = 1;
                v2[2] = 0;

            }
        }
    }
}

void TPZSolveMatrix::Normalize(double *sigma, double &maxel) {
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

//Project Sigma
bool TPZSolveMatrix::PhiPlane(double *eigenvalues, double *sigma_projected) {
    REAL mc_phi = fMaterialData.FrictionAngle();
    REAL mc_cohesion = fMaterialData.Cohesion();

    const REAL sinphi = sin(mc_phi);
    const REAL cosphi = cos(mc_phi);

    REAL phi = eigenvalues[0] - eigenvalues[2] + (eigenvalues[0] + eigenvalues[2]) * sinphi - 2. * mc_cohesion *cosphi;

    sigma_projected[0] = eigenvalues[0];
    sigma_projected[1] = eigenvalues[1];
    sigma_projected[2] = eigenvalues[2];

    bool check_validity = (fabs(phi) < 1.e-12) || (phi < 0.0);
    return check_validity;
}

bool TPZSolveMatrix::ReturnMappingMainPlane(double *eigenvalues, double *sigma_projected, double &m_hardening) {
    REAL mc_phi = fMaterialData.FrictionAngle();
    REAL mc_psi = mc_phi;
    REAL mc_cohesion = fMaterialData.Cohesion();
    REAL G = fMaterialData.ElasticResponse().G();
    REAL K = fMaterialData.ElasticResponse().K();

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

bool TPZSolveMatrix::ReturnMappingRightEdge(double *eigenvalues, double *sigma_projected, double &m_hardening) {
    REAL mc_phi = fMaterialData.FrictionAngle();
    REAL mc_psi = mc_phi;
    REAL mc_cohesion = fMaterialData.Cohesion();
    REAL G = fMaterialData.ElasticResponse().G();
    REAL K = fMaterialData.ElasticResponse().K();

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

bool TPZSolveMatrix::ReturnMappingLeftEdge(double *eigenvalues, double *sigma_projected, double &m_hardening) {
    REAL mc_phi = fMaterialData.FrictionAngle();
    REAL mc_psi = mc_phi;
    REAL mc_cohesion = fMaterialData.Cohesion();
    REAL G = fMaterialData.ElasticResponse().G();
    REAL K = fMaterialData.ElasticResponse().K();

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

void TPZSolveMatrix::ReturnMappingApex(double *eigenvalues, double *sigma_projected, double &m_hardening) {
    REAL mc_phi = fMaterialData.FrictionAngle();
    REAL mc_psi = mc_phi;
    REAL mc_cohesion = fMaterialData.Cohesion();
    REAL K = fMaterialData.ElasticResponse().K();

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

//Gather solution
void TPZSolveMatrix::GatherSolution(TPZFMatrix<REAL> &global_solution, TPZFMatrix<REAL> &gather_solution) {
    int64_t n_globalsol = fDim*fCol;

    gather_solution.Resize(n_globalsol,1);
    gather_solution.Zero();

    cblas_dgthr(n_globalsol, global_solution, &gather_solution(0,0), &fIndexes[0]);

}

//Strain
void TPZSolveMatrix::DeltaStrain(TPZFMatrix<REAL> &expandsolution, TPZFMatrix<REAL> &delta_strain) {
    int64_t nelem = fRowSizes.size();
    int64_t n_globalsol = fDim*fCol;

    delta_strain.Resize(2*n_globalsol,1);
    delta_strain.Zero();

    for (int64_t iel = 0; iel < nelem; iel++) {
        for (int i = 0; i < fRowSizes[iel]; i++) {
            for (int j = 0; j < 1; j++) {
                for (int k = 0; k < fColSizes[iel]; k++) {
                    delta_strain(j * fRowSizes[iel] + i + fRowFirstIndex[iel], 0) += fStorage[k * fRowSizes[iel] + i + fMatrixPosition[iel]] * expandsolution(j * fColSizes[iel] + k + fColFirstIndex[iel],0);
                    delta_strain(j * fRowSizes[iel] + i + fRowFirstIndex[iel] + n_globalsol, 0) += fStorage[k * fRowSizes[iel] + i + fMatrixPosition[iel]] * expandsolution(j * fColSizes[iel] + k + fColFirstIndex[iel] + n_globalsol/2,0);
                }
            }
        }
    }
}

void TPZSolveMatrix::TotalStrain (TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &total_strain) {
    total_strain = total_strain + delta_strain;
}

void TPZSolveMatrix::ElasticStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &total_strain, TPZFMatrix<REAL> &plastic_strain, TPZFMatrix<REAL> &elastic_strain) {
    elastic_strain = total_strain - plastic_strain;
}

//Compute stress
void TPZSolveMatrix::ComputeStress(TPZFMatrix<REAL> &elastic_strain, TPZFMatrix<REAL> &sigma) {
    REAL E = fMaterialData.ElasticResponse().E();
    REAL nu = fMaterialData.ElasticResponse().Poisson();
    int64_t npts = fRow/fDim;
    sigma.Resize(4*npts,1);

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
        sigma(4*ipts,0) = fWeight[ipts]*(elastic_strain(2*ipts,0)*E*(1.-nu)/((1.-2*nu)*(1.+nu)) + elastic_strain(2*ipts+2*npts+1,0)*E*nu/((1.-2*nu)*(1.+nu))); // Sigma xx
        sigma(4*ipts+1,0) = fWeight[ipts]*(elastic_strain(2*ipts+2*npts+1,0)*E*(1.-nu)/((1.-2*nu)*(1.+nu)) + elastic_strain(2*ipts,0)*E*nu/((1.-2*nu)*(1.+nu))); // Sigma yy
        sigma(4*ipts+2,0) = fWeight[ipts]*(E*nu/((1.+nu)*(1.-2*nu))*(elastic_strain(2*ipts,0) + elastic_strain(2*ipts+2*npts+1,0))); // Sigma zz
        sigma(4*ipts+3,0) = fWeight[ipts]*E/(2*(1.+nu))*(elastic_strain(2*ipts+1,0)+elastic_strain(2*ipts+2*npts,0)); // Sigma xy

//    sigma(2*ipts,0) = weight[ipts]*E/(1.-nu*nu)*(result(2*ipts,0)+nu*result(2*ipts+npts_tot+1,0)); // Sigma x
//    sigma(2*ipts+1,0) = weight[ipts]*E/(1.-nu*nu)*(1.-nu)/2*(result(2*ipts+1,0)+result(2*ipts+npts_tot,0))*0.5; // Sigma xy
//    sigma(2*ipts+npts_tot,0) = sigma(2*ipts+1,0); //Sigma xy
//    sigma(2*ipts+npts_tot+1,0) = weight[ipts]*E/(1.-nu*nu)*(result(2*ipts+npts_tot+1,0)+nu*result(2*ipts,0)); // Sigma y
    }
#endif
}

//Compute strain
void TPZSolveMatrix::ComputeStrain(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &elastic_strain) {
    REAL E = fMaterialData.ElasticResponse().E();
    REAL nu = fMaterialData.ElasticResponse().Poisson();
    int64_t npts = fRow/fDim;

    for (int ipts = 0; ipts < npts; ipts++) {
        elastic_strain(2 * ipts + 0, 0) = 1. / E * (sigma(4*ipts + 0, 0) - nu * (sigma(4*ipts + 1, 0) + sigma(4*ipts + 2, 0))); //exx
        elastic_strain(2 * ipts + 1, 0) = (1. + nu) / E * sigma(4*ipts + 3, 0); //exy
        elastic_strain(2 * ipts + fRow + 0, 0) = (1. + nu) / E * sigma(4*ipts + 3, 0); //exy
        elastic_strain(2 * ipts + fRow + 1, 0) = 1. / E * (sigma(4*ipts + 1, 0) - nu * (sigma(4*ipts + 0, 0) + sigma(4*ipts + 2, 0))); //eyy
    }
}

void TPZSolveMatrix::SpectralDecomposition(TPZFMatrix<REAL> &sigma_trial, TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &eigenvectors) {
    int64_t npts = fRow/fDim;

    REAL maxel;
    TPZVec<REAL> interval(2);
    eigenvalues.Resize(3*npts,1);
    eigenvectors.Resize(9*npts,1);

    for (int64_t ipts = 0; ipts < npts; ipts++) {
        Normalize(&sigma_trial(4*ipts, 0), maxel);
        Interval(&sigma_trial(4*ipts, 0), &interval[0]);
        NewtonIterations(&interval[0], &sigma_trial(4*ipts, 0), &eigenvalues(3*ipts, 0), maxel);
        Eigenvectors(&sigma_trial(4*ipts, 0), &eigenvalues(3*ipts, 0), &eigenvectors(9*ipts,0));
    }
}

void TPZSolveMatrix::ProjectSigma(TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &sigma_projected, TPZFMatrix<REAL> &plastic_strain) {
    int npts = fRow/fDim;

    REAL mc_psi = fMaterialData.FrictionAngle();

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

void TPZSolveMatrix::StressCompleteTensor(TPZFMatrix<REAL> &sigma_projected, TPZFMatrix<REAL> &eigenvectors, TPZFMatrix<REAL> &sigma){
    int npts = fRow/fDim;
    sigma.Resize(4*npts,1);

    for (int ipts = 0; ipts < npts; ipts++) {
        sigma(4*ipts + 0,0) = sigma_projected(3*ipts + 0,0)*eigenvectors(9*ipts + 0,0)*eigenvectors(9*ipts + 0,0) + sigma_projected(3*ipts + 1,0)*eigenvectors(9*ipts + 3,0)*eigenvectors(9*ipts + 3,0) + sigma_projected(3*ipts + 2,0)*eigenvectors(9*ipts + 6,0)*eigenvectors(9*ipts + 6,0);
        sigma(4*ipts + 1,0) = sigma_projected(3*ipts + 0,0)*eigenvectors(9*ipts + 1,0)*eigenvectors(9*ipts + 1,0) + sigma_projected(3*ipts + 1,0)*eigenvectors(9*ipts + 4,0)*eigenvectors(9*ipts + 4,0) + sigma_projected(3*ipts + 2,0)*eigenvectors(9*ipts + 7,0)*eigenvectors(9*ipts + 7,0);
        sigma(4*ipts + 2,0) = sigma_projected(3*ipts + 0,0)*eigenvectors(9*ipts + 2,0)*eigenvectors(9*ipts + 2,0) + sigma_projected(3*ipts + 1,0)*eigenvectors(9*ipts + 5,0)*eigenvectors(9*ipts + 5,0) + sigma_projected(3*ipts + 2,0)*eigenvectors(9*ipts + 8,0)*eigenvectors(9*ipts + 8,0);
        sigma(4*ipts + 3,0) = sigma_projected(3*ipts + 0,0)*eigenvectors(9*ipts + 0,0)*eigenvectors(9*ipts + 1,0) + sigma_projected(3*ipts + 1,0)*eigenvectors(9*ipts + 3,0)*eigenvectors(9*ipts + 4,0) + sigma_projected(3*ipts + 2,0)*eigenvectors(9*ipts + 6,0)*eigenvectors(9*ipts + 7,0);
    }
}

void TPZSolveMatrix::NodalForces(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &nodal_forces) {
    int64_t nelem = fRowSizes.size();
    int64_t npts = fRow/fDim;
    nodal_forces.Resize(fRow,1);
    nodal_forces.Zero();

    for (int iel = 0; iel < nelem; iel++) {
        TPZFMatrix<REAL> sigma_x(fRowSizes[iel], 1);
        TPZFMatrix<REAL> sigma_y(fRowSizes[iel], 1);

        for (int icol = 0; icol < fColSizes[iel]; icol++) {
            sigma_x(2 * icol, 0) = sigma(4 * icol + 4 * fColFirstIndex[iel] + 0, 0);
            sigma_x(2 * icol + 1, 0) = sigma(4 * icol + 4 * fColFirstIndex[iel] + 3, 0);

            sigma_y(2 * icol, 0) = sigma(4 * icol + 4 * fColFirstIndex[iel] + 3, 0);
            sigma_y(2 * icol + 1, 0) = sigma(4 * icol + 4 * fColFirstIndex[iel] + 1, 0);
        }

        for (int i = 0; i < fColSizes[iel]; i++) {
            for (int k = 0; k < fRowSizes[iel]; k++) {
                nodal_forces(i + fColFirstIndex[iel], 0) += fStorage[k + i * fRowSizes[iel] + fMatrixPosition[iel]] * sigma_x(k, 0);
                nodal_forces(i + fColFirstIndex[iel] + npts, 0) +=  fStorage[k + i * fRowSizes[iel] + fMatrixPosition[iel]] * sigma_y(k, 0);
            }
        }
    }
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


