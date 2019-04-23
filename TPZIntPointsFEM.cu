#include "TPZIntPointsFEM.h"
#include "TPZTensor.h"
#include "pzmatrix.h"
#include <stdlib.h>
#include "TPZTensor.h"
#include "TPZVTKGeoMesh.h"
#include "pzintel.h"
#include "pzskylstrmatrix.h"
#include "TPZVecGPU.h"

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>


#ifdef USING_MKL
#include <mkl.h>
#include <algorithm>
#endif

#define NT 128



__device__ void Normalize(REAL *sigma, REAL &maxel) {
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

__device__ void Interval(REAL *sigma, REAL *interval) {
	__shared__ REAL lower_vec[3];
	__shared__ REAL upper_vec[3];

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

__device__ void NewtonIterations(REAL *interval, REAL *sigma, REAL *eigenvalues, REAL &maxel) {
    int numiterations = 20;
    REAL tol = 10e-12;

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

    //sorting in descending order
    for (int i = 0; i < 3; ++i) {
		for (int j = i + 1; j < 3; ++j) {
			if (eigenvalues[i] < eigenvalues[j]) {
				REAL a = eigenvalues[i];
				eigenvalues[i] = eigenvalues[j];
				eigenvalues[j] = a;
			}
		}
	}
}

__device__ void Multiplicity1(REAL *sigma, REAL eigenvalue, REAL *eigenvector) {
    __shared__ REAL det[3];
    det[0] = (sigma[0] - eigenvalue)*(sigma[1] - eigenvalue) - sigma[3]*sigma[3];
    det[1] = (sigma[0] - eigenvalue)*(sigma[2] - eigenvalue);
    det[2] = (sigma[1] - eigenvalue)*(sigma[2] - eigenvalue);

    REAL maxdet = fabs(det[0]);
    for (int i = 1; i < 3; i++) {
        if (fabs(det[i]) > fabs(maxdet)) {
            maxdet = fabs(det[i]);
        }
    }
    __shared__ REAL v[3];
    if (maxdet == fabs(det[0])) {
        v[0] = 0;
        v[1] = 0;
        v[2] = 1;

    }
    else if (maxdet == fabs(det[1])) {
        v[0] = 1/det[1]*(-(sigma[2] - eigenvalue)*sigma[3]);
        v[1] = 1;
        v[2] = 0;

    }
    else {
        v[0] = 1;
        v[1] = 1/det[2]*(-(sigma[2] - eigenvalue)*sigma[3]);
        v[2] = 0;
    }
    REAL norm = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    eigenvector[0] = v[0]/norm;
    eigenvector[1] = v[1]/norm;
    eigenvector[2] = v[2]/norm;
}

__device__ void Multiplicity2(REAL *sigma, REAL eigenvalue, REAL *eigenvector1, REAL *eigenvector2) {
    __shared__ REAL x[3];
    x[0] = sigma[0] - eigenvalue;
    x[1] = sigma[1] - eigenvalue;
    x[2] = sigma[2] - eigenvalue;

    REAL maxx = fabs(x[0]);
    for (int i = 1; i < 3; i++) {
        if (fabs(x[i]) > fabs(maxx)) {
            maxx = fabs(x[i]);
        }
    }

    __shared__ REAL v1[3];
    __shared__ REAL v2[3];

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
    REAL norm1 = sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]);
    REAL norm2 = sqrt(v2[0]*v2[0] + v2[1]*v1[1] + v2[2]*v2[2]);

    eigenvector1[0] = v1[0]/norm1;
    eigenvector1[1] = v1[1]/norm1;
    eigenvector1[2] = v1[2]/norm1;

    eigenvector2[0] = v2[0]/norm2;
    eigenvector2[1] = v2[1]/norm2;
    eigenvector2[2] = v2[2]/norm2;
}

__device__ void Eigenvectors(REAL *sigma, REAL *eigenvalues, REAL *eigenvectors, REAL &maxel) {
    sigma[0]*=maxel;
    sigma[1]*=maxel;
    sigma[2]*=maxel;
    sigma[3]*=maxel;

    if ((eigenvalues[0] == eigenvalues[1]) && (eigenvalues[1] == eigenvalues[2])) {
        eigenvectors[0] = 1.;
        eigenvectors[1] = 0.;
        eigenvectors[2] = 0.;

        eigenvectors[3] = 0.;
        eigenvectors[4] = 1.;
        eigenvectors[5] = 0.;

        eigenvectors[6] = 0.;
        eigenvectors[7] = 0.;
        eigenvectors[8] = 1.;
    }
    else {
        if (eigenvalues[0] != eigenvalues[1] && eigenvalues[0] != eigenvalues[2]) {
            Multiplicity1(sigma, eigenvalues[0], &eigenvectors[0]);
        } else if (eigenvalues[0] == eigenvalues[1]) {
            Multiplicity2(sigma, eigenvalues[0], &eigenvectors[0], &eigenvectors[3]);
        } else if (eigenvalues[0] == eigenvalues[2]) {
            Multiplicity2(sigma, eigenvalues[0], &eigenvectors[0], &eigenvectors[6]);
        }
        if (eigenvalues[1] != eigenvalues[0] && eigenvalues[1] != eigenvalues[2]) {
            Multiplicity1(sigma, eigenvalues[1], &eigenvectors[3]);
        } else if (eigenvalues[1] == eigenvalues[2]) {
            Multiplicity2(sigma, eigenvalues[1], &eigenvectors[3], &eigenvectors[6]);
        }
        if (eigenvalues[2] != eigenvalues[0] && eigenvalues[2] != eigenvalues[1]) {
            Multiplicity1(sigma, eigenvalues[2], &eigenvectors[6]);
        }
    }
}

__global__ void DeltaStrainKernel(int64_t nelem, REAL *storage, int *rowsizes, int *colsizes, int *matrixpos, int *rowfirstindex, int* colfirstindex, int npts, int nphis, REAL *gather_solution, REAL *delta_strain) {
    int iel = blockIdx.x * blockDim.x + threadIdx.x;

    if(iel < nelem) {
        for (int i = 0; i < rowsizes[iel]; i++) {
            for (int k = 0; k < colsizes[iel]; k++) {
                delta_strain[i + rowfirstindex[iel]] += storage[k * rowsizes[iel] + i + matrixpos[iel]] * gather_solution[k + colfirstindex[iel]];
                delta_strain[i + rowfirstindex[iel] + npts] += storage[k * rowsizes[iel] + i + matrixpos[iel]] * gather_solution[k + colfirstindex[iel] + nphis];
            }
        }
    }
}

__global__ void NodalForcesKernel(int64_t nelem, REAL *storage, int *rowsizes, int *colsizes, int *matrixpos, int *rowfirstindex, int* colfirstindex, int npts, int nphis, REAL *sigma, REAL *nodal_forces) {
    int iel = blockIdx.x * blockDim.x + threadIdx.x;

    if(iel < nelem) {
        for (int i = 0; i < colsizes[iel]; i++) {
            for (int k = 0; k < rowsizes[iel]; k++) {
                nodal_forces[i + colfirstindex[iel]] -= storage[k + i * rowsizes[iel] + matrixpos[iel]] * sigma[k + rowfirstindex[iel]];
                nodal_forces[i + colfirstindex[iel] + nphis] -=  storage[k + i * rowsizes[iel] + matrixpos[iel]] * sigma[k + rowfirstindex[iel] + npts];
            }
        }
    }
}

__global__ void ComputeStressKernel(int64_t fNpts, int fDim, REAL *elastic_strain, REAL *sigma, REAL mu, REAL lambda){
    int ipts = blockIdx.x * blockDim.x + threadIdx.x;

	if (ipts < fNpts/fDim) {
	    sigma[4 * ipts] = elastic_strain[2 * ipts] * (lambda + 2. * mu) + elastic_strain[2 * ipts + fNpts + 1] * lambda; // Sigma xx
	    sigma[4 * ipts + 1] = elastic_strain[2 * ipts + fNpts + 1] * (lambda + 2. * mu) + elastic_strain[2 * ipts] * lambda; // Sigma yy
	    sigma[4 * ipts + 2] = lambda * (elastic_strain[2 * ipts] + elastic_strain[2 * ipts + fNpts + 1]); // Sigma zz
	    sigma[4 * ipts + 3] = mu * (elastic_strain[2 * ipts + 1] + elastic_strain[2 * ipts + fNpts]); // Sigma xy
	}
}

__global__ void ComputeStrainKernel(int64_t fNpts, int fDim, REAL *sigma, REAL *elastic_strain, REAL nu, REAL E, REAL *weight) {
    int ipts = blockIdx.x * blockDim.x + threadIdx.x;

	if (ipts < fNpts/fDim) {
        elastic_strain[2 * ipts + 0] = 1 / weight[ipts] * (1. / E * (sigma[2 * ipts] * (1. - nu * nu) - sigma[2 * ipts + fNpts + 1] * (nu + nu * nu))); //exx
        elastic_strain[2 * ipts + 1] = 1 / weight[ipts] * ((1. + nu) / E * sigma[2 * ipts + 1]); //exy
        elastic_strain[2 * ipts + fNpts + 0] = elastic_strain[2 * ipts + 1]; //exy
        elastic_strain[2 * ipts + fNpts + 1] = 1 / weight[ipts] * (1. / E * (sigma[2 * ipts + fNpts + 1] * (1. - nu * nu) - sigma[2 * ipts] * (nu + nu * nu))); //eyy
    }
}

__global__ void SpectralDecompositionKernel(int64_t fNpts, int fDim, REAL *sigma_trial, REAL *eigenvalues, REAL *eigenvectors) {
	int ipts = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ REAL maxel;
	__shared__ REAL interval[2];
	if (ipts < fNpts/fDim) {
		Normalize(&sigma_trial[4 * ipts], maxel);
		Interval(&sigma_trial[4 * ipts], &interval[0]);
		NewtonIterations(&interval[0], &sigma_trial[4 * ipts], &eigenvalues[3 * ipts], maxel);
		Eigenvectors(&sigma_trial[4 * ipts], &eigenvalues[3 * ipts], &eigenvectors[9 * ipts], maxel);
	}
}

__device__ bool PhiPlane(REAL *eigenvalues, REAL *sigma_projected, REAL mc_phi, REAL mc_cohesion) {
    const REAL sinphi = sin(mc_phi);
    const REAL cosphi = cos(mc_phi);

    REAL phi = eigenvalues[0] - eigenvalues[2] + (eigenvalues[0] + eigenvalues[2]) * sinphi - 2. * mc_cohesion *cosphi;

    sigma_projected[0] = eigenvalues[0];
    sigma_projected[1] = eigenvalues[1];
    sigma_projected[2] = eigenvalues[2];

    bool check_validity = (fabs(phi) < 1.e-12) || (phi < 0.0);
    return check_validity;
}

__device__ bool ReturnMappingMainPlane(REAL *eigenvalues, REAL *sigma_projected, REAL &m_hardening, REAL mc_phi, REAL mc_psi, REAL mc_cohesion, REAL K, REAL G) {
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

    eigenvalues[0] -= (2. * G *(1 + sinpsi / 3.) + 2. * K * sinpsi) * gamma;
    eigenvalues[1] += (4. * G / 3. - K * 2.) * sinpsi * gamma;
    eigenvalues[2] += (2. * G * (1 - sinpsi / 3.) - 2. * K * sinpsi) * gamma;
    sigma_projected[0] = eigenvalues[0];
    sigma_projected[1] = eigenvalues[1];
    sigma_projected[2] = eigenvalues[2];

    m_hardening += gamma * 2. * cosphi;

    bool check_validity = (eigenvalues[0] > eigenvalues[1] || fabs(eigenvalues[0]-eigenvalues[1]) < 1.e-12) && (eigenvalues[1] > eigenvalues[2] || fabs(eigenvalues[1]-eigenvalues[2]) < 1.e-12);
    return check_validity;
}

__device__ bool ReturnMappingRightEdge(REAL *eigenvalues, REAL *sigma_projected, REAL &m_hardening, REAL mc_phi, REAL mc_psi, REAL mc_cohesion, REAL K, REAL G) {
    const REAL sinphi = sin(mc_phi);
    const REAL sinpsi = sin(mc_psi);
    const REAL cosphi = cos(mc_phi);

    __shared__ REAL gamma[2], phi[2], sigma_bar[2], ab[2];

    __shared__ REAL jac[2][2], jac_inv[2][2];


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

    eigenvalues[0] -= (2. * G * (1 + sinpsi / 3.) + 2. * K * sinpsi) * (gamma[0] + gamma[1]);
    eigenvalues[1] += ((4. * G / 3. - K * 2.) * sinpsi) * gamma[0] + (2. * G * (1. - sinpsi / 3.) - 2. * K * sinpsi) * gamma[1];
    eigenvalues[2] += (2. * G * (1 - sinpsi / 3.) - 2. * K * sinpsi) * gamma[0] + ((4. * G / 3. - 2. * K) * sinpsi) * gamma[1];
    sigma_projected[0] = eigenvalues[0];
    sigma_projected[1] = eigenvalues[1];
    sigma_projected[2] = eigenvalues[2];

    m_hardening += (gamma[0] + gamma[1]) * 2. * cosphi;

    bool check_validity = (eigenvalues[0] > eigenvalues[1] || fabs(eigenvalues[0]-eigenvalues[1]) < 1.e-12) && (eigenvalues[1] > eigenvalues[2] || fabs(eigenvalues[1]-eigenvalues[2]) < 1.e-12);
    return check_validity;
}

__device__ bool ReturnMappingLeftEdge(REAL *eigenvalues, REAL *sigma_projected, REAL &m_hardening, REAL mc_phi, REAL mc_psi, REAL mc_cohesion, REAL K, REAL G) {
    const REAL sinphi = sin(mc_phi);
    const REAL sinpsi = sin(mc_psi);
    const REAL cosphi = cos(mc_phi);
    const REAL sinphi2 = sinphi*sinphi;
    const REAL cosphi2 = 1. - sinphi2;

    __shared__ REAL gamma[2], phi[2], sigma_bar[2], ab[2];

    __shared__ REAL jac[2][2], jac_inv[2][2];

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

    eigenvalues[0] += -(2. * G * (1 + sinpsi / 3.) + 2. * K * sinpsi) * gamma[0] + ((4. * G / 3. - 2. * K) * sinpsi) * gamma[1];
    eigenvalues[1] += ((4. * G / 3. - K * 2.) * sinpsi) * gamma[0] - (2. * G * (1. + sinpsi / 3.) + 2. * K * sinpsi) * gamma[1];
    eigenvalues[2] += (2. * G * (1 - sinpsi / 3.) - 2. * K * sinpsi) * (gamma[0] + gamma[1]);
    sigma_projected[0] = eigenvalues[0];
    sigma_projected[1] = eigenvalues[1];
    sigma_projected[2] = eigenvalues[2];

    m_hardening += (gamma[0] + gamma[1]) * 2. * cosphi;

    bool check_validity = (eigenvalues[0] > eigenvalues[1] || fabs(eigenvalues[0]-eigenvalues[1]) < 1.e-12) && (eigenvalues[1] > eigenvalues[2] || fabs(eigenvalues[1]-eigenvalues[2]) < 1.e-12);
    return check_validity;
}

__device__ bool ReturnMappingApex(REAL *eigenvalues, REAL *sigma_projected, REAL &m_hardening, REAL mc_phi, REAL mc_psi, REAL mc_cohesion, REAL K) {
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

__global__ void ProjectSigmaKernel(int64_t fNpts, int fDim, REAL mc_phi, REAL mc_psi, REAL mc_cohesion, REAL K, REAL G, REAL *eigenvalues, REAL *sigma_projected, REAL *m_type, REAL *alpha) {
	int ipts = blockIdx.x * blockDim.x + threadIdx.x;

    bool check = false;
	if (ipts < fNpts/fDim) {
        m_type[ipts] = 0;
        check = PhiPlane(&eigenvalues[3*ipts], &sigma_projected[3*ipts], mc_phi, mc_cohesion); //elastic domain
        if (!check) { //plastic domain
            m_type[ipts] = 1;
            check = ReturnMappingMainPlane(&eigenvalues[3*ipts], &sigma_projected[3*ipts], alpha[ipts], mc_phi, mc_psi, mc_cohesion, K, G); //main plane
            if (!check) { //edges or apex
                if  (((1 - sin(mc_psi)) * eigenvalues[0 + 3*ipts] - 2. * eigenvalues[1 + 3*ipts] + (1 + sin(mc_psi)) * eigenvalues[2 + 3*ipts]) > 0) { // right edge
                    check = ReturnMappingRightEdge(&eigenvalues[3*ipts], &sigma_projected[3*ipts], alpha[ipts], mc_phi, mc_psi, mc_cohesion, K, G);
                } else { //left edge
                    check = ReturnMappingLeftEdge(&eigenvalues[3*ipts], &sigma_projected[3*ipts], alpha[ipts], mc_phi, mc_psi, mc_cohesion, K, G);
                }
                if (!check) { //apex
                    m_type[ipts] = -1;
                    ReturnMappingApex(&eigenvalues[3*ipts], &sigma_projected[3*ipts], alpha[ipts], mc_phi, mc_psi, mc_cohesion, K);
                }
            }
        }
    }
}

__global__ void StressCompleteTensorKernel(int64_t fNpts, int fDim, REAL *sigma_projected, REAL *eigenvectors, REAL *sigma, REAL *weight) {
    int ipts = blockIdx.x * blockDim.x + threadIdx.x;

	if (ipts < fNpts/fDim) {
	    sigma[2*ipts + 0] = weight[ipts]*(sigma_projected[3*ipts + 0]*eigenvectors[9*ipts + 0]*eigenvectors[9*ipts + 0] + sigma_projected[3*ipts + 1]*eigenvectors[9*ipts + 3]*eigenvectors[9*ipts + 3] + sigma_projected[3*ipts + 2]*eigenvectors[9*ipts + 6]*eigenvectors[9*ipts + 6]);
	    sigma[2*ipts + 1] = weight[ipts]*(sigma_projected[3*ipts + 0]*eigenvectors[9*ipts + 0]*eigenvectors[9*ipts + 1] + sigma_projected[3*ipts + 1]*eigenvectors[9*ipts + 3]*eigenvectors[9*ipts + 4] + sigma_projected[3*ipts + 2]*eigenvectors[9*ipts + 6]*eigenvectors[9*ipts + 7]);
	    sigma[2*ipts + fNpts] = sigma[2*ipts + 1];
	    sigma[2*ipts + fNpts + 1] = weight[ipts]*(sigma_projected[3*ipts + 0]*eigenvectors[9*ipts + 1]*eigenvectors[9*ipts + 1] + sigma_projected[3*ipts + 1]*eigenvectors[9*ipts + 4]*eigenvectors[9*ipts + 4] + sigma_projected[3*ipts + 2]*eigenvectors[9*ipts + 7]*eigenvectors[9*ipts + 7]);
	}
}

//Gather solution
void TPZIntPointsFEM::GatherSolutionGPU(TPZVecGPU<REAL> &global_solution, TPZVecGPU<REAL> &gather_solution) {
	gather_solution.Resize(fNpts);
	gather_solution.Zero();
    cusparseDgthr(handle_cusparse, fDim*fNphis, global_solution.GetData(), &gather_solution.GetData()[0], &dIndexes.GetData()[0], CUSPARSE_INDEX_BASE_ZERO);
}

//Strain
void TPZIntPointsFEM::DeltaStrainGPU(TPZVecGPU<REAL> &gather_solution, TPZVecGPU<REAL> &delta_strain) {
    int64_t nelem = fRowSizes.size();

    delta_strain.Resize(fDim*fNpts);
    delta_strain.Zero();

    int numBlocks = (nelem + NT - 1)/NT;
    DeltaStrainKernel <<< numBlocks, NT >>> (nelem, dStorage.GetData(), dRowSizes.GetData(), dColSizes.GetData(), dMatrixPosition.GetData(), dRowFirstIndex.GetData(), dColFirstIndex.GetData(), fNpts, fNphis, gather_solution.GetData(), delta_strain.GetData());
    cudaDeviceSynchronize();
}

void TPZIntPointsFEM::ElasticStrainGPU(TPZVecGPU<REAL> &delta_strain, TPZVecGPU<REAL> &plastic_strain, TPZVecGPU<REAL> &elastic_strain) {
	elastic_strain.Resize(fDim*fNpts);
	elastic_strain.Zero();

	REAL a = -1.;
	cublasDaxpy(handle_cublas,fDim*fNpts,&a, &plastic_strain.GetData()[0],1,&delta_strain.GetData()[0],1);
	elastic_strain = delta_strain; //because the op is y = ax + y in which y = delta_strain
}

void TPZIntPointsFEM::PlasticStrainGPU(TPZVecGPU<REAL> &delta_strain, TPZVecGPU<REAL> &elastic_strain, TPZVecGPU<REAL> &plastic_strain) {
	REAL a = -1.;
	cublasDaxpy(handle_cublas,fDim*fNpts,&a, &elastic_strain.GetData()[0],1,&delta_strain.GetData()[0],1);
	plastic_strain = delta_strain; //because the op is y = ax + y in which y = delta_strain
}

//Compute stress
void TPZIntPointsFEM::ComputeStressGPU(TPZVecGPU<REAL> &elastic_strain, TPZVecGPU<REAL> &sigma) {
    REAL lambda = fMaterial->GetPlasticModel().fER.Lambda();
    REAL mu = fMaterial->GetPlasticModel().fER.Mu();
    sigma.Resize(fDim*fNpts);

    int numBlocks = (fNpts/fDim + NT - 1)/NT;
    ComputeStressKernel <<< numBlocks, NT >>> (fNpts, fDim, elastic_strain.GetData(), sigma.GetData(), mu, lambda);
    cudaDeviceSynchronize();
}

//Compute strain
void TPZIntPointsFEM::ComputeStrainGPU(TPZVecGPU<REAL> &sigma, TPZVecGPU<REAL> &elastic_strain) {
    REAL E = fMaterial->GetPlasticModel().fER.E();
    REAL nu = fMaterial->GetPlasticModel().fER.Poisson();

    int numBlocks = (fNpts/fDim + NT - 1)/NT;
    ComputeStrainKernel <<< numBlocks, NT >>> (fNpts, fDim, sigma.GetData(), elastic_strain.GetData(), nu, E, dWeight.GetData());
    cudaDeviceSynchronize();
}

void TPZIntPointsFEM::SpectralDecompositionGPU(TPZVecGPU<REAL> &sigma_trial, TPZVecGPU<REAL> &eigenvalues, TPZVecGPU<REAL> &eigenvectors) {

    eigenvalues.Resize(3*fNpts/fDim);
    eigenvectors.Resize(9*fNpts/fDim);

    int numBlocks = (fNpts/fDim + NT - 1)/NT;
    SpectralDecompositionKernel <<< numBlocks, NT >>> (fNpts, fDim, sigma_trial.GetData(), eigenvalues.GetData(), eigenvectors.GetData());
    cudaDeviceSynchronize();
}

void TPZIntPointsFEM::ProjectSigmaGPU(TPZVecGPU<REAL> &eigenvalues, TPZVecGPU<REAL> &sigma_projected) {

	REAL mc_phi = fMaterial->GetPlasticModel().fYC.Phi();
	REAL mc_psi = fMaterial->GetPlasticModel().fYC.Psi();
	REAL mc_cohesion = fMaterial->GetPlasticModel().fYC.Cohesion();
	REAL K = fMaterial->GetPlasticModel().fER.K();
	REAL G = fMaterial->GetPlasticModel().fER.G();

    sigma_projected.Resize(3*fNpts/fDim);
    sigma_projected.Zero();

    TPZVecGPU<REAL> m_type(fNpts/fDim);
    m_type.Zero();

    TPZVecGPU<REAL> alpha(fNpts/fDim);
    alpha.Zero();

    int numBlocks = (fNpts/fDim + NT - 1)/NT;
    ProjectSigmaKernel <<< numBlocks, NT >>> (fNpts, fDim, mc_phi, mc_psi, mc_cohesion, K, G, eigenvalues.GetData(), sigma_projected.GetData(), m_type.GetData(), alpha.GetData());
    cudaDeviceSynchronize();


}

void TPZIntPointsFEM::StressCompleteTensorGPU(TPZVecGPU<REAL> &sigma_projected, TPZVecGPU<REAL> &eigenvectors, TPZVecGPU<REAL> &sigma){
    sigma.Resize(fDim*fNpts);

    int numBlocks = (fNpts/fDim + NT - 1)/NT;
    StressCompleteTensorKernel <<< numBlocks, NT >>> (fNpts, fDim, sigma_projected.GetData(), eigenvectors.GetData(), sigma.GetData(), dWeight.GetData());
    cudaDeviceSynchronize();
}

void TPZIntPointsFEM::NodalForcesGPU(TPZVecGPU<REAL> &sigma, TPZVecGPU<REAL> &nodal_forces) {
    int64_t nelem = fRowSizes.size();

    nodal_forces.Resize(fDim*fNphis);
    nodal_forces.Zero();

    int numBlocks = (nelem + NT - 1)/NT;
    DeltaStrainKernel <<< numBlocks, NT >>> (nelem, dStorage.GetData(), dRowSizes.GetData(), dColSizes.GetData(), dMatrixPosition.GetData(), dRowFirstIndex.GetData(), dColFirstIndex.GetData(), fNpts, fNphis, sigma.GetData(), nodal_forces.GetData());
    cudaDeviceSynchronize();
}

void TPZIntPointsFEM::ColoredAssembleGPU(TPZVecGPU<STATE>  &nodal_forces, TPZVecGPU<STATE> &residual) {
    int64_t ncolor = *std::max_element(fElemColor.begin(), fElemColor.end())+1;
    int64_t sz = fIndexes.size();
    int64_t neq = fCmesh->NEquations();
    residual.Resize(neq*ncolor);
    residual.Zero();

    cusparseDsctr(handle_cusparse, sz, nodal_forces.GetData(), &dIndexesColor.GetData()[0], &residual.GetData()[0], CUSPARSE_INDEX_BASE_ZERO);

    int64_t colorassemb = ncolor / 2.;
    REAL alpha = 1.;
    while (colorassemb > 0) {

        int64_t firsteq = (ncolor - colorassemb) * neq;
        cublasDaxpy(handle_cublas, firsteq, &alpha, &residual.GetData()[firsteq], 1., &residual.GetData()[0], 1.);


        ncolor -= colorassemb;
        colorassemb = ncolor/2;
    }
    residual.Resize(neq);
}

void TPZIntPointsFEM::ColoringElements() const {
    int64_t nelem_c = fCmesh->NElements();
    int64_t nconnects = fCmesh->NConnects();
    TPZVec<int64_t> connects_vec(nconnects,0);

    int64_t contcolor = 0;
    bool needstocontinue = true;

    while (needstocontinue)
    {
        int it = 0;
        needstocontinue = false;
        for (int64_t iel = 0; iel < nelem_c; iel++) {
            TPZCompEl *cel = fCmesh->Element(iel);
            if (!cel || cel->Dimension() != fCmesh->Dimension()) continue;

            it++;
            if (fElemColor[it-1] != -1) continue;

            TPZStack<int64_t> connectlist;
            fCmesh->Element(iel)->BuildConnectList(connectlist);
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
//    ofstream file("colored.vtk");
//    TPZVTKGeoMesh::PrintGMeshVTK(fCmesh->Reference(),file);


    int64_t nelem = fRowSizes.size();
    int64_t neq = fCmesh->NEquations();
    for (int64_t iel = 0; iel < nelem; iel++) {
        int64_t cols = fColSizes[iel];
        int64_t cont_cols = fColFirstIndex[iel];

        for (int64_t icols = 0; icols < cols; icols++) {
            fIndexesColor[cont_cols + icols] = fIndexes[cont_cols + icols] + fElemColor[iel]*neq;
            fIndexesColor[cont_cols+ fNphis + icols] = fIndexes[cont_cols + fNphis + icols] + fElemColor[iel]*neq;
        }
    }
}

void TPZIntPointsFEM::AssembleResidual() {
	TPZVecGPU<REAL> gather_solution;
	TPZVecGPU<REAL> delta_strain;
	TPZVecGPU<REAL> elastic_strain;
	TPZVecGPU<REAL> sigma_trial;
	TPZVecGPU<REAL> eigenvalues;
	TPZVecGPU<REAL> eigenvectors;
	TPZVecGPU<REAL> sigma_projected;
	TPZVecGPU<REAL> sigma;
	TPZVecGPU<REAL> nodal_forces;
	TPZVecGPU<REAL> residual;

	//residual assemble
	int64_t neq = fCmesh->NEquations();

	dSolution.Set(&fSolution(0, 0), neq);
	GatherSolutionGPU(dSolution, gather_solution);
	DeltaStrainGPU(gather_solution, delta_strain);
	ElasticStrainGPU(delta_strain, dPlasticStrain, elastic_strain);
	ComputeStressGPU(elastic_strain, sigma_trial);
	SpectralDecompositionGPU(sigma_trial, eigenvalues, eigenvectors);
	ProjectSigmaGPU(eigenvalues, sigma_projected);
	StressCompleteTensorGPU(sigma_projected, eigenvectors, sigma);
	NodalForcesGPU(sigma, nodal_forces);
	ColoredAssembleGPU(nodal_forces, residual);

	//update strain
	ComputeStrainGPU(sigma, elastic_strain);
	PlasticStrainGPU(delta_strain, elastic_strain, dPlasticStrain);

	REAL a = 1.;
	cublasDaxpy(handle_cublas,neq,&a, &dRhsBoundary.GetData()[0],1,&residual.GetData()[0],1);
	dRhs = residual;

    fRhs.Resize(neq,1);
    dRhs.Get(&fRhs(0,0),neq);
}

void TPZIntPointsFEM::SetDataStructure(){
    int dim_mesh = (fCmesh->Reference())->Dimension(); // Mesh dimension
    this->SetMeshDimension(dim_mesh);
    int64_t nelem_c = fCmesh->NElements(); // Number of computational elements
    std::vector<int64_t> cel_indexes;

// Number of domain geometric elements
    for (int64_t i = 0; i < nelem_c; i++) {
        TPZCompEl *cel = fCmesh->Element(i);
        if (!cel) continue;
        TPZGeoEl *gel = fCmesh->Element(i)->Reference();
        if (!gel) continue;
        if( gel->Dimension() == dim_mesh) cel_indexes.push_back(cel->Index());
        if( gel->Dimension() < dim_mesh) fBoundaryElements.Push(cel->Index());
    }

    if (cel_indexes.size() == 0) {
        DebugStop();
    }

// RowSizes and ColSizes vectors
    int64_t nelem = cel_indexes.size();
    TPZVec<int> rowsizes(nelem);
    TPZVec<int> colsizes(nelem);

    int64_t npts_tot = 0;
    int64_t nf_tot = 0;
    int it = 0;
    for (auto iel : cel_indexes) {
        //Verification
        TPZCompEl *cel = fCmesh->Element(iel);

        //Integration rule
        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement * >(cel);
        if (!cel_inter) DebugStop();
        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());

        int64_t npts = int_rule->NPoints(); // number of integration points of the element
        int64_t dim = cel_inter->Dimension(); //dimension of the element
        int64_t nf = cel_inter->NShapeF(); // number of shape functions of the element

        rowsizes[it] = dim * npts;
        colsizes[it] = nf;

        it++;

        npts_tot += npts;
        nf_tot += nf;
    }
    this->SetNumberofIntPoints(dim_mesh*npts_tot);
    this->SetNumberofPhis(nf_tot);
    this->SetRowandColSizes(rowsizes, colsizes);

// Dphi matrix, weight and indexes vectors
    TPZFMatrix<REAL> elmatrix;
    TPZStack<REAL> weight;
    TPZManVector<int> indexes(dim_mesh * nf_tot);

    int64_t cont1 = 0;
    int64_t cont2 = 0;
    it = 0;
    for (auto iel : cel_indexes) {
        //Verification
        TPZCompEl *cel = fCmesh->Element(iel);

        //Integration rule
        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement * >(cel);
        if (!cel_inter) DebugStop();
        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());

        int64_t npts = int_rule->NPoints(); // number of integration points of the element
        int64_t dim = cel_inter->Dimension(); //dimension of the element
        int64_t nf = cel_inter->NShapeF(); // number of shape functions of the element

        TPZMaterialData data;
        cel_inter->InitMaterialData(data);

        elmatrix.Resize(dim * npts, nf);
        for (int64_t inpts = 0; inpts < npts; inpts++) {
            TPZManVector<REAL> qsi(dim, 1);
            REAL w;
            int_rule->Point(inpts, qsi, w);
            cel_inter->ComputeRequiredData(data, qsi);
            weight.Push(w * std::abs(data.detjac)); //weight = w * detjac

            TPZFMatrix<REAL> axes = data.axes;
            TPZFMatrix<REAL> dphix = data.dphix;
            TPZFMatrix<REAL> dphiXY;
            axes.Transpose();
            axes.Multiply(dphix,dphiXY);

            for (int inf = 0; inf < nf; inf++) {
                for (int idim = 0; idim < dim; idim++)
                    elmatrix(inpts * dim + idim, inf) = dphiXY(idim, inf);
            }
        }
        this->SetElementMatrix(it, elmatrix);
        it++;

        //Indexes vector
        int64_t ncon = cel->NConnects();
        for (int64_t icon = 0; icon < ncon; icon++) {
            int64_t id = cel->ConnectIndex(icon);
            TPZConnect &df = fCmesh->ConnectVec()[id];
            int64_t conid = df.SequenceNumber();
            if (df.NElConnected() == 0 || conid < 0 || fCmesh->Block().Size(conid) == 0) continue;
            else {
                int64_t pos = fCmesh->Block().Position(conid);
                int64_t nsize = fCmesh->Block().Size(conid);
                for (int64_t isize = 0; isize < nsize; isize++) {
                    if (isize % 2 == 0) {
                        indexes[cont1] = pos + isize;
                        cont1++;
                    } else {
                        indexes[cont2 + nf_tot] = pos + isize;
                        cont2++;
                    }
                }
            }
        }
    }
    this->SetIndexes(indexes);
    this->SetWeightVector(weight);
    this->ColoringElements();
    this->AssembleRhsBoundary();

    fPlasticStrain.Resize(fDim * fNpts, 1);
    fPlasticStrain.Zero();

    TransferDataStructure();S
}

void TPZIntPointsFEM::AssembleRhsBoundary() {
    int64_t neq = fCmesh->NEquations();
    fRhsBoundary.Resize(neq, 1);
    fRhsBoundary.Zero();

    for (auto iel : fBoundaryElements) {
        TPZCompEl *cel = fCmesh->Element(iel);
        if (!cel) continue;
        TPZElementMatrix ef(fCmesh, TPZElementMatrix::EF);
        cel->CalcResidual(ef);
        ef.ComputeDestinationIndices();
        fRhsBoundary.AddFel(ef.fMat, ef.fSourceIndex, ef.fDestinationIndex);
    }
}

void TPZIntPointsFEM::TransferDataStructure() {
    cuSparseHandle();
    cuBlasHandle();

    int64_t neq = fCmesh->NEquations();
    int64_t nelem = fColSizes.size();
    int64_t ncolor = *std::max_element(fElemColor.begin(), fElemColor.end())+1;

    dRhs.Resize(neq);
    dRhs.Zero();

    dRhsBoundary.Resize(neq);
    dRhsBoundary.Set(&fRhsBoundary(0,0), neq);

	dSolution.Resize(neq);
	dSolution.Zero();

	dPlasticStrain.Resize(fDim * fNpts);
	dPlasticStrain.Zero();

	dStorage.Resize(fStorage.size());
	dStorage.Set(&fStorage[0], fStorage.size());

	dRowSizes.Resize(nelem);
    dRowSizes.Set(&fRowSizes[0], nelem);

    dColSizes.Resize(nelem);
	dColSizes.Set(&fColSizes[0], nelem);

	dMatrixPosition.Resize(nelem);
	dMatrixPosition.Set(&fMatrixPosition[0], nelem);

	dRowFirstIndex.Resize(nelem);
	dRowFirstIndex.Set(&fRowFirstIndex[0], nelem);

	dColFirstIndex.Resize(nelem);
	dColFirstIndex.Set(&fColFirstIndex[0], nelem);

	dIndexes.Resize(fIndexes.size());
	dIndexes.Set(&fIndexes[0], fIndexes.size());

	dIndexesColor.Resize(fIndexesColor.size());
	dIndexesColor.Set(&fIndexesColor[0], fIndexesColor.size());

	dWeight.Resize(fWeight.size());
	dWeight.Set(&fWeight[0], fWeight.size());
}
