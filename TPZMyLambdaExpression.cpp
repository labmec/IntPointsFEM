//
// Created by natalia on 17/05/19.
//
#include <omp.h>

#include "TPZMyLambdaExpression.h"
#include "SpectralDecomp.h"
#include "SigmaProjection.h"

TPZMyLambdaExpression::TPZMyLambdaExpression() : fMaterial(), fPlasticStrain(0,0), fMType(0,0), fAlpha(0,0), fIntPoints() {
}

TPZMyLambdaExpression::TPZMyLambdaExpression(TPZIntPointsStructMatrix *IntPoints, int materialid) : fMaterial(), fPlasticStrain(0,0), fMType(0,0), fAlpha(0,0), fIntPoints() {
    SetIntPoints(IntPoints);
    SetMaterialId(materialid);
}

TPZMyLambdaExpression::~TPZMyLambdaExpression() {

}

TPZMyLambdaExpression::TPZMyLambdaExpression(const TPZMyLambdaExpression &copy) {
    fMaterial = copy.fMaterial;
    fPlasticStrain = copy.fPlasticStrain;
    fMType = copy.fMType;
    fAlpha = copy.fAlpha;
    fIntPoints = copy.fIntPoints;
}

TPZMyLambdaExpression &TPZMyLambdaExpression::operator=(const TPZMyLambdaExpression &copy) {
    if(&copy == this){
        return *this;
    }

    fMaterial = copy.fMaterial;
    fPlasticStrain = copy.fPlasticStrain;
    fMType = copy.fMType;
    fAlpha = copy.fAlpha;
    fIntPoints = copy.fIntPoints;

    return *this;
}

void TPZMyLambdaExpression::ElasticStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &elastic_strain) {
    int dim = fIntPoints->CompMesh()->Dimension();
    int rows = fIntPoints->BlockMatrix().Rows();

    elastic_strain.Resize(dim * rows, 1);
    elastic_strain.Zero();

    elastic_strain = delta_strain - fPlasticStrain;
}

void TPZMyLambdaExpression::PlasticStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &elastic_strain) {
    int dim = fIntPoints->CompMesh()->Dimension();
    int rows = fIntPoints->BlockMatrix().Rows();

    fPlasticStrain.Resize(dim * rows, 1);

    fPlasticStrain = delta_strain - elastic_strain;
}

void TPZMyLambdaExpression::ComputeStress(TPZFMatrix<REAL> &elastic_strain, TPZFMatrix<REAL> &sigma) {
    int dim = fIntPoints->CompMesh()->Dimension();
    int rows = fIntPoints->BlockMatrix().Rows();

    REAL lambda = fMaterial->GetPlasticModel().fER.Lambda();
    REAL mu =  fMaterial->GetPlasticModel().fER.Mu();
    sigma.Resize(dim*rows,1);

//#pragma omp parallel for
    for (int64_t ipts=0; ipts < rows/dim; ipts++) {
        //plane strain
        sigma(4 * ipts, 0) = elastic_strain(2 * ipts, 0) * (lambda + 2. * mu) + elastic_strain(2 * ipts + rows + 1, 0) * lambda; // Sigma xx
        sigma(4 * ipts + 1, 0) = elastic_strain(2 * ipts + rows + 1, 0) * (lambda + 2. * mu) + elastic_strain(2 * ipts, 0) * lambda; // Sigma yy
        sigma(4 * ipts + 2, 0) = lambda * (elastic_strain(2 * ipts, 0) + elastic_strain(2 * ipts + rows + 1, 0)); // Sigma zz
        sigma(4 * ipts + 3, 0) = mu * (elastic_strain(2 * ipts + 1, 0) + elastic_strain(2 * ipts + rows, 0)); // Sigma xy
    }
}

void TPZMyLambdaExpression::ComputeStrain(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &elastic_strain) {
    int dim = fIntPoints->CompMesh()->Dimension();
    int rows = fIntPoints->BlockMatrix().Rows();

    REAL E = fMaterial->GetPlasticModel().fER.E();
    REAL nu = fMaterial->GetPlasticModel().fER.Poisson();

    TPZVec<REAL> weight;
    weight = fIntPoints->Weight();

//#pragma omp parallel for
    for (int ipts = 0; ipts < rows / dim; ipts++) {
        elastic_strain(2 * ipts + 0, 0) = 1 / weight[ipts] * (1. / E * (sigma(2 * ipts, 0) * (1. - nu * nu) - sigma(2 * ipts + rows + 1, 0) * (nu + nu * nu))); //exx
        elastic_strain(2 * ipts + 1, 0) = 1 / weight[ipts] * ((1. + nu) / E * sigma(2 * ipts + 1, 0)); //exy
        elastic_strain(2 * ipts + rows + 0, 0) = elastic_strain(2 * ipts + 1, 0); //exy
        elastic_strain(2 * ipts + rows + 1, 0) = 1 / weight[ipts] * (1. / E * (sigma(2 * ipts + rows + 1, 0) * (1. - nu * nu) - sigma(2 * ipts, 0) * (nu + nu * nu))); //eyy
    }
}

void TPZMyLambdaExpression::SpectralDecomposition(TPZFMatrix<REAL> &sigma_trial, TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &eigenvectors) {
    int dim = fIntPoints->CompMesh()->Dimension();
    int rows = fIntPoints->BlockMatrix().Rows();

    REAL maxel;
    TPZVec<REAL> interval(2*rows/dim);
    eigenvalues.Resize(3*rows/dim,1);
    eigenvectors.Resize(9*rows/dim,1);

//#pragma omp parallel for private(maxel)
    for (int64_t ipts = 0; ipts < rows/dim; ipts++) {
        Normalize(&sigma_trial(4*ipts, 0), maxel);
        Interval(&sigma_trial(4*ipts, 0), &interval[2*ipts]);
        NewtonIterations(&interval[2*ipts], &sigma_trial(4*ipts, 0), &eigenvalues(3*ipts, 0), maxel);
        Eigenvectors(&sigma_trial(4*ipts, 0), &eigenvalues(3*ipts, 0), &eigenvectors(9*ipts,0),maxel);
    }
}

void TPZMyLambdaExpression::ProjectSigma(TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &sigma_projected) {
    int dim = fIntPoints->CompMesh()->Dimension();
    int rows = fIntPoints->BlockMatrix().Rows();

    REAL mc_psi = fMaterial->GetPlasticModel().fYC.Psi();
    REAL mc_phi = fMaterial->GetPlasticModel().fYC.Phi();
    REAL mc_cohesion = fMaterial->GetPlasticModel().fYC.Cohesion();
    REAL K = fMaterial->GetPlasticModel().fER.K();
    REAL G = fMaterial->GetPlasticModel().fER.G();

    sigma_projected.Resize(3*rows/dim,1);
    sigma_projected.Zero();
    TPZFMatrix<REAL> elastic_strain_np1(dim*rows);


    bool check = false;
//#pragma omp parallel for
    for (int ipts = 0; ipts < rows/dim; ipts++) {
        fMType(ipts,0) = 0;
        check = PhiPlane(&eigenvalues(3*ipts, 0), &sigma_projected(3*ipts, 0), mc_phi, mc_cohesion); //elastic domain
        if (!check) { //plastic domain
            fMType(ipts,0) = 1;
            check = ReturnMappingMainPlane(&eigenvalues(3*ipts, 0), &sigma_projected(3*ipts, 0), fAlpha(ipts,0), mc_phi, mc_psi, mc_cohesion, K, G); //main plane
            if (!check) { //edges or apex
                if  (((1 - sin(mc_psi)) * eigenvalues(0 + 3*ipts, 0) - 2. * eigenvalues(1 + 3*ipts, 0) + (1 + sin(mc_psi)) * eigenvalues(2 + 3*ipts, 0)) > 0) { // right edge
                    check = ReturnMappingRightEdge(&eigenvalues(3*ipts, 0), &sigma_projected(3*ipts, 0), fAlpha(ipts,0), mc_phi, mc_psi, mc_cohesion, K, G);
                } else { //left edge
                    check = ReturnMappingLeftEdge(&eigenvalues(3*ipts, 0), &sigma_projected(3*ipts, 0), fAlpha(ipts,0), mc_phi, mc_psi, mc_cohesion, K, G);
                }
                if (!check) { //apex
                    fMType(ipts,0) = -1;
                    ReturnMappingApex(&eigenvalues(3*ipts, 0), &sigma_projected(3*ipts, 0), fAlpha(ipts,0), mc_phi, mc_psi, mc_cohesion, K);
                }
            }
        }
    }
}

void TPZMyLambdaExpression::StressCompleteTensor(TPZFMatrix<REAL> &sigma_projected, TPZFMatrix<REAL> &eigenvectors, TPZFMatrix<REAL> &sigma){
    int dim = fIntPoints->CompMesh()->Dimension();
    int rows = fIntPoints->BlockMatrix().Rows();

    TPZVec<REAL> weight;
    weight = fIntPoints->Weight();

    sigma.Resize(dim*rows,1);

//#pragma omp parallel for
    for (int ipts = 0; ipts < rows/dim; ipts++) {
        sigma(2*ipts + 0,0) = weight[ipts]*(sigma_projected(3*ipts + 0,0)*eigenvectors(9*ipts + 0,0)*eigenvectors(9*ipts + 0,0) + sigma_projected(3*ipts + 1,0)*eigenvectors(9*ipts + 3,0)*eigenvectors(9*ipts + 3,0) + sigma_projected(3*ipts + 2,0)*eigenvectors(9*ipts + 6,0)*eigenvectors(9*ipts + 6,0));
        sigma(2*ipts + 1,0) = weight[ipts]*(sigma_projected(3*ipts + 0,0)*eigenvectors(9*ipts + 0,0)*eigenvectors(9*ipts + 1,0) + sigma_projected(3*ipts + 1,0)*eigenvectors(9*ipts + 3,0)*eigenvectors(9*ipts + 4,0) + sigma_projected(3*ipts + 2,0)*eigenvectors(9*ipts + 6,0)*eigenvectors(9*ipts + 7,0));
        sigma(2*ipts + rows,0) = sigma(2*ipts + 1,0);
        sigma(2*ipts + rows + 1,0) = weight[ipts]*(sigma_projected(3*ipts + 0,0)*eigenvectors(9*ipts + 1,0)*eigenvectors(9*ipts + 1,0) + sigma_projected(3*ipts + 1,0)*eigenvectors(9*ipts + 4,0)*eigenvectors(9*ipts + 4,0) + sigma_projected(3*ipts + 2,0)*eigenvectors(9*ipts + 7,0)*eigenvectors(9*ipts + 7,0));
    }
}

void TPZMyLambdaExpression::ComputeSigma(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &sigma) {
    TPZFMatrix<REAL> elastic_strain;
    TPZFMatrix<REAL> sigma_trial;
    TPZFMatrix<REAL> eigenvalues;
    TPZFMatrix<REAL> eigenvectors;
    TPZFMatrix<REAL> sigma_projected;

    int dim = fIntPoints->CompMesh()->Dimension();
    int rows = fIntPoints->BlockMatrix().Rows();

    fPlasticStrain.Resize(dim * rows, 1);
    fPlasticStrain.Zero();

    fMType.Resize(rows / dim, 1);
    fMType.Zero();

    fAlpha.Resize(rows / dim, 1);
    fAlpha.Zero();

    // Compute sigma
    ElasticStrain(delta_strain, elastic_strain);
    ComputeStress(elastic_strain, sigma_trial);
    SpectralDecomposition(sigma_trial, eigenvalues, eigenvectors); //check open mp usage in this method
    ProjectSigma(eigenvalues, sigma_projected);
    StressCompleteTensor(sigma_projected, eigenvectors, sigma);

    // Update plastic strain
    ComputeStrain(sigma, elastic_strain);
    PlasticStrain(delta_strain, elastic_strain);
}