#include "TPZIntPointsFEM.h"
#include "TPZTensor.h"
#include "pzmatrix.h"
#include <stdlib.h>
#include "TPZTensor.h"
#include "TPZVTKGeoMesh.h"
#include "pzintel.h"
#include "pzskylstrmatrix.h"
#include <omp.h>
#include "Timer.h"

REAL timeGatherSolution = 0;
REAL timeDeltaStrain = 0;
REAL timeElasticStrain = 0;
REAL timePlasticStrain = 0;
REAL timeComputeStress = 0;
REAL timeComputeStrain = 0;
REAL timeSpectralDecomposition = 0;
REAL timeProjectSigma = 0;
REAL timeStressCompleteTensor = 0;
REAL timeNodalForces = 0;
REAL timeColoredAssemble = 0;

#ifdef USING_MKL
#include <mkl.h>
#include <algorithm>
#endif

#include "SpectralDecomp.h"
#include "SigmaProjection.h"


TPZIntPointsFEM::TPZIntPointsFEM() :
        fDim(-1), fNpts(-1), fNphis(-1), fNColor(-1), fMaterial(0), fRhs(0, 0),
        fRhsBoundary(0, 0), fPlasticStrain(0, 0), fIndexes(0), fIndexesColor(0), fWeight(0), fBMatrix(), fTimer() {

}

TPZIntPointsFEM::TPZIntPointsFEM(TPZIrregularBlockMatrix *Bmatrix, int materialid) :
        fDim(-1), fNpts(-1), fNphis(-1), fNColor(-1), fMaterial(0), fRhs(0, 0),
        fRhsBoundary(0, 0), fPlasticStrain(0, 0), fIndexes(0), fIndexesColor(0), fWeight(0), fBMatrix(), fTimer() {

	SetBMatrix(Bmatrix);
	SetMaterialId(materialid);
}

TPZIntPointsFEM::~TPZIntPointsFEM() {

}

TPZIntPointsFEM::TPZIntPointsFEM(const TPZIntPointsFEM &copy) {
    fDim = copy.fDim;
    fNpts = copy.fNpts;
    fNphis = copy.fNphis;
    fNColor = copy.fNColor;
    fMaterial = copy.fMaterial;
    fRhs = copy.fRhs;
    fRhsBoundary = copy.fRhsBoundary;
    fIndexes = copy.fIndexes;
    fIndexesColor = copy.fIndexesColor;
    fWeight = copy.fWeight;
    fBMatrix = copy.fBMatrix;

    fPlasticStrain = copy.fPlasticStrain;

    fTimer = copy.fTimer;


}

TPZIntPointsFEM &TPZIntPointsFEM::operator=(const TPZIntPointsFEM &copy) {
    if(&copy == this){
        return *this;
    }
    fDim = copy.fDim;
    fNpts = copy.fNpts;
    fNphis = copy.fNphis;
    fNColor = copy.fNColor;
    fMaterial = copy.fMaterial;
    fRhs = copy.fRhs;
    fRhsBoundary = copy.fRhsBoundary;
    fIndexes = copy.fIndexes;
    fIndexesColor = copy.fIndexesColor;
    fWeight = copy.fWeight;
    fBMatrix = copy.fBMatrix;

    fPlasticStrain = copy.fPlasticStrain;

    return *this;
}

void TPZIntPointsFEM::SetTimerConfig(Timer::WhichUnit unit) {
	fTimer.TimerConfig(unit);
}

void TPZIntPointsFEM::GatherSolution(TPZFMatrix<REAL> &global_solution, TPZFMatrix<REAL> &gather_solution) {
    gather_solution.Resize(fNpts,1);
    gather_solution.Zero();

    fTimer.Start();
    cblas_dgthr(fDim*fNphis, global_solution, &gather_solution(0,0), &fIndexes[0]);
    fTimer.Stop();
    timeGatherSolution+= fTimer.ElapsedTime();
}

void TPZIntPointsFEM::ElasticStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &plastic_strain, TPZFMatrix<REAL> &elastic_strain) {
    elastic_strain.Resize(fDim*fNpts,1);
    elastic_strain.Zero();

    plastic_strain.Zero();

    fTimer.Start();
    elastic_strain = delta_strain - plastic_strain;
    fTimer.Stop();
    timeElasticStrain+= fTimer.ElapsedTime();
}

void TPZIntPointsFEM::PlasticStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &elastic_strain, TPZFMatrix<REAL> &plastic_strain) {
    plastic_strain.Resize(fDim*fNpts,1);

    fTimer.Start();
    plastic_strain = delta_strain - elastic_strain;
    fTimer.Stop();
    timePlasticStrain+= fTimer.ElapsedTime();
}

void TPZIntPointsFEM::ComputeStress(TPZFMatrix<REAL> &elastic_strain, TPZFMatrix<REAL> &sigma) {
    REAL lambda = fMaterial->GetPlasticModel().fER.Lambda();
    REAL mu = fMaterial->GetPlasticModel().fER.Mu();
    sigma.Resize(fDim*fNpts,1);

    fTimer.Start();
#pragma omp parallel for
    for (int64_t ipts=0; ipts < fNpts/fDim; ipts++) {
        //plane strain
        sigma(4 * ipts, 0) = elastic_strain(2 * ipts, 0) * (lambda + 2. * mu) + elastic_strain(2 * ipts + fNpts + 1, 0) * lambda; // Sigma xx
        sigma(4 * ipts + 1, 0) = elastic_strain(2 * ipts + fNpts + 1, 0) * (lambda + 2. * mu) + elastic_strain(2 * ipts, 0) * lambda; // Sigma yy
        sigma(4 * ipts + 2, 0) = lambda * (elastic_strain(2 * ipts, 0) + elastic_strain(2 * ipts + fNpts + 1, 0)); // Sigma zz
        sigma(4 * ipts + 3, 0) = mu * (elastic_strain(2 * ipts + 1, 0) + elastic_strain(2 * ipts + fNpts, 0)); // Sigma xy
    }
    fTimer.Stop();
    timeComputeStress+= fTimer.ElapsedTime();
}

void TPZIntPointsFEM::ComputeStrain(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &elastic_strain) {
    REAL E = fMaterial->GetPlasticModel().fER.E();
    REAL nu = fMaterial->GetPlasticModel().fER.Poisson();

    fTimer.Start();
#pragma omp parallel for
    for (int ipts = 0; ipts < fNpts / fDim; ipts++) {
        elastic_strain(2 * ipts + 0, 0) = 1 / fWeight[ipts] * (1. / E * (sigma(2 * ipts, 0) * (1. - nu * nu) - sigma(2 * ipts + fNpts + 1, 0) * (nu + nu * nu))); //exx
        elastic_strain(2 * ipts + 1, 0) = 1 / fWeight[ipts] * ((1. + nu) / E * sigma(2 * ipts + 1, 0)); //exy
        elastic_strain(2 * ipts + fNpts + 0, 0) = elastic_strain(2 * ipts + 1, 0); //exy
        elastic_strain(2 * ipts + fNpts + 1, 0) = 1 / fWeight[ipts] * (1. / E * (sigma(2 * ipts + fNpts + 1, 0) * (1. - nu * nu) - sigma(2 * ipts, 0) * (nu + nu * nu))); //eyy
    }
    fTimer.Stop();
    timeComputeStrain+= fTimer.ElapsedTime();
}

void TPZIntPointsFEM::SpectralDecomposition(TPZFMatrix<REAL> &sigma_trial, TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &eigenvectors) {
    REAL maxel;
    TPZVec<REAL> interval(2*fNpts/fDim);
    eigenvalues.Resize(3*fNpts/fDim,1);
    eigenvectors.Resize(9*fNpts/fDim,1);
    fTimer.Start();

#pragma omp parallel for private(maxel)
    for (int64_t ipts = 0; ipts < fNpts/fDim; ipts++) {
        Normalize(&sigma_trial(4*ipts, 0), maxel);
        Interval(&sigma_trial(4*ipts, 0), &interval[2*ipts]);
        NewtonIterations(&interval[2*ipts], &sigma_trial(4*ipts, 0), &eigenvalues(3*ipts, 0), maxel);
        Eigenvectors(&sigma_trial(4*ipts, 0), &eigenvalues(3*ipts, 0), &eigenvectors(9*ipts,0),maxel);
    }
    fTimer.Stop();
    timeSpectralDecomposition+= fTimer.ElapsedTime();
}

void TPZIntPointsFEM::ProjectSigma(TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &sigma_projected) {
    REAL mc_psi = fMaterial->GetPlasticModel().fYC.Psi();
    REAL mc_phi = fMaterial->GetPlasticModel().fYC.Phi();
    REAL mc_cohesion = fMaterial->GetPlasticModel().fYC.Cohesion();
    REAL K = fMaterial->GetPlasticModel().fER.K();
    REAL G = fMaterial->GetPlasticModel().fER.G();

    sigma_projected.Resize(3*fNpts/fDim,1);
    sigma_projected.Zero();
    TPZFMatrix<REAL> elastic_strain_np1(fDim*fNpts);

    TPZFMatrix<REAL> m_type(fNpts/fDim, 1, 0.);
    TPZFMatrix<REAL> alpha(fNpts/fDim, 1, 0.);
    bool check = false;
    fTimer.Start();
    #pragma omp parallel for
    for (int ipts = 0; ipts < fNpts/fDim; ipts++) {
        m_type(ipts,0) = 0;
        check = PhiPlane(&eigenvalues(3*ipts, 0), &sigma_projected(3*ipts, 0), mc_phi, mc_cohesion); //elastic domain
        if (!check) { //plastic domain
            m_type(ipts,0) = 1;
            check = ReturnMappingMainPlane(&eigenvalues(3*ipts, 0), &sigma_projected(3*ipts, 0), alpha(ipts,0), mc_phi, mc_psi, mc_cohesion, K, G); //main plane
            if (!check) { //edges or apex
                if  (((1 - sin(mc_psi)) * eigenvalues(0 + 3*ipts, 0) - 2. * eigenvalues(1 + 3*ipts, 0) + (1 + sin(mc_psi)) * eigenvalues(2 + 3*ipts, 0)) > 0) { // right edge
                    check = ReturnMappingRightEdge(&eigenvalues(3*ipts, 0), &sigma_projected(3*ipts, 0), alpha(ipts,0), mc_phi, mc_psi, mc_cohesion, K, G);
                } else { //left edge
                    check = ReturnMappingLeftEdge(&eigenvalues(3*ipts, 0), &sigma_projected(3*ipts, 0), alpha(ipts,0), mc_phi, mc_psi, mc_cohesion, K, G);
                }
                if (!check) { //apex
                    m_type(ipts,0) = -1;
                    ReturnMappingApex(&eigenvalues(3*ipts, 0), &sigma_projected(3*ipts, 0), alpha(ipts,0), mc_phi, mc_psi, mc_cohesion, K);
                }
            }
        }
    }
    fTimer.Stop();
    timeProjectSigma+= fTimer.ElapsedTime();
}

void TPZIntPointsFEM::StressCompleteTensor(TPZFMatrix<REAL> &sigma_projected, TPZFMatrix<REAL> &eigenvectors, TPZFMatrix<REAL> &sigma){
    sigma.Resize(fDim*fNpts,1);

    fTimer.Start();
#pragma omp parallel for
    for (int ipts = 0; ipts < fNpts/fDim; ipts++) {
        sigma(2*ipts + 0,0) = fWeight[ipts]*(sigma_projected(3*ipts + 0,0)*eigenvectors(9*ipts + 0,0)*eigenvectors(9*ipts + 0,0) + sigma_projected(3*ipts + 1,0)*eigenvectors(9*ipts + 3,0)*eigenvectors(9*ipts + 3,0) + sigma_projected(3*ipts + 2,0)*eigenvectors(9*ipts + 6,0)*eigenvectors(9*ipts + 6,0));
        sigma(2*ipts + 1,0) = fWeight[ipts]*(sigma_projected(3*ipts + 0,0)*eigenvectors(9*ipts + 0,0)*eigenvectors(9*ipts + 1,0) + sigma_projected(3*ipts + 1,0)*eigenvectors(9*ipts + 3,0)*eigenvectors(9*ipts + 4,0) + sigma_projected(3*ipts + 2,0)*eigenvectors(9*ipts + 6,0)*eigenvectors(9*ipts + 7,0));
        sigma(2*ipts + fNpts,0) = sigma(2*ipts + 1,0);
        sigma(2*ipts + fNpts + 1,0) = fWeight[ipts]*(sigma_projected(3*ipts + 0,0)*eigenvectors(9*ipts + 1,0)*eigenvectors(9*ipts + 1,0) + sigma_projected(3*ipts + 1,0)*eigenvectors(9*ipts + 4,0)*eigenvectors(9*ipts + 4,0) + sigma_projected(3*ipts + 2,0)*eigenvectors(9*ipts + 7,0)*eigenvectors(9*ipts + 7,0));
    }
    fTimer.Stop();
    timeStressCompleteTensor+= fTimer.ElapsedTime();
}

void TPZIntPointsFEM::ColoredAssemble(TPZFMatrix<STATE>  &nodal_forces) {
    int64_t ncolor = fNColor;
    int64_t sz = fIndexes.size();
    int64_t neq = fBMatrix->CompMesh()->NEquations();
    fRhs.Resize(neq*ncolor,1);
    fRhs.Zero();

    fTimer.Start();
    cblas_dsctr(sz, nodal_forces, &fIndexesColor[0], &fRhs(0,0));

    int64_t colorassemb = ncolor / 2.;
    while (colorassemb > 0) {

        int64_t firsteq = (ncolor - colorassemb) * neq;
        cblas_daxpy(colorassemb * neq, 1., &fRhs(firsteq, 0), 1., &fRhs(0, 0), 1.);

        ncolor -= colorassemb;
        colorassemb = ncolor/2;
    }
    fRhs.Resize(neq, 1);

    fRhs = fRhs + fRhsBoundary;

    fTimer.Stop();
    timeColoredAssemble+= fTimer.ElapsedTime();
}

void TPZIntPointsFEM::ComputeSigma(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &sigma) {
    TPZFMatrix<REAL> elastic_strain;
    TPZFMatrix<REAL> sigma_trial;
    TPZFMatrix<REAL> eigenvalues;
    TPZFMatrix<REAL> eigenvectors;
    TPZFMatrix<REAL> sigma_projected;

    ElasticStrain(delta_strain, fPlasticStrain, elastic_strain);
    ComputeStress(elastic_strain, sigma_trial);
    SpectralDecomposition(sigma_trial, eigenvalues, eigenvectors); //check open mp usage in this method
    ProjectSigma(eigenvalues, sigma_projected);
    StressCompleteTensor(sigma_projected, eigenvectors, sigma);

//    update strain
    ComputeStrain(sigma, elastic_strain);
    PlasticStrain(delta_strain, elastic_strain, fPlasticStrain);
}

void TPZIntPointsFEM::SetDataStructure(){
    fDim = fBMatrix->Dimension();
    fNpts = fBMatrix->Rows();
    fNphis = fBMatrix->Cols();

    fWeight.resize(fNpts/fDim);
    fIndexes.resize(fDim * fNphis);

    int64_t cont1 = 0;
    int64_t cont2 = 0;
    int it = 0;
    for (auto iel : fBMatrix->ElemIndexes()) {
        //Verification
        TPZCompEl *cel = fBMatrix->CompMesh()->Element(iel);

        //Integration rule
        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement * >(cel);
        if (!cel_inter) DebugStop();
        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());

        int64_t npts = int_rule->NPoints(); // number of integration points of the element
        int64_t dim = cel_inter->Dimension(); //dimension of the element
        int64_t nf = cel_inter->NShapeF(); // number of shape functions of the element

        TPZMaterialData data;
        cel_inter->InitMaterialData(data);

        for (int64_t inpts = 0; inpts < npts; inpts++) {
            TPZManVector<REAL> qsi(dim, 1);
            REAL w;
            int_rule->Point(inpts, qsi, w);
            cel_inter->ComputeRequiredData(data, qsi);
            fWeight[it] = w * std::abs(data.detjac);
            it++;
        }


        //Indexes vector
        int64_t ncon = cel->NConnects();
        for (int64_t icon = 0; icon < ncon; icon++) {
            int64_t id = cel->ConnectIndex(icon);
            TPZConnect &df = fBMatrix->CompMesh()->ConnectVec()[id];
            int64_t conid = df.SequenceNumber();
            if (df.NElConnected() == 0 || conid < 0 || fBMatrix->CompMesh()->Block().Size(conid) == 0) continue;
            else {
                int64_t pos = fBMatrix->CompMesh()->Block().Position(conid);
                int64_t nsize = fBMatrix->CompMesh()->Block().Size(conid);
                for (int64_t isize = 0; isize < nsize; isize++) {
                    if (isize % 2 == 0) {
                        fIndexes[cont1] = pos + isize;
                        cont1++;
                    } else {
                        fIndexes[cont2 + fNphis] = pos + isize;
                        cont2++;
                    }
                }
            }
        }
    }

    this->ColoringElements();
    this->AssembleRhsBoundary();

    fPlasticStrain.Resize(fDim * fNpts, 1);
    fPlasticStrain.Zero();
}

void TPZIntPointsFEM::ColoringElements()  {
    int64_t nelem_c = fBMatrix->CompMesh()->NElements();
    int64_t nconnects = fBMatrix->CompMesh()->NConnects();
    TPZVec<int64_t> connects_vec(nconnects,0);
    TPZVec<int64_t> elemcolor(fBMatrix->NumBlocks(),-1);

    int64_t contcolor = 0;
    bool needstocontinue = true;

    while (needstocontinue)
    {
        int it = 0;
        needstocontinue = false;
        for (int64_t iel = 0; iel < nelem_c; iel++) {
            TPZCompEl *cel = fBMatrix->CompMesh()->Element(iel);
            if (!cel || cel->Dimension() != fBMatrix->CompMesh()->Dimension()) continue;

            it++;
            if (elemcolor[it-1] != -1) continue;

            TPZStack<int64_t> connectlist;
            fBMatrix->CompMesh()->Element(iel)->BuildConnectList(connectlist);
            int64_t ncon = connectlist.size();

            int64_t icon;
            for (icon = 0; icon < ncon; icon++) {
                if (connects_vec[connectlist[icon]] != 0) break;
            }
            if (icon != ncon) {
                needstocontinue = true;
                continue;
            }
            elemcolor[it-1] = contcolor;
//            cel->Reference()->SetMaterialId(contcolor);

            for (icon = 0; icon < ncon; icon++) {
                connects_vec[connectlist[icon]] = 1;
            }
        }
        contcolor++;
        connects_vec.Fill(0);
    }
//    ofstream file("colored.vtk");
//    TPZVTKGeoMesh::PrintGMeshVTK(fBMatrix->CompMesh()->Reference(),file);
    fNColor = contcolor;
    fIndexesColor.resize(fDim * fNphis);
    int64_t nelem = fBMatrix->NumBlocks();
    int64_t neq = fBMatrix->CompMesh()->NEquations();
    for (int64_t iel = 0; iel < nelem; iel++) {
        int64_t cols = fBMatrix->ColSizes()[iel];
        int64_t cont_cols = fBMatrix->ColFirstIndex()[iel];

        for (int64_t icols = 0; icols < cols; icols++) {
            fIndexesColor[cont_cols + icols] = fIndexes[cont_cols + icols] + elemcolor[iel]*neq;
            fIndexesColor[cont_cols+ fNphis + icols] = fIndexes[cont_cols + fNphis + icols] + elemcolor[iel]*neq;
        }
    }
}

void TPZIntPointsFEM::AssembleRhsBoundary() {
    int64_t neq = fBMatrix->CompMesh()->NEquations();
    fRhsBoundary.Resize(neq, 1);
    fRhsBoundary.Zero();

    for (auto iel : fBMatrix->BoundaryElemIndexes()) {
        TPZCompEl *cel = fBMatrix->CompMesh()->Element(iel);
        if (!cel) continue;
        TPZElementMatrix ef(fBMatrix->CompMesh(), TPZElementMatrix::EF);
        cel->CalcResidual(ef);
        ef.ComputeDestinationIndices();
        fRhsBoundary.AddFel(ef.fMat, ef.fSourceIndex, ef.fDestinationIndex);
    }
}
