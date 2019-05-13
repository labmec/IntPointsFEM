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

ofstream file("timing-cpu.txt");
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
		fDim(-1), fBoundaryElements(), fCmesh(0), fNpts(-1), fNphis(-1), fElemColor(
				0), fMaterial(0), fRhs(0, 0), fRhsBoundary(0, 0), fSolution(0,
				0), fPlasticStrain(0, 0), fStorage(0), fRowSizes(0), fColSizes(
				0), fMatrixPosition(0), fRowFirstIndex(0), fColFirstIndex(0), fIndexes(
				0), fIndexesColor(0), fWeight(),
				fTimer() {

}

TPZIntPointsFEM::TPZIntPointsFEM(TPZCompMesh *cmesh, int materialid) :
		fDim(-1), fBoundaryElements(), fCmesh(0), fNpts(-1), fNphis(-1), fElemColor(
				0), fMaterial(0), fRhs(0, 0), fRhsBoundary(0, 0), fSolution(0,
				0), fPlasticStrain(0, 0), fStorage(0), fRowSizes(0), fColSizes(
				0), fMatrixPosition(0), fRowFirstIndex(0), fColFirstIndex(0), fIndexes(
				0), fIndexesColor(0), fWeight(),
				fTimer() {

	SetCompMesh(cmesh);
	SetMaterialId(materialid);
}

TPZIntPointsFEM::~TPZIntPointsFEM() {

}

TPZIntPointsFEM::TPZIntPointsFEM(const TPZIntPointsFEM &copy) {
	fTimer = copy.fTimer;

    fDim = copy.fDim;
    fBoundaryElements = copy.fBoundaryElements;
    fCmesh = copy.fCmesh;
    fNpts = copy.fNpts;
    fNphis = copy.fNphis;
    fElemColor = copy.fElemColor;
    fMaterial = copy.fMaterial;

    fRhs = copy.fRhs;
    fRhsBoundary = copy.fRhsBoundary;
    fSolution = copy.fSolution;
    fPlasticStrain = copy.fPlasticStrain;
    fStorage = copy.fStorage;
    fColSizes = copy.fColSizes;
    fRowSizes = copy.fRowSizes;
    fMatrixPosition = copy.fMatrixPosition;
    fRowFirstIndex = copy.fRowFirstIndex;
    fColFirstIndex = copy.fColFirstIndex;
    fIndexes = copy.fIndexes;
    fIndexesColor = copy.fIndexesColor;
    fWeight = copy.fWeight;
}

TPZIntPointsFEM &TPZIntPointsFEM::operator=(const TPZIntPointsFEM &copy) {
    if(&copy == this){
        return *this;
    }
	fTimer = copy.fTimer;

    fDim = copy.fDim;
    fBoundaryElements = copy.fBoundaryElements;
    fCmesh = copy.fCmesh;
    fNpts = copy.fNpts;
    fNphis = copy.fNphis;
    fElemColor = copy.fElemColor;
    fMaterial = copy.fMaterial;

    fRhs = copy.fRhs;
    fRhsBoundary = copy.fRhsBoundary;
    fSolution = copy.fSolution;
    fPlasticStrain = copy.fPlasticStrain;
    fStorage = copy.fStorage;
    fColSizes = copy.fColSizes;
    fRowSizes = copy.fRowSizes;
    fMatrixPosition = copy.fMatrixPosition;
    fRowFirstIndex = copy.fRowFirstIndex;
    fColFirstIndex = copy.fColFirstIndex;
    fIndexes = copy.fIndexes;
    fIndexesColor = copy.fIndexesColor;
    fWeight = copy.fWeight;

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
    file << "GatherSolution	" << timeGatherSolution << std::endl;
}

void TPZIntPointsFEM::DeltaStrain(TPZFMatrix<REAL> &gather_solution, TPZFMatrix<REAL> &delta_strain) {
    int64_t nelem = fRowSizes.size();

    REAL alpha = 1.;
    REAL beta = 0.;

    delta_strain.Resize(fDim*fNpts,1);
    delta_strain.Zero();
//    char trans = 'N';
//    char matdescra[6] = {
//            'G', // type of matrix
//            ' ', // triangular indicator (ignored in multiplication)
//            ' ', // diagonal indicator (ignored in multiplication)
//            'C', // type of indexing
//            ' ',
//            ' '
//    };

    fTimer.Start();
//    mkl_dcsrmv(&trans, (const int*) fNpts, (const int*) fNphis , &alpha, matdescra , &fStorage[0] , &fColInd[0], &fRowPtr[0], &fRowPtr_last[0], &gather_solution(0,0) , &beta, &delta_strain(0,0));
//    mkl_dcsrmv(&trans, (const int*) fNpts, (const int*) fNphis , &alpha, matdescra , &fStorage[0] , &fColInd[0], &fRowPtr[0], &fRowPtr_last[0], &gather_solution(fNphis,0) , &beta, &delta_strain(fNpts,0));

#pragma omp parallel for
    for (int64_t iel = 0; iel < nelem; iel++) {
        for (int i = 0; i < fRowSizes[iel]; i++) {
            for (int k = 0; k < fColSizes[iel]; k++) {
                delta_strain(i + fRowFirstIndex[iel], 0) += fStorage[k * fRowSizes[iel] + i + fMatrixPosition[iel]] * gather_solution(k + fColFirstIndex[iel], 0);
                delta_strain(i + fRowFirstIndex[iel] + fNpts, 0) += fStorage[k * fRowSizes[iel] + i + fMatrixPosition[iel]] * gather_solution(k + fColFirstIndex[iel] + fNphis, 0);
            }
        }
    }
    fTimer.Stop();
    timeDeltaStrain+= fTimer.ElapsedTime();
    file << "MatMult	" << timeDeltaStrain << std::endl;
}

void TPZIntPointsFEM::ElasticStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &plastic_strain, TPZFMatrix<REAL> &elastic_strain) {
    elastic_strain.Resize(fDim*fNpts,1);
    elastic_strain.Zero();

    plastic_strain.Zero();

    fTimer.Start();
    elastic_strain = delta_strain - plastic_strain;
    fTimer.Stop();
    timeElasticStrain+= fTimer.ElapsedTime();
    file << "ElasticStrain	" << timeElasticStrain << std::endl;
}

void TPZIntPointsFEM::PlasticStrain(TPZFMatrix<REAL> &delta_strain, TPZFMatrix<REAL> &elastic_strain, TPZFMatrix<REAL> &plastic_strain) {
    plastic_strain.Resize(fDim*fNpts,1);

    fTimer.Start();
    plastic_strain = delta_strain - elastic_strain;
    fTimer.Stop();
    timePlasticStrain+= fTimer.ElapsedTime();
    file << "PlasticStrain	" << timeElasticStrain << std::endl;
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
    file << "ComputeStress	" << timeComputeStress << std::endl;
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
    file << "ComputeStrain	" << timeComputeStrain << std::endl;
}

void TPZIntPointsFEM::SpectralDecomposition(TPZFMatrix<REAL> &sigma_trial, TPZFMatrix<REAL> &eigenvalues, TPZFMatrix<REAL> &eigenvectors) {
    REAL maxel;
    TPZVec<REAL> interval(2*fNpts/fDim);
    eigenvalues.Resize(3*fNpts/fDim,1);
    eigenvectors.Resize(9*fNpts/fDim,1);
    fTimer.Start();

    int64_t ipts;
#pragma omp parallel for private(maxel)
    for (ipts = 0; ipts < fNpts/fDim; ipts++) {
        Normalize(&sigma_trial(4*ipts, 0), maxel);
        Interval(&sigma_trial(4*ipts, 0), &interval[2*ipts]);
        NewtonIterations(&interval[2*ipts], &sigma_trial(4*ipts, 0), &eigenvalues(3*ipts, 0), maxel);
        Eigenvectors(&sigma_trial(4*ipts, 0), &eigenvalues(3*ipts, 0), &eigenvectors(9*ipts,0),maxel);
    }
    fTimer.Stop();
    timeSpectralDecomposition+= fTimer.ElapsedTime();
    file << "SpectralDecomposition	" << timeSpectralDecomposition << std::endl;
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
    file << "ProjectSigma	" << timeProjectSigma << std::endl;
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
    file << "StressCompleteTensor	" << timeStressCompleteTensor << std::endl;
}

void TPZIntPointsFEM::NodalForces(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &nodal_forces) {
    int64_t nelem = fRowSizes.size();
    nodal_forces.Resize(fDim*fNphis,1);
    nodal_forces.Zero();

    fTimer.Start();
#pragma omp parallel for
    for (int iel = 0; iel < nelem; iel++) {
        for (int i = 0; i < fColSizes[iel]; i++) {
            for (int k = 0; k < fRowSizes[iel]; k++) {
                nodal_forces(i + fColFirstIndex[iel], 0) -= fStorage[k + i * fRowSizes[iel] + fMatrixPosition[iel]] * sigma(k + fRowFirstIndex[iel], 0);
                nodal_forces(i + fColFirstIndex[iel] + fNphis, 0) -=  fStorage[k + i * fRowSizes[iel] + fMatrixPosition[iel]] * sigma(k + fRowFirstIndex[iel] + fNpts, 0);
            }
        }
    }
    fTimer.Stop();
    timeNodalForces+= fTimer.ElapsedTime();
    file << "NodalForces	" << timeNodalForces << std::endl;
}

void TPZIntPointsFEM::ColoredAssemble(TPZFMatrix<STATE>  &nodal_forces, TPZFMatrix<STATE> &residual) {
    int64_t ncolor = *std::max_element(fElemColor.begin(), fElemColor.end())+1;
    int64_t sz = fIndexes.size();
    int64_t neq = fCmesh->NEquations();
    residual.Resize(neq*ncolor,1);
    residual.Zero();

    fTimer.Start();
    cblas_dsctr(sz, nodal_forces, &fIndexesColor[0], &residual(0,0));

    int64_t colorassemb = ncolor / 2.;
    while (colorassemb > 0) {

        int64_t firsteq = (ncolor - colorassemb) * neq;
        cblas_daxpy(colorassemb * neq, 1., &residual(firsteq, 0), 1., &residual(0, 0), 1.);

        ncolor -= colorassemb;
        colorassemb = ncolor/2;
    }
    residual.Resize(neq, 1);
    fTimer.Stop();
    timeColoredAssemble+= fTimer.ElapsedTime();
    file << "ColoredAssemble	" << timeColoredAssemble << std::endl;

}

void TPZIntPointsFEM::AssembleResidual() {
    TPZFMatrix<REAL> gather_solution;
    TPZFMatrix<REAL> delta_strain;
    TPZFMatrix<REAL> elastic_strain;
    TPZFMatrix<REAL> sigma_trial;
    TPZFMatrix<REAL> eigenvalues;
    TPZFMatrix<REAL> eigenvectors;
    TPZFMatrix<REAL> sigma_projected;
    TPZFMatrix<REAL> sigma;
    TPZFMatrix<REAL> nodal_forces;
    TPZFMatrix<REAL> residual;

    //residual assemble
    GatherSolution(fSolution, gather_solution);
    DeltaStrain(gather_solution, delta_strain);
    ElasticStrain(delta_strain, fPlasticStrain, elastic_strain);
    ComputeStress(elastic_strain, sigma_trial);
    SpectralDecomposition(sigma_trial, eigenvalues, eigenvectors); //check open mp usage in this method
    ProjectSigma(eigenvalues, sigma_projected);
    StressCompleteTensor(sigma_projected, eigenvectors, sigma);
    NodalForces(sigma, nodal_forces);
    ColoredAssemble(nodal_forces,residual);

    //update strain
    ComputeStrain(sigma, elastic_strain);
    PlasticStrain(delta_strain, elastic_strain, fPlasticStrain);

    fRhs = residual + fRhsBoundary;
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
    TPZVec<MKL_INT> rowsizes(nelem);
    TPZVec<MKL_INT> colsizes(nelem);

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
    TPZManVector<MKL_INT> indexes(dim_mesh * nf_tot);

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

//        elmatrix.Transpose();
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
    this->CSRInfo();
    this->AssembleRhsBoundary();

    fPlasticStrain.Resize(fDim * fNpts, 1);
    fPlasticStrain.Zero();
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
