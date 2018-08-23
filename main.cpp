
#include <iostream>
#include <string.h>
#include <ctime>
#include <algorithm>
#include <iterator>

// Neopz
#include "pzgmesh.h"
#include "pzgeoelbc.h"
#include "pzbndcond.h"
#include "pzanalysis.h"
#include "pzskylstrmatrix.h"
#include "pzstepsolver.h"
#include "pzinterpolationspace.h"
#include "TPZVTKGeoMesh.h"
#include "pzintel.h"
#include "tpzintpoints.h"
#include "TPZMatElasticity2D.h"


#include "TPZSolveMatrix.h"

#ifdef USING_TBB
#include "tbb/parallel_for_each.h"
#endif

TPZGeoMesh * geometry_2D(int nelem_x, int nelem_y, REAL len, int ndivide);
TPZCompMesh * cmesh_2D(TPZGeoMesh * gmesh, int nelem_x, int nelem_y, int pOrder);
void sol_teste(TPZCompMesh * cmesh);

int main(){
    
    // ------------------------ DATA INPUT ------------------------------
    // NUMBER OF ELEMENTS IN X AND Y DIRECTIONS
    int nelem_x = 2;
    int nelem_y = 2;
    
    // DOMAIN LENGTH
    REAL len = 2;

    // COMPUTATIONAL MESH ORDER
    int pOrder = 1;

    // SUBDIVISIONS OF THE ELEMENTS
    int ndivide = 0;

    // ENTER THE FILE NAME
    std::string namefile = "Elasticity_teste";
    // -------------------------------------------------------------------

    // Generating the geometry
    TPZGeoMesh *gmesh = geometry_2D(nelem_x, nelem_y, len, ndivide);
    std::ofstream vtk_file_00(namefile + ".vtk");
    TPZVTKGeoMesh::PrintGMeshVTK(gmesh, vtk_file_00);

    // Creating the computational mesh
    TPZCompMesh *cmesh = cmesh_2D(gmesh, nelem_x, nelem_y, pOrder);

    // Defining the analysis
    bool optimizeBandwidth = true;
    TPZAnalysis an(cmesh, optimizeBandwidth);
    TPZSkylineStructMatrix strskyl(cmesh);
    an.SetStructuralMatrix(strskyl);

    // Solve
    TPZStepSolver<STATE> *direct = new TPZStepSolver<STATE>;
    direct->SetDirect(ELDLt);
    an.SetSolver(*direct);
    delete direct;
    direct = 0;

    an.Run();

    TPZFMatrix<STATE> sol = cmesh->Solution();

    // Post processing in Paraview
    TPZManVector<std::string> scalarnames(2), vecnames(1);
    scalarnames[0] = "SigmaX";
    scalarnames[1] = "SigmaY";
    vecnames[0] = "Displacement";
    an.DefineGraphMesh(2, scalarnames, vecnames, namefile + "ElasticitySolutions.vtk");
    an.PostProcess(1);

    // Residual Calculation
    std::clock_t begin = clock();

    sol_teste(cmesh);

    std::clock_t end = clock();

    REAL elapsed_secs = REAL(end - begin) / CLOCKS_PER_SEC;
    std::cout << "Time elapsed: " << elapsed_secs << std::endl;

    return 0;
}

TPZGeoMesh *geometry_2D(int nelem_x, int nelem_y, REAL len, int ndivide) {

    // Creates the geometric mesh
    TPZGeoMesh *gmesh = new TPZGeoMesh();
    int dim = 2;
    gmesh->SetDimension(dim);

    // Geometry definitions
    int nnodes_x = nelem_x + 1; //Number of elements in x direction
    int nnodes_y = nelem_y + 1; //Number of elements in x direction
    int64_t nelem = nelem_x * nelem_y; //Total number of elements

    // Nodes initialization
    // Enumeration: vertical order - from the below to the top, and from the left to the right
    TPZManVector<REAL> coord(3, 0.);
    int64_t id, index;
    for (int i = 0; i < nnodes_x; i++) {
        for (int j = 0; j < nnodes_y; j++) {
            id = i * nnodes_y + j;
            coord[0] = (i) * len / (nnodes_x - 1);
            coord[1] = (j) * len / (nnodes_y - 1);
            index = gmesh->NodeVec().AllocateNewElement();
            gmesh->NodeVec()[index] = TPZGeoNode(id, coord, *gmesh);
        }
    }

    // Element connectivities
    // Enumeration: vertical order - from the below to the top, and from the left to the right
    TPZManVector<int64_t> connect(4, 0);
    for (int i = 0; i < (nnodes_x - 1); i++) {
        for (int j = 0; j < (nnodes_y - 1); j++) {
            index = (i) * (nnodes_y - 1) + (j);
            connect[0] = (i) * nnodes_y + (j);
            connect[1] = connect[0] + (nnodes_y);
            connect[2] = connect[1] + 1;
            connect[3] = connect[0] + 1;
            gmesh->CreateGeoElement(EQuadrilateral, connect, 1, id); //Allocates and define the geometric element
        }
    }

    // Generate neighborhood information
    gmesh->BuildConnectivity();

    // Creating the boundary conditions
    // Dirichlet
    for (int64_t i = 0; i < nelem_y; i++) {
        TPZGeoEl *gelem = gmesh->Element(i);
        TPZGeoElBC el_boundary(gelem, 7, -1); //Left side of the plane
    }
    for (int64_t i = 0; i < nelem_x; i++) {
        int64_t n = nelem_y * (i + 1) - 1;
        TPZGeoEl *gelem = gmesh->Element(n);
        TPZGeoElBC el_boundary(gelem, 6, -2); //Top side of the plane
    }

    // Neumann
    for (int64_t i = nelem - nelem_y; i < nelem; i++) {
        TPZGeoEl *gelem = gmesh->Element(i);
        TPZGeoElBC el_boundary(gelem, 5, -4); //Right side of the plane - tension
    }
    for (int64_t i = 0; i < nelem_x; i++) {
        int64_t n = nelem_y * (i + 1) - (nelem_y);
        TPZGeoEl *gelem = gmesh->Element(n);
        TPZGeoElBC el_boundary(gelem, 4, -3); //Bottom side of the plane - tension
    }

    // HP adaptativity
    if (ndivide != 0) {
        // Finding the elements which will be subdivided
        TPZGeoEl *gel; // Defining the element
        TPZVec<REAL> x(3, 0.); // Defining the coordinate at the end of the node
        x[0] = 0;
        x[1] = len;
        TPZVec<REAL> qsi(3, 0.); // Defining the parametric coordinate
        int64_t InitialElIndex = 0;
        int targetDim = 2;
        gel = gmesh->FindElement(x, qsi, InitialElIndex,
                                 targetDim); // Finding the element which is related to the coordinate
        int64_t elid = gel->Index(); // Atention: this procedure catchs the first element which is related to the coordinate

        TPZVec<TPZGeoEl *> subelindex;

        gel = gmesh->Element(elid);
        gel->Divide(subelindex);
        for (int i = 0; i < ndivide - 1; i++) {
            subelindex[3]->Divide(subelindex);
        }

    }
    return gmesh;
}

TPZCompMesh *cmesh_2D(TPZGeoMesh *gmesh, int nelem_x, int nelem_y, int pOrder) {

    // Creating the computational mesh
    TPZCompMesh *cmesh = new TPZCompMesh(gmesh);
    cmesh->SetDefaultOrder(pOrder);

    // Creating elasticity material
    TPZMatElasticity2D *mat = new TPZMatElasticity2D(1);
    mat->SetElasticParameters(200000000., 0.3);

    // Setting the boundary conditions
    TPZMaterial *bcBottom, *bcRight, *bcTop, *bcLeft;
    TPZFMatrix<REAL> val1(2, 1, 0.);
    TPZFMatrix<REAL> val2(2, 1, 0.);

    bcLeft = mat->CreateBC(mat, -1, 7, val1, val2); // X displacement = 0
    bcTop = mat->CreateBC(mat, -2, 8, val1, val2); // Y displacement = 0

    val2(1, 0) = -1000000.;
    bcBottom = mat->CreateBC(mat, -3, 1, val1, val2); // Tension in y

    val2(0, 0) = 1000000.;
    val2(1, 0) = 0.0;
    bcRight = mat->CreateBC(mat, -4, 1, val1, val2); // Tension in x

    cmesh->InsertMaterialObject(mat);

    cmesh->InsertMaterialObject(bcBottom);
    cmesh->InsertMaterialObject(bcRight);
    cmesh->InsertMaterialObject(bcTop);
    cmesh->InsertMaterialObject(bcLeft);

    cmesh->SetAllCreateFunctionsContinuous();
    cmesh->AutoBuild();

    int64_t nelem = cmesh->NElements();

    for (int64_t i = 0; i < nelem; i++) {

        // If there is no computational element, continue to the next element
        TPZCompEl *cel = cmesh->ElementVec()[i];
        if (!cel) continue;

        // If there is no geometric element, continue to the next element
        TPZGeoEl *gel = cel->Reference();
        if (!gel) continue;

        // If the element has no "father", continue to the next element (It means it is not a subelement)
        TPZGeoEl *father = gel->Father();
        if (!father) continue;

        // Gets the element level
        int level = gel->Level();

        // Defines the element
        TPZInterpolatedElement *intel = dynamic_cast<TPZInterpolatedElement *>(cel);
        if (!intel) continue;
        intel->PRefine(level + pOrder + 1);

    }

    cmesh->AutoBuild();
    cmesh->AdjustBoundaryElements();
    cmesh->CleanUpUnconnectedNodes();

    return cmesh;
}

void sol_teste(TPZCompMesh *cmesh) {

    int dim_mesh = (cmesh->Reference())->Dimension(); // Mesh dimension

    int64_t nelem = 0; // Number of geometric elements
    int64_t nelem_c = cmesh->NElements(); // Number of computational elements

    int64_t cont_elem = 0;

    // -----------------------------------------------------------------------
    // NUMBER OF GEOMETRIC ELEMENTS
    for (int64_t i = 0; i < nelem_c; i++) {
        TPZCompEl *cel = cmesh->Element(i);
        if (!cel) continue;
        TPZGeoEl *gel = cmesh->Element(i)->Reference();
        if (!gel || gel->Dimension() != dim_mesh) continue;
        nelem++;
    }
    // -----------------------------------------------------------------------
    // PHI AND DPHI MATRICES AND INDEXES VECTOR
    // Vector of matrices
    TPZManVector<TPZManVector<int64_t>> indexes_el(nelem);
    TPZManVector<TPZFMatrix<REAL>> AVec(nelem);
    TPZManVector<TPZFMatrix<REAL>> AdVec(nelem);
    TPZStack<REAL> weight;

    int64_t npts_tot = 0;
    int64_t nf_tot = 0;

    for (int64_t iel = 0; iel < nelem_c; iel++) {
        int64_t cont_coef = 0;

        // Verifications
        TPZCompEl *cel = cmesh->Element(iel);
        if (!cel) continue;
        TPZGeoEl *gel = cel->Reference();
        if (!gel || gel->Dimension() != dim_mesh) continue;

        // Integration rule
        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement * >(cel);
        if (!cel_inter) DebugStop();
        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());

        int64_t npts = int_rule->NPoints(); // number of integration points of the element
        int64_t nf = cel_inter->NShapeF(); // number of shape functions of the element

        TPZMaterialData data;
        cel_inter->InitMaterialData(data);

        AVec[cont_elem].Redim(npts, nf);
        AdVec[cont_elem].Redim(npts * dim_mesh, nf);

        for (int i_npts = 0; i_npts < npts; i_npts++) {
            TPZManVector<REAL> qsi(dim_mesh, 1);
            REAL w;
            int_rule->Point(i_npts, qsi, w);

            cel_inter->ComputeRequiredData(data, qsi);
            weight.Push(w * std::abs(data.detjac)); //weight = w * detjac
            TPZFMatrix<REAL> &phi = data.phi;
            TPZFMatrix<REAL> &dphix = data.dphix;

            for (int i_nf = 0; i_nf < nf; i_nf++) {
                AVec[cont_elem](i_npts, i_nf) = phi(i_nf, 0); //phi matrix: Avec[iel](npts,nf)
                for (int i_dim = 0; i_dim < dim_mesh; i_dim++)
                    AdVec[cont_elem](i_npts * dim_mesh + i_dim, i_nf) = dphix(i_dim, i_nf); //dphi matrix: Advec[iel](dim*npts,nf)
            }
        }

        // Vector of indexes for each element
        indexes_el[cont_elem].Resize(dim_mesh * nf);
        int64_t ncon_i = cel->NConnects();

        for (int64_t l = 0; l < ncon_i; l++) {
            int64_t id = cel->ConnectIndex(l);
            TPZConnect &df = cmesh->ConnectVec()[id];
            int64_t con_id = df.SequenceNumber();
            if (df.NElConnected() == 0 || con_id < 0 || cmesh->Block().Size(con_id) == 0) continue;
            else {
                int64_t pos = cmesh->Block().Position(con_id);
                int64_t nk = cmesh->Block().Size(con_id);
                for (int64_t k = 0; k < nk; k++) {
                    indexes_el[cont_elem][cont_coef] = pos + k;
                    cont_coef++;
                }
            }
        }
        npts_tot += npts;
        nf_tot += nf;
        cont_elem++;
    }

    // Global vector of indexes
    TPZManVector<MKL_INT> indexes(nf_tot*dim_mesh, 0);
    int64_t pos = 0;
    for (int64_t iel = 0; iel<nelem; iel++) {
        int64_t n_ind_el = (indexes_el[iel]).size();
        for (int64_t jind=0; jind<(n_ind_el/2); jind++){
            indexes[pos] = indexes_el[iel][dim_mesh*jind];
            indexes[pos+nf_tot] = indexes_el[iel][dim_mesh*jind+1];
            pos++;
        }
    }
    // -----------------------------------------------------------------------
    //SOLVE ADVEC*COEF_SOL
    TPZSolveMatrix * SolMat = new TPZSolveMatrix(npts_tot, nf_tot, AdVec, indexes);
    TPZFMatrix<REAL> coef_sol = cmesh->Solution();
    TPZFMatrix<REAL> result;

    SolMat->Multiply(coef_sol, result); //result = [du_0, ..., du_nelem-1, dv_0,..., dv_nelem-1]
    // -----------------------------------------------------------------------
    // SIGMA CALCULATION
    TPZFMatrix<REAL> sigma;

    SolMat->ComputeSigma(weight,result,sigma); //sigma = [sigmax_0, ..., sigmax_nelem-1, sigmay_0, ..., sigmay_nelem-1, sigmaxy_0, ..., sigmaxy_nelem-1]
    // -----------------------------------------------------------------------
    // COMPUTE NODAL FORCES
    int neq= cmesh->NEquations();
    TPZFMatrix<REAL> nodal_forces_vec(npts_tot*dim_mesh, 1, 0.);

    SolMat->MultiplyTranspose(sigma, nodal_forces_vec); //nodal_forces_vec = [fx_0, ..., fx_nelem-1, fy_0, ..., fy_nelem-1]
    // -----------------------------------------------------------------------
    //ASSEMBLE: RESIDUAL CALCULATION
    TPZFMatrix<REAL> nodal_forces_global1(neq, 1, 0.);
    TPZFMatrix<REAL> nodal_forces_global2(neq, 1, 0.);

    SolMat->TraditionalAssemble(nodal_forces_vec,nodal_forces_global1); //traditional assemble
    SolMat->ColoredAssemble(cmesh, indexes_el, nodal_forces_vec,nodal_forces_global2); //colored assemble

    //Compare assemble methods
    for (int j = 0; j < nodal_forces_global1.Rows(); ++j) {
        std::cout << nodal_forces_global2[j]-nodal_forces_global1[j] << std::endl;
    }
}
