
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
TPZCompMesh * cmesh_2D(TPZGeoMesh * gmesh, int pOrder);
void sol_teste(TPZCompMesh * cmesh);

#ifdef USING_TBB
std::ofstream timing("timingtbb.txt");
#elif USING_CUDA
std::ofstream timing("timingcuda.txt");
#else
std::ofstream timing("timing.txt");
#endif

int main(){
#ifdef USING_TBB
    timing << "--------------------USING TBB--------------------"<< std::endl;
    std::cout << "--------------------USING TBB--------------------" << std::endl;
#elif USING_CUDA
    timing << "--------------------USING CUDA-------------------"  << std::endl;
    std::cout << "--------------------USING CUDA-------------------" << std::endl;
#endif

    for (int i = 0; i < 4; i++) {
        //// ------------------------ DATA INPUT ------------------------------
        //// NUMBER OF ELEMENTS IN X AND Y DIRECTIONS
        int nelem_x = pow(10,i);
        int nelem_y = pow(10,i);

        timing << "-------------------------------------------------" << std::endl;
        timing << "MESH SIZE: " << nelem_x << "x" << nelem_y << std::endl;
        std::cout << "-------------------------------------------------" << std::endl;
        std::cout << "MESH SIZE: " << nelem_x << "x" << nelem_y << std::endl;

    //// DOMAIN LENGTH
    REAL len = 2;

    //// COMPUTATIONAL MESH ORDER
    int pOrder = 1;

    //// SUBDIVISIONS OF THE ELEMENTS
    int ndivide = 0;

    //// ENTER THE FILE NAME
    std::string namefile = "Elasticity_teste";
    //// ------------------------------------------------------------------

    //// Generating the geometry
    TPZGeoMesh *gmesh = geometry_2D(nelem_x, nelem_y, len, ndivide);
//    std::ofstream vtk_file_00(namefile + ".vtk");
//    TPZVTKGeoMesh::PrintGMeshVTK(gmesh, vtk_file_00);

    //// Creating the computational mesh
    TPZCompMesh *cmesh = cmesh_2D(gmesh, pOrder);

    //// Defining the analysis
    bool optimizeBandwidth = true;
    TPZAnalysis an(cmesh, optimizeBandwidth);
    TPZSkylineStructMatrix strskyl(cmesh);
    an.SetStructuralMatrix(strskyl);

    //// Solve
    TPZStepSolver<STATE> *direct = new TPZStepSolver<STATE>;
    direct->SetDirect(ELDLt);
    an.SetSolver(*direct);
    delete direct;
    an.Run();

    std::clock_t begin = clock();
    an.AssembleResidual();
    std::clock_t end = clock();
    REAL elapsed_secs = REAL(end - begin) / CLOCKS_PER_SEC;
    timing << "Time elapsed (AssembleResidual): " << elapsed_secs << " s" << std::endl;

//    //// Post processing in Paraview
//    TPZManVector<std::string> scalarnames(2), vecnames(1);
//    scalarnames[0] = "SigmaX";
//    scalarnames[1] = "SigmaY";
//    vecnames[0] = "Displacement";
//    an.DefineGraphMesh(2, scalarnames, vecnames, namefile + "ElasticitySolutions.vtk");
//    an.PostProcess(1);

    //// Residual Calculation
    sol_teste(cmesh);
    }
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

TPZCompMesh *cmesh_2D(TPZGeoMesh *gmesh, int pOrder) {

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

    //// -------------------------------------------------------------------------------
    //// NUMBER OF GEOMETRIC ELEMENTS
    for (int64_t i = 0; i < nelem_c; i++) {
        TPZCompEl *cel = cmesh->Element(i);
        if (!cel) continue;
        TPZGeoEl *gel = cmesh->Element(i)->Reference();
        if (!gel || gel->Dimension() != dim_mesh) continue;
        nelem++;
    }
    //// -------------------------------------------------------------------------------

    //// ROWSIZES AND COLSIZES VECTORS--------------------------------------------------
    TPZVec<MKL_INT> rowsizes(nelem);
    TPZVec<MKL_INT> colsizes(nelem);

    int64_t npts_tot = 0;
    int64_t nf_tot = 0;

    for (int64_t iel = 0; iel < nelem; ++iel) {
        //Verification
        TPZCompEl *cel = cmesh->Element(iel);
        if (!cel) continue;
        TPZGeoEl *gel = cel->Reference();
        if (!gel) continue;

        //Integration rule
        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement * >(cel);
        if (!cel_inter) DebugStop();
        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());

        int64_t npts = int_rule->NPoints(); // number of integration points of the element
        int64_t dim = cel_inter->Dimension(); //dimension of the element
        int64_t nf = cel_inter->NShapeF(); // number of shape functions of the element

        rowsizes[iel] = dim*npts;
        colsizes[iel] = nf;

        npts_tot += npts;
        nf_tot += nf;
    }

    TPZSolveMatrix * SolMat = new TPZSolveMatrix(dim_mesh*npts_tot, nf_tot, rowsizes, colsizes);
    //// -------------------------------------------------------------------------------

    //// DPHI MATRIX FOR EACH ELEMENT, WEIGHT AND INDEXES VECTORS-----------------------
    TPZFMatrix<REAL> elmatrix;
    TPZStack<REAL> weight;
    TPZManVector<MKL_INT> indexes(dim_mesh*nf_tot);

    int64_t cont1 = 0;
    int64_t cont2 = 0;

    for (int64_t iel = 0; iel < nelem; ++iel) {
        //Verification
        TPZCompEl *cel = cmesh->Element(iel);
        if (!cel) continue;
        TPZGeoEl *gel = cel->Reference();
        if (!gel) continue;

        //Integration rule
        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement * >(cel);
        if (!cel_inter) DebugStop();
        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());

        int64_t npts = int_rule->NPoints(); // number of integration points of the element
        int64_t dim = cel_inter->Dimension(); //dimension of the element
        int64_t nf = cel_inter->NShapeF(); // number of shape functions of the element


        TPZMaterialData data;
        cel_inter->InitMaterialData(data);

        elmatrix.Resize(dim*npts,nf);
        for (int64_t inpts = 0; inpts < npts; inpts++) {
            TPZManVector<REAL> qsi(dim, 1);
            REAL w;
            int_rule->Point(inpts, qsi, w);
            cel_inter->ComputeRequiredData(data, qsi);
            weight.Push(w * std::abs(data.detjac)); //weight = w * detjac

            TPZFMatrix<REAL> &dphix = data.dphix;
            for (int inf = 0; inf < nf; inf++) {
                for (int idim = 0; idim < dim; idim++)
                    elmatrix(inpts * dim + idim, inf) = dphix(idim, inf);
            }
        }
        SolMat->SetElementMatrix(iel,elmatrix);

        //Indexes vector
        int64_t ncon = cel->NConnects();
        for (int64_t icon = 0; icon < ncon; icon++) {
            int64_t id = cel->ConnectIndex(icon);
            TPZConnect &df = cmesh->ConnectVec()[id];
            int64_t conid = df.SequenceNumber();
            if (df.NElConnected() == 0 || conid < 0 || cmesh->Block().Size(conid) == 0) continue;
            else {
                int64_t pos = cmesh->Block().Position(conid);
                int64_t nsize = cmesh->Block().Size(conid);
                for (int64_t isize = 0; isize < nsize; isize++) {
                    if(isize%2==0){
                        indexes[cont1] = pos + isize;
                        cont1++;
                    }
                    else{
                        indexes[cont2 + nf_tot] = pos + isize;
                        cont2++;
                    }
                }
            }
        }
    }
    SolMat->SetIndexes(indexes);
    //// -------------------------------------------------------------------------------

    //// TIMING START-------------------------------------------------------------------
    std::clock_t begin = clock();
    //// -------------------------------------------------------------------------------

    //// SOLVE ADVEC*COEF_SOL-----------------------------------------------------------
    TPZFMatrix<REAL> coef_sol = cmesh->Solution();
    TPZFMatrix<REAL> result;
    SolMat->Multiply(coef_sol,result); //result = [du_0, ..., du_nelem-1, dv_0,..., dv_nelem-1]
    //// -------------------------------------------------------------------------------

    //// SIGMA CALCULATION
    TPZFMatrix<REAL> sigma;

    SolMat->ComputeSigma(weight,result,sigma); //sigma = [sigmax_0, sigmaxy_0, ..., sigmax_nelem-1, sigmaxy_nelem-1, sigmaxy_0, sigmay_0, ..., sigmaxy_nelem-1, sigmay_nelem-1]
    //// -------------------------------------------------------------------------------

    //// COMPUTE NODAL FORCES-----------------------------------------------------------
    TPZFMatrix<REAL> nodal_forces_vec;

    SolMat->MultiplyTranspose(sigma, nodal_forces_vec); //nodal_forces_vec = [fx_0, ..., fx_nelem-1, fy_0, ..., fy_nelem-1]
    //// -------------------------------------------------------------------------------

    //// ASSEMBLE: RESIDUAL CALCULATION-------------------------------------------------
    int neq= cmesh->NEquations();
    TPZFMatrix<REAL> nodal_forces_global1(neq, 1, 0.);

    SolMat->TraditionalAssemble(nodal_forces_vec,nodal_forces_global1); //traditional assemble
    //// -------------------------------------------------------------------------------

    //// TIMING END---------------------------------------------------------------------
    std::clock_t end = clock();
    REAL elapsed_secs = REAL(end - begin) / CLOCKS_PER_SEC;
    timing << "Time elapsed (integ points): " << elapsed_secs << " s" << std::endl;
    //// -------------------------------------------------------------------------------
}
