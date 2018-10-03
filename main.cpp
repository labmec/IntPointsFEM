
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
#include "TPZSSpStructMatrix.h"

#include "TPZSolveMatrix.h"

#ifdef USING_TBB
#include "tbb/parallel_for_each.h"
#endif

TPZGeoMesh * geometry_2D(int nelem_x, int nelem_y, REAL len, int ndivide);
TPZCompMesh * cmesh_2D(TPZGeoMesh * gmesh, int pOrder);
TPZCompMesh * cmesh_mat_2D(TPZGeoMesh * gmesh, int pOrder);
void ComputeRhsPointbyPoint(TPZCompMesh *cmesh);
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

for (int i = 0; i < 1; i++) {
    //// ------------------------ DATA INPUT ------------------------------
    //// NUMBER OF ELEMENTS IN X AND Y DIRECTIONS
    int nelem_x = 60;
    int nelem_y = 60;

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
//std::ofstream vtk_file_00(namefile + ".vtk");
//TPZVTKGeoMesh::PrintGMeshVTK(gmesh, vtk_file_00);

//// Creating the computational mesh
TPZCompMesh *cmesh = cmesh_2D(gmesh, pOrder);
TPZCompMesh *cmesh_d = cmesh_mat_2D(gmesh, pOrder);

//// Defining the analysis
bool optimizeBandwidth = true;
int n_threads = 12;
TPZAnalysis an(cmesh, optimizeBandwidth);
TPZAnalysis an_d(cmesh_d, optimizeBandwidth);
#ifdef USING_MKL
TPZSymetricSpStructMatrix strskyl(cmesh);
TPZSymetricSpStructMatrix strskyl_d(cmesh_d);
#else
TPZSkylineStructMatrix strskyl(cmesh);
TPZSkylineStructMatrix strskyl_d(cmesh_d);
#endif
strskyl.SetNumThreads(n_threads);
strskyl_d.SetNumThreads(n_threads);
an.SetStructuralMatrix(strskyl);
an_d.SetStructuralMatrix(strskyl_d);
    
//// Solve
TPZStepSolver<STATE> *direct = new TPZStepSolver<STATE>;
direct->SetDirect(ECholesky);
an.SetSolver(*direct);
an_d.SetSolver(*direct);
delete direct;
an.Assemble();
an.Solve();
    
// Computing global K
an_d.Assemble();
TPZFMatrix<STATE> res_d;
// Computing K u
an_d.Solver().Matrix()->Multiply(an.Solution(), res_d);

//    res_d.Print("ku = ",std::cout,EMathematicaInput);
    
    
//// Post processing in Paraview
TPZManVector<std::string> scalarnames(2), vecnames(1);
scalarnames[0] = "SigmaX";
scalarnames[1] = "SigmaY";
vecnames[0] = "Displacement";
an.DefineGraphMesh(2, scalarnames, vecnames, namefile + "ElasticitySolutions.vtk");
an.PostProcess(0);

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

//int64_t nelem = cmesh->NElements();
//for (int64_t i = 0; i < nelem; i++) {
//
//    // If there is no computational element, continue to the next element
//    TPZCompEl *cel = cmesh->ElementVec()[i];
//    if (!cel) continue;
//
//    // If there is no geometric element, continue to the next element
//    TPZGeoEl *gel = cel->Reference();
//    if (!gel) continue;
//
//    // If the element has no "father", continue to the next element (It means it is not a subelement)
//    TPZGeoEl *father = gel->Father();
//    if (!father) continue;
//
//    // Gets the element level
//    int level = gel->Level();
//
//    // Defines the element
//    TPZInterpolatedElement *intel = dynamic_cast<TPZInterpolatedElement *>(cel);
//    if (!intel) continue;
//    intel->PRefine(level + pOrder + 1);
//
//}

cmesh->AutoBuild();
cmesh->AdjustBoundaryElements();
cmesh->CleanUpUnconnectedNodes();

return cmesh;
}
TPZCompMesh * cmesh_mat_2D(TPZGeoMesh * gmesh, int pOrder){
    
    // Creating the computational mesh
    TPZCompMesh *cmesh = new TPZCompMesh(gmesh);
    cmesh->SetDefaultOrder(pOrder);
    
    // Creating elasticity material
    TPZMatElasticity2D *mat = new TPZMatElasticity2D(1);
    mat->SetElasticParameters(200000000., 0.3);
    cmesh->InsertMaterialObject(mat);
    
    cmesh->SetAllCreateFunctionsContinuous();
    cmesh->AutoBuild();
    return cmesh;
}

void ComputeRhsPointbyPoint(TPZCompMesh *cmesh) {

    int64_t neq = cmesh->NEquations();
    TPZFMatrix <REAL> rhs(neq, 1);
    rhs.Zero();
    int64_t nelem = cmesh->NElements();

    for (int64_t iel = 0; iel < nelem; ++iel) {
        TPZCompEl *cel = cmesh->Element(iel);
        if (!cel || cel->Dimension() != cmesh->Dimension()) continue;

        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement * >(cel);
        if (!cel_inter) DebugStop();

        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());
        int64_t npts = int_rule->NPoints();
        int64_t dim = cel_inter->Dimension();
        int64_t nf = cel_inter->NShapeF();

        TPZMaterialData data;
        cel_inter->InitMaterialData(data);

        TPZFMatrix <REAL> ef(dim * nf, 1);
        ef.Zero();

        for (int64_t ipts = 0; ipts < npts; ipts++) {
            TPZManVector <REAL> qsi(dim, 1);
            REAL w;
            int_rule->Point(ipts, qsi, w);
            cel_inter->ComputeRequiredData(data, qsi);
            TPZMaterial *material = cel_inter->Material();
            material->Contribute(data, w, ef);
        }

        int64_t ncon = cel->NConnects();
        TPZVec<int> iglob(dim * nf, 0);
        int ni = 0;

        for (int64_t icon = 0; icon < ncon; icon++) {
            int64_t id = cel->ConnectIndex(icon);
            TPZConnect &df = cmesh->ConnectVec()[id];
            int64_t conid = df.SequenceNumber();
            if (df.NElConnected() == 0 || conid < 0 || cmesh->Block().Size(conid) == 0) continue;
            else {
                int64_t pos = cmesh->Block().Position(conid);
                int64_t nsize = cmesh->Block().Size(conid);
                for (int64_t isize = 0; isize < nsize; isize++) {
                    iglob[ni] = pos + isize;
                    ni++;
                }
            }
        }

        for (int i = 0; i < ef.Rows(); i++) {
            rhs(iglob[i], 0) += ef(i, 0);
        }
    }
    
    rhs.Print("r = ",std::cout,EMathematicaInput);
}


void sol_teste(TPZCompMesh *cmesh) {

int dim_mesh = (cmesh->Reference())->Dimension(); // Mesh dimension
int64_t nelem_c = cmesh->NElements(); // Number of computational elements
std::vector<int64_t> cel_indexes;

//// -------------------------------------------------------------------------------
//// NUMBER OF DOMAIN GEOMETRIC ELEMENTS
for (int64_t i = 0; i < nelem_c; i++) {
    TPZCompEl *cel = cmesh->Element(i);
    if (!cel) continue;
    TPZGeoEl *gel = cmesh->Element(i)->Reference();
    if (!gel || gel->Dimension() != dim_mesh) continue;
    cel_indexes.push_back(cel->Index());
}
//// -------------------------------------------------------------------------------

    if (cel_indexes.size() == 0) {
        DebugStop();
    }
    
//// ROWSIZES AND COLSIZES VECTORS--------------------------------------------------
int64_t nelem = cel_indexes.size(); // Number of domain geometric elements
TPZVec<int64_t> rowsizes(nelem);
TPZVec<int64_t> colsizes(nelem);

int64_t npts_tot = 0;
int64_t nf_tot = 0;

for (auto iel : cel_indexes) {
    //Verification
    TPZCompEl *cel = cmesh->Element(iel);

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

for (auto iel : cel_indexes) {
        //Verification
    TPZCompEl *cel = cmesh->Element(iel);

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

TPZFMatrix<REAL> coef_sol = cmesh->Solution();
int neq= cmesh->NEquations();
TPZFMatrix<REAL> nodal_forces_global1(neq,1,0.);
TPZFMatrix<REAL> nodal_forces_global2(neq,1,0.);
TPZFMatrix<REAL> result;
TPZFMatrix<REAL> sigma;
TPZFMatrix<REAL> nodal_forces_vec;

//std::cout << "SOLVING WITH GPU" << std::endl;
//std::clock_t begingpu = clock();
//SolMat->SolveWithCUDA(cmesh, coef_sol, weight, nodal_forces_global1);
//std::clock_t endgpu = clock();

std::cout << "SOLVING WITH CPU" << std::endl;
std::clock_t begincpu = clock();
SolMat->Multiply(coef_sol,result);
SolMat->ComputeSigma(weight,result,sigma);
SolMat->MultiplyTranspose(sigma, nodal_forces_vec);
SolMat->TraditionalAssemble(nodal_forces_vec,nodal_forces_global1); // ok
//nodal_forces_global1.Print("f = ",std::cout,EMathematicaInput);
SolMat->ColoringElements(cmesh);
SolMat->ColoredAssemble(nodal_forces_vec,nodal_forces_global2);
std::clock_t endcpu = clock();


    
//REAL elapsed_secs_gpu = REAL(endgpu - begingpu) / CLOCKS_PER_SEC;
REAL elapsed_secs_cpu = REAL(endcpu - begincpu) / CLOCKS_PER_SEC;
//timing << "Time elapsed (GPU): " << std::setprecision(5) << std::fixed << elapsed_secs_gpu << " s" << std::endl;
timing << "Time elapsed (CPU): "<< std::setprecision(5) << std::fixed << elapsed_secs_cpu << " s" << std::endl;

//for(int i = 0; i < neq; i++){
//    std::cout << nodal_forces_global1(i,0) - nodal_forces_global2(i,0)<< std::endl;
//}

    std::cout << "norm of diff = " << Norm(nodal_forces_global1-nodal_forces_global2) << std::endl;


}
