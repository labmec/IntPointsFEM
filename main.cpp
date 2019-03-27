
#include <iostream>
#include <string.h>
#include <ctime>
#include <algorithm>
#include <iterator>

// Neopz
#include "pzgmesh.h"
#include "pzcmesh.h"
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
#include "TPZGmshReader.h"
#include "pzpostprocanalysis.h"
#include "pzfstrmatrix.h"

#include "TPZMatElastoPlastic2D.h"
#include "TPZMatElastoPlastic.h"
#include "TPZElastoPlasticMem.h"
#include "TPZElasticCriterion.h"
#include "TPZPlasticStepPV.h"
#include "TPZSandlerExtended.h"
#include "TPZYCMohrCoulombPV.h"

#include "TPZSolveMatrix.h"
#include "TPZSolveVector.h"
#include "TElastoPlasticData.h"

#ifdef USING_TBB
#include "tbb/parallel_for_each.h"

#endif
/// Geometric mesh (quadrilateral elements)
TPZGeoMesh *Geometry2D(int nelem_x, int nelem_y, REAL len, int ndivide);

/// Gmsh mesh
TPZGeoMesh * ReadGeometry(std::string geometry_file);
void PrintGeometry(TPZGeoMesh * geometry);

/// CompMesh elasticity
TPZCompMesh *CmeshElasticity(TPZGeoMesh *gmesh, int pOrder);
TPZCompMesh *CmeshElasticityNoBoundary(TPZGeoMesh *gmesh, int pOrder);

/// CompMesh elastoplasticity
TPZCompMesh *CmeshElastoplasticity(TPZGeoMesh *gmesh, int pOrder, TElastoPlasticData & material_data);
TPZCompMesh *CmeshElastoplasticityNoBoundary(TPZGeoMesh *gmesh, int pOrder);

/// Material configuration
TElastoPlasticData WellboreConfig();

/// Residual calculation
void SolMatrix(TPZFMatrix<REAL> residual, TPZCompMesh *cmesh);
void SolVector(TPZFMatrix<REAL> residual, TPZCompMesh *cmesh);
TPZFMatrix<REAL>  Residual(TPZCompMesh *cmesh, TPZCompMesh *cmesh_noboundary);

/// Accept solution
void AcceptPseudoTimeStepSolution(TPZAnalysis * an, TPZCompMesh * cmesh);

/// Post process
void PostProcess(TPZCompMesh *cmesh, TElastoPlasticData material, int n_threads);

int main(int argc, char *argv[]) {
    int pOrder = 3; // Computational mesh order

// Generates the geometry
    std::string file("wellbore.msh");
    TPZGeoMesh *gmesh = ReadGeometry(file);
    PrintGeometry(gmesh);

// Creates the computational mesh
    TElastoPlasticData wellbore_material = WellboreConfig();
    TPZCompMesh *cmesh = CmeshElastoplasticity(gmesh, pOrder, wellbore_material);
//    TPZCompMesh *cmesh_noboundary = CmeshElastoplasticityNoBoundary(gmesh, pOrder);

// Defines the analysis
    bool optimizeBandwidth = true;
    int n_threads = 12;
    TPZAnalysis * an = new TPZAnalysis(cmesh, optimizeBandwidth);
    TPZSymetricSpStructMatrix strskyl(cmesh);
    strskyl.SetNumThreads(n_threads);
    an->SetStructuralMatrix(strskyl);

// Solve
    TPZStepSolver<STATE> step;
    step.SetDirect(ELDLt);
    an->SetSolver(step);
    an->Assemble();
    an->Solve();
    an->LoadSolution();
    AcceptPseudoTimeStepSolution(an, cmesh);

// Post process
    PostProcess(cmesh, wellbore_material, n_threads);

// Calculates residual without boundary conditions
//    TPZFMatrix<REAL> residual = Residual(cmesh, cmesh_noboundary);
//    TPZFMatrix<REAL> residual(1,1,0.);

// Calculates residual using matrix operations and check if the result is ok
//    SolMatrix(residual, cmesh);
//    SolVector(residual, cmesh);

    return 0;
}

void PostProcess(TPZCompMesh *cmesh, TElastoPlasticData wellbore_material, int n_threads) {
    int div = 1;
    TPZPostProcAnalysis * post_processor = new TPZPostProcAnalysis;
    post_processor->SetCompMesh(cmesh);

    int n_regions = 1;
    TPZManVector<int,1> post_mat_id(n_regions);
    post_mat_id[0] = wellbore_material.Id();

    TPZStack<std::string,50> names, scalnames,vecnames,tensnames;
    vecnames.push_back("Displacement");
    tensnames.push_back("Stress");

    for (auto i : scalnames) {
        names.push_back(i);
    }
    for (auto i : vecnames) {
        names.push_back(i);
    }
    for (auto i : tensnames) {
        names.push_back(i);
    }


    std::string vtk_file("Approximation.vtk");

    post_processor->SetPostProcessVariables(post_mat_id, names);
    TPZFStructMatrix structmatrix(post_processor->Mesh());
    structmatrix.SetNumThreads(n_threads);
    post_processor->SetStructuralMatrix(structmatrix);

    post_processor->TransferSolution(); /// Computes the L2 projection.
    post_processor->DefineGraphMesh(2,scalnames,vecnames,tensnames,vtk_file);
    post_processor->PostProcess(div,2);
}

TElastoPlasticData WellboreConfig(){
    TPZElasticResponse LER;
    
    REAL Ey = 2000.0;
    REAL nu = 0.2;
    LER.SetEngineeringData(Ey, nu);
    
    REAL mc_cohesion    = 1000000000000.0;
    REAL mc_phi         = (20.0*M_PI/180);
    

    std::vector<TBCData> bc_data;
    TBCData bc_inner, bc_outer, bc_ux_fixed, bc_uy_fixed;
    bc_inner.SetId(2);
    bc_inner.SetType(6);
    bc_inner.SetValue({-10.}); /// tr(sigma)/3
    
    bc_outer.SetId(3);
    bc_outer.SetType(6);
    bc_outer.SetValue({-50.}); /// tr(sigma)/3
    
    bc_ux_fixed.SetId(4);
    bc_ux_fixed.SetType(3);
    bc_ux_fixed.SetValue({1,0});
    
    bc_uy_fixed.SetId(5);
    bc_uy_fixed.SetType(3);
    bc_uy_fixed.SetValue({0,1});
    
    bc_data.push_back(bc_inner);
    bc_data.push_back(bc_outer);
    bc_data.push_back(bc_ux_fixed);
    bc_data.push_back(bc_uy_fixed);
    
    TElastoPlasticData rock;
    rock.SetParameters(LER, mc_phi, mc_cohesion);
    rock.SetId(1);

    rock.SetBoundaryData(bc_data);
    
    return rock;
}

TPZGeoMesh * ReadGeometry(std::string geometry_file) {
    TPZGmshReader Geometry;
    REAL l = 1.0;
    Geometry.SetCharacteristiclength(l);
    Geometry.SetFormatVersion("4.0");
    TPZGeoMesh * geometry = Geometry.GeometricGmshMesh(geometry_file);
    Geometry.PrintPartitionSummary(std::cout);
#ifdef PZDEBUG
    if (!geometry)
    {
        std::cout << "The geometrical mesh was not generated." << std::endl;
        DebugStop();
    }
#endif
    
    return geometry;
    
}

void PrintGeometry(TPZGeoMesh * geometry) {
    std::stringstream text_name;
    std::stringstream vtk_name;
    text_name  << "geometry" << ".txt";
    vtk_name   << "geometry"  << ".vtk";
    std::ofstream textfile(text_name.str().c_str());
    geometry->Print(textfile);
    std::ofstream vtkfile(vtk_name.str().c_str());
    TPZVTKGeoMesh::PrintGMeshVTK(geometry, vtkfile, true);
    
#ifdef PZDEBUG
    TPZCheckGeom checker(geometry);
    checker.CheckUniqueId();
    if(checker.PerformCheck())
    {
        DebugStop();
    }
#endif
    
}

TPZGeoMesh *Geometry2D(int nelem_x, int nelem_y, REAL len, int ndivide) {
// Creates the geometric mesh
    TPZGeoMesh *gmesh = new TPZGeoMesh();
    int dim = 2;
    gmesh->SetDimension(dim);

// Geometry definitions
    int64_t nnodes_x = nelem_x + 1; //Number of nodes in x direction
    int64_t nnodes_y = nelem_y + 1; //Number of nodes in x direction
    int64_t nelem = nelem_x * nelem_y; //Total number of elements

// Nodes initialization
// Enumeration: vertical order - from the below to the top, and from the left to the right
    TPZManVector<REAL> coord(3, 0.);
    int64_t id, index;
    for (int64_t i = 0; i < nnodes_x; i++) {
        for (int64_t j = 0; j < nnodes_y; j++) {
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
    for (int64_t i = 0; i < (nnodes_x - 1); i++) {
        for (int64_t j = 0; j < (nnodes_y - 1); j++) {
            index = (i) * (nnodes_y - 1) + (j);
            connect[0] = (i) * nnodes_y + (j);
            connect[1] = connect[0] + (nnodes_y);
            connect[2] = connect[1] + 1;
            connect[3] = connect[0] + 1;
            gmesh->CreateGeoElement(EQuadrilateral, connect, 1, id); //Allocates and define the geometric element
        }
    }

// Generates neighborhood information
    gmesh->BuildConnectivity();

// Creates the boundary conditions
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
    for (int64_t i = 0; i < nelem_x; i++) {
        int64_t n = nelem_y * (i + 1) - (nelem_y);
        TPZGeoEl *gelem = gmesh->Element(n);
        TPZGeoElBC el_boundary(gelem, 4, -3); //Bottom side of the plane - tension
    }
    for (int64_t i = nelem - nelem_y; i < nelem; i++) {
        TPZGeoEl *gelem = gmesh->Element(i);
        TPZGeoElBC el_boundary(gelem, 5, -4); //Right side of the plane - tension
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

void AcceptPseudoTimeStepSolution(TPZAnalysis * an, TPZCompMesh * cmesh){
    
    bool update = true;
    {
        std::map<int, TPZMaterial *> & refMatVec = cmesh->MaterialVec();
        std::map<int, TPZMaterial * >::iterator mit;
        TPZMatWithMem<TPZElastoPlasticMem> * pMatWithMem;
        for(mit=refMatVec.begin(); mit!= refMatVec.end(); mit++)
        {
            pMatWithMem = dynamic_cast<TPZMatWithMem<TPZElastoPlasticMem> *>( mit->second );
            if(pMatWithMem != NULL)
            {
                pMatWithMem->SetUpdateMem(update);
            }
        }
    }
    an->AssembleResidual();
    update = false;
    {
        std::map<int, TPZMaterial *> & refMatVec = cmesh->MaterialVec();
        std::map<int, TPZMaterial * >::iterator mit;
        TPZMatWithMem<TPZElastoPlasticMem> * pMatWithMem;
        for(mit=refMatVec.begin(); mit!= refMatVec.end(); mit++)
        {
            pMatWithMem = dynamic_cast<TPZMatWithMem<TPZElastoPlasticMem> *>( mit->second );
            if(pMatWithMem != NULL)
            {
                pMatWithMem->SetUpdateMem(update);
//                for(auto memory: *pMatWithMem->GetMemory()){
//                    memory.Print();
//                }
            }
        }
    }
    
}

TPZCompMesh *CmeshElasticity(TPZGeoMesh *gmesh, int pOrder) {

    // Creating the computational mesh
    TPZCompMesh *cmesh = new TPZCompMesh(gmesh);
    cmesh->SetDefaultOrder(pOrder);

// Creating elasticity material
    TPZMatElasticity2D *mat = new TPZMatElasticity2D(1);
    mat->SetElasticParameters(200000000., 0.3);
    mat->SetPlaneStrain();

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
    cmesh->AdjustBoundaryElements();
    cmesh->CleanUpUnconnectedNodes();

    return cmesh;


//// Creates the computational mesh
//    TPZCompMesh *cmesh = new TPZCompMesh(gmesh);
//    cmesh->SetDefaultOrder(pOrder);
//
//// Creates elastic material
//    TPZMatElasticity2D *material = new TPZMatElasticity2D(1);
//    material->SetElasticParameters(200000., 0.3);
//    material->SetPlaneStrain();
//
//// Set the boundary conditions
//    TPZMaterial *bcBottom, *bcRight, *bcTop, *bcLeft;
//    TPZFMatrix<REAL> val1(2, 2), val2(2, 2);
//
//    val2(0, 0) = 0;
//    val2(1, 0) = 0;
//    bcLeft = material->CreateBC(material, -1, 0, val1, val2); // X displacement = 0
//
//    val2(0,0) = 0;
//    val2(1,0) = 0;
//    bcTop = material->CreateBC(material, -2, 0, val1, val2); // Y displacement = 0
//
//    val2(0, 0) = 0.0;
//    val2(1, 0) = -1000.;
//    bcBottom = material->CreateBC(material, -3, 1, val1, val2); // Tension in y
//
//    val2(0, 0) = 0.0;
//    val2(1, 0) = 0.0;
//    bcRight = material->CreateBC(material, -4, 0, val1, val2); // Tension in x
//
//    cmesh->InsertMaterialObject(material);
//    cmesh->InsertMaterialObject(bcBottom);
//    cmesh->InsertMaterialObject(bcRight);
//    cmesh->InsertMaterialObject(bcTop);
//    cmesh->InsertMaterialObject(bcLeft);
//
//    cmesh->SetAllCreateFunctionsContinuous();
//    cmesh->AutoBuild();
//    cmesh->AdjustBoundaryElements();
//    cmesh->CleanUpUnconnectedNodes();
//
//    return cmesh;
}

TPZCompMesh *CmeshElasticityNoBoundary(TPZGeoMesh *gmesh, int pOrder) {

    // Creating the computational mesh
    TPZCompMesh *cmesh = new TPZCompMesh(gmesh);
    cmesh->SetDefaultOrder(pOrder);

    // Creating elasticity material
    TPZMatElasticity2D *mat = new TPZMatElasticity2D(1);
    mat->SetElasticParameters(200000000., 0.3);
    mat->SetPlaneStrain();
    cmesh->InsertMaterialObject(mat);

    cmesh->SetAllCreateFunctionsContinuous();
    cmesh->AutoBuild();
    return cmesh;

//    // Creating the computational mesh
//    TPZCompMesh *cmesh = new TPZCompMesh(gmesh);
//    cmesh->SetDefaultOrder(pOrder);
//
//    // Creating elasticity material
//    TPZMatElasticity2D *mat = new TPZMatElasticity2D(1);
//    mat->SetElasticParameters(200000000., 0.3);
//    cmesh->InsertMaterialObject(mat);
//
//    cmesh->SetAllCreateFunctionsContinuous();
//    cmesh->AutoBuild();
//    return cmesh;
}

TPZCompMesh *CmeshElastoplasticity(TPZGeoMesh *gmesh, int p_order, TElastoPlasticData & material_data) {

// Creates the computational mesh
    TPZCompMesh * cmesh = new TPZCompMesh(gmesh);
    cmesh->SetDefaultOrder(p_order);
    int dim = gmesh->Dimension();
    int matid = material_data.Id();
    
    // Mohr Coulomb data
    REAL mc_cohesion    = material_data.Cohesion();
    REAL mc_phi         = material_data.FrictionAngle();
    REAL mc_psi         = mc_phi;

    // ElastoPlastic Material using Mohr Coulomb
    // Elastic predictor
    TPZElasticResponse ER = material_data.ElasticResponse();

    TPZPlasticStepPV<TPZYCMohrCoulombPV, TPZElasticResponse> LEMC;
    LEMC.SetElasticResponse(ER);
    LEMC.fYC.SetUp(mc_phi, mc_psi, mc_cohesion, ER);
    int PlaneStrain = 1;
    LEMC.fN.m_eps_t.Zero();
    LEMC.fN.m_eps_p.Zero();
    
    TPZElastoPlasticMem default_memory;
    default_memory.m_ER = ER;
    default_memory.m_sigma.Zero();
    default_memory.m_elastoplastic_state = LEMC.fN;

    // Creates elastoplatic material
    TPZMatElastoPlastic2D < TPZPlasticStepPV<TPZYCMohrCoulombPV, TPZElasticResponse>, TPZElastoPlasticMem > * material = new TPZMatElastoPlastic2D < TPZPlasticStepPV<TPZYCMohrCoulombPV, TPZElasticResponse>, TPZElastoPlasticMem >(matid,PlaneStrain);
    material->SetPlasticityModel(LEMC);
    material->SetDefaultMem(default_memory);
    cmesh->InsertMaterialObject(material);
    // Set the boundary conditions
    
    TPZFNMatrix<3,REAL> val1(dim,dim), val2(dim,dim);
    val1.Zero();
    val2.Zero();
    int n_bc = material_data.BoundaryData().size();
    for (int i = 0; i < n_bc; i++) {
        int bc_id = material_data.BoundaryData()[i].Id();
        int type = material_data.BoundaryData()[i].Type();
        
        int n_values = material_data.BoundaryData()[i].Value().size();

        for (int k = 0; k < n_values; k++) {
            val2(k,0) = material_data.BoundaryData()[i].Value()[k];
        }
        TPZMaterial *bc = material->CreateBC(material, bc_id, type, val1, val2);
        cmesh->InsertMaterialObject(bc);
    }

    cmesh->SetDimModel(dim);
    cmesh->SetAllCreateFunctionsContinuousWithMem();
    cmesh->ApproxSpace().CreateWithMemory(true);
    cmesh->AutoBuild();
    
#ifdef PZDEBUG
    std::ofstream out("cmesh.txt");
    cmesh->Print(out);
#endif
    return cmesh;
}

TPZCompMesh *CmeshElastoplasticityNoBoundary(TPZGeoMesh * gmesh, int p_order) {

// Creates the computational mesh
    TPZCompMesh * cmesh = new TPZCompMesh(gmesh);
    cmesh->SetDefaultOrder(p_order);

// Mohr Coulomb data
    REAL mc_cohesion    = 10.0;
    REAL mc_phi         = (20.0*M_PI/180);
    REAL mc_psi         = mc_phi;

// ElastoPlastic Material using Mohr Coulomb
// Elastic predictor
    TPZElasticResponse ER;
    REAL G = 400*mc_cohesion;
    REAL nu = 0.3;
    REAL E = 2.0*G*(1+nu);

    TPZPlasticStepPV<TPZYCMohrCoulombPV, TPZElasticResponse> LEMC;
    ER.SetEngineeringData(E,nu);
    LEMC.SetElasticResponse(ER);
    LEMC.fYC.SetUp(mc_phi, mc_psi, mc_cohesion, ER);
    int PlaneStrain = 1;
    int matid = 1;

// Creates elastoplatic material
    TPZMatElastoPlastic2D < TPZPlasticStepPV<TPZYCMohrCoulombPV, TPZElasticResponse>, TPZElastoPlasticMem > * material = new TPZMatElastoPlastic2D < TPZPlasticStepPV<TPZYCMohrCoulombPV, TPZElasticResponse>, TPZElastoPlasticMem >(matid,PlaneStrain);
    material->SetPlasticityModel(LEMC);

    cmesh->InsertMaterialObject(material);
    cmesh->SetAllCreateFunctionsContinuousWithMem();
    cmesh->AutoBuild();

    return cmesh;
}

void SolVector(TPZFMatrix<REAL> residual, TPZCompMesh *cmesh) {

    int dim_mesh = (cmesh->Reference())->Dimension(); // Mesh dimension
    int64_t nelem_c = cmesh->NElements(); // Number of computational elements
    std::vector<int64_t> cel_indexes;

// Number of domain geometric elements
    for (int64_t i = 0; i < nelem_c; i++) {
        TPZCompEl *cel = cmesh->Element(i);
        if (!cel) continue;
        TPZGeoEl *gel = cmesh->Element(i)->Reference();
        if (!gel || gel->Dimension() != dim_mesh) continue;
        cel_indexes.push_back(cel->Index());
    }

    if (cel_indexes.size() == 0) {
        DebugStop();
    }

// RowSizes and ColSizes vectors
    int64_t nelem = cel_indexes.size();
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

        rowsizes[iel] = dim * npts;
        colsizes[iel] = nf;

        npts_tot += npts;
        nf_tot += nf;
    }

    TPZSolveVector *SolVec = new TPZSolveVector(dim_mesh * npts_tot, nf_tot, rowsizes, colsizes);

// Dphi matrix, weight and indexes vectors
    TPZFMatrix<REAL> elmatrix;
    TPZVec<REAL> weight(npts_tot);
    TPZManVector<MKL_INT> indexes(dim_mesh * nf_tot);
    int cont = 0;
    for (auto iel : cel_indexes) {
        int64_t cont1 = 0;
        int64_t cont2 = 0;
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

        elmatrix.Resize(dim * npts, nf);
        for (int64_t inpts = 0; inpts < npts; inpts++) {
            TPZManVector<REAL> qsi(dim, 1);
            REAL w;
            int_rule->Point(inpts, qsi, w);
            cel_inter->ComputeRequiredData(data, qsi);
            weight[iel + nelem*inpts] = w * std::abs(data.detjac);

            TPZFMatrix<REAL> &dphix = data.dphix;
            for (int inf = 0; inf < nf; inf++) {
                for (int idim = 0; idim < dim; idim++)
                    elmatrix(inpts * dim + idim, inf) = dphix(idim, inf);
            }
        }
        SolVec->SetElementMatrix(iel, elmatrix);

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
                    if (isize % 2 == 0) {
                        indexes[cont1*nelem + cont] = pos + isize;
                        cont1++;
                    } else {
                        indexes[cont2*nelem + nf_tot + cont] = pos + isize;
                        cont2++;
                    }
                }
            }
        }
        cont++;
    }
    SolVec->SetIndexes(indexes);
    SolVec->ColoringElements(cmesh);

    TPZFMatrix<REAL> coef_sol = cmesh->Solution();
    int neq = cmesh->NEquations();

    TPZFMatrix<REAL> nodal_forces_global1(neq, 1, 0.);
    TPZFMatrix<REAL> nodal_forces_global2(neq, 1, 0.);
    TPZFMatrix<REAL> result;
    TPZFMatrix<REAL> sigma;
    TPZFMatrix<REAL> nodal_forces_vec;

#ifdef __CUDACC__
    std::cout << "\n\nSOLVING WITH GPU" << std::endl;
    SolVec->AllocateMemory(cmesh);
    SolVec->MultiplyCUDA(coef_sol,result);
    SolVec->ComputeSigmaCUDA(weight, result, sigma);
    SolVec->MultiplyTransposeCUDA(sigma,nodal_forces_vec);
    SolVec->ColoredAssembleCUDA(nodal_forces_vec,nodal_forces_global1);
    SolVec->FreeMemory();

#endif

    std::cout << "\n\nSOLVING WITH CPU" << std::endl;
    SolVec->Multiply(coef_sol, result);
    SolVec->ComputeSigma(weight, result, sigma);
    SolVec->MultiplyTranspose(sigma,nodal_forces_vec);
    SolVec->ColoredAssemble(nodal_forces_vec,nodal_forces_global2);

    //Check the result
    int rescpu = Norm(nodal_forces_global2 - residual);
    if(rescpu == 0){
        std::cout << "\nAssemble done in the CPU is ok." << std::endl;
    } else {
        std::cout << "\nAssemble done in the CPU is not ok." << std::endl;
    }

#ifdef __CUDACC__
    int resgpu = Norm(nodal_forces_global1 - residual);
    if(resgpu == 0){
        std::cout << "\nAssemble done in the GPU is ok." << std::endl;
    } else {
        std::cout << "\nAssemble done in the GPU is not ok." << std::endl;
    }
#endif
}

void SolMatrix(TPZFMatrix<REAL> residual, TPZCompMesh *cmesh) {

    int dim_mesh = (cmesh->Reference())->Dimension(); // Mesh dimension
    int64_t nelem_c = cmesh->NElements(); // Number of computational elements
    std::vector<int64_t> cel_indexes;

// Number of domain geometric elements
    for (int64_t i = 0; i < nelem_c; i++) {
        TPZCompEl *cel = cmesh->Element(i);
        if (!cel) continue;
        TPZGeoEl *gel = cmesh->Element(i)->Reference();
        if (!gel || gel->Dimension() != dim_mesh) continue;
        cel_indexes.push_back(cel->Index());
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

        rowsizes[iel] = dim * npts;
        colsizes[iel] = nf;

        npts_tot += npts;
        nf_tot += nf;
    }

    TPZSolveMatrix *SolMat = new TPZSolveMatrix(dim_mesh * npts_tot, nf_tot, rowsizes, colsizes);

// Dphi matrix, weight and indexes vectors
    TPZFMatrix<REAL> elmatrix;
    TPZStack<REAL> weight;
    TPZManVector<MKL_INT> indexes(dim_mesh * nf_tot);

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

        elmatrix.Resize(dim * npts, nf);
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
        SolMat->SetElementMatrix(iel, elmatrix);

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
    SolMat->SetIndexes(indexes);
    SolMat->ColoringElements(cmesh);

    TPZFMatrix<REAL> solution = cmesh->Solution();
    int neq = cmesh->NEquations();

    TPZFMatrix<REAL> nodal_forces_global1(neq, 1, 0.);
    TPZFMatrix<REAL> nodal_forces_global2(neq, 1, 0.);
    TPZFMatrix<REAL> sigma_trial;
    TPZFMatrix<REAL> eigenvalues;
    TPZFMatrix<REAL> nodal_forces_vec;
    TPZFMatrix<REAL> delta_strain;
    TPZFMatrix<REAL> total_strain(dim_mesh * dim_mesh * npts_tot, 1, 0.);
    TPZFMatrix<REAL> plastic_strain(dim_mesh * dim_mesh * npts_tot, 1, 0.);
    TPZFMatrix<REAL> elastic_strain(dim_mesh * dim_mesh * npts_tot, 1, 0.);
    TPZFMatrix<REAL> phi;
    TPZFMatrix<REAL> sigma_projected;

    #ifdef __CUDACC__
    std::cout << "\n\nSOLVING WITH GPU" << std::endl;
    SolMat->AllocateMemory(cmesh);
    SolMat->MultiplyCUDA(coef_sol, result);
    SolMat->ComputeSigmaCUDA(weight, result, sigma);
    SolMat->MultiplyTransposeCUDA(sigma, nodal_forces_vec);
    SolMat->ColoredAssembleCUDA(nodal_forces_vec, nodal_forces_global1);
    SolMat->FreeMemory();
    #endif

    std::cout << "\n\nSOLVING WITH CPU" << std::endl;

    SolMat->DeltaStrain(solution, delta_strain);
    SolMat->ElasticStrain(delta_strain, total_strain, plastic_strain, elastic_strain);
    SolMat->SigmaTrial(weight, delta_strain, sigma_trial);
    SolMat->PrincipalStress(sigma_trial, eigenvalues);
    SolMat->ProjectSigma(total_strain, plastic_strain, eigenvalues, sigma_projected);

//    SolMat->Multiply(coef_sol, result);
//    SolMat->ComputeSigma(weight, result, sigma);
//    SolMat->MultiplyTranspose(sigma, nodal_forces_vec);
//    SolMat->ColoredAssemble(nodal_forces_vec, nodal_forces_global2);

//    //Check the result
//    int rescpu = Norm(nodal_forces_global2 - residual);
//    if(rescpu == 0){
//        std::cout << "\nAssemble done in the CPU is ok." << std::endl;
//    } else {
//        std::cout << "\nAssemble done in the CPU is not ok." << std::endl;
//    }

    #ifdef __CUDACC__
    int resgpu = Norm(nodal_forces_global1 - residual);
    if(resgpu == 0){
        std::cout << "\nAssemble done in the GPU is ok." << std::endl;
    } else {
        std::cout << "\nAssemble done in the GPU is not ok." << std::endl;
    }
    #endif
}

TPZFMatrix<REAL> Residual(TPZCompMesh *cmesh, TPZCompMesh *cmesh_noboundary) {
    bool optimizeBandwidth = true;
    int n_threads = 16;

    TPZAnalysis an_d(cmesh_noboundary, optimizeBandwidth);
    TPZSymetricSpStructMatrix strskyl(cmesh_noboundary);
    strskyl.SetNumThreads(n_threads);
    an_d.SetStructuralMatrix(strskyl);

    TPZStepSolver<STATE> step;
    step.SetDirect(ELDLt);
    an_d.SetSolver(step);
    an_d.Assemble();
    an_d.Solve();

    TPZFMatrix<STATE> res;
    an_d.Solver().Matrix()->Multiply(cmesh->Solution(), res);
    return res;
}
