
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

    // Cálculo do resíduo
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

    // Generate neighborhod information
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
//    TPZMatGriffith * mat = new TPZMatGriffith(1);
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

    int64_t nelem = cmesh->NElements();// nelem_y*nelem_x;

    for (int64_t i = 0; i < nelem; i++) {

        // Se não existe elemento computacional, continua para o próximo elemento
        TPZCompEl *cel = cmesh->ElementVec()[i];
        if (!cel) continue;

        // Se não existe elemento geométrico, continua para o próximo elemento
        TPZGeoEl *gel = cel->Reference();
        if (!gel) continue;

        // Se o elemento não tem "pai", continua para o próximo elemento (SIGNIFICA QUE NÃO É SUBELEMENTO!!)
        TPZGeoEl *father = gel->Father();
        if (!father) continue;

        // Pega o "level" do elemento
        int level = gel->Level();

        // Define elemento de classe de métodos de adaptatividade hp
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

    int dim_mesh = (cmesh->Reference())->Dimension(); // Dimensão da malha

    int64_t nelem = 0; // numero de elementos geométricos
    int64_t nelem_c = cmesh->NElements(); // Número de elementos computacionais

    int64_t cont_elem = 0;

    // CÁLCULO DO NÚMERO DE ELEMENTOS
    for (int64_t i = 0; i < nelem_c; i++) {
        TPZCompEl *cel = cmesh->Element(i);
        if (!cel) continue;
        TPZGeoEl *gel = cmesh->Element(i)->Reference();
        if (!gel || gel->Dimension() != dim_mesh) continue;
        nelem++;
    }

    // -----------------------------------------------------------------------
    // CÁLCULO DE SIGMA

    // Vetor de matrizes para armazenar
    TPZManVector<TPZManVector<int64_t>> indexes_el(nelem);
    TPZManVector<TPZFMatrix<REAL>> AVec(nelem);
    TPZManVector<TPZFMatrix<REAL>> AdVec(nelem);
    TPZStack<REAL> weight;

    int64_t npts_tot = 0;
    int64_t nf_tot = 0;

    for (int64_t iel = 0; iel < nelem_c; iel++) {
        int64_t cont_coef = 0;

        // Verificações
        TPZCompEl *cel = cmesh->Element(iel);
        if (!cel) continue;
        TPZGeoEl *gel = cel->Reference();
        if (!gel || gel->Dimension() != dim_mesh) continue;

        // Def da regra de integração
        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement * >(cel);
        if (!cel_inter) DebugStop();
        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());

        int64_t npts = int_rule->NPoints(); // número de pontos de integração
        int64_t nf = cel_inter->NShapeF(); // número de funções de forma

        // MATRIZES POR ELEMENTO

        // Inicializando o MaterialData
        TPZMaterialData data;
        cel_inter->InitMaterialData(data);

        // Montando a matriz dos phis e dphis
        AVec[cont_elem].Redim(npts, nf);
        AdVec[cont_elem].Redim(npts * dim_mesh, nf);
//        weight[cont_elem].resize(npts);
        for (int i_npts = 0; i_npts < npts; i_npts++) {
            TPZManVector<REAL> qsi(dim_mesh, 1);
            REAL w;
            int_rule->Point(i_npts, qsi, w);



            cel_inter->ComputeRequiredData(data, qsi);
            weight.Push(w * std::abs(data.detjac));
//            weight[cont_elem][i_npts] = w*std::abs(data.detjac);
            TPZFMatrix<REAL> &phi = data.phi;
            TPZFMatrix<REAL> &dphix = data.dphix;

            for (int i_nf = 0; i_nf < nf; i_nf++) {
                AVec[cont_elem](i_npts, i_nf) = phi(i_nf, 0);
                for (int i_dim = 0; i_dim < dim_mesh; i_dim++)
                    AdVec[cont_elem](i_npts * dim_mesh + i_dim, i_nf) = dphix(i_dim, i_nf);
            }
        }


        // VETOR DE ÍNDICES DO ELEMENTO (relacionado com a posição no vetor solução)
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

    // VETOR DE  INDICES GLOBAL
    // De 0 a dim*nf_tot/2 -> índices relativos aos graus de liberdade x
    // De dim*nf_tot/2 a dim*nf_tot -> índices relativos aos graus de liberdade y
    // A ordem dos índices é a ordem dos elementos no AdVec.
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

    //RESOLVE AdVec*coef_sol
    TPZSolveMatrix * SolMat = new TPZSolveMatrix(npts_tot, nf_tot, AdVec, indexes);
    TPZFMatrix<REAL> coef_sol = cmesh->Solution();
    TPZFMatrix<REAL> result;
    SolMat->Multiply(coef_sol, result);

    // Cálculo do sigma
    REAL E = 200000000.;
    REAL nu =0.30;
    TPZFMatrix<REAL> sigma(npts_tot, 3, 0.);
#ifdef USING_TBB
    using namespace tbb;
    parallel_for(size_t(0),size_t(npts_tot),size_t(1),[&](size_t ipts)
                      {
                          sigma(ipts,0) = weight[ipts]*E/((1.-2.*nu)*(1.+nu))*((1.-nu)*result(2*ipts,0)+nu*result(2*ipts+1,1)); // Sigma x
                          sigma(ipts,1) = weight[ipts]*E/((1.-2.*nu)*(1.+nu))*((1.-nu)*result(2*ipts+1,1)+nu*result(2*ipts,0)); // Sigma y
                          sigma(ipts,2) = weight[ipts]*2*E/(2.*(1.+nu))*(result(2*ipts,1)+result(2*ipts+1,0))*0.5; // Sigma xy
                      }
                      );
#else
    for (int64_t ipts=0; ipts<npts_tot; ipts++) {
        sigma(ipts,0) = weight[ipts]*E/((1.-2.*nu)*(1.+nu))*((1.-nu)*result(2*ipts,0)+nu*result(2*ipts+1,1)); // Sigma x
        sigma(ipts,1) = weight[ipts]*E/((1.-2.*nu)*(1.+nu))*((1.-nu)*result(2*ipts+1,1)+nu*result(2*ipts,0)); // Sigma y
        sigma(ipts,2) = weight[ipts]*2.*E/(2.*(1.+nu))*(result(2*ipts,1)+result(2*ipts+1,0))*0.5; // Sigma xy
    }
#endif

    // -----------------------------------------------------------------------
    // CÁLCULO DAS FORÇAS NODAIS
    int64_t cont_cols=0;
    cont_elem=0;
    TPZManVector<int64_t> elem_vec_ids(nelem);

    // Vetor formado pela matriz de forças por elemento
    TPZVec<TPZFMatrix<REAL>> nodal_forces_el(nelem);

    for (int64_t iel=0; iel<nelem_c; iel++) {

        // Verificações
        TPZCompEl * cel = cmesh->Element(iel);
        if(!cel) continue;
        TPZGeoEl * gel = cel->Reference();
        if(!gel || gel->Dimension() != dim_mesh) continue;

        // Def da regra de integração
        TPZInterpolatedElement * cel_inter = dynamic_cast<TPZInterpolatedElement * >(cel);
        if (!cel_inter) DebugStop();
        TPZIntPoints * int_rule = &(cel_inter->GetIntegrationRule());

//        AdVec[cont_elem].Transpose();

        // Forças nodais na direção x
        nodal_forces_el[cont_elem].Redim(cel_inter->NShapeF(), 2);
        int64_t rows = AdVec[cont_elem].Cols();
        TPZFMatrix<STATE> nodal_forcex(rows,1,&nodal_forces_el[cont_elem](0,0), rows);
        TPZFMatrix<REAL> fv(2*int_rule->NPoints(),1,0.);
        for (int64_t ipts=0; ipts<int_rule->NPoints(); ipts++) {
            fv(2*ipts,0) = sigma(ipts+cont_cols,0); // Sigma x
            fv(2*ipts+1,0) = sigma(ipts+cont_cols,2); // Sigma xy
        }
        bool transpose = true;

        AdVec[cont_elem].MultAdd(fv, nodal_forcex, nodal_forcex,1.,1.,transpose);
//        nodal_forces_el[cont_elem].AddSub(0, 0, AdVec[cont_elem].operator*(fv));

        // Forças nodais na direção y
        for (int64_t ipts=0; ipts<int_rule->NPoints(); ipts++) {
            fv(2*ipts,0) = sigma(ipts+cont_cols,2); // Sigma xy
            fv(2*ipts+1,0) = sigma(ipts+cont_cols,1); // Sigma y
        }
        TPZFMatrix<STATE> nodal_forcey(rows,1,&nodal_forces_el[cont_elem](0,1), rows);
        AdVec[cont_elem].MultAdd(fv, nodal_forcex, nodal_forcex,1.,1.,transpose);
//        nodal_forces_el[cont_elem].AddSub(0, 1, AdVec[cont_elem].operator*(fv));

        cont_cols+=int_rule->NPoints();
        cont_elem++;
    }

    // Segunda versao utilizando cores
    // -----------------------------------------------------------------------
    // INÍCIO DA ASSEMBLAGEM
    int64_t nnodes_tot = cmesh->Reference()->NNodes();
    TPZManVector<REAL> nnodes_vec(nnodes_tot,0.);

    cont_elem = 0;

    TPZManVector<int> nelem_cor(nelem,-1); // vetor de cores
    TPZManVector<int64_t> elem_neighbour(8,0.); // definindo que um elem pode ter no maximo 8 elem vizinhos

    for (int64_t iel1=0; iel1<nelem_c; iel1++) {
        if(!cmesh->Element(iel1)) continue;
        TPZGeoEl * gel1 = cmesh->Element(iel1)->Reference();
        if(!gel1 ||  gel1->Dimension() != dim_mesh) continue;

        TPZManVector<int64_t> nodeindices;
        gel1->GetNodeIndices(nodeindices); // Armazena os nós do elemento finito

        // ** Início da verificação de qual coord é repetida:
        TPZGeoEl * gel2;

        // contadores
        int64_t cont_elem_cor = 0;
        int64_t cont_elem_neighbour = 0;

        // inicializa com nnodes_vec nulo, e preenche com 1 os nós repetidos
        nnodes_vec.Fill(0);
        for (int64_t iel2=0; iel2<nelem_c; iel2++) {
            if(!cmesh->Element(iel2)) continue;
            gel2 = cmesh->Element(iel2)->Reference();
            if(!gel2 ||  gel2->Dimension() != dim_mesh) continue;

            for (int64_t inode=0; inode<gel2->NNodes(); inode++) {
                if(std::find (nodeindices.begin(), nodeindices.end(), gel2->NodeIndex(inode)) != nodeindices.end()){
                    nnodes_vec[gel2->NodeIndex(inode)] = 1; // preenchendo nnodes_vec
                    elem_neighbour[cont_elem_neighbour] = cont_elem_cor; // preenche o vetor de elementos vizinhos ao elemento de análise
                    cont_elem_neighbour++;
                }
            }
            cont_elem_cor++;
        }
        // ** fim da verificação

        // Preenche a cor
        for (int64_t inodes_tot=0; inodes_tot<nnodes_tot; inodes_tot++) {
            cont_elem_cor = cont_elem;
            if (nnodes_vec[inodes_tot] == 1){
                for (int64_t iel2=iel1; iel2<nelem_c; iel2++) {
                    if(!cmesh->Element(iel2)) continue;
                    gel2 = cmesh->Element(iel2)->Reference();
                    if(!gel2 ||  gel2->Dimension() != dim_mesh) continue;

                    gel2->GetNodeIndices(nodeindices);
                    if (std::find(nodeindices.begin(), nodeindices.end(), inodes_tot) != nodeindices.end()){
                        nelem_cor[cont_elem_cor] = 1+nelem_cor[cont_elem];
                    }
                }
            }

            // Verifica se pode ser uma cor menor
            for (int64_t icor=0; icor<nelem_cor[cont_elem_cor]; icor++) {
                if (std::find(elem_neighbour.begin(), elem_neighbour.end(), icor) == elem_neighbour.end())
                    nelem_cor[cont_elem_cor] = icor;
                if (cont_elem==0)
                    nelem_cor[cont_elem_cor] = 0;
            }
            cont_elem_cor++;
        }
        cont_elem++;
    }

    // -----------------------------------------------------------------------
    // ASSEMBLAGEM POR COR
    int64_t ncoef = coef_sol.Rows();
    TPZManVector<REAL> nodal_forces_global2(ncoef,0.);
    int ncor = *std::max_element(nelem_cor.begin(), nelem_cor.end());

    TPZManVector<int64_t> assemble_cores(nf_tot*2, 0.);
    TPZManVector<int64_t> nf_por_cor(ncor+1, 0.);
    pos = 0;

    // VETOR CORES E NEQ
    for (int icor=0; icor<ncor+1; icor++) {
        auto it = std::find (nelem_cor.begin(), nelem_cor.end(), icor);
        while (std::find (it, nelem_cor.end(), icor) != nelem_cor.end()) {
            it = std::find (it, nelem_cor.end(), icor);
            int poscor = std::distance(nelem_cor.begin(), it);
            for (int iassemblecor = 0; iassemblecor<indexes_el[poscor].size(); iassemblecor++)
                assemble_cores[iassemblecor+pos] = indexes_el[poscor][iassemblecor];
            it++;
            pos += indexes_el[poscor].size();
            nf_por_cor[icor] += indexes_el[poscor].size()/2;
        }
    }

    // SE TERMINAR EM NÚMERO IMPAR DE CORES
    int64_t nf_cor = 0;
    if ( (ncor+1)%2 != 0) {

        auto it = std::find (nelem_cor.begin(), nelem_cor.end(), ncor);
        while (std::find (it, nelem_cor.end(), ncor) != nelem_cor.end()) {

            it = std::find (it, nelem_cor.end(), ncor);

            int iel = std::distance(nelem_cor.begin(), it);
            int64_t nf_el = indexes_el[iel].size()/2;
            nf_cor += nf_el;

            for (int64_t inf_el = 0; inf_el<nf_el*2; inf_el++) {
                int64_t id = assemble_cores[nf_tot - nf_cor + inf_el];
                if (id%2 == 0)
                    nodal_forces_global2[id] += nodal_forces_el[iel](inf_el/2,0);
                else
                    nodal_forces_global2[id] += nodal_forces_el[iel](inf_el/2,1);
            }
            it++;
        }
    }

    REAL ncor_to_assemble = (ncor+1)/2;
    while(ncor_to_assemble >= 0.5){
        for (int64_t cor = 2*ncor_to_assemble-1; cor>=int(ncor_to_assemble); cor--) {

            nf_cor += nf_por_cor[cor];
            auto it = std::find (nelem_cor.begin(), nelem_cor.end(), cor);
            int64_t nf_el = 0;

            while (std::find (it, nelem_cor.end(), cor) != nelem_cor.end()) {
                it = std::find (it, nelem_cor.end(), cor);
                int iel = std::distance(nelem_cor.begin(), it);

                for (int64_t inf_el = 0; inf_el<indexes_el[iel].size(); inf_el++) {
                    int64_t id = assemble_cores[nf_tot*2 - nf_cor*2 + nf_el*2 + inf_el];
                    if (id%2 == 0)
                        nodal_forces_global2[id] += nodal_forces_el[iel](inf_el/2,0);
                    else
                        nodal_forces_global2[id] += nodal_forces_el[iel](inf_el/2,1);
                }
                it++;
                nf_el += indexes_el[iel].size()/2;
            }
        }
        ncor_to_assemble = ncor_to_assemble/2;
    }

    // -----------------------------------------------------------------------
    // ASSEMBLAGEM "TRADICIONAL"
    TPZManVector<REAL> nodal_forces_global(ncoef, 0.);
    for (int64_t iel=0; iel<nelem; iel++) {
        int64_t nf_el = nodal_forces_el[iel].Rows();

        for (int64_t inf=0; inf<nf_el; inf++) {
            int64_t id = indexes_el[iel][2*inf];
            nodal_forces_global[id] += nodal_forces_el[iel](inf,0);

            id = indexes_el[iel][2*inf+1];
            nodal_forces_global[id] += nodal_forces_el[iel](inf,1);
        }
    }

    // -----------------------------------------------------------------------
    // COMPARANDO OS DOIS TIPOS DE ASSEMBLAGEM
    for (int64_t i=0; i<nodal_forces_global2.size(); i++) {
        std::cout << nodal_forces_global2[i]-nodal_forces_global[i] << std::endl;
    }

}
