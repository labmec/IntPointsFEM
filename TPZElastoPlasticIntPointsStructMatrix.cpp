#include "TPZElastoPlasticIntPointsStructMatrix.h"
#include "pzintel.h"
#include "pzskylstrmatrix.h"
#include "pzmetis.h"
#include "TPZConstitutiveLawProcessor.h"
#include "TPZElasticCriterion.h"
#include "pzbndcond.h"

#ifdef USING_MKL
#include <mkl.h>
#endif


TPZElastoPlasticIntPointsStructMatrix::TPZElastoPlasticIntPointsStructMatrix(TPZCompMesh *cmesh) : TPZSymetricSpStructMatrix(cmesh), fSparseMatrixLinear(), fRhsLinear(), fCoefToGradSol(), fBCMaterialIds() {

    if (!cmesh->Reference()->Dimension()) {
        DebugStop();
    }
    fDimension = cmesh->Reference()->Dimension();
}

TPZElastoPlasticIntPointsStructMatrix::~TPZElastoPlasticIntPointsStructMatrix() {
}

TPZStructMatrix * TPZElastoPlasticIntPointsStructMatrix::Clone(){
    return new TPZElastoPlasticIntPointsStructMatrix(*this);
}

TPZMatrix<STATE> * TPZElastoPlasticIntPointsStructMatrix::Create(){

    if(!isBuilt()) {
        this->SetUpDataStructure(); // When basis functions are computed and storaged
    }
    
    TPZStack<int64_t> elgraph;
    TPZVec<int64_t> elgraphindex;
    fMesh->ComputeElGraph(elgraph,elgraphindex,fMaterialIds); // This method seems to be efficient.
    TPZMatrix<STATE> * mat = SetupMatrixData(elgraph, elgraphindex);
    
    /// Sparsify global indexes
    // Filling local std::map
    TPZSYsmpMatrix<STATE> *stiff = dynamic_cast<TPZSYsmpMatrix<STATE> *> (mat);
    TPZVec<int64_t> &IA = stiff->IA();
    TPZVec<int64_t> &JA = stiff->JA();
    {
        int64_t n_ia = IA.size();
        int64_t l = 0;
        for (int64_t i = 0; i < n_ia - 1 ; i++) {
            int NNZ = IA[i+1] - IA[i];
            for (int64_t j = IA[i]; j < NNZ + IA[i]; j++) {
                m_i_j_to_sequence[i][JA[j]] = l;
                l++;
            }
        }
        
    }
    
    return mat;
}

TPZMatrix<STATE> *TPZElastoPlasticIntPointsStructMatrix::CreateAssemble(TPZFMatrix<STATE> &rhs, TPZAutoPointer<TPZGuiInterface> guiInterface) {

    int64_t neq = fMesh->NEquations();
    TPZMatrix<STATE> *stiff = Create(); // @TODO:: Requires optimization.
    rhs.Redim(neq,1);
    Assemble(*stiff,rhs,guiInterface);
    return stiff;
}

void TPZElastoPlasticIntPointsStructMatrix::SetUpDataStructure() {

    if(isBuilt()) {
        std::cout << __PRETTY_FUNCTION__ << " Data structure has been setup." << std::endl;
        return;
    }
    
    TPZVec<int> element_indexes;
    ComputeDomainElementIndexes(element_indexes);
    
    ClassifyMaterialsByDimension();

    TPZIrregularBlocksMatrix::IrregularBlocks blocksData;
    this->SetUpIrregularBlocksData(element_indexes, blocksData);

    int64_t rows = blocksData.fRowFirstIndex[blocksData.fNumBlocks];
    int64_t cols = blocksData.fColFirstIndex[blocksData.fNumBlocks];
    TPZIrregularBlocksMatrix blocksMatrix(rows, cols);
    blocksMatrix.SetBlocks(blocksData);
    fCoefToGradSol.SetIrregularBlocksMatrix(blocksMatrix);

    TPZVec<int> dof_indexes;
    this->SetUpIndexes(element_indexes, dof_indexes);
    fCoefToGradSol.SetDoFIndexes(dof_indexes);

    TPZVec<int> colored_element_indexes;
    int ncolor;
    
    this->ColoredIndexes(element_indexes, dof_indexes, colored_element_indexes, ncolor);
    fCoefToGradSol.SetColorIndexes(colored_element_indexes);
    fCoefToGradSol.SetNColors(ncolor);

    AssembleBoundaryData();

#ifdef USING_CUDA
    std::cout << "Transfering data to GPU..." << std::endl;
    fCoefToGradSol.TransferDataToGPU();
    std::cout << "Done!" << std::endl;
#endif

}



void TPZElastoPlasticIntPointsStructMatrix::Assemble(TPZMatrix<STATE> & mat, TPZFMatrix<STATE> & rhs, TPZAutoPointer<TPZGuiInterface> guiInterface) {
    
    TPZSYsmpMatrix<STATE> &stiff = dynamic_cast<TPZSYsmpMatrix<STATE> &> (mat);
    TPZVec<STATE> &Kg = stiff.A();
    
    TPZVec<int> &indexes = fCoefToGradSol.DoFIndexes();
    int64_t n_vols = fCoefToGradSol.IrregularBlocksMatrix().Blocks().fNumBlocks;
    TPZVec<int> & el_n_dofs = fCoefToGradSol.IrregularBlocksMatrix().Blocks().fColSizes;
    TPZVec<int> & cols_first_index = fCoefToGradSol.IrregularBlocksMatrix().Blocks().fColFirstIndex;


    
    /// implement OptV2
    if (1) {

        /// Serial
        for (int iel = 0; iel < n_vols; iel++) {
            
            int el_dof = el_n_dofs[iel];
            int pos = cols_first_index[iel];

            /// Compute Elementary Matrix.
            TPZFMatrix<STATE> K;
            fCoefToGradSol.ComputeTangetMatrix(iel,K);
            for (int i_dof = 0; i_dof < el_dof; i_dof++) {
                
                int i_dest = indexes[pos + i_dof];

                for (int j_dof = 0; j_dof < el_dof; j_dof++) {

                    int j_dest = indexes[pos + j_dof];
                    
                    STATE val = K(i_dof,j_dof);
                    int64_t  index = m_i_j_to_sequence[i_dest][j_dest];
                    if (i_dest <= j_dest) Kg[index] += val;
                }
            }
        }

    }

    
//    {
//        int n_ia = IA.size();
//        int n_ja = JA.size();
//        int l = 0;
//        std::ofstream kg_out("kg.txt");
//        for (int i = 0; i < n_ia - 1 ; i++) {
//            int NNZ = IA[i+1] - IA[i];
//            kg_out << "Row i = " << i << ", NNZ = " << IA[i+1] - IA[i] << " " ;
//            kg_out << "IA[" << i << "] = " << IA[i] << std::endl;
//            for (int j = IA[i]; j < NNZ + IA[i]; j++) {
//                kg_out << "     JA[" << j << "] = " << JA[j] << " " ;
//                kg_out << "     k " << stiff.GetVal(i, JA[j]) << " and K_g[" << l << "] = " <<  Kg[l] << std::endl;
//                l++;
//            }
//        }
//        for (int l = 0; l < Kg.size(); l++) {
//            kg_out << " K_g[" << l << "] = " <<  Kg[l] << std::endl;
//        }
//    }
    
    auto it_end = fSparseMatrixLinear.MapEnd();

    for (auto it = fSparseMatrixLinear.MapBegin(); it!=it_end; it++) {
        int64_t row = it->first.first;
        int64_t col = it->first.second;
        STATE val = it->second;
        Kg[m_i_j_to_sequence[row][col]] += val;
    }

    Assemble(rhs,guiInterface);
}

int TPZElastoPlasticIntPointsStructMatrix::StressRateVectorSize(){
    
    switch (fDimension) {
        case 1:{
            return 1;
        }
        break;
        case 2:{
            return 3;
        }
            break;
        case 3:{
            return 6;
        }
            break;
        default:
        {
            DebugStop();
            return 0;
        }
        break;
    }
}

void TPZElastoPlasticIntPointsStructMatrix::Assemble(TPZFMatrix<STATE> & rhs, TPZAutoPointer<TPZGuiInterface> guiInterface){

    if(!isBuilt()) {
        this->SetUpDataStructure();
    }

    int neq = fMesh->NEquations();

#ifdef USING_CUDA
    
    TPZVecGPU<REAL> solution(neq);
    solution.set(&fMesh->Solution()(0,0), neq);

    TPZVecGPU<REAL> dgrad_u;
    TPZVecGPU<REAL> drhs(neq);
    rhs.Resize(neq, 1);
    rhs.Zero();

    fCoefToGradSol.Multiply(solution, dgrad_u);

    TPZFMatrix<REAL> grad_u(dgrad_u.getSize(),1);
    TPZFMatrix<REAL> sigma;
    dgrad_u.get(&grad_u(0,0), dgrad_u.getSize());

    fConstitutiveLawProcessor.ComputeSigma(grad_u, sigma);

    TPZVecGPU<REAL> dsigma(sigma.Rows());
    dsigma.set(&sigma(0,0), sigma.Rows());

    fCoefToGradSol.MultiplyTranspose(dsigma, drhs);
    drhs.get(&rhs(0,0), neq);
    
#else
    
    
    rhs.Resize(neq, 1);
    rhs.Zero();
    fCoefToGradSol.ResidualIntegration(fMesh->Solution(),rhs);
    
#endif
    rhs += fRhsLinear;
}

void TPZElastoPlasticIntPointsStructMatrix::AssembleBoundaryData() {
    
    int64_t neq = fMesh->NEquations();
    TPZStructMatrix str(fMesh);
    str.SetMaterialIds(fBCMaterialIds);
    TPZAutoPointer<TPZGuiInterface> guiInterface;
    fRhsLinear.Resize(neq, 1);
    fRhsLinear.Zero();
    fSparseMatrixLinear.Resize(neq, neq);
    str.Assemble(fSparseMatrixLinear, fRhsLinear, guiInterface);
}

void TPZElastoPlasticIntPointsStructMatrix::ComputeDomainElementIndexes(TPZVec<int> &element_indexes) {
    
    TPZStack<int> el_indexes_loc;
    for (int64_t i = 0; i < fMesh->NElements(); i++) {
        TPZCompEl *cel = fMesh->Element(i);
        if (!cel) continue;
        TPZGeoEl *gel = cel->Reference();
        if (!gel) continue;
        if(gel->Dimension() == fDimension){
            el_indexes_loc.Push(cel->Index());
        }
    }
    element_indexes = el_indexes_loc;
}

void TPZElastoPlasticIntPointsStructMatrix::ClassifyMaterialsByDimension() {
    
    for (auto material : fMesh->MaterialVec()) {
        TPZBndCond * bc_mat = dynamic_cast<TPZBndCond *>(material.second);
        bool domain_material_Q = !bc_mat;
        if (domain_material_Q) {
            fMaterialIds.insert(material.first);
        }else{
            fBCMaterialIds.insert(material.first);
        }
    }
}

void TPZElastoPlasticIntPointsStructMatrix::SetUpIrregularBlocksData(TPZVec<int> &element_indexes, TPZIrregularBlocksMatrix::IrregularBlocks &blocksData) {
    
    /// Number of elements
    int nblocks = element_indexes.size();

    blocksData.fNumBlocks = nblocks;
    blocksData.fRowSizes.resize(nblocks);
    blocksData.fColSizes.resize(nblocks);
    blocksData.fMatrixPosition.resize(nblocks + 1);
    blocksData.fRowFirstIndex.resize(nblocks + 1);
    blocksData.fColFirstIndex.resize(nblocks + 1);

    blocksData.fMatrixPosition[0] = 0;
    blocksData.fRowFirstIndex[0] = 0;
    blocksData.fColFirstIndex[0] = 0;

    blocksData.fRowRowPosition.resize(nblocks + 1);
    blocksData.fColColPosition.resize(nblocks + 1);
    blocksData.fRowRowPosition[0] = 0;
    blocksData.fColColPosition[0] = 0;
    
    // @TODO Candidate for TBB ParallelScan
    // Example with lambda expression https://www.threadingbuildingblocks.org/docs/help/reference/algorithms/parallel_scan_func.html
    for(int iel = 0; iel < nblocks; iel++) {
        
        TPZCompEl *cel = fMesh->Element(element_indexes[iel]);
        
        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement *>(cel);
        if (!cel_inter) DebugStop();
#ifdef PZDEBUG
        int dim = cel->Reference()->Dimension();
        if (dim !=fDimension) {
            DebugStop();
        }
#endif
        
        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());
        int64_t npts = int_rule->NPoints(); // number of integration points of the element
        int64_t n_el_dof = cel_inter->NShapeF(); // number of shape functions of the element (Element DoF)

        blocksData.fRowSizes[iel] = StressRateVectorSize() * npts;
        blocksData.fColSizes[iel] = n_el_dof * fDimension;
        
        blocksData.fMatrixPosition[iel + 1] = blocksData.fMatrixPosition[iel] + blocksData.fRowSizes[iel] * blocksData.fColSizes[iel];
        blocksData.fRowFirstIndex[iel + 1] =  blocksData.fRowFirstIndex[iel] + blocksData.fRowSizes[iel];
        blocksData.fColFirstIndex[iel + 1] = blocksData.fColFirstIndex[iel] + blocksData.fColSizes[iel];

        blocksData.fRowRowPosition[iel + 1] = blocksData.fRowRowPosition[iel] + blocksData.fRowSizes[iel] * blocksData.fRowSizes[iel];
        blocksData.fColColPosition[iel + 1] = blocksData.fColColPosition[iel] + blocksData.fColSizes[iel] * blocksData.fColSizes[iel];
    }

    blocksData.fStorage.resize(blocksData.fMatrixPosition[nblocks]);
    
    
    
    TPZFNMatrix<8,REAL> dphiXY;
    TPZMaterialData data;
    TPZVec<REAL> qsi(fDimension);
    
    // @TODO Candidate for TBB ParallelFor
    for (int iel = 0; iel < nblocks; ++iel) {
        
        TPZCompEl *cel = fMesh->Element(element_indexes[iel]);
        
        int sigma_entries_times_npts      = blocksData.fRowSizes[iel];
        int n_el_dof            = blocksData.fColSizes[iel]/fDimension;
        int pos_el              = blocksData.fMatrixPosition[iel];
        
        
        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement *>(cel);
        if (!cel_inter) DebugStop();
        
#ifdef PZDEBUG
        int dim = cel->Reference()->Dimension();
        if (dim !=fDimension) {
            DebugStop();
        }
#endif
        
        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());
        
        int npts = int_rule->NPoints();
        int n_sigma_entries = StressRateVectorSize();
        
#ifdef PZDEBUG
        if (npts != sigma_entries_times_npts / n_sigma_entries) {
            DebugStop();
        }
#endif

        cel_inter->InitMaterialData(data);
        TPZFMatrix<REAL> B_el(npts*n_sigma_entries,n_el_dof * fDimension,0.0);
        for (int64_t i_pts = 0; i_pts < npts; i_pts++) {

            REAL w;
            int_rule->Point(i_pts, qsi, w);
            cel_inter->ComputeRequiredData(data, qsi);

            data.axes.Transpose();
            data.axes.Multiply(data.dphix, dphiXY);

//            dphiXY.Print("b = ",std::cout,EMathematicaInput);
            
            for (int j_dim = 0; j_dim < 2; j_dim++){
                
                for (int i_dof = 0; i_dof < n_el_dof; i_dof++) {
                    
                    REAL val = dphiXY(j_dim, i_dof);
                    for (int i_sigma_comp = 0; i_sigma_comp < fDimension; i_sigma_comp++){

                        B_el(i_pts * n_sigma_entries + i_sigma_comp + j_dim, i_dof*2 + i_sigma_comp) = val;
                    }
                }
            }

        }
        B_el.Transpose();
//        B_el.Print("Bel = ",std::cout,EMathematicaInput);
        TPZFMatrix<REAL> B_el_loc(npts * n_sigma_entries, n_el_dof * fDimension, &blocksData.fStorage[pos_el], npts * n_sigma_entries * n_el_dof * fDimension);
        B_el_loc = B_el;
    }
}

void TPZElastoPlasticIntPointsStructMatrix::SetUpIndexes(TPZVec<int> &element_indexes, TPZVec<int> & dof_indexes) {
    
    int64_t nblocks = fCoefToGradSol.IrregularBlocksMatrix().Blocks().fNumBlocks;
    int64_t rows = fCoefToGradSol.IrregularBlocksMatrix().Rows();
    int64_t cols = fCoefToGradSol.IrregularBlocksMatrix().Cols();
    TPZVec<int> & dof_positions = fCoefToGradSol.IrregularBlocksMatrix().Blocks().fColFirstIndex;

    dof_indexes.resize(cols);
    int64_t npts = rows / StressRateVectorSize();
    TPZVec<REAL> weight(npts);

    int64_t wit = 0;
    for (int iel = 0; iel < nblocks; ++iel) {
        
        int dof_pos = dof_positions[iel];
        TPZCompEl *cel = fMesh->Element(element_indexes[iel]);
        
        
        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement *>(cel);
        if (!cel_inter) DebugStop();
        TPZGeoEl * gel = cel->Reference();
        if (!gel) DebugStop();
        
        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());

        int64_t el_npts = int_rule->NPoints(); // number of integration points of the element
        int64_t dim = cel_inter->Dimension(); //dimension of the element

        TPZFMatrix<REAL> jac,axes, jacinv;
        REAL detjac;
        for (int64_t ipts = 0; ipts < el_npts; ipts++) {
            TPZVec<REAL> qsi(dim);
            REAL w;
            int_rule->Point(ipts, qsi, w);
            gel->Jacobian(qsi, jac, axes, detjac, jacinv);
            weight[wit] = w * std::abs(detjac);
            wit++;
        }

        int64_t ncon = cel->NConnects();
        int i_dof = 0;
        for (int64_t icon = 0; icon < ncon; icon++) {
            int64_t id = cel->ConnectIndex(icon);
            TPZConnect &df = fMesh->ConnectVec()[id];
            int64_t conid = df.SequenceNumber();
            if (df.NElConnected() == 0 || conid < 0 || fMesh->Block().Size(conid) == 0) continue;
            int64_t pos = fMesh->Block().Position(conid);
            int64_t nsize = fMesh->Block().Size(conid);
            for (int64_t isize = 0; isize < nsize; isize++) {
                dof_indexes[dof_pos+i_dof] = pos + isize;
                i_dof++;
            }
            
        }
    }

    TPZMaterial *material = fMesh->FindMaterial(1);
    fCoefToGradSol.ConstitutiveLawProcessor().SetMaterial(material);
    fCoefToGradSol.ConstitutiveLawProcessor().SetIntPoints(npts);
    fCoefToGradSol.ConstitutiveLawProcessor().SetWeightVector(weight);
}

void TPZElastoPlasticIntPointsStructMatrix::ColoredIndexes(TPZVec<int> &element_indexes, TPZVec<int> &indexes, TPZVec<int> &coloredindexes, int &ncolor) {
    
    int64_t nblocks = fCoefToGradSol.IrregularBlocksMatrix().Blocks().fNumBlocks;
    int64_t cols = fCoefToGradSol.IrregularBlocksMatrix().Cols();

    TPZVec<int64_t> connects_vec(fMesh->NConnects(),0);
    TPZVec<int64_t> elemcolor(nblocks,-1);

    int64_t contcolor = 0;
    bool needstocontinue = true;

    while (needstocontinue)
    {
        int it = 0;
        needstocontinue = false;
        for (auto iel : element_indexes) {
            TPZCompEl *cel = fMesh->Element(iel);
            if (!cel || cel->Dimension() != fMesh->Dimension()) continue;

            it++;
            if (elemcolor[it-1] != -1) continue;

            TPZStack<int64_t> connectlist;
            fMesh->Element(iel)->BuildConnectList(connectlist);
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
            for (icon = 0; icon < ncon; icon++) {
                connects_vec[connectlist[icon]] = 1;
            }
        }
        contcolor++;
        connects_vec.Fill(0);
    }

    ncolor = contcolor;
    coloredindexes.resize(cols);
    int64_t neq = fMesh->NEquations();
    for (int64_t iel = 0; iel < nblocks; iel++) {
        int64_t elem_col = fCoefToGradSol.IrregularBlocksMatrix().Blocks().fColSizes[iel];
        int64_t cont_cols = fCoefToGradSol.IrregularBlocksMatrix().Blocks().fColFirstIndex[iel];

        for (int64_t icols = 0; icols < elem_col; icols++) {
            coloredindexes[cont_cols + icols] = indexes[cont_cols + icols] + elemcolor[iel]*neq;
        }
    }
}

void TPZElastoPlasticIntPointsStructMatrix::Dep(TPZVec<REAL> &depxx, TPZVec<REAL> &depyy, TPZVec<REAL> &depxy) {
    
    int nblocks = fCoefToGradSol.IrregularBlocksMatrix().Blocks().fNumBlocks;
    int sizedep = fCoefToGradSol.IrregularBlocksMatrix().Blocks().fRowRowPosition[nblocks];
    // for (int i = 0; i < nblocks; ++i) {
    //     int rows = fCoefToGradSol.IrregularBlocksMatrix().Blocks().fRowSizes[i];
    //     sizedep += rows * rows;
    // }

    depxx.resize(sizedep);
    depyy.resize(sizedep);
    depxy.resize(sizedep);
    depxx.Fill(0.);
    depyy.Fill(0.);
    depxy.Fill(0.);

    int depel_pos = 0;
    for (int iel = 0; iel < fMesh->NElements(); ++iel) {
        TPZCompEl *cel = fMesh->Element(iel);
        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement *>(cel);
        if (!cel_inter) DebugStop();
        if (cel->Reference()->Dimension() != fMesh->Dimension()) continue;
        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());
        TPZMaterialData data;
        cel_inter->InitMaterialData(data);

        TPZMaterial *cel_mat = cel->Material();
        TPZMatElastoPlastic2D<TPZPlasticStepPV<TPZYCMohrCoulombPV, TPZElasticResponse>, TPZElastoPlasticMem> *mat = dynamic_cast<TPZMatElastoPlastic2D<TPZPlasticStepPV<TPZYCMohrCoulombPV, TPZElasticResponse>, TPZElastoPlasticMem> *>(cel_mat);

        int64_t npts = int_rule->NPoints(); // number of integration points of the element
        int64_t dim = cel_inter->Dimension(); //dimension of the element

        for (int64_t ipts = 0; ipts < npts; ipts++) {
            TPZVec<REAL> qsi(dim);
            REAL w;
            int_rule->Point(ipts, qsi, w);
            cel_inter->ComputeRequiredData(data, qsi);
            REAL weight = w * fabs(data.detjac);

            TPZTensor<REAL> deltastrain;
            TPZTensor<REAL> stress;
            TPZFMatrix<REAL> dep(6,6);
            deltastrain.Zero();
            stress.Zero();
            dep.Zero();
            mat->GetPlasticModel().ApplyStrainComputeSigma(deltastrain,stress,&dep);

            int pos1 = depel_pos + ipts * (dim * dim * npts + dim);
            int pos2 = depel_pos + ipts * (dim * dim * npts + dim) + 1;
            int pos3 = depel_pos + ipts * (dim * dim * npts + dim) + dim * npts;
            int pos4 = depel_pos + ipts * (dim * dim * npts + dim) + dim * npts + 1;

            depxx[pos1] = weight * dep.GetVal(_XX_, _XX_);
            depxx[pos2] = dep.GetVal(_XX_, _XY_) * 0.5;
            depxx[pos3] = dep.GetVal(_XY_, _XX_);
            depxx[pos4] = weight * dep.GetVal(_XY_, _XY_) * 0.5;

            depyy[pos1] = weight * dep.GetVal(_XY_, _XY_) * 0.5;
            depyy[pos2] = dep.GetVal(_XY_, _YY_);
            depyy[pos3] = dep.GetVal(_YY_, _XY_) * 0.5;
            depyy[pos4] = weight * dep.GetVal(_YY_, _YY_);

            depxy[pos1] = dep.GetVal(_XX_, _XY_) * 0.5;
            depxy[pos2] = weight * dep.GetVal(_XX_, _YY_);
            depxy[pos3] = weight * dep.GetVal(_XY_, _XY_) * 0.5;
            depxy[pos4] = dep.GetVal(_XY_, _YY_);
        }
        depel_pos = depel_pos + (npts * dim) * (npts * dim);
    }
}
