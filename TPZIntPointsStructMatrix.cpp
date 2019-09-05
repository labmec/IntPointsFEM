//
// Created by natalia on 03/09/2019.
//
#ifdef USING_TBB
#include <tbb/parallel_for.h>
#endif

#include "TPZIntPointsStructMatrix.h"
#include "pzskylstrmatrix.h"
#include "pzbndcond.h"
#include "pzintel.h"
#include "pzsysmp.h"

TPZIntPointsStructMatrix::TPZIntPointsStructMatrix(TPZCompMesh *cmesh) : TPZSpStructMatrix(cmesh), fSparseMatrixLinear(), fRhsLinear(), fIntegrator(), fBCMaterialIds() {

    if (!cmesh->Reference()->Dimension()) {
        DebugStop();
    }
    fDimension = cmesh->Reference()->Dimension();
    fIAToSequence.resize(0);
    fJAToSequence.resize(0);
    fElColorIndexes.resize(0);
    fFirstColorIndex.resize(0);
}

TPZIntPointsStructMatrix::~TPZIntPointsStructMatrix() {

}

TPZStructMatrix * TPZIntPointsStructMatrix::Clone(){
    return new TPZIntPointsStructMatrix(*this);
}

TPZMatrix<STATE> * TPZIntPointsStructMatrix::Create(){

    if(!isBuilt()) {
        this->SetUpDataStructure(); // When basis functions are computed and storaged
    }

    TPZStack<int64_t> elgraph;
    TPZVec<int64_t> elgraphindex;
    fMesh->ComputeElGraph(elgraph,elgraphindex,fMaterialIds); // This method seems to be efficient.
    TPZMatrix<STATE> * mat = SetupMatrixData(elgraph, elgraphindex);

    TPZFYsmpMatrix<STATE> *stiff = dynamic_cast<TPZFYsmpMatrix<STATE> *> (mat);

    fIAToSequence = stiff->IA();
    fJAToSequence = stiff->JA();

    return mat;
}

TPZMatrix<STATE> *TPZIntPointsStructMatrix::CreateAssemble(TPZFMatrix<STATE> &rhs, TPZAutoPointer<TPZGuiInterface> guiInterface) {

    int64_t neq = fMesh->NEquations();
    TPZMatrix<STATE> *stiff = Create();
    rhs.Redim(neq,1);
    Assemble(*stiff,rhs,guiInterface);
    return stiff;
}

bool TPZIntPointsStructMatrix::isBuilt() {
    if(fIntegrator.IrregularBlocksMatrix().Rows() != 0) return true;
    else return false;
}

void TPZIntPointsStructMatrix::Assemble(TPZMatrix<STATE> & mat, TPZFMatrix<STATE> & rhs, TPZAutoPointer<TPZGuiInterface> guiInterface) {

    TPZFYsmpMatrix<STATE> &stiff = dynamic_cast<TPZFYsmpMatrix<STATE> &> (mat);
    TPZVec<STATE> &Kg = stiff.A();

    TPZVec<int> & dof_id = fIntegrator.DoFIndexes();
    TPZVec<int> & el_n_dofs = fIntegrator.IrregularBlocksMatrix().Blocks().fColSizes;
    TPZVec<int> & col_pos = fIntegrator.IrregularBlocksMatrix().Blocks().fColFirstIndex;

    int n_colors = fFirstColorIndex.size() - 1;

    for (int ic = 0; ic < n_colors; ic++) { //Serial by color

        #ifdef USING_TBB
            tbb::parallel_for(size_t(fFirstColorIndex[ic]),size_t(fFirstColorIndex[ic+1]),size_t(1),[&](size_t i) // Each set of colors in parallel
        #else
            for (int i = fFirstColorIndex[ic]; i < fFirstColorIndex[ic+1]; i++)
        #endif
            {
                int iel = fElColorIndexes[i];

                TPZFMatrix<STATE> Kel;
                fIntegrator.ComputeTangentMatrix(iel,Kel);

                int el_dof = el_n_dofs[iel];
                int pos = col_pos[iel];

                for (int i_dof = 0; i_dof < el_dof; i_dof++) {
                    int64_t i_dest = dof_id[pos + i_dof];

                    for (int j_dof = 0; j_dof < el_dof; j_dof++) {
                        int64_t j_dest = dof_id[pos + j_dof];

                        STATE val = Kel(i_dof,j_dof);
                        int64_t index = me(fIAToSequence, fJAToSequence, i_dest, j_dest);
                        Kg[index] += val;
                    }
                }
            }
        #ifdef USING_TBB
            );
        #endif
    }

    // Add boundary contribution
    auto it_end = fSparseMatrixLinear.MapEnd();
    for (auto it = fSparseMatrixLinear.MapBegin(); it!=it_end; it++) {
        int64_t row = it->first.first;
        int64_t col = it->first.second;
        STATE val = it->second + stiff.GetVal(row, col);
        stiff.PutVal(row, col, val);
    }

    // Residual assemble
    Assemble(rhs,guiInterface);
}

void TPZIntPointsStructMatrix::Assemble(TPZFMatrix<STATE> & rhs, TPZAutoPointer<TPZGuiInterface> guiInterface){
    int neq = fMesh->NEquations();

    rhs.Resize(neq, 1);
    rhs.Zero();
    fIntegrator.ResidualIntegration(fMesh->Solution(),rhs);
    rhs += fRhsLinear;
}

void TPZIntPointsStructMatrix::SetUpDataStructure() {

    if(isBuilt()) {
        std::cout << __PRETTY_FUNCTION__ << " Data structure has been setup." << std::endl;
        return;
    }

    TPZVec<int> element_indexes;
    ComputeDomainElementIndexes(element_indexes);

    ClassifyMaterialsByDimension();

    TPZIrregularBlocksMatrix::IrregularBlocks blocksData;
    SetUpIrregularBlocksData(element_indexes, blocksData);

    int64_t rows = blocksData.fRowFirstIndex[blocksData.fNumBlocks];
    int64_t cols = blocksData.fColFirstIndex[blocksData.fNumBlocks];
    TPZIrregularBlocksMatrix blocksMatrix(rows, cols);
    blocksMatrix.SetBlocks(blocksData);
    fIntegrator.SetIrregularBlocksMatrix(blocksMatrix);

    TPZVec<int> dof_indexes;
    SetUpIndexes(element_indexes, dof_indexes);
    fIntegrator.SetDoFIndexes(dof_indexes);

    TPZVec<int> colored_element_indexes;
    int ncolor;

    ColoredIndexes(element_indexes, dof_indexes, colored_element_indexes, ncolor);

    fIntegrator.SetColorIndexes(colored_element_indexes);
    fIntegrator.SetNColors(ncolor);

    AssembleBoundaryData();
}

void TPZIntPointsStructMatrix::ComputeDomainElementIndexes(TPZVec<int> &element_indexes) {

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

void TPZIntPointsStructMatrix::ClassifyMaterialsByDimension() {

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

void TPZIntPointsStructMatrix::AssembleBoundaryData() {

    int64_t neq = fMesh->NEquations();
    TPZStructMatrix str(fMesh);
    str.SetMaterialIds(fBCMaterialIds);
    str.SetNumThreads(fNumThreads);
    TPZAutoPointer<TPZGuiInterface> guiInterface;
    fRhsLinear.Resize(neq, 1);
    fRhsLinear.Zero();
    fSparseMatrixLinear.Resize(neq, neq);
    str.Assemble(fSparseMatrixLinear, fRhsLinear, guiInterface);
}

void TPZIntPointsStructMatrix::SetUpIrregularBlocksData(TPZVec<int> &element_indexes, TPZIrregularBlocksMatrix::IrregularBlocks &blocksData) {

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
    }

    blocksData.fStorage.resize(blocksData.fMatrixPosition[nblocks]);

#ifdef USING_TBB
    tbb::parallel_for(size_t(0),size_t(nblocks),size_t(1),[&](size_t iel)
#else
    for (int iel = 0; iel < nblocks; iel++)
#endif
    {

        TPZFNMatrix<8,REAL> dphiXY;
        TPZMaterialData data;
        TPZVec<REAL> qsi(fDimension);

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
        TPZFMatrix<REAL> B_el_loc(npts * n_sigma_entries, n_el_dof * fDimension, &blocksData.fStorage[pos_el], npts * n_sigma_entries * n_el_dof * fDimension);
        B_el_loc = B_el;
    }
#ifdef USING_TBB
    );
#endif
}

int TPZIntPointsStructMatrix::StressRateVectorSize(){

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

void TPZIntPointsStructMatrix::SetUpIndexes(TPZVec<int> &element_indexes, TPZVec<int> & dof_indexes) {

    int64_t nblocks = fIntegrator.IrregularBlocksMatrix().Blocks().fNumBlocks;
    int64_t rows = fIntegrator.IrregularBlocksMatrix().Rows();
    int64_t cols = fIntegrator.IrregularBlocksMatrix().Cols();
    TPZVec<int> & dof_positions = fIntegrator.IrregularBlocksMatrix().Blocks().fColFirstIndex;
    TPZVec<int> & npts_positions = fIntegrator.IrregularBlocksMatrix().Blocks().fRowFirstIndex;

    dof_indexes.resize(cols);
    int64_t npts = rows / StressRateVectorSize();
    TPZVec<REAL> weight(npts);

#ifdef USING_TBB
    tbb::parallel_for(size_t(0),size_t(nblocks),size_t(1),[&](size_t iel)
#else
    for (int iel = 0; iel < nblocks; iel++)
#endif
    {
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
        int64_t wit = 0;
        for (int64_t ipts = 0; ipts < el_npts; ipts++) {
            TPZVec<REAL> qsi(dim);
            REAL w;
            int_rule->Point(ipts, qsi, w);
            gel->Jacobian(qsi, jac, axes, detjac, jacinv);
            weight[npts_positions[iel]/3 + wit] = w * std::abs(detjac);
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
#ifdef USING_TBB
    );
#endif

    TPZMaterial *material = fMesh->FindMaterial(1);
    fIntegrator.ConstitutiveLawProcessor().SetMaterial(material);
    fIntegrator.ConstitutiveLawProcessor().SetUpDataByIntPoints(npts);
    fIntegrator.ConstitutiveLawProcessor().SetWeightVector(weight);
}

void TPZIntPointsStructMatrix::ColoredIndexes(TPZVec<int> &element_indexes, TPZVec<int> &indexes, TPZVec<int> &coloredindexes, int &ncolor) {

    int64_t nblocks = fIntegrator.IrregularBlocksMatrix().Blocks().fNumBlocks;
    int64_t cols = fIntegrator.IrregularBlocksMatrix().Cols();

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

#ifdef USING_TBB
    tbb::parallel_for(size_t(0),size_t(nblocks),size_t(1),[&](size_t iel)
#else
    for (int iel = 0; iel < nblocks; iel++)
#endif
    {
        int64_t elem_col = fIntegrator.IrregularBlocksMatrix().Blocks().fColSizes[iel];
        int64_t cont_cols = fIntegrator.IrregularBlocksMatrix().Blocks().fColFirstIndex[iel];

        for (int64_t icols = 0; icols < elem_col; icols++) {
            coloredindexes[cont_cols + icols] = indexes[cont_cols + icols] + elemcolor[iel]*neq;
        }
    }
#ifdef USING_TBB
    );
#endif

    std::map<int64_t, std::vector<int64_t> > color_map;
    for (int64_t iel = 0; iel < nblocks; iel++) {
        color_map[elemcolor[iel]].push_back(iel);
    }

    if (contcolor != color_map.size()) {
        DebugStop();
    }

    fElColorIndexes.resize(nblocks);
    fFirstColorIndex.resize(color_map.size()+1);

    int c_color = 0;

    fFirstColorIndex[c_color] = 0;
    for (auto color_data : color_map) {
        int n_el_per_color = color_data.second.size();
        int iel = fFirstColorIndex[c_color];
        for (int i = 0; i < n_el_per_color ; i++) {
            int el_index = color_data.second[i];
            fElColorIndexes[iel] = el_index;
            iel++;
        }
        c_color++;
        fFirstColorIndex[c_color] = n_el_per_color + fFirstColorIndex[c_color-1];
    }
}

int64_t TPZIntPointsStructMatrix::me(TPZVec<int64_t> &IA, TPZVec<int64_t> &JA, int64_t & i_dest, int64_t & j_dest) {
    for (int64_t k = IA[i_dest]; k < IA[i_dest + 1]; k++) {
        if (JA[k] == j_dest) {
            return k;
        }
    }
}

