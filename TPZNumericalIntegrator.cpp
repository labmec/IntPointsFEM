//
// Created by natalia on 24/05/19.
//

#include "TPZNumericalIntegrator.h"
#include "pzcmesh.h"
#include "Timer.h"
#include "pzintel.h"

TPZNumericalIntegrator::TPZNumericalIntegrator() : fBlockMatrix(0,0), fNColor(-1), fDoFIndexes(0), fColorIndexes(0), fConstitutiveLawProcessor(), fElColorIndex(0), fFirstColorIndex(0),
                                                   fColorLSequence(0), fFirstColorLIndex(0) {
#ifdef USING_CUDA
    dDoFIndexes.resize(0);
    dColorIndexes.resize(0);
    dElColorIndex.resize(0);
    dColorLSequence.resize(0);
#endif
}

TPZNumericalIntegrator::~TPZNumericalIntegrator() {

}

#ifdef USING_CUDA
void TPZNumericalIntegrator::Multiply(TPZVecGPU<REAL> &coef, TPZVecGPU<REAL> &delta_strain) {

    int64_t rows = fBlockMatrix.Rows();
    int64_t cols = fBlockMatrix.Cols();

    TPZVecGPU<REAL> gather_solution(cols);
    gather_solution.Zero();
    fCudaCalls.GatherOperation(cols, coef.getData(), gather_solution.getData(), dDoFIndexes.getData());

    delta_strain.resize(rows);
    delta_strain.Zero();
    fBlockMatrix.MultiplyVector(&gather_solution.getData()[0], &delta_strain.getData()[0], false);
}
#endif 

void TPZNumericalIntegrator::Multiply(TPZFMatrix<REAL> &coef, TPZFMatrix<REAL> &delta_strain) {

    int64_t rows = fBlockMatrix.Rows();
    int64_t cols = fBlockMatrix.Cols();

    TPZFMatrix<REAL> gather_solution(cols, 1);
    gather_solution.Zero();
    cblas_dgthr(cols, coef, &gather_solution(0, 0), &fDoFIndexes[0]);

    delta_strain.Resize(rows, 1);
    fBlockMatrix.MultiplyVector(&gather_solution(0,0), &delta_strain(0, 0), false);
}

#ifdef USING_CUDA
void TPZNumericalIntegrator::MultiplyTranspose(TPZVecGPU<REAL> &sigma, TPZVecGPU<REAL> &res) {
    int64_t cols = fBlockMatrix.Cols();

    int64_t ncolor = fNColor;
    int64_t neq = res.getSize();    

    TPZVecGPU<REAL> forces(cols);
    forces.Zero();
    res.resize(ncolor * neq);
    res.Zero();

    fBlockMatrix.MultiplyVector(&sigma.getData()[0], &forces.getData()[0], true);

    fCudaCalls.ScatterOperation(cols, forces.getData(), res.getData(), dColorIndexes.getData());

    int64_t colorassemb = ncolor / 2.;
    while (colorassemb > 0) {

        int64_t firsteq = (ncolor - colorassemb) * neq;
        fCudaCalls.DaxpyOperation(colorassemb * neq, 1., &res.getData()[firsteq], &res.getData()[0]); 

        ncolor -= colorassemb;
        colorassemb = ncolor/2;
    }
}
#endif

void TPZNumericalIntegrator::MultiplyTranspose(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &res) {
    int64_t cols = fBlockMatrix.Cols();

    int64_t ncolor = fNColor;
    int64_t neq = res.Rows();

    TPZFMatrix<REAL> forces(cols, 1);
    res.Resize(ncolor * neq, 1);
    res.Zero();

    fBlockMatrix.MultiplyVector(&sigma(0, 0), &forces(0, 0), true);

    cblas_dsctr(cols, forces, &fColorIndexes[0], &res(0,0));

    int64_t colorassemb = ncolor / 2.;
    while (colorassemb > 0) {

        int64_t firsteq = (ncolor - colorassemb) * neq;
        cblas_daxpy(colorassemb * neq, 1., &res(firsteq, 0), 1., &res(0, 0), 1.);

        ncolor -= colorassemb;
        colorassemb = ncolor/2;
    }
    res.Resize(neq, 1);
}

void TPZNumericalIntegrator::ResidualIntegration(TPZFMatrix<REAL> & solution ,TPZFMatrix<REAL> &rhs) {
    TPZFMatrix<REAL> delta_strain;
    TPZFMatrix<REAL> sigma;

    Multiply(solution, delta_strain);
    fConstitutiveLawProcessor.ComputeSigma(delta_strain, sigma);
    MultiplyTranspose(sigma, rhs); // Perform Residual integration using a global linear application B
}

#ifdef USING_CUDA
void TPZNumericalIntegrator::ResidualIntegration(TPZFMatrix<REAL> & solution ,TPZVecGPU<REAL> &rhs) {

    TPZVecGPU<REAL> d_solution(solution.Rows());
    d_solution.set(&solution(0,0), solution.Rows());

    TPZVecGPU<REAL> d_delta_strain;
    TPZVecGPU<REAL> d_sigma;

    Multiply(d_solution, d_delta_strain);
    fConstitutiveLawProcessor.ComputeSigma(d_delta_strain, d_sigma);
    MultiplyTranspose(d_sigma, rhs);
}
#endif

void TPZNumericalIntegrator::KAssembly(TPZFMatrix<REAL> & solution, TPZVec<STATE> & Kg, TPZFMatrix<STATE> & rhs) {
    TPZFMatrix<REAL> delta_strain, sigma, dep;

    Multiply(solution, delta_strain);
    fConstitutiveLawProcessor.ComputeSigmaDep(delta_strain, sigma, dep);
    MultiplyTranspose(sigma, rhs);

    Timer timer;   
    timer.TimeUnit(Timer::ESeconds);
    timer.TimerOption(Timer::EChrono);
    timer.Start();

    // Compute Kc
    int size = fBlockMatrix.Blocks().fMatrixStride[fBlockMatrix.Blocks().fNumBlocks];
    TPZVec<REAL> Kc(size,0.0);
#ifdef USING_TBB
     tbb::parallel_for(size_t(0),size_t(fBlockMatrix.Blocks().fNumBlocks),size_t(1),[&](size_t i)
#else
     for (int i = 0; i < fBlockMatrix.Blocks().fNumBlocks; i++)
#endif
     {
         int iel = fElColorIndex[i];
         int el_dofs = fBlockMatrix.Blocks().fColSizes[iel];

         // Compute Elementary Matrix.
         TPZFMatrix<STATE> K;
         ComputeTangentMatrix(iel,dep, K);
//         K.Print(std::cout);
         int stride = fBlockMatrix.Blocks().fMatrixStride[i];
         int c = stride;
        for(int i_dof = 0 ; i_dof < el_dofs; i_dof++){
            for(int j_dof = i_dof; j_dof < el_dofs; j_dof++){
                Kc[c] += K(i_dof,j_dof);
                c++;
            }
        }
    }
#ifdef USING_TBB
     );
#endif
     // Assemble K
     for (int ic = 0; ic < fNColor; ic++) {
         int first_l = fFirstColorLIndex[ic];
         int last_l = fFirstColorLIndex[ic + 1];
         int n_l_indexes = last_l - first_l;
         // Gather from Kg
         TPZVec<REAL> aux(n_l_indexes);
         cblas_dgthr(n_l_indexes, &Kg[0], &aux[0], &fColorLSequence[first_l]);

         // Contributing
         cblas_daxpy(n_l_indexes, 1., &Kc[first_l], 1., &aux[0],1);

         // Scatter to Kg
         cblas_dsctr(n_l_indexes, &aux[0], &fColorLSequence[first_l], &Kg[0]);
     }

         timer.Stop();
    std::cout << "K: Elasped time [sec] = " << timer.ElapsedTime() << std::endl;
}

#ifdef USING_CUDA
void TPZNumericalIntegrator::KAssembly(TPZFMatrix<REAL> & solution, TPZVecGPU<STATE> & Kg, TPZVecGPU<STATE> & rhs) {
    TPZVecGPU<REAL> d_solution(solution.Rows());
    d_solution.set(&solution(0,0), solution.Rows());

    TPZVecGPU<REAL> d_delta_strain, d_sigma, d_dep;

        Timer timer;   
    timer.TimeUnit(Timer::ESeconds);
    timer.TimerOption(Timer::EChrono);
    timer.Start();

    //Compute rhs
    Multiply(d_solution, d_delta_strain);
    fConstitutiveLawProcessor.ComputeSigmaDep(d_delta_strain, d_sigma, d_dep);
    MultiplyTranspose(d_sigma, rhs);

    // // Compute Kc
    TPZVecGPU<REAL> d_Kc(fColorLSequence.size());
    d_Kc.Zero();
    fCudaCalls.MatrixAssemble(d_Kc.getData(), d_dep.getData(), fBlockMatrix.Blocks().fNumBlocks, dElColorIndex.getData(), 
    fBlockMatrix.BlocksDev().dStorage.getData(), fBlockMatrix.BlocksDev().dRowSizes.getData(), fBlockMatrix.BlocksDev().dColSizes.getData(), 
    fBlockMatrix.BlocksDev().dRowFirstIndex.getData(), fBlockMatrix.BlocksDev().dColFirstIndex.getData(), fBlockMatrix.BlocksDev().dMatrixPosition.getData(),
    fBlockMatrix.BlocksDev().dMatrixStride.getData());

    // Assemble K
    for (int ic = 0; ic < fNColor; ic++) {
        int first_l = fFirstColorLIndex[ic];
        int last_l = fFirstColorLIndex[ic + 1];
        int n_l_indexes = last_l - first_l;
        TPZVecGPU<REAL> aux(n_l_indexes);
        fCudaCalls.GatherOperation(n_l_indexes, Kg.getData(), aux.getData(), &dColorLSequence.getData()[first_l]);
        fCudaCalls.DaxpyOperation(n_l_indexes, 1., &d_Kc.getData()[first_l], aux.getData());
        fCudaCalls.ScatterOperation(n_l_indexes, aux.getData(), Kg.getData(), &dColorLSequence.getData()[first_l]);
    }
             timer.Stop();
    std::cout << "K: Elasped time [sec] = " << timer.ElapsedTime() << std::endl;
}
#endif

#ifdef USING_CUDA
void TPZNumericalIntegrator::TransferDataToGPU() {
    fBlockMatrix.TransferDataToGPU();
    fConstitutiveLawProcessor.TransferDataToGPU();

    dDoFIndexes.resize(fDoFIndexes.size());
    dDoFIndexes.set(&fDoFIndexes[0], fDoFIndexes.size());

    dColorIndexes.resize(fColorIndexes.size());
    dColorIndexes.set(&fColorIndexes[0], fColorIndexes.size());

    dElColorIndex.resize(fElColorIndex.size());
    dElColorIndex.set(&fElColorIndex[0], fElColorIndex.size());    

    dColorLSequence.resize(fColorLSequence.size());
    dColorLSequence.set(&fColorLSequence[0], fColorLSequence.size());    
}
#endif

void TPZNumericalIntegrator::ComputeTangentMatrix(int64_t iel, TPZFMatrix<REAL> &Dep, TPZFMatrix<REAL> &K){

    int n_sigma_comps = 3;
    int el_npts = fBlockMatrix.Blocks().fRowSizes[iel]/n_sigma_comps;
    int el_dofs = fBlockMatrix.Blocks().fColSizes[iel];
    int first_el_ip = fBlockMatrix.Blocks().fRowFirstIndex[iel]/n_sigma_comps;

    K.Resize(el_dofs, el_dofs);
    K.Zero();

    int pos = fBlockMatrix.Blocks().fMatrixPosition[iel];
    TPZFMatrix<STATE> dep(3,3);
    TPZFMatrix<STATE> Bip(n_sigma_comps,el_dofs,0.0);
    TPZFMatrix<STATE> DeBip;
    int c1 = 0;
    int c2 = 0;
    for (int ip = 0; ip < el_npts; ip++) {
        for (int i = 0; i < n_sigma_comps; i++) {
            for (int j = 0; j < el_dofs; j++) {
                Bip(i,j) = fBlockMatrix.Blocks().fStorage[pos + c1];
                c1++;
            }
        }

        for (int i = 0; i < n_sigma_comps; i++) {
            for (int j = 0; j < n_sigma_comps; j++) {
                dep(i,j) = Dep(first_el_ip * n_sigma_comps * n_sigma_comps + c2, 0);
                c2++;
            }
        }

        dep.Multiply(Bip, DeBip);
        Bip.MultAdd(DeBip, K, K, 1., 1.0, 1);
    }
}

void TPZNumericalIntegrator::SetUpIrregularBlocksData(TPZCompMesh * cmesh) {

    int dimension = cmesh->Dimension();

    /// Number of elements
    int nblocks = fElementIndex.size();

    fBlockMatrix.Blocks().fNumBlocks = nblocks;
    fBlockMatrix.Blocks().fRowSizes.resize(nblocks);
    fBlockMatrix.Blocks().fColSizes.resize(nblocks);
    fBlockMatrix.Blocks().fMatrixPosition.resize(nblocks + 1);
    fBlockMatrix.Blocks().fRowFirstIndex.resize(nblocks + 1);
    fBlockMatrix.Blocks().fColFirstIndex.resize(nblocks + 1);
    fBlockMatrix.Blocks().fMatrixStride.resize(nblocks + 1);

    fBlockMatrix.Blocks().fMatrixPosition[0] = 0;
    fBlockMatrix.Blocks().fRowFirstIndex[0] = 0;
    fBlockMatrix.Blocks().fColFirstIndex[0] = 0;
    fBlockMatrix.Blocks().fMatrixStride[0] = 0;

    // @TODO Candidate for TBB ParallelScan
    // Example with lambda expression https://www.threadingbuildingblocks.org/docs/help/reference/algorithms/parallel_scan_func.html
    for(int iel = 0; iel < nblocks; iel++) {

        TPZCompEl *cel = cmesh->Element(fElementIndex[iel]);

        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement *>(cel);
        if (!cel_inter) DebugStop();
#ifdef PZDEBUG
        int dim = cel->Reference()->Dimension();
        if (dim !=cmesh->Dimension()) {
            DebugStop();
        }
#endif

        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());
        int64_t npts = int_rule->NPoints(); // number of integration points of the element
        int64_t n_el_dof = cel_inter->NShapeF(); // number of shape functions of the element (Element DoF)

        fBlockMatrix.Blocks().fRowSizes[iel] = StressRateVectorSize(dimension) * npts;
        fBlockMatrix.Blocks().fColSizes[iel] = n_el_dof * dimension;

        fBlockMatrix.Blocks().fMatrixPosition[iel + 1] = fBlockMatrix.Blocks().fMatrixPosition[iel] + fBlockMatrix.Blocks().fRowSizes[iel] * fBlockMatrix.Blocks().fColSizes[iel];
        fBlockMatrix.Blocks().fRowFirstIndex[iel + 1] =  fBlockMatrix.Blocks().fRowFirstIndex[iel] + fBlockMatrix.Blocks().fRowSizes[iel];
        fBlockMatrix.Blocks().fColFirstIndex[iel + 1] = fBlockMatrix.Blocks().fColFirstIndex[iel] + fBlockMatrix.Blocks().fColSizes[iel];
    }

    fBlockMatrix.Blocks().fStorage.resize(fBlockMatrix.Blocks().fMatrixPosition[nblocks]);

    int64_t rows = fBlockMatrix.Blocks().fRowFirstIndex[fBlockMatrix.Blocks().fNumBlocks];
    int64_t cols = fBlockMatrix.Blocks().fColFirstIndex[fBlockMatrix.Blocks().fNumBlocks];
    fBlockMatrix.Resize(rows, cols);

    TPZFNMatrix<8,REAL> dphiXY;
    TPZMaterialData data;
    TPZVec<REAL> qsi(dimension);

    // @TODO Candidate for TBB ParallelFor
    for (int iel = 0; iel < nblocks; ++iel) {

        TPZCompEl *cel = cmesh->Element(fElementIndex[iel]);

        int sigma_entries_times_npts      = fBlockMatrix.Blocks().fRowSizes[iel];
        int n_el_dof            = fBlockMatrix.Blocks().fColSizes[iel]/dimension;
        int pos_el              = fBlockMatrix.Blocks().fMatrixPosition[iel];


        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement *>(cel);
        if (!cel_inter) DebugStop();

#ifdef PZDEBUG
        int dim = cel->Reference()->Dimension();
        if (dim !=cmesh->Dimension()) {
            DebugStop();
        }
#endif

        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());

        int npts = int_rule->NPoints();
        int n_sigma_entries = StressRateVectorSize(dimension);

#ifdef PZDEBUG
        if (npts != sigma_entries_times_npts / n_sigma_entries) {
            DebugStop();
        }
#endif

        cel_inter->InitMaterialData(data);
        TPZFMatrix<REAL> B_el(npts*n_sigma_entries,n_el_dof * dimension,0.0);
        for (int64_t i_pts = 0; i_pts < npts; i_pts++) {

            REAL w;
            int_rule->Point(i_pts, qsi, w);
            cel_inter->ComputeRequiredData(data, qsi);

            data.axes.Transpose();
            data.axes.Multiply(data.dphix, dphiXY);

            for (int j_dim = 0; j_dim < 2; j_dim++){

                for (int i_dof = 0; i_dof < n_el_dof; i_dof++) {

                    REAL val = dphiXY(j_dim, i_dof);
                    for (int i_sigma_comp = 0; i_sigma_comp < dimension; i_sigma_comp++){

                        B_el(i_pts * n_sigma_entries + i_sigma_comp + j_dim, i_dof*2 + i_sigma_comp) = val;
                    }
                }
            }

        }
        B_el.Transpose();
        TPZFMatrix<REAL> B_el_loc(npts * n_sigma_entries, n_el_dof * dimension, &fBlockMatrix.Blocks().fStorage[pos_el], npts * n_sigma_entries * n_el_dof * dimension);
        B_el_loc = B_el;
    }
}

void TPZNumericalIntegrator::SetUpIndexes(TPZCompMesh * cmesh) {

    int64_t nblocks = fBlockMatrix.Blocks().fNumBlocks;
    int64_t rows = fBlockMatrix.Rows();
    int64_t cols = fBlockMatrix.Cols();
    TPZVec<int> & dof_positions = fBlockMatrix.Blocks().fColFirstIndex;

    fDoFIndexes.resize(cols);
    int64_t npts = rows / StressRateVectorSize(cmesh->Dimension());
    TPZVec<REAL> weight(npts);

    int64_t wit = 0;
    for (int iel = 0; iel < nblocks; ++iel) {

        int dof_pos = dof_positions[iel];
        TPZCompEl *cel = cmesh->Element(fElementIndex[iel]);


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
            TPZConnect &df = cmesh->ConnectVec()[id];
            int64_t conid = df.SequenceNumber();
            if (df.NElConnected() == 0 || conid < 0 || cmesh->Block().Size(conid) == 0) continue;
            int64_t pos = cmesh->Block().Position(conid);
            int64_t nsize = cmesh->Block().Size(conid);
            for (int64_t isize = 0; isize < nsize; isize++) {
                fDoFIndexes[dof_pos+i_dof] = pos + isize;
                i_dof++;
            }

        }
    }

    TPZMaterial *material = cmesh->FindMaterial(1);
    fConstitutiveLawProcessor.SetMaterial(material);
    fConstitutiveLawProcessor.SetUpDataByIntPoints(npts);
    fConstitutiveLawProcessor.SetWeightVector(weight);
}

void TPZNumericalIntegrator::SetUpColoredIndexes(TPZCompMesh * cmesh) {

    int64_t nblocks = fBlockMatrix.Blocks().fNumBlocks;
    int64_t cols = fBlockMatrix.Cols();

    TPZVec<int64_t> connects_vec(cmesh->NConnects(),0);
    TPZVec<int64_t> elemcolor(nblocks,-1);

    int64_t contcolor = 0;
    bool needstocontinue = true;

    while (needstocontinue)
    {
        int it = 0;
        needstocontinue = false;
        for (auto iel : fElementIndex) {
            TPZCompEl *cel = cmesh->Element(iel);
            if (!cel || cel->Dimension() != cmesh->Dimension()) continue;

            it++;
            if (elemcolor[it-1] != -1) continue;

            TPZStack<int64_t> connectlist;
            cmesh->Element(iel)->BuildConnectList(connectlist);
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

    fNColor = contcolor;
    fColorIndexes.resize(cols);
    int64_t neq = cmesh->NEquations();
    for (int64_t iel = 0; iel < nblocks; iel++) {
        int64_t elem_col = fBlockMatrix.Blocks().fColSizes[iel];
        int64_t cont_cols = fBlockMatrix.Blocks().fColFirstIndex[iel];

        for (int64_t icols = 0; icols < elem_col; icols++) {
            fColorIndexes[cont_cols + icols] = fDoFIndexes[cont_cols + icols] + elemcolor[iel]*neq;
        }
    }

    std::map<int64_t, std::vector<int64_t> > color_map;
    for (int64_t iel = 0; iel < nblocks; iel++) {
        color_map[elemcolor[iel]].push_back(iel);
    }

    if (contcolor != color_map.size()) {
        DebugStop();
    }

    fElColorIndex.resize(nblocks+1);
    fElColorIndex[nblocks] = 0;
    fFirstColorIndex.resize(color_map.size() + 1);

    int c_color = 0;

    fFirstColorIndex[c_color] = 0;
    for (auto color_data : color_map) {
        int n_el_per_color = color_data.second.size();
        int iel = fFirstColorIndex[c_color];
        for (int i = 0; i < n_el_per_color ; i++) {
            int el_index = color_data.second[i];
            fElColorIndex[iel] = el_index;
            iel++;
        }
        c_color++;
        fFirstColorIndex[c_color] = n_el_per_color + fFirstColorIndex[c_color - 1];
    }

    std::cout << "Number of colors = " << color_map.size() << std::endl;
}

void TPZNumericalIntegrator::FillLIndexes(TPZVec<int64_t> & IA, TPZVec<int64_t> & JA){

    TPZVec<int> & el_n_dofs = fBlockMatrix.Blocks().fColSizes;
    TPZVec<int> & cols_first_index = fBlockMatrix.Blocks().fColFirstIndex;

    int n_colors = fFirstColorIndex.size() - 1;
    fFirstColorLIndex.resize(n_colors + 1);
    fFirstColorLIndex[0]=0;
    int cont = 1;
    for (int ic = 0; ic < n_colors; ic++) {

        int first = fFirstColorIndex[ic];
        int last = fFirstColorIndex[ic + 1];
        int nel_per_color = last - first;
        int64_t c = 0;
        for (int i = 0; i < nel_per_color; i++) {
            int iel = fElColorIndex[first + i];
            int el_dof = el_n_dofs[iel];
            int n_entries = (el_dof*el_dof + el_dof)/2;
            c += n_entries;

            int val = fBlockMatrix.Blocks().fMatrixStride[cont-1];
            int value = val + (el_dof*el_dof + el_dof)/2;
            fBlockMatrix.Blocks().fMatrixStride[cont] = value;
            cont++;


        }
        fFirstColorLIndex[ic + 1] = c + fFirstColorLIndex[ic];
    }
    fColorLSequence.resize(fFirstColorLIndex[n_colors]);

    for (int ic = 0; ic < n_colors; ic++) {
        int first = fFirstColorIndex[ic];
        int last = fFirstColorIndex[ic + 1];
        int nel_per_color = last - first;
        int64_t c = fFirstColorLIndex[ic];
        for (int i = 0; i < nel_per_color; i++) {
            int iel = fElColorIndex[first + i];
            int el_dof = el_n_dofs[iel];
            int pos = cols_first_index[iel];
            for (int i_dof = 0; i_dof < el_dof; i_dof++) {
                int64_t i_dest = fDoFIndexes[pos + i_dof];
                for (int j_dof = i_dof; j_dof < el_dof; j_dof++) {
                    int64_t j_dest = fDoFIndexes[pos + j_dof];
                    int64_t l_index = me(IA, JA, i_dest, j_dest);
                    fColorLSequence[c] = l_index;
                    c++;
                }
            }
        }
    }
}

int64_t TPZNumericalIntegrator::me(TPZVec<int64_t> &IA, TPZVec<int64_t> &JA, int64_t & i_dest, int64_t & j_dest) {
    int64_t row(i_dest),col(j_dest);
    if (i_dest > j_dest) {
        int64_t temp = i_dest;
        row = col;
        col = temp;
    }
    for(int ic=IA[row] ; ic < IA[row+1]; ic++ ) {
        if ( JA[ic] == col )
        {
            return ic;
        }
    }
    return 0;
}

int TPZNumericalIntegrator::StressRateVectorSize(int dim){

    switch (dim) {
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