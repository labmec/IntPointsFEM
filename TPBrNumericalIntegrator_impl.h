//
//  TPBrNumericalIntegrator.hpp
//  IntPointsFEM
//
//  Created by Omar Durán on 9/5/19.
//

#include "TPBrNumericalIntegrator.h"
#include "mkl.h"
#include "pzsysmp.h"
#include "pzintel.h"
#include "pzcmesh.h"

#ifdef USING_TBB
#include <tbb/parallel_for.h>
#endif

template<class T, class MEM>
TPBrNumericalIntegrator<T, MEM>::TPBrNumericalIntegrator() : fElementIndexes(0), fBlockMatrix(0, 0), fMaterial(), fConstitutiveLawProcessor(),
                                                            fDoFIndexes(0), fColorIndexes(0), fNColor(-1), fMaterialRegionElColorIndexes(0),
                                                             fMaterialRegionFirstColorIndex(0) {
}

template<class T, class MEM>
TPBrNumericalIntegrator<T, MEM>::~TPBrNumericalIntegrator() {

}

template<class T, class MEM>
bool TPBrNumericalIntegrator<T, MEM>::isBuilt() {
    if (fBlockMatrix.Rows() != 0) return true;
    else return false;
}

template<class T, class MEM>
void TPBrNumericalIntegrator<T, MEM>::Multiply(TPZFMatrix<REAL> &coef, TPZFMatrix<REAL> &delta_strain) {
    int64_t rows = fBlockMatrix.Rows();
    int64_t cols = fBlockMatrix.Cols();

    TPZFMatrix<REAL> gather_solution(cols, 1);
    gather_solution.Zero();
    cblas_dgthr(cols, coef, &gather_solution(0, 0), &fDoFIndexes[0]);

    delta_strain.Resize(rows, 1);
    fBlockMatrix.MultiplyVector(gather_solution, &delta_strain(0, 0), false);
}

template<class T, class MEM>
void TPBrNumericalIntegrator<T, MEM>::MultiplyTranspose(TPZFMatrix<REAL> &sigma, TPZFMatrix<REAL> &res) {
    int64_t cols = fBlockMatrix.Cols();

    int64_t ncolor = fNColor;
    int64_t neq = res.Rows();

    TPZFMatrix<REAL> forces(cols, 1);
    res.Resize(ncolor * neq, 1);
    res.Zero();

    fBlockMatrix.MultiplyVector(sigma, &forces(0, 0), true);

    cblas_dsctr(cols, forces, &fColorIndexes[0], &res(0, 0));

    int64_t colorassemb = ncolor / 2.;
    while (colorassemb > 0) {

        int64_t firsteq = (ncolor - colorassemb) * neq;
        cblas_daxpy(colorassemb * neq, 1., &res(firsteq, 0), 1., &res(0, 0), 1.);

        ncolor -= colorassemb;
        colorassemb = ncolor / 2;
    }
    res.Resize(neq, 1);

}

template<class T, class MEM>
void TPBrNumericalIntegrator<T, MEM>::ResidualIntegration(TPZFMatrix<REAL> &solution, TPZFMatrix<REAL> &rhs) {
    TPZFMatrix<REAL> delta_strain;
    TPZFMatrix<REAL> sigma;

    Multiply(solution, delta_strain);
    ofstream out("tpbr.txt");
    delta_strain.Print(out);
    out.flush();
    fConstitutiveLawProcessor.ComputeSigma(delta_strain, sigma);
    MultiplyTranspose(sigma, rhs); // Perform Residual integration using a global linear application B
}

template<class T, class MEM>
void TPBrNumericalIntegrator<T, MEM>::KAssembly(TPZVec<STATE> &Kg, TPZVec<int64_t> &IAToSequence,
                                                TPZVec<int64_t> &JAToSequence) {

    TPZVec<int> &dof_id = DoFIndexes();
    TPZVec<int> &el_n_dofs = IrregularBlocksMatrix().Blocks().fColSizes;
    TPZVec<int> &col_pos = IrregularBlocksMatrix().Blocks().fColFirstIndex;

    for (int ic = 0; ic < fNColor; ic++) { //Serial by color

#ifdef USING_TBB
        tbb::parallel_for(size_t(fMaterialRegionFirstColorIndex[ic]),size_t(fMaterialRegionFirstColorIndex[ic+1]),size_t(1),[&](size_t i) // Each set of colors in parallel
#else
        for (int i = fMaterialRegionFirstColorIndex[ic]; i < fMaterialRegionFirstColorIndex[ic + 1]; i++)
#endif
        {
            int iel = fMaterialRegionElColorIndexes[i];

            TPZFMatrix<STATE> Kel;
            ComputeTangentMatrix(iel, Kel);

            int el_dof = el_n_dofs[iel];
            int pos = col_pos[iel];

            for (int i_dof = 0; i_dof < el_dof; i_dof++) {
                int64_t i_dest = dof_id[pos + i_dof];

                for (int j_dof = 0; j_dof < el_dof; j_dof++) {
                    int64_t j_dest = dof_id[pos + j_dof];

                    STATE val = Kel(i_dof, j_dof);
                    int64_t index = me(IAToSequence, JAToSequence, i_dest, j_dest);
                    Kg[index] += val;
                }
            }
        }
#ifdef USING_TBB
        );
#endif
    }

}

template<class T, class MEM>
int64_t
TPBrNumericalIntegrator<T, MEM>::me(TPZVec<int64_t> &IA, TPZVec<int64_t> &JA, int64_t &i_dest, int64_t &j_dest) {
    for (int64_t k = IA[i_dest]; k < IA[i_dest + 1]; k++) {
        if (JA[k] == j_dest) {
            return k;
        }
    }
}

template<class T, class MEM>
void TPBrNumericalIntegrator<T, MEM>::ComputeConstitutiveMatrix(TPZFMatrix<REAL> &De) {
    De.Zero();
    REAL lambda = fConstitutiveLawProcessor.GetPlasticModel().fER.Lambda();
    REAL mu = fConstitutiveLawProcessor.GetPlasticModel().fER.Mu();

    De(0, 0) = lambda + 2.0 * mu;
    De(1, 1) = mu;
    De(2, 2) = lambda + 2.0 * mu;
    De(0, 2) = lambda;
    De(2, 0) = lambda;
}

template<class T, class MEM>
void TPBrNumericalIntegrator<T, MEM>::ComputeTangentMatrix(int64_t iel, TPZFMatrix<REAL> &K) {
    TPZFMatrix<REAL> De(3, 3);
    ComputeConstitutiveMatrix(De);

    int n_sigma_comps = 3;
    int el_npts = fBlockMatrix.Blocks().fRowSizes[iel] / n_sigma_comps;
    int el_dofs = fBlockMatrix.Blocks().fColSizes[iel];
    int first_el_ip = fBlockMatrix.Blocks().fRowFirstIndex[iel] / n_sigma_comps;

    K.Resize(el_dofs, el_dofs);
    K.Zero();

    int pos = fBlockMatrix.Blocks().fMatrixPosition[iel];
    TPZFMatrix<STATE> Bip(n_sigma_comps, el_dofs, 0.0);
    TPZFMatrix<STATE> DeBip;
    int c = 0;
    for (int ip = 0; ip < el_npts; ip++) {
        for (int i = 0; i < n_sigma_comps; i++) {
            for (int j = 0; j < el_dofs; j++) {
                Bip(i, j) = fBlockMatrix.Blocks().fStorage[pos + c];
                c++;
            }
        }

        REAL omega = fConstitutiveLawProcessor.WeightVector()[first_el_ip + ip];
        De.Multiply(Bip, DeBip);
        Bip.MultAdd(DeBip, K, K, omega, 1.0, 1);
    }
}

template<class T, class MEM>
int TPBrNumericalIntegrator<T, MEM>::StressRateVectorSize(int dim) {

    switch (dim) {
        case 1: {
            return 1;
        }
            break;
        case 2: {
            return 3;
        }
            break;
        case 3: {
            return 6;
        }
            break;
        default: {
            DebugStop();
            return 0;
        }
            break;
    }
}

template<class T, class MEM>
void TPBrNumericalIntegrator<T, MEM>::SetUpIrregularBlocksMatrix(TPZCompMesh *cmesh) {
    /// Number of elements
    int nblocks = fElementIndexes.size();

    int dimension = cmesh->Dimension();

    fBlockMatrix.Blocks().fNumBlocks = nblocks;
    fBlockMatrix.Blocks().fRowSizes.resize(nblocks);
    fBlockMatrix.Blocks().fColSizes.resize(nblocks);
    fBlockMatrix.Blocks().fMatrixPosition.resize(nblocks + 1);
    fBlockMatrix.Blocks().fRowFirstIndex.resize(nblocks + 1);
    fBlockMatrix.Blocks().fColFirstIndex.resize(nblocks + 1);

    fBlockMatrix.Blocks().fMatrixPosition[0] = 0;
    fBlockMatrix.Blocks().fRowFirstIndex[0] = 0;
    fBlockMatrix.Blocks().fColFirstIndex[0] = 0;

    // @TODO Candidate for TBB ParallelScan
    // Example with lambda expression https://www.threadingbuildingblocks.org/docs/help/reference/algorithms/parallel_scan_func.html
    for (int iel = 0; iel < nblocks; iel++) {

        TPZCompEl *cel = cmesh->Element(fElementIndexes[iel]);

        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement *>(cel);
        if (!cel_inter) DebugStop();
#ifdef PZDEBUG
        int dim = cel->Reference()->Dimension();
        if (dim !=dimension) {
            DebugStop();
        }
#endif

        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());
        int64_t npts = int_rule->NPoints(); // number of integration points of the element
        int64_t n_el_dof = cel_inter->NShapeF(); // number of shape functions of the element (Element DoF)

        fBlockMatrix.Blocks().fRowSizes[iel] = StressRateVectorSize(dimension) * npts;
        fBlockMatrix.Blocks().fColSizes[iel] = n_el_dof * cmesh->Dimension();

        fBlockMatrix.Blocks().fMatrixPosition[iel + 1] = fBlockMatrix.Blocks().fMatrixPosition[iel] +
                                                         fBlockMatrix.Blocks().fRowSizes[iel] *
                                                         fBlockMatrix.Blocks().fColSizes[iel];
        fBlockMatrix.Blocks().fRowFirstIndex[iel + 1] =
                fBlockMatrix.Blocks().fRowFirstIndex[iel] + fBlockMatrix.Blocks().fRowSizes[iel];
        fBlockMatrix.Blocks().fColFirstIndex[iel + 1] =
                fBlockMatrix.Blocks().fColFirstIndex[iel] + fBlockMatrix.Blocks().fColSizes[iel];
    }

    fBlockMatrix.Blocks().fStorage.resize(fBlockMatrix.Blocks().fMatrixPosition[nblocks]);

#ifdef USING_TBB
    tbb::parallel_for(size_t(0),size_t(nblocks),size_t(1),[&](size_t iel)
#else
    for (int iel = 0; iel < nblocks; iel++)
#endif
    {

        TPZFNMatrix<8, REAL> dphiXY;
        TPZMaterialData data;
        TPZVec<REAL> qsi(cmesh->Dimension());

        TPZCompEl *cel = cmesh->Element(fElementIndexes[iel]);

        int sigma_entries_times_npts = fBlockMatrix.Blocks().fRowSizes[iel];
        int n_el_dof = fBlockMatrix.Blocks().fColSizes[iel] / cmesh->Dimension();
        int pos_el = fBlockMatrix.Blocks().fMatrixPosition[iel];


        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement *>(cel);
        if (!cel_inter) DebugStop();

#ifdef PZDEBUG
        int dim = cel->Reference()->Dimension();
        if (dim !=dimension) {
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
        TPZFMatrix<REAL> B_el(npts * n_sigma_entries, n_el_dof * cmesh->Dimension(), 0.0);
        for (int64_t i_pts = 0; i_pts < npts; i_pts++) {

            REAL w;
            int_rule->Point(i_pts, qsi, w);
            cel_inter->ComputeRequiredData(data, qsi);

            data.axes.Transpose();
            data.axes.Multiply(data.dphix, dphiXY);

            for (int j_dim = 0; j_dim < dimension; j_dim++) {

                for (int i_dof = 0; i_dof < n_el_dof; i_dof++) {

                    REAL val = dphiXY(j_dim, i_dof);
                    for (int i_sigma_comp = 0; i_sigma_comp < dimension; i_sigma_comp++) {

                        B_el(i_pts * n_sigma_entries + i_sigma_comp + j_dim, i_dof * 2 + i_sigma_comp) = val;
                    }
                }
            }

        }
        B_el.Transpose();
        TPZFMatrix<REAL> B_el_loc(npts * n_sigma_entries, n_el_dof * dimension, &fBlockMatrix.Blocks().fStorage[pos_el],
                                  npts * n_sigma_entries * n_el_dof * dimension);
        B_el_loc = B_el;
    }
#ifdef USING_TBB
    );
#endif

    int64_t rows = fBlockMatrix.Blocks().fRowFirstIndex[fBlockMatrix.Blocks().fNumBlocks];
    int64_t cols = fBlockMatrix.Blocks().fColFirstIndex[fBlockMatrix.Blocks().fNumBlocks];

    fBlockMatrix.Resize(rows, cols);

}

template<class T, class MEM>
void TPBrNumericalIntegrator<T, MEM>::SetUpIndexes(TPZCompMesh * cmesh) {

    int dimension = cmesh->Dimension();

    int64_t nblocks = fBlockMatrix.Blocks().fNumBlocks;
    int64_t rows = fBlockMatrix.Rows();
    int64_t cols = fBlockMatrix.Cols();
    TPZVec<int> &dof_positions = fBlockMatrix.Blocks().fColFirstIndex;
    TPZVec<int> &npts_positions = fBlockMatrix.Blocks().fRowFirstIndex;

    fDoFIndexes.resize(cols);
    int64_t npts = rows / StressRateVectorSize(dimension);
    TPZVec<REAL> weight(npts);

#ifdef USING_TBB
    tbb::parallel_for(size_t(0),size_t(nblocks),size_t(1),[&](size_t iel)
#else
    for (int iel = 0; iel < nblocks; iel++)
#endif
    {
        int dof_pos = dof_positions[iel];
        TPZCompEl *cel = cmesh->Element(fElementIndexes[iel]);


        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement *>(cel);
        if (!cel_inter) DebugStop();
        TPZGeoEl *gel = cel->Reference();
        if (!gel) DebugStop();

        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());

        int64_t el_npts = int_rule->NPoints(); // number of integration points of the element
        int64_t dim = cel_inter->Dimension(); //dimension of the element

        TPZFMatrix<REAL> jac, axes, jacinv;
        REAL detjac;
        int64_t wit = 0;
        for (int64_t ipts = 0; ipts < el_npts; ipts++) {
            TPZVec<REAL> qsi(dim);
            REAL w;
            int_rule->Point(ipts, qsi, w);
            gel->Jacobian(qsi, jac, axes, detjac, jacinv);
            weight[npts_positions[iel] / 3 + wit] = w * std::abs(detjac);
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
                fDoFIndexes[dof_pos + i_dof] = pos + isize;
                i_dof++;
            }

        }
    }
#ifdef USING_TBB
    );
#endif

    fConstitutiveLawProcessor.SetPlasticModel(fMaterial->GetPlasticModel());
    fConstitutiveLawProcessor.SetUpDataByIntPoints(npts);
    fConstitutiveLawProcessor.SetWeightVector(weight);
}

template<class T, class MEM>
void TPBrNumericalIntegrator<T, MEM>::ColoredIndexes(TPZCompMesh * cmesh) {

    int64_t nblocks = fBlockMatrix.Blocks().fNumBlocks;
    int64_t cols = fBlockMatrix.Cols();

    TPZVec<int64_t> connects_vec(cmesh->NConnects(), 0);
    TPZVec<int64_t> elemcolor(nblocks, -1);

    int64_t contcolor = 0;
    bool needstocontinue = true;

    while (needstocontinue) {
        int it = 0;
        needstocontinue = false;
        for (auto iel : fElementIndexes) {
            TPZCompEl *cel = cmesh->Element(iel);
            if (!cel || cel->Dimension() != cmesh->Dimension()) continue;

            it++;
            if (elemcolor[it - 1] != -1) continue;

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
            elemcolor[it - 1] = contcolor;
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

#ifdef USING_TBB
    tbb::parallel_for(size_t(0),size_t(nblocks),size_t(1),[&](size_t iel)
#else
    for (int iel = 0; iel < nblocks; iel++)
#endif
    {
        int64_t elem_col = fBlockMatrix.Blocks().fColSizes[iel];
        int64_t cont_cols = fBlockMatrix.Blocks().fColFirstIndex[iel];

        for (int64_t icols = 0; icols < elem_col; icols++) {
            fColorIndexes[cont_cols + icols] = fDoFIndexes[cont_cols + icols] + elemcolor[iel] * neq;
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

    fMaterialRegionElColorIndexes.resize(nblocks);
    fMaterialRegionFirstColorIndex.resize(color_map.size() + 1);

    int c_color = 0;

    fMaterialRegionFirstColorIndex[c_color] = 0;
    for (auto color_data : color_map) {
        int n_el_per_color = color_data.second.size();
        int iel = fMaterialRegionFirstColorIndex[c_color];
        for (int i = 0; i < n_el_per_color; i++) {
            int el_index = color_data.second[i];
            fMaterialRegionElColorIndexes[iel] = el_index;
            iel++;
        }
        c_color++;
        fMaterialRegionFirstColorIndex[c_color] = n_el_per_color + fMaterialRegionFirstColorIndex[c_color - 1];
    }
}
