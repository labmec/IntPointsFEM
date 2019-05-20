#include "TPZIntPointsFEM.h"
#include "pzintel.h"
#include "pzskylstrmatrix.h"

#ifdef USING_MKL
#include <mkl.h>
#endif


TPZIntPointsFEM::TPZIntPointsFEM() : fNColor(-1), fRhs(0, 0), fRhsBoundary(0, 0), fIndexes(0), fIndexesColor(0), fWeight(0), fBMatrix() {

}

TPZIntPointsFEM::TPZIntPointsFEM(TPZIrregularBlockMatrix *Bmatrix) : fNColor(-1), fRhs(0, 0), fRhsBoundary(0, 0), fIndexes(0), fIndexesColor(0), fWeight(0), fBMatrix() {
    SetBMatrix(Bmatrix);
}

TPZIntPointsFEM::~TPZIntPointsFEM() {

}

TPZIntPointsFEM::TPZIntPointsFEM(const TPZIntPointsFEM &copy) {
    fNColor = copy.fNColor;
    fRhs = copy.fRhs;
    fRhsBoundary = copy.fRhsBoundary;
    fIndexes = copy.fIndexes;
    fIndexesColor = copy.fIndexesColor;
    fWeight = copy.fWeight;
    fBMatrix = copy.fBMatrix;
}

TPZIntPointsFEM &TPZIntPointsFEM::operator=(const TPZIntPointsFEM &copy) {
    if(&copy == this){
        return *this;
    }
    fNColor = copy.fNColor;
    fRhs = copy.fRhs;
    fRhsBoundary = copy.fRhsBoundary;
    fIndexes = copy.fIndexes;
    fIndexesColor = copy.fIndexesColor;
    fWeight = copy.fWeight;
    fBMatrix = copy.fBMatrix;

    return *this;
}

void TPZIntPointsFEM::GatherSolution(TPZFMatrix<REAL> &global_solution, TPZFMatrix<REAL> &gather_solution) {
    int dim = fBMatrix->Dimension();
    int rows = fBMatrix->Rows();
    int cols = fBMatrix->Cols();

    gather_solution.Resize(rows, 1);
    gather_solution.Zero();

    cblas_dgthr(dim * cols, global_solution, &gather_solution(0, 0), &fIndexes[0]);
}

void TPZIntPointsFEM::ColoredAssemble(TPZFMatrix<STATE>  &nodal_forces) {
    int64_t ncolor = fNColor;
    int64_t sz = fIndexes.size();
    int64_t neq = fBMatrix->CompMesh()->NEquations();
    fRhs.Resize(neq*ncolor,1);
    fRhs.Zero();

    cblas_dsctr(sz, nodal_forces, &fIndexesColor[0], &fRhs(0,0));

    int64_t colorassemb = ncolor / 2.;
    while (colorassemb > 0) {

        int64_t firsteq = (ncolor - colorassemb) * neq;
        cblas_daxpy(colorassemb * neq, 1., &fRhs(firsteq, 0), 1., &fRhs(0, 0), 1.);

        ncolor -= colorassemb;
        colorassemb = ncolor/2;
    }
    fRhs.Resize(neq, 1);

    fRhs = fRhs + fRhsBoundary;
}

void TPZIntPointsFEM::SetIntPointsInfo(){
    int dim = fBMatrix->Dimension();
    int rows = fBMatrix->Rows();
    int cols = fBMatrix->Cols();

    fWeight.resize(rows / dim);
    fIndexes.resize(dim * cols);

    int64_t cont1 = 0;
    int64_t cont2 = 0;
    int it = 0;
    for (auto iel : fBMatrix->ElemIndexes()) {
        TPZCompEl *cel = fBMatrix->CompMesh()->Element(iel);
        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement * >(cel);
        if (!cel_inter) DebugStop();
        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());

        int64_t npts = int_rule->NPoints(); // number of integration points of the element
        int64_t dim = cel_inter->Dimension(); //dimension of the element

        TPZMaterialData data;
        cel_inter->InitMaterialData(data);

        //Weight vector
        for (int64_t inpts = 0; inpts < npts; inpts++) {
            TPZManVector<REAL> qsi(dim, 1);
            REAL w;
            int_rule->Point(inpts, qsi, w);
            cel_inter->ComputeRequiredData(data, qsi);
            fWeight[it] = w * std::abs(data.detjac);
            it++;
        }

        //Indexes vector
        int64_t ncon = cel->NConnects();
        for (int64_t icon = 0; icon < ncon; icon++) {
            int64_t id = cel->ConnectIndex(icon);
            TPZConnect &df = fBMatrix->CompMesh()->ConnectVec()[id];
            int64_t conid = df.SequenceNumber();
            if (df.NElConnected() == 0 || conid < 0 || fBMatrix->CompMesh()->Block().Size(conid) == 0) continue;
            else {
                int64_t pos = fBMatrix->CompMesh()->Block().Position(conid);
                int64_t nsize = fBMatrix->CompMesh()->Block().Size(conid);
                for (int64_t isize = 0; isize < nsize; isize++) {
                    if (isize % 2 == 0) {
                        fIndexes[cont1] = pos + isize;
                        cont1++;
                    } else {
                        fIndexes[cont2 + cols] = pos + isize;
                        cont2++;
                    }
                }
            }
        }
    }

    this->ColoringElements();
    this->AssembleRhsBoundary();
}

void TPZIntPointsFEM::ColoringElements()  {
    int dim = fBMatrix->Dimension();
    int cols = fBMatrix->Cols();

    int64_t nconnects = fBMatrix->CompMesh()->NConnects();
    TPZVec<int64_t> connects_vec(nconnects,0);
    TPZVec<int64_t> elemcolor(fBMatrix->NumBlocks(),-1);

    int64_t contcolor = 0;
    bool needstocontinue = true;

    //Elements coloring
    while (needstocontinue)
    {
        int it = 0;
        needstocontinue = false;
        for (auto iel : fBMatrix->ElemIndexes()) {
            TPZCompEl *cel = fBMatrix->CompMesh()->Element(iel);
            if (!cel || cel->Dimension() != fBMatrix->CompMesh()->Dimension()) continue;

            it++;
            if (elemcolor[it-1] != -1) continue;

            TPZStack<int64_t> connectlist;
            fBMatrix->CompMesh()->Element(iel)->BuildConnectList(connectlist);
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
//            cel->Reference()->SetMaterialId(contcolor);

            for (icon = 0; icon < ncon; icon++) {
                connects_vec[connectlist[icon]] = 1;
            }
        }
        contcolor++;
        connects_vec.Fill(0);
    }
//    ofstream file("colored.vtk");
//    TPZVTKGeoMesh::PrintGMeshVTK(fBMatrix->CompMesh()->Reference(),file);

    //Indexes coloring
    fNColor = contcolor;
    fIndexesColor.resize(dim * cols);
    int64_t nelem = fBMatrix->NumBlocks();
    int64_t neq = fBMatrix->CompMesh()->NEquations();
    for (int64_t iel = 0; iel < nelem; iel++) {
        int64_t elem_col = fBMatrix->ColSizes()[iel];
        int64_t cont_cols = fBMatrix->ColFirstIndex()[iel];

        for (int64_t icols = 0; icols < elem_col; icols++) {
            fIndexesColor[cont_cols + icols] = fIndexes[cont_cols + icols] + elemcolor[iel]*neq;
            fIndexesColor[cont_cols+ cols + icols] = fIndexes[cont_cols + cols + icols] + elemcolor[iel]*neq;
        }
    }
}

void TPZIntPointsFEM::AssembleRhsBoundary() {
    int64_t neq = fBMatrix->CompMesh()->NEquations();
    fRhsBoundary.Resize(neq, 1);
    fRhsBoundary.Zero();

    for (auto iel : fBMatrix->BoundaryElemIndexes()) {
        TPZCompEl *cel = fBMatrix->CompMesh()->Element(iel);
        if (!cel) continue;
        TPZElementMatrix ef(fBMatrix->CompMesh(), TPZElementMatrix::EF);
        cel->CalcResidual(ef);
        ef.ComputeDestinationIndices();
        fRhsBoundary.AddFel(ef.fMat, ef.fSourceIndex, ef.fDestinationIndex);
    }
}
