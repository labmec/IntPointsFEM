#include "TPZIntPointsStructMatrix.h"
#include "pzintel.h"
#include "pzskylstrmatrix.h"

#ifdef USING_MKL
#include <mkl.h>
#endif
#include "TPZMyLambdaExpression.h"

TPZIntPointsStructMatrix::TPZIntPointsStructMatrix() : fRhs(0,0), fRhsBoundary(0,0), fMaterialIds(0), fBlockMatrix(), fIntPointsData(), fElemIndexes(0){

}

TPZIntPointsStructMatrix::TPZIntPointsStructMatrix(TPZCompMesh *cmesh) : fRhs(0,0), fRhsBoundary(0,0), fMaterialIds(0), fBlockMatrix(), fIntPointsData(), fElemIndexes(0) {
    SetMesh(cmesh);
}

TPZIntPointsStructMatrix::~TPZIntPointsStructMatrix() {

}

TPZIntPointsStructMatrix::TPZIntPointsStructMatrix(const TPZIntPointsStructMatrix &copy) {
    fRhs = copy.fRhs;
    fRhsBoundary = copy.fRhsBoundary;
    fMaterialIds = copy.fMaterialIds;
    fBlockMatrix = copy.fBlockMatrix;
    fElemIndexes = copy.fElemIndexes;
}

TPZIntPointsStructMatrix &TPZIntPointsStructMatrix::operator=(const TPZIntPointsStructMatrix &copy) {
    if(&copy == this){
        return *this;
    }

    fRhs = copy.fRhs;
    fRhsBoundary = copy.fRhsBoundary;
    fMaterialIds = copy.fMaterialIds;
    fBlockMatrix = copy.fBlockMatrix;
    fElemIndexes = copy.fElemIndexes;

    return *this;
}

void TPZIntPointsStructMatrix::ElementsToAssemble() {
    int nmaterials = fMaterialIds.size();
    fElemIndexes.resize(nmaterials);

    for (int imat = 0; imat < nmaterials; ++imat) {
        for (int64_t i = 0; i < fMesh->NElements(); i++) {
            TPZCompEl *cel = fMesh->Element(i);
            if (!cel) continue;
            TPZGeoEl *gel = fMesh->Element(i)->Reference();
            if (!gel) continue;
            if(cel->Material()->Id() == fMaterialIds[imat]) fElemIndexes[imat].Push(cel->Index());
        }
    }
}

void TPZIntPointsStructMatrix::BlocksInfo(TPZStack<int64_t> elemindexes) {
    int nblocks = elemindexes.size();
    fBlockMatrix.SetNumBlocks(nblocks);

    TPZVec<int> rowsizes(nblocks);
    TPZVec<int> colsizes(nblocks);
    TPZVec<int> matrixpos(nblocks+1);
    TPZVec<int> rowfirstindex(nblocks+1);
    TPZVec<int> colfirstindex(nblocks+1);

    matrixpos[0] = 0;
    rowfirstindex[0] = 0;
    colfirstindex[0] = 0;

    int64_t rows = 0;
    int64_t cols = 0;

    int iel = 0;
    for (auto elem_index : elemindexes) {
        TPZCompEl *cel = fMesh->Element(elem_index);
        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement*>(cel);
        if (!cel_inter) DebugStop();
        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());

        int64_t npts = int_rule->NPoints(); // number of integration points of the element
        int64_t dim = cel_inter->Dimension(); //dimension of the element
        int64_t nf = cel_inter->NShapeF(); // number of shape functions of the element

        rowsizes[iel] = dim * npts;
        colsizes[iel] = nf;
        matrixpos[iel + 1] = matrixpos[iel] + rowsizes[iel] * colsizes[iel];
        rowfirstindex[iel + 1] = rowfirstindex[iel] + rowsizes[iel];
        colfirstindex[iel + 1] = colfirstindex[iel] + colsizes[iel];

        rows += rowsizes[iel];
        cols += colsizes[iel];
        iel++;
    }

    fBlockMatrix.SetRowSizes(rowsizes);
    fBlockMatrix.SetColSizes(colsizes);
    fBlockMatrix.SetMatrixPosition(matrixpos);
    fBlockMatrix.SetRowFirstIndex(rowfirstindex);
    fBlockMatrix.SetColFirstIndex(colfirstindex);
    fBlockMatrix.SetRows(rows);
    fBlockMatrix.SetCols(cols);

    TPZVec<int> rowptr(rows + 1);
    TPZVec<int> colind(matrixpos[nblocks]);

    for (int iel = 0; iel < nblocks; ++iel) {
        for (int irow = 0; irow < rowsizes[iel]; ++irow) {
            rowptr[irow + rowfirstindex[iel]] = matrixpos[iel] + irow*colsizes[iel];

            for (int icol = 0; icol < colsizes[iel]; ++icol) {
                colind[icol + matrixpos[iel] + irow*colsizes[iel]] = icol + colfirstindex[iel];
            }
        }
    }
    rowptr[rows] = matrixpos[nblocks];

    fBlockMatrix.SetRowPtr(rowptr);
    fBlockMatrix.SetColInd(colind);
}

void TPZIntPointsStructMatrix::FillBlocks(TPZStack<int64_t> elemindexes) {
    TPZVec<REAL> storage(fBlockMatrix.MatrixPosition()[fBlockMatrix.NumBlocks()]);

    int iel = 0;
    for(auto elem_index : elemindexes) {
        TPZCompEl *cel = fMesh->Element(elem_index);
        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement *>(cel);
        if (!cel_inter) DebugStop();
        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());

        int64_t npts = int_rule->NPoints(); // number of integration points of the element
        int64_t dim = cel_inter->Dimension(); //dimension of the element
        int64_t nf = cel_inter->NShapeF(); // number of shape functions of the element

        TPZFMatrix<REAL> elmatrix;
        int rows = fBlockMatrix.RowSizes()[iel];
        int cols = fBlockMatrix.ColSizes()[iel];
        int pos = fBlockMatrix.MatrixPosition()[iel];
        elmatrix.Resize(rows, cols);

        TPZMaterialData data;
        cel_inter->InitMaterialData(data);

        for (int64_t ipts = 0; ipts < npts; ipts++) {
            TPZVec<REAL> qsi(dim);
            REAL w;
            int_rule->Point(ipts, qsi, w);
            cel_inter->ComputeRequiredData(data, qsi);

            TPZFMatrix<REAL> axes = data.axes;
            TPZFMatrix<REAL> dphix = data.dphix;
            TPZFMatrix<REAL> dphiXY;

            axes.Transpose();
            axes.Multiply(dphix, dphiXY);

            for (int inf = 0; inf < nf; inf++) {
                for (int idim = 0; idim < dim; idim++)
                    elmatrix(ipts * dim + idim, inf) = dphiXY(idim, inf);
            }
        }
        elmatrix.Transpose(); // Using CSR format

        TPZFMatrix<REAL> elmatloc(rows, cols, &storage[pos], rows * cols);
        elmatloc = elmatrix;
        iel++;
    }

    fBlockMatrix.SetStorage(storage);
}

void TPZIntPointsStructMatrix::IntPointsInfo(TPZStack<int64_t> elemindexes) {
    TPZVec<REAL> weight(fBlockMatrix.Rows() / fMesh->Dimension());
    TPZVec<int> indexes(fMesh->Dimension() * fBlockMatrix.Cols());

    int iel = 0;
    int iw = 0;
    int64_t cont1 = 0;
    int64_t cont2 = 0;
    for(auto elem_index : elemindexes) {
        TPZCompEl *cel = fMesh->Element(elem_index);
        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement*>(cel);
        if (!cel_inter) DebugStop();
        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());

        int64_t npts = int_rule->NPoints(); // number of integration points of the element
        int64_t dim = cel_inter->Dimension(); //dimension of the element

        TPZMaterialData data;
        cel_inter->InitMaterialData(data);

        for (int64_t ipts = 0; ipts < npts; ipts++) {
            TPZVec<REAL> qsi(dim);
            REAL w;
            int_rule->Point(ipts, qsi, w);
            cel_inter->ComputeRequiredData(data, qsi);
            weight[iw] = w * std::abs(data.detjac);
            iw++;
        }

        int64_t ncon = cel->NConnects();
        for (int64_t icon = 0; icon < ncon; icon++) {
            int64_t id = cel->ConnectIndex(icon);
            TPZConnect &df = fMesh->ConnectVec()[id];
            int64_t conid = df.SequenceNumber();
            if (df.NElConnected() == 0 || conid < 0 || fMesh->Block().Size(conid) == 0) continue;
            else {
                int64_t pos = fMesh->Block().Position(conid);
                int64_t nsize = fMesh->Block().Size(conid);
                for (int64_t isize = 0; isize < nsize; isize++) {
                    if (isize % 2 == 0) {
                        indexes[cont1] = pos + isize;
                        cont1++;
                    } else {
                        indexes[cont2 + fBlockMatrix.Cols()] = pos + isize;
                        cont2++;
                    }
                }
            }
        }
        iel++;
    }
    fIntPointsData.SetIndexes(indexes);
    fIntPointsData.SetWeight(weight);
}

void TPZIntPointsStructMatrix::Initialize() {
    ElementsToAssemble();
    AssembleRhsBoundary();
    int nmat = fMaterialIds.size();

    for (int imat = 0; imat < nmat; ++imat) {
        BlocksInfo(fElemIndexes[imat]);
        FillBlocks(fElemIndexes[imat]);
        IntPointsInfo(fElemIndexes[imat]);
        ColoringElements(fElemIndexes[imat]);
    }
}

void TPZIntPointsStructMatrix::Assemble(TPZFMatrix<REAL> &solution) {
    int nmat = fMaterialIds.size();
    int dim = fMesh->Dimension();
    int neq = fMesh->NEquations();

    TPZFMatrix<REAL> gather_solution;
    TPZFMatrix<REAL> grad_u;
    TPZFMatrix<REAL> sigma;
    TPZFMatrix<REAL> forces;
    TPZFMatrix<REAL> residual;

    fRhs.Resize(neq, 1);
    fRhs.Zero();

    for (int imat = 0; imat < nmat; ++imat) {
        int rows = fBlockMatrix.Rows();
        int cols = fBlockMatrix.Cols();

        gather_solution.Resize(dim * cols, 1);
        fIntPointsData.GatherSolution(solution, gather_solution);

        grad_u.Resize(dim * rows, 1);
        fBlockMatrix.Multiply(&gather_solution(0,0), &grad_u(0,0), 0);
        fBlockMatrix.Multiply(&gather_solution(cols,0), &grad_u(rows,0), 0);

        sigma.Resize(dim * rows, 1);
        TPZMyLambdaExpression lambdaexp(this);
        lambdaexp.ComputeSigma(grad_u, sigma);

        forces.Resize(dim * cols, 1);
        fBlockMatrix.Multiply(&sigma(0,0), &forces(0,0), true);
        fBlockMatrix.Multiply(&sigma(rows,0), &forces(cols,0), true);

        residual.Resize(neq, 1);
        fIntPointsData.ColoredAssemble(forces, residual);

        fRhs += residual;
    }
    fRhs += fRhsBoundary;
}

void TPZIntPointsStructMatrix::ColoringElements(TPZStack<int64_t> elemindexes)  {
    int dim = fMesh->Dimension();
    int cols = fBlockMatrix.Cols();

    int64_t nconnects = fMesh->NConnects();
    TPZVec<int64_t> connects_vec(nconnects,0);
    TPZVec<int64_t> elemcolor(fBlockMatrix.NumBlocks(),-1);

    int64_t contcolor = 0;
    bool needstocontinue = true;

    //Elements coloring
    while (needstocontinue)
    {
        int it = 0;
        needstocontinue = false;
        for (auto iel : elemindexes) {
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
    fIntPointsData.SetNColor(contcolor);
    TPZVec<int> indexescolor(dim * cols);
    int64_t nelem = fBlockMatrix.NumBlocks();
    int64_t neq = fMesh->NEquations();
    for (int64_t iel = 0; iel < nelem; iel++) {
        int64_t elem_col = fBlockMatrix.ColSizes()[iel];
        int64_t cont_cols = fBlockMatrix.ColFirstIndex()[iel];

        for (int64_t icols = 0; icols < elem_col; icols++) {
            indexescolor[cont_cols + icols] = fIntPointsData.Indexes()[cont_cols + icols] + elemcolor[iel]*neq;
            indexescolor[cont_cols+ cols + icols] = fIntPointsData.Indexes()[cont_cols + cols + icols] + elemcolor[iel]*neq;
        }
    }
    fIntPointsData.SetIndexesColor(indexescolor);
}

void TPZIntPointsStructMatrix::AssembleRhsBoundary() {
    int64_t neq = fMesh->NEquations();
    fRhsBoundary.Resize(neq, 1);
    fRhsBoundary.Zero();


    for (int64_t i = 0; i < fMesh->NElements(); i++) {
        TPZCompEl *cel = fMesh->Element(i);
        if (!cel) continue;
        TPZGeoEl *gel = fMesh->Element(i)->Reference();
        if (!gel) continue;
        if(cel->Dimension() < fMesh->Dimension()) {
            TPZCompEl *cel = fMesh->Element(i);
            if (!cel) continue;
            TPZElementMatrix ef(fMesh, TPZElementMatrix::EF);
            cel->CalcResidual(ef);
            ef.ComputeDestinationIndices();
            fRhsBoundary.AddFel(ef.fMat, ef.fSourceIndex, ef.fDestinationIndex);
        }
    }
}