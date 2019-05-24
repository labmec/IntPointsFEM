#include "TPZIntPointsStructMatrix.h"
#include "pzintel.h"
#include "pzskylstrmatrix.h"

#ifdef USING_MKL
#include <mkl.h>
#endif
#include "TPZMyLambdaExpression.h"

TPZIntPointsStructMatrix::TPZIntPointsStructMatrix() : TPZStructMatrix(), fIntPointsData(), fBlockMatrix(), fRhsBoundary(0,0), fElemIndexes() {
}

TPZIntPointsStructMatrix::TPZIntPointsStructMatrix(TPZCompMesh *cmesh) : TPZStructMatrix(cmesh), fIntPointsData(), fBlockMatrix(), fRhsBoundary(0,0), fElemIndexes() {
    ElementsToAssemble();
    AssembleRhsBoundary();
    TPZMatrix<STATE> *blockMatrix = Create();
    fBlockMatrix = dynamic_cast<TPZIrregularBlocksMatrix *> (blockMatrix);

    IntPointsInfo(*fBlockMatrix);
    ColoringElements(*fBlockMatrix);
}

TPZIntPointsStructMatrix::TPZIntPointsStructMatrix(TPZAutoPointer<TPZCompMesh> cmesh) : TPZStructMatrix(cmesh), fIntPointsData(), fBlockMatrix(), fRhsBoundary(0,0), fElemIndexes() {
}

TPZIntPointsStructMatrix::~TPZIntPointsStructMatrix() {
}

TPZStructMatrix * TPZIntPointsStructMatrix::Clone(){
    return new TPZIntPointsStructMatrix(*this);
}

TPZIntPointsStructMatrix::TPZIntPointsStructMatrix(const TPZIntPointsStructMatrix &copy) {
    fIntPointsData = copy.fIntPointsData;
    fBlockMatrix = copy.fBlockMatrix;
    fRhsBoundary = copy.fRhsBoundary;
    fElemIndexes = copy.fElemIndexes;
}

TPZIntPointsStructMatrix &TPZIntPointsStructMatrix::operator=(const TPZIntPointsStructMatrix &copy) {
    if(&copy == this){
        return *this;
    }

    fIntPointsData = copy.fIntPointsData;
    fBlockMatrix = copy.fBlockMatrix;
    fRhsBoundary = copy.fRhsBoundary;
    fElemIndexes = copy.fElemIndexes;

    return *this;
}

void TPZIntPointsStructMatrix::ElementsToAssemble() {
    std::map<int, TPZMaterial*> & matvec = fMesh->MaterialVec();
    std::map<int, TPZMaterial* >::iterator mit;

    for(mit=matvec.begin(); mit!= matvec.end(); mit++)
    {
        for (int64_t i = 0; i < fMesh->NElements(); i++) {
            TPZCompEl *cel = fMesh->Element(i);
            if (!cel) continue;
            TPZGeoEl *gel = fMesh->Element(i)->Reference();
            if (!gel) continue;
            if(cel->Material()->Id() == mit->second->Id() && cel->Dimension() == fMesh->Dimension()){
                fElemIndexes.Push(cel->Index());
            }
        }
    }
}

TPZMatrix<REAL> * TPZIntPointsStructMatrix::Create() {
    int nblocks = fElemIndexes.size();

    TPZIrregularBlocksMatrix *blockMatrix = new TPZIrregularBlocksMatrix();

    TPZIrregularBlocksMatrix::IrregularBlocks blocksData;

    blocksData.fNumBlocks = nblocks;

    blocksData.fRowSizes.resize(nblocks);
    blocksData.fColSizes.resize(nblocks);
    blocksData.fMatrixPosition.resize(nblocks + 1);
    blocksData.fRowFirstIndex.resize(nblocks + 1);
    blocksData.fColFirstIndex.resize(nblocks + 1);

    blocksData.fMatrixPosition[0] = 0;
    blocksData.fRowFirstIndex[0] = 0;
    blocksData.fColFirstIndex[0] = 0;

    int64_t rows = 0;
    int64_t cols = 0;

    int iel = 0;
    for(auto elem_index : fElemIndexes) {
        TPZCompEl *cel = fMesh->Element(elem_index);
        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement *>(cel);
        if (!cel_inter) DebugStop();
        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());

        int64_t npts = int_rule->NPoints(); // number of integration points of the element
        int64_t dim = cel_inter->Dimension(); //dimension of the element
        int64_t nf = cel_inter->NShapeF(); // number of shape functions of the element

        blocksData.fRowSizes[iel] = dim * npts;
        blocksData.fColSizes[iel] = nf;
        blocksData.fMatrixPosition[iel + 1] = blocksData.fMatrixPosition[iel] + blocksData.fRowSizes[iel] * blocksData.fColSizes[iel];
        blocksData.fRowFirstIndex[iel + 1] =  blocksData.fRowFirstIndex[iel] + blocksData.fRowSizes[iel];
        blocksData.fColFirstIndex[iel + 1] = blocksData.fColFirstIndex[iel] + blocksData.fColSizes[iel];

        rows += blocksData.fRowSizes[iel];
        cols += blocksData.fColSizes[iel];
        iel++;
    }

    blocksData.fRowPtr.resize(rows + 1);
    blocksData.fColInd.resize(blocksData.fMatrixPosition[nblocks]);

    for (int iel = 0; iel < nblocks; ++iel) {
        for (int irow = 0; irow < blocksData.fRowSizes[iel]; ++irow) {
            blocksData.fRowPtr[irow + blocksData.fRowFirstIndex[iel]] = blocksData.fMatrixPosition[iel] + irow*blocksData.fColSizes[iel];

            for (int icol = 0; icol < blocksData.fColSizes[iel]; ++icol) {
                blocksData.fColInd[icol + blocksData.fMatrixPosition[iel] + irow*blocksData.fColSizes[iel]] = icol + blocksData.fColFirstIndex[iel];
            }
        }
    }
    blocksData.fRowPtr[rows] = blocksData.fMatrixPosition[nblocks];

    blocksData.fStorage.resize(blocksData.fMatrixPosition[nblocks]);

    iel = 0;
    for(auto elem_index : fElemIndexes) {
        TPZCompEl *cel = fMesh->Element(elem_index);
        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement *>(cel);
        if (!cel_inter) DebugStop();
        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());

        int64_t npts = int_rule->NPoints(); // number of integration points of the element
        int64_t dim = cel_inter->Dimension(); //dimension of the element
        int64_t nf = cel_inter->NShapeF(); // number of shape functions of the element

        TPZFMatrix<REAL> elmatrix;
        int rows = blocksData.fRowSizes[iel];
        int cols = blocksData.fColSizes[iel];
        int pos = blocksData.fMatrixPosition[iel];
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

        TPZFMatrix<REAL> elmatloc(rows, cols, &blocksData.fStorage[pos], rows * cols);
        elmatloc = elmatrix;
        iel++;
    }

    blockMatrix->Resize(rows, cols);
    blockMatrix->SetBlocks(blocksData);

    return blockMatrix;
}

void TPZIntPointsStructMatrix::Assemble(TPZFMatrix<REAL> & rhs) {
    int dim = fMesh->Dimension();
    int neq = fMesh->NEquations();

    TPZFMatrix<REAL> gather_solution;
    TPZFMatrix<REAL> grad_u;
    TPZFMatrix<REAL> sigma;
    TPZFMatrix<REAL> forces;

    rhs.Resize(neq, 1);
    rhs.Zero();

    int rows = fBlockMatrix->Rows();
    int cols = fBlockMatrix->Cols();

    gather_solution.Resize(dim * cols, 1);
    fIntPointsData.GatherSolution(fMesh->Solution(), gather_solution);

    TPZFMatrix<REAL> gather_x(cols, 1, &gather_solution(0, 0), cols);
    TPZFMatrix<REAL> gather_y(cols, 1, &gather_solution(cols, 0), cols);

    grad_u.Resize(dim * rows, 1);
    TPZFMatrix<REAL> grad_u_x(rows, 1, &grad_u(0, 0), rows);
    TPZFMatrix<REAL> grad_u_y(rows, 1, &grad_u(rows, 0), rows);

    fBlockMatrix->Multiply(gather_x, grad_u_x, 0);
    fBlockMatrix->Multiply(gather_y, grad_u_y, 0);

    sigma.Resize(dim * rows, 1);
    TPZMyLambdaExpression lambdaexp(this);
    lambdaexp.ComputeSigma(grad_u, sigma);

    TPZFMatrix<REAL> sigma_x(rows, 1, &sigma(0, 0), rows);
    TPZFMatrix<REAL> sigma_y(rows, 1, &sigma(rows, 0), rows);

    forces.Resize(dim * cols, 1);
    TPZFMatrix<REAL> forces_x(cols, 1, &forces(0, 0), cols);
    TPZFMatrix<REAL> forces_y(cols, 1, &forces(cols, 0), cols);

    fBlockMatrix->Multiply(sigma_x, forces_x, true);
    fBlockMatrix->Multiply(sigma_y, forces_y, true);

    fIntPointsData.ColoredAssemble(forces, rhs);

    rhs += fRhsBoundary;
}

void TPZIntPointsStructMatrix::ColoringElements(TPZIrregularBlocksMatrix &blockMatrix)  {
    int dim = fMesh->Dimension();
    int cols = blockMatrix.Cols();

    int64_t nconnects = fMesh->NConnects();
    TPZVec<int64_t> connects_vec(nconnects,0);
    TPZVec<int64_t> elemcolor(blockMatrix.Blocks().fNumBlocks,-1);

    int64_t contcolor = 0;
    bool needstocontinue = true;

    //Elements coloring
    while (needstocontinue)
    {
        int it = 0;
        needstocontinue = false;
        for (auto iel : fElemIndexes) {
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
    int64_t nelem = blockMatrix.Blocks().fNumBlocks;
    int64_t neq = fMesh->NEquations();
    for (int64_t iel = 0; iel < nelem; iel++) {
        int64_t elem_col = blockMatrix.Blocks().fColSizes[iel];
        int64_t cont_cols = blockMatrix.Blocks().fColFirstIndex[iel];

        for (int64_t icols = 0; icols < elem_col; icols++) {
            indexescolor[cont_cols + icols] = fIntPointsData.Indexes()[cont_cols + icols] + elemcolor[iel]*neq;
            indexescolor[cont_cols+ cols + icols] = fIntPointsData.Indexes()[cont_cols + cols + icols] + elemcolor[iel]*neq;
        }
    }
    fIntPointsData.SetIndexesColor(indexescolor);
}

void TPZIntPointsStructMatrix::IntPointsInfo(TPZIrregularBlocksMatrix &blockMatrix) {
    TPZVec<REAL> weight(blockMatrix.Rows() / fMesh->Dimension());
    TPZVec<int> indexes(fMesh->Dimension() * blockMatrix.Cols());

    int iel = 0;
    int iw = 0;
    int64_t cont1 = 0;
    int64_t cont2 = 0;
    for(auto elem_index : fElemIndexes) {
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
                        indexes[cont2 + blockMatrix.Cols()] = pos + isize;
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