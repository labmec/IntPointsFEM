#include "TPZElastoPlasticIntPointsStructMatrix.h"
#include "pzintel.h"
#include "pzskylstrmatrix.h"

#ifdef USING_MKL
#include <mkl.h>
#endif
#include "TPZMyLambdaExpression.h"

TPZElastoPlasticIntPointsStructMatrix::TPZElastoPlasticIntPointsStructMatrix(TPZCompMesh *cmesh) : TPZSymetricSpStructMatrix(cmesh), fLambdaExp(), fStructMatrix(), fCoefToGradSol() {

}

TPZElastoPlasticIntPointsStructMatrix::~TPZElastoPlasticIntPointsStructMatrix() {
}

TPZStructMatrix * TPZElastoPlasticIntPointsStructMatrix::Clone(){
    return new TPZElastoPlasticIntPointsStructMatrix(*this);
}

TPZMatrix<STATE> *TPZElastoPlasticIntPointsStructMatrix::CreateAssemble(TPZFMatrix<STATE> &rhs, TPZAutoPointer<TPZGuiInterface> guiInterface) {
//    fStructMatrix.SetMesh(this->Mesh());
//    TPZMatrix<STATE> *matrix = fStructMatrix.Create();
//    TPZMatrix<STATE> *matrix2 = fStructMatrix.CreateAssemble(rhs, guiInterface);
//    return matrix;
}

void TPZElastoPlasticIntPointsStructMatrix::SetUpDataStructure() {
    TPZStack<REAL> elindex_domain;
    this->GetDomainElements(elindex_domain);

    TPZIrregularBlocksMatrix::IrregularBlocks blocksData;
    this->SetUpIrregularBlocksData(elindex_domain, blocksData);

    int64_t rows = blocksData.fRowFirstIndex[blocksData.fNumBlocks];
    int64_t cols = blocksData.fColFirstIndex[blocksData.fNumBlocks];
    TPZIrregularBlocksMatrix blocksMatrix(rows, cols);
    blocksMatrix.SetBlocks(blocksData);
    fCoefToGradSol.SetIrregularBlocksMatrix(blocksMatrix);

    TPZVec<int> indexes;
    this->SetUpIndexes(elindex_domain, indexes);
    fCoefToGradSol.SetIndexes(indexes);

    TPZVec<int> coloredindexes;
    int ncolor;
    this->ColoredIndexes(elindex_domain, indexes, coloredindexes, ncolor);
    fCoefToGradSol.SetIndexesColor(coloredindexes);
    fCoefToGradSol.SetNColors(ncolor);
}

void TPZElastoPlasticIntPointsStructMatrix::CalcResidual(TPZFMatrix<REAL> & rhs) {
    if(!isBuilt()) {
        this->SetUpDataStructure();
    }

    int neq = fMesh->NEquations();

    TPZFMatrix<REAL> grad_u;
    TPZFMatrix<REAL> sigma;
    rhs.Resize(neq, 1);
    rhs.Zero();

    fCoefToGradSol.CoefToGradU(fMesh->Solution(), grad_u);
    fLambdaExp.ComputeSigma(grad_u, sigma);
    fCoefToGradSol.SigmaToRes(sigma, rhs);

    TPZFMatrix<REAL> rhsboundary;
    AssembleRhsBoundary(rhsboundary);

    rhs += rhsboundary;
}

void TPZElastoPlasticIntPointsStructMatrix::AssembleRhsBoundary(TPZFMatrix<REAL> &rhsboundary) {
    int64_t neq = fMesh->NEquations();
    rhsboundary.Resize(neq, 1);
    rhsboundary.Zero();

    for (int iel = 0; iel < fMesh->NElements(); iel++) {
        TPZCompEl *cel = fMesh->Element(iel);
        if (!cel) continue;
        if(cel->Dimension() < fMesh->Dimension()) {
            TPZElementMatrix ef(fMesh, TPZElementMatrix::EF);
            cel->CalcResidual(ef);
            ef.ComputeDestinationIndices();
            rhsboundary.AddFel(ef.fMat, ef.fSourceIndex, ef.fDestinationIndex);
        }
    }
}

void TPZElastoPlasticIntPointsStructMatrix::GetDomainElements(TPZStack<REAL> &elindex_domain) {
    // Store the domain elements of a same material in elem_domain
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
                elindex_domain.Push(cel->Index());
            }
        }
    }
}

void TPZElastoPlasticIntPointsStructMatrix::SetUpIrregularBlocksData(TPZStack<REAL> &elindex_domain, TPZIrregularBlocksMatrix::IrregularBlocks &blocksData) {
    int nblocks = elindex_domain.size();

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
    for(int iel = 0; iel < nblocks; iel++) {
        TPZCompEl *cel = fMesh->Element(elindex_domain[iel]);
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
    }

    blocksData.fStorage.resize(blocksData.fMatrixPosition[nblocks]);

    for (int iel = 0; iel < nblocks; ++iel) {
        TPZCompEl *cel = fMesh->Element(elindex_domain[iel]);
        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement *>(cel);
        if (!cel_inter) DebugStop();
        TPZIntPoints *int_rule = &(cel_inter->GetIntegrationRule());

        int row_el = blocksData.fRowSizes[iel];
        int col_el = blocksData.fColSizes[iel];
        int pos_el = blocksData.fMatrixPosition[iel];
        int dim = cel->Dimension();

        TPZFMatrix<REAL> elmatrix;
        elmatrix.Resize(row_el, col_el);

        TPZMaterialData data;
        cel_inter->InitMaterialData(data);

        for (int64_t ipts = 0; ipts < row_el / dim; ipts++) {
            TPZVec<REAL> qsi(dim);
            REAL w;
            int_rule->Point(ipts, qsi, w);
            cel_inter->ComputeRequiredData(data, qsi);

            TPZFMatrix<REAL> dphiXY;
            data.axes.Transpose();
            data.axes.Multiply(data.dphix, dphiXY);

            for (int inf = 0; inf < col_el; inf++) {
                for (int idim = 0; idim < dim; idim++)
                    elmatrix(ipts * dim + idim, inf) = dphiXY(idim, inf);
            }
        }
        elmatrix.Transpose(); // Using CSR format
        TPZFMatrix<REAL> elmatloc(row_el, col_el, &blocksData.fStorage[pos_el], row_el * col_el);
        elmatloc = elmatrix;
    }

    this->CSRVectors(blocksData);
}

void TPZElastoPlasticIntPointsStructMatrix::CSRVectors(TPZIrregularBlocksMatrix::IrregularBlocks &blocksData) {
    int64_t nblocks = blocksData.fNumBlocks;
    int64_t rows = blocksData.fRowFirstIndex[nblocks];

    blocksData.fRowPtr.resize(rows + 1);
    blocksData.fColInd.resize(blocksData.fMatrixPosition[nblocks]);

    for (int iel = 0; iel < nblocks; ++iel) {
        for (int irow = 0; irow < blocksData.fRowSizes[iel]; ++irow) {
            blocksData.fRowPtr[irow + blocksData.fRowFirstIndex[iel]] = blocksData.fMatrixPosition[iel] + irow * blocksData.fColSizes[iel];

            for (int icol = 0; icol < blocksData.fColSizes[iel]; ++icol) {
                blocksData.fColInd[icol + blocksData.fMatrixPosition[iel] + irow * blocksData.fColSizes[iel]] = icol + blocksData.fColFirstIndex[iel];
            }
        }
    }
    blocksData.fRowPtr[rows] = blocksData.fMatrixPosition[nblocks];
}

void TPZElastoPlasticIntPointsStructMatrix::SetUpIndexes(TPZStack<REAL> &elindex_domain, TPZVec<int> &indexes) {
    int64_t nblocks = fCoefToGradSol.IrregularBlocksMatrix().Blocks().fNumBlocks;
    int64_t rows = fCoefToGradSol.IrregularBlocksMatrix().Rows();
    int64_t cols = fCoefToGradSol.IrregularBlocksMatrix().Cols();

    indexes.resize(fMesh->Dimension() * cols);
    TPZVec<REAL> weight(rows / fMesh->Dimension());

    int64_t cont1 = 0;
    int64_t cont2 = 0;
    int64_t wit = 0;
    for (int iel = 0; iel < nblocks; ++iel) {
        TPZCompEl *cel = fMesh->Element(elindex_domain[iel]);
        TPZInterpolatedElement *cel_inter = dynamic_cast<TPZInterpolatedElement *>(cel);
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
            weight[wit] = w * std::abs(data.detjac);
            wit++;
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
                        indexes[cont2 + cols] = pos + isize;
                        cont2++;
                    }
                }
            }
        }
    }

    TPZMaterial *material = fMesh->FindMaterial(1);
    fLambdaExp.SetMaterial(material);
    fLambdaExp.SetIntPoints(rows / fMesh->Dimension());
    fLambdaExp.SetWeightVector(weight);
}

void TPZElastoPlasticIntPointsStructMatrix::ColoredIndexes(TPZStack<REAL> &elindex_domain, TPZVec<int> &indexes, TPZVec<int> &coloredindexes, int &ncolor) {
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
        for (auto iel : elindex_domain) {
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
    coloredindexes.resize(fMesh->Dimension() * cols);
    int64_t neq = fMesh->NEquations();
    for (int64_t iel = 0; iel < nblocks; iel++) {
        int64_t elem_col = fCoefToGradSol.IrregularBlocksMatrix().Blocks().fColSizes[iel];
        int64_t cont_cols = fCoefToGradSol.IrregularBlocksMatrix().Blocks().fColFirstIndex[iel];

        for (int64_t icols = 0; icols < elem_col; icols++) {
            coloredindexes[cont_cols + icols] = indexes[cont_cols + icols] + elemcolor[iel]*neq;
            coloredindexes[cont_cols + cols + icols] = indexes[cont_cols + cols + icols] + elemcolor[iel]*neq;
        }
    }
}