//
// Created by natalia on 03/09/2019.
//
#ifdef USING_TBB
#include <tbb/parallel_for.h>
#endif

#include "TPBrIntPointsStructMatrix.h"
#include "pzskylstrmatrix.h"
#include "pzbndcond.h"
#include "pzintel.h"
#include "pzsysmp.h"

template<class T, class MEM>
TPBrIntPointsStructMatrix<T, MEM>::TPBrIntPointsStructMatrix(TPZCompMesh *cmesh) : TPZSpStructMatrix(cmesh), fNMaterials(-1), fIntegrator(0), fSparseMatrixLinear(),
                                                                                fRhsLinear(),fBCMaterialIds(), fIAToSequence(0), fJAToSequence(0) {

    if (!cmesh->Reference()->Dimension()) {
        DebugStop();
    }
}

template<class T, class MEM>
TPBrIntPointsStructMatrix<T, MEM>::~TPBrIntPointsStructMatrix() {

}

template<class T, class MEM>
TPZStructMatrix * TPBrIntPointsStructMatrix<T, MEM>::Clone(){
    return new TPBrIntPointsStructMatrix(*this);
}

template<class T, class MEM>
TPZMatrix<STATE> * TPBrIntPointsStructMatrix<T, MEM>::Create(){

    SetUpDataStructure();

    TPZStack<int64_t> elgraph;
    TPZVec<int64_t> elgraphindex;
    fMesh->ComputeElGraph(elgraph,elgraphindex,fMaterialIds);
    TPZMatrix<STATE> * mat = SetupMatrixData(elgraph, elgraphindex);

    TPZFYsmpMatrix<STATE> *stiff = dynamic_cast<TPZFYsmpMatrix<STATE> *> (mat);

    fIAToSequence = stiff->IA();
    fJAToSequence = stiff->JA();

    return mat;
}

template<class T, class MEM>
TPZMatrix<STATE> *TPBrIntPointsStructMatrix<T, MEM>::CreateAssemble(TPZFMatrix<STATE> &rhs, TPZAutoPointer<TPZGuiInterface> guiInterface) {

    int64_t neq = fMesh->NEquations();
    TPZMatrix<STATE> *stiff = Create();
    rhs.Redim(neq,1);
    Assemble(*stiff,rhs,guiInterface);
    return stiff;
}

template<class T, class MEM>
void TPBrIntPointsStructMatrix<T, MEM>::Assemble(TPZMatrix<STATE> & mat, TPZFMatrix<STATE> & rhs, TPZAutoPointer<TPZGuiInterface> guiInterface) {

    TPZFYsmpMatrix<STATE> & stiff = dynamic_cast<TPZFYsmpMatrix<STATE> &> (mat);

    for (auto & numerical_integrator : fIntegrator) {
        numerical_integrator.KAssembly(stiff.A(),fIAToSequence,fJAToSequence);
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

template<class T, class MEM>
void TPBrIntPointsStructMatrix<T, MEM>::Assemble(TPZFMatrix<STATE> & rhs, TPZAutoPointer<TPZGuiInterface> guiInterface){
    for (auto & numerical_integrator : fIntegrator) {
        TPZFMatrix<REAL> rhs_mat(fMesh->NEquations(), 1);
        numerical_integrator.ResidualIntegration(fMesh->Solution(), rhs_mat);
        rhs += rhs_mat;
    }
    rhs += fRhsLinear;
}

template<class T, class MEM>
void TPBrIntPointsStructMatrix<T, MEM>::SetUpDataStructure() {

    TPZVec<int> element_indexes;
    ClassifyMaterialsByDimension();
    fNMaterials = fMaterialIds.size();
    fIntegrator.resize(fNMaterials);

    int imat = 0;
    std::set<int>::iterator it;
    for (it = fMaterialIds.begin(); it != fMaterialIds.end(); it++) {

        if(fIntegrator[imat].isBuilt()) continue;

        ComputeDomainElementIndexes(element_indexes, *it);
        fIntegrator[imat].SetElementIndexes(element_indexes);

        TPZMaterial * material = fMesh->FindMaterial(*it);
        TPZMatElastoPlastic2D < T, MEM > *mat = dynamic_cast<TPZMatElastoPlastic2D < T, MEM > *>(material);
        fIntegrator[imat].SetMaterial(mat);

        fIntegrator[imat].SetUpIrregularBlocksMatrix(fMesh);
        fIntegrator[imat].SetUpIndexes(fMesh);
        fIntegrator[imat].ColoredIndexes(fMesh);

        imat++;
    }
    
    AssembleBoundaryData();
}

template<class T, class MEM>
void TPBrIntPointsStructMatrix<T, MEM>::ComputeDomainElementIndexes(TPZVec<int> &element_indexes, int mat_id) {

    TPZStack<int> el_indexes_loc;
    for (int64_t i = 0; i < fMesh->NElements(); i++) {
        TPZCompEl *cel = fMesh->Element(i);
        if (!cel) continue;
        TPZGeoEl *gel = cel->Reference();
        if (!gel) continue;
        if(cel->Material()->Id() == mat_id){
            el_indexes_loc.Push(cel->Index());
        }
    }
    element_indexes = el_indexes_loc;
}

template<class T, class MEM>
void TPBrIntPointsStructMatrix<T, MEM>::ClassifyMaterialsByDimension() {

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


template<class T, class MEM>
void TPBrIntPointsStructMatrix<T, MEM>::AssembleBoundaryData() {

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



