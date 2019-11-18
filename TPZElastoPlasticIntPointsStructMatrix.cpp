#include "TPZElastoPlasticIntPointsStructMatrix.h"
#include "pzintel.h"
#include "pzskylstrmatrix.h"
#include "pzmetis.h"
#include "TPZConstitutiveLawProcessor.h"
#include "TPZElasticCriterion.h"
#include "pzbndcond.h"
#include "Timer.h"

#ifdef USING_MKL
#include <mkl.h>
#endif


TPZElastoPlasticIntPointsStructMatrix::TPZElastoPlasticIntPointsStructMatrix(TPZCompMesh *cmesh) : TPZSymetricSpStructMatrix(cmesh), fSparseMatrixLinear(), fRhsLinear(), fIntegrator(), fBCMaterialIds() {

    if (!cmesh->Reference()->Dimension()) {
        DebugStop();
    }
    fDimension = cmesh->Reference()->Dimension();

    #ifdef USING_CUDA
    dRhsLinear.resize(0);
    #endif
}

TPZElastoPlasticIntPointsStructMatrix::~TPZElastoPlasticIntPointsStructMatrix() {
}

TPZStructMatrix * TPZElastoPlasticIntPointsStructMatrix::Clone(){
    return new TPZElastoPlasticIntPointsStructMatrix(*this);
}

TPZMatrix<STATE> * TPZElastoPlasticIntPointsStructMatrix::Create(){

    if(!fIntegrator.isBuilt()) {
        this->SetUpDataStructure(); // When basis functions are computed and stored
    }
    
    TPZStack<int64_t> elgraph;
    TPZVec<int64_t> elgraphindex;
    fMesh->ComputeElGraph(elgraph,elgraphindex,fMaterialIds);
    TPZMatrix<STATE> * mat = SetupMatrixData(elgraph, elgraphindex);

    TPZSYsmpMatrix<STATE> *stiff = dynamic_cast<TPZSYsmpMatrix<STATE> *> (mat);
    fIntegrator.FillLIndexes(stiff->IA(), stiff->JA());
    
#ifdef USING_CUDA
    Timer timer;   
    timer.TimeUnit(Timer::ESeconds);
    timer.TimerOption(Timer::ECudaEvent);
    timer.Start();
    std::cout << "Transfering data to GPU..." << std::endl;
    #ifdef USING_SPARSE
        fIntegrator.IrregularBlockMatrix().CSRVectors();
    #endif
    this->TransferDataToGPU();
    timer.Stop();
    std::cout << "Done! It took " <<  timer.ElapsedTime() << timer.Unit() << std::endl;
#endif

    return mat;
}

#ifdef USING_CUDA
void TPZElastoPlasticIntPointsStructMatrix::TransferDataToGPU() {
    fIntegrator.TransferDataToGPU();

    dRhsLinear.resize(fRhsLinear.Rows());
    dRhsLinear.set(&fRhsLinear(0,0), fRhsLinear.Rows());
}
#endif

TPZMatrix<STATE> *TPZElastoPlasticIntPointsStructMatrix::CreateAssemble(TPZFMatrix<STATE> &rhs, TPZAutoPointer<TPZGuiInterface> guiInterface) {

    int64_t neq = fMesh->NEquations();
    TPZMatrix<STATE> *stiff = Create();
    rhs.Redim(neq,1);
    Assemble(*stiff,rhs,guiInterface);
    return stiff;
}

void TPZElastoPlasticIntPointsStructMatrix::SetUpDataStructure() {

    if(fIntegrator.isBuilt()) {
        std::cout << __PRETTY_FUNCTION__ << " Data structure has been setup." << std::endl;
        return;
    }

    ClassifyMaterialsByDimension();

    TPZVec<int> element_indexes;
    ComputeDomainElementIndexes(element_indexes);
    fIntegrator.SetElementIndexes(element_indexes);
    fIntegrator.SetUpIrregularBlocksData(fMesh);
    fIntegrator.SetUpIndexes(fMesh);
    fIntegrator.SetUpColoredIndexes(fMesh);

    AssembleBoundaryData();
}

 void TPZElastoPlasticIntPointsStructMatrix::Assemble(TPZMatrix<STATE> & mat, TPZFMatrix<STATE> & rhs, TPZAutoPointer<TPZGuiInterface> guiInterface) {
    
    int neq = fMesh->NEquations();

    rhs.Resize(neq, 1);
    rhs.Zero();  

    TPZSYsmpMatrix<STATE> &stiff = dynamic_cast<TPZSYsmpMatrix<STATE> &> (mat);
    TPZVec<STATE> &Kg = stiff.A();

    // Timer timer;   
    // timer.TimeUnit(Timer::ESeconds);
    // timer.TimerOption(Timer::EChrono);
    // timer.Start();

#ifdef USING_CUDA
    TPZVecGPU<REAL> d_Kg(Kg.size());
    d_Kg.Zero();

    TPZVecGPU<REAL> d_rhs(rhs.Rows());
    d_rhs.Zero();

    fIntegrator.KAssembly(fMesh->Solution(), d_Kg, d_rhs);
    fCudaCalls.DaxpyOperation(neq, 1., dRhsLinear.getData(), d_rhs.getData());

    // back to CPU
    d_rhs.get(&rhs(0,0), neq);
    d_Kg.get(&Kg[0], d_Kg.getSize());
#else
    fIntegrator.KAssembly(fMesh->Solution(), Kg, rhs);
    rhs += fRhsLinear;
#endif

    // timer.Stop();
    // std::cout << "K + R: Elasped time [sec] = " << timer.ElapsedTime() << std::endl;

    // Add Klinear contribution
    auto it_end = fSparseMatrixLinear.MapEnd();
    for (auto it = fSparseMatrixLinear.MapBegin(); it!=it_end; it++) {
        int64_t row = it->first.first;
        int64_t col = it->first.second;
        STATE val = it->second + stiff.GetVal(row, col);
        stiff.PutVal(row, col, val);
    }
}

void TPZElastoPlasticIntPointsStructMatrix::Assemble(TPZFMatrix<STATE> & rhs, TPZAutoPointer<TPZGuiInterface> guiInterface){

    int neq = fMesh->NEquations();

    Timer timer;   
    timer.TimeUnit(Timer::ESeconds);
    timer.TimerOption(Timer::EChrono);
    timer.Start();

    rhs.Resize(neq, 1);
    rhs.Zero();  
#ifdef USING_CUDA
    TPZVecGPU<REAL> d_rhs(rhs.Rows());
    d_rhs.Zero();
    fIntegrator.ResidualIntegration(fMesh->Solution(),d_rhs);
    fCudaCalls.DaxpyOperation(neq, 1., dRhsLinear.getData(), d_rhs.getData());
    d_rhs.get(&rhs(0,0), neq); //back to CPU
#else
    fIntegrator.ResidualIntegration(fMesh->Solution(),rhs);
    rhs += fRhsLinear;
#endif

    timer.Stop();
    std::cout << "R: Elasped time [sec] = " << timer.ElapsedTime() << std::endl;

}

void TPZElastoPlasticIntPointsStructMatrix::AssembleBoundaryData() {

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
