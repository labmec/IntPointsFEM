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
        this->SetUpDataStructure(); // When basis functions are computed and storaged
    }
    
    TPZStack<int64_t> elgraph;
    TPZVec<int64_t> elgraphindex;
    fMesh->ComputeElGraph(elgraph,elgraphindex,fMaterialIds); // This method seems to be efficient.
    TPZMatrix<STATE> * mat = SetupMatrixData(elgraph, elgraphindex);
    
    /// Sparsify global indexes
    // Filling local std::map
    TPZSYsmpMatrix<STATE> *stiff = dynamic_cast<TPZSYsmpMatrix<STATE> *> (mat);
    fIntegrator.FillLIndexes(stiff->IA(), stiff->JA());
    
#ifdef USING_CUDA
    Timer timer;   
    timer.TimeUnit(Timer::ESeconds);
    timer.TimerOption(Timer::ECudaEvent);
    timer.Start();
    std::cout << "Transfering data to GPU..." << std::endl;
    fIntegrator.TransferDataToGPU();
    this->TransferDataToGPU();
    timer.Stop();
    std::cout << "Done! It took " <<  timer.ElapsedTime() << timer.Unit() << std::endl;
#endif

    return mat;
}

#ifdef USING_CUDA
void TPZElastoPlasticIntPointsStructMatrix::TransferDataToGPU() {
    dRhsLinear.resize(fRhsLinear.Rows());
    dRhsLinear.set(&fRhsLinear(0,0), fRhsLinear.Rows());
}
#endif



TPZMatrix<STATE> *TPZElastoPlasticIntPointsStructMatrix::CreateAssemble(TPZFMatrix<STATE> &rhs, TPZAutoPointer<TPZGuiInterface> guiInterface) {

    int64_t neq = fMesh->NEquations();
    TPZMatrix<STATE> *stiff = Create(); // @TODO:: Requires optimization.
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


#ifdef USING_CUDA
    TPZVecGPU<REAL> d_solution(fMesh->Solution().Rows());
    d_solution.set(&fMesh->Solution()(0,0), fMesh->Solution().Rows());

    TPZVecGPU<REAL> d_delta_strain, d_sigma, d_dep;

    //Compute rhs
    TPZVecGPU<REAL> d_rhs(rhs.Rows());
    d_rhs.Zero();
    fIntegrator.Multiply(d_solution, d_delta_strain);
    fIntegrator.ConstitutiveLawProcessor().ComputeSigmaDep(d_delta_strain, d_sigma, d_dep);
    fIntegrator.MultiplyTranspose(d_sigma, d_rhs);

    // Compute Kc
    TPZVecGPU<REAL> d_Kc(fIntegrator.ColorLSequence().size());
    d_Kc.Zero();
    fCudaCalls.MatrixAssemble(d_Kc.getData(), d_dep.getData(), fIntegrator.IrregularBlocksMatrix().Blocks().fNumBlocks, fIntegrator.ElColorIndexDev().getData(), fIntegrator.ConstitutiveLawProcessor().WeightVectorDev().getData(),
    fIntegrator.DoFIndexesDev().getData(), fIntegrator.IrregularBlocksMatrix().BlocksDev().dStorage.getData(),
    fIntegrator.IrregularBlocksMatrix().BlocksDev().dRowSizes.getData(), fIntegrator.IrregularBlocksMatrix().BlocksDev().dColSizes.getData(),
    fIntegrator.IrregularBlocksMatrix().BlocksDev().dRowFirstIndex.getData(), fIntegrator.IrregularBlocksMatrix().BlocksDev().dColFirstIndex.getData(),
    fIntegrator.IrregularBlocksMatrix().BlocksDev().dMatrixPosition.getData());

    // Assemble K
    TPZVecGPU<REAL> d_Kg(Kg.size());
    d_Kg.Zero();
    for (int ic = 0; ic < fIntegrator.NColors(); ic++) {
        int first_l = fIntegrator.FirstColorLIndex()[ic];
        int last_l = fIntegrator.FirstColorLIndex()[ic + 1];
        int n_l_indexes = last_l - first_l;
        TPZVecGPU<REAL> aux(n_l_indexes);
        fCudaCalls.GatherOperation(n_l_indexes, d_Kg.getData(), aux.getData(), &fIntegrator.ColorLSequenceDev().getData()[first_l]);
        fCudaCalls.DaxpyOperation(n_l_indexes, 1., &d_Kc.getData()[first_l], aux.getData());
        fCudaCalls.ScatterOperation(n_l_indexes, aux.getData(), d_Kg.getData(), &fIntegrator.ColorLSequenceDev().getData()[first_l]);
    }

    // back to CPU
    d_rhs.get(&rhs(0,0), neq);
    d_Kg.get(&Kg[0], d_Kg.getSize()); // back to CPU
#else
    TPZFMatrix<REAL> delta_strain, sigma, dep;

    fIntegrator.Multiply(fMesh->Solution(), delta_strain);
    fIntegrator.ConstitutiveLawProcessor().ComputeSigmaDep(delta_strain, sigma, dep);
    fIntegrator.MultiplyTranspose(sigma, rhs);

    int el_dofs = fIntegrator.IrregularBlocksMatrix().Blocks().fColSizes[0];

     // Compute Kc
    TPZVec<REAL> Kc(fIntegrator.ColorLSequence().size(),0.0);
#ifdef USING_TBB
     tbb::parallel_for(size_t(0),size_t(fIntegrator.IrregularBlocksMatrix().Blocks().fNumBlocks),size_t(1),[&](size_t i)
#else
     for (int i = 0; i < fIntegrator.IrregularBlocksMatrix().Blocks().fNumBlocks; i++)
#endif
     {
         int iel = fIntegrator.ElColorIndexes()[i];

         // Compute Elementary Matrix.
         TPZFMatrix<STATE> K;
         fIntegrator.ComputeTangentMatrix(iel,dep, K);
         int stride = i*(el_dofs * el_dofs + el_dofs)/2;
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
     for (int ic = 0; ic < fIntegrator.NColors(); ic++) {
         int first_l = fIntegrator.FirstColorLIndex()[ic];
         int last_l = fIntegrator.FirstColorLIndex()[ic + 1];
         int n_l_indexes = last_l - first_l;
         // Gather from Kg
         TPZVec<REAL> aux(n_l_indexes);
         cblas_dgthr(n_l_indexes, &Kg[0], &aux[0], &fIntegrator.ColorLSequence()[first_l]);

         // Contributing
         cblas_daxpy(n_l_indexes, 1., &Kc[first_l], 1., &aux[0],1);

         // Scatter to Kg
         cblas_dsctr(n_l_indexes, &aux[0], &fIntegrator.ColorLSequence()[first_l], &Kg[0]);
     }
#endif
    rhs += fRhsLinear;

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
