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
    m_IA_to_sequence.resize(0);
    m_JA_to_sequence.resize(0);

    #ifdef USING_CUDA
    d_IA_to_sequence.resize(0);    
    d_JA_to_sequence.resize(0);
    d_el_color_indexes.resize(0); 
    #endif
}

TPZElastoPlasticIntPointsStructMatrix::~TPZElastoPlasticIntPointsStructMatrix() {
}

TPZStructMatrix * TPZElastoPlasticIntPointsStructMatrix::Clone(){
    return new TPZElastoPlasticIntPointsStructMatrix(*this);
}

TPZMatrix<STATE> * TPZElastoPlasticIntPointsStructMatrix::Create(){

    if(!isBuilt()) {
        this->SetUpDataStructure(); // When basis functions are computed and storaged
    }
    
    TPZStack<int64_t> elgraph;
    TPZVec<int64_t> elgraphindex;
    fMesh->ComputeElGraph(elgraph,elgraphindex,fMaterialIds); // This method seems to be efficient.
    TPZMatrix<STATE> * mat = SetupMatrixData(elgraph, elgraphindex);
    
    /// Sparsify global indexes
    // Filling local std::map
    TPZSYsmpMatrix<STATE> *stiff = dynamic_cast<TPZSYsmpMatrix<STATE> *> (mat);

    m_IA_to_sequence = stiff->IA();
    m_JA_to_sequence = stiff->JA();


    fIntegrator.FillLIndexes(m_IA_to_sequence, m_JA_to_sequence);
    
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
    d_IA_to_sequence.resize(m_IA_to_sequence.size());
    d_IA_to_sequence.set(&m_IA_to_sequence[0], m_IA_to_sequence.size());

    d_JA_to_sequence.resize(m_JA_to_sequence.size());
    d_JA_to_sequence.set(&m_JA_to_sequence[0], m_JA_to_sequence.size());

    d_el_color_indexes.resize(m_el_color_indexes.size());
    d_el_color_indexes.set(&m_el_color_indexes[0], m_el_color_indexes.size());

    d_RhsLinear.resize(fRhsLinear.Rows());
    d_RhsLinear.set(&fRhsLinear(0,0), fRhsLinear.Rows());

    d_color_l_sequence.resize(m_color_l_sequence.size());
    d_color_l_sequence.set(&m_color_l_sequence[0], m_color_l_sequence.size());

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

    if(isBuilt()) {
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

#ifdef USING_CUDA
    TPZVecGPU<REAL> d_solution(fMesh->Solution().Rows());
    d_solution.set(&fMesh->Solution()(0,0), fMesh->Solution().Rows());

    TPZVecGPU<REAL> d_rhs(rhs.Rows());
    d_rhs.Zero();

    TPZVecGPU<REAL> d_delta_strain;
    TPZVecGPU<REAL> d_sigma;
    TPZVecGPU<REAL> d_dep;

    fIntegrator.Multiply(d_solution, d_delta_strain);
    fIntegrator.ConstitutiveLawProcessor().ComputeSigmaDep(d_delta_strain, d_sigma, d_dep);
    fIntegrator.MultiplyTranspose(d_sigma, d_rhs);
    fCudaCalls.DaxpyOperation(neq, 1., d_RhsLinear.getData(), d_rhs.getData());
    d_rhs.get(&rhs(0,0), neq); //back to CPU

    TPZFMatrix<REAL> dep(d_dep.getSize(), 1);   
    d_dep.get(&dep(0,0), d_dep.getSize()); //back to CPU

#else
    TPZFMatrix<REAL> delta_strain;
    TPZFMatrix<REAL> sigma;
    TPZFMatrix<REAL> dep;

    fIntegrator.Multiply(fMesh->Solution(), delta_strain);
    fIntegrator.ConstitutiveLawProcessor().ComputeSigmaDep(delta_strain, sigma, dep);
    fIntegrator.MultiplyTranspose(sigma, rhs); // Perform Residual integration using a global linear application B
#endif
    rhs += fRhsLinear;

    TPZSYsmpMatrix<STATE> &stiff = dynamic_cast<TPZSYsmpMatrix<STATE> &> (mat);
    TPZVec<STATE> &Kg = stiff.A();
    int64_t nnz = Kg.size();
    
    TPZVec<int> & indexes = fIntegrator.DoFIndexes();
    TPZVec<int> & el_n_dofs = fIntegrator.IrregularBlocksMatrix().Blocks().fColSizes;
    TPZVec<int> & cols_first_index = fIntegrator.IrregularBlocksMatrix().Blocks().fColFirstIndex;

    int n_colors = fIntegrator.FirstColorIndex().size()-1;
     
    int first = fIntegrator.FirstColorIndex()[0];
    int last = fIntegrator.FirstColorIndex()[n_colors];
    int el_dofs = el_n_dofs[0];
    int nel_per_color = last - first;
    TPZVec<REAL> Kc(fIntegrator.ColorLSequence().size(),0.0);
     
#ifdef USING_TBB
    tbb::parallel_for(size_t(0),size_t(nel_per_color),size_t(1),[&](size_t i)
#else
    for (int i = 0; i < nel_per_color; i++)
#endif
        {
            int iel = fIntegrator.ElColorIndexes()[first + i];

            // Compute Elementary Matrix.
            TPZFMatrix<STATE> K;
            // fIntegrator.ComputeTangentMatrix(iel,K);
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
         
    for (int ic = 0; ic < n_colors; ic++) {
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
    fCudaCalls.DaxpyOperation(neq, 1., d_RhsLinear.getData(), d_rhs.getData());
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
