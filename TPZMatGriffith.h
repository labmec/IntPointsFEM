//
//  TPZMatGriffith.h
//  Fracture
//
//  Created by Karolinne Oliveira Coelho on 2/7/18.
//
//

#ifndef TPZMatGriffith_h
#define TPZMatGriffith_h

#include "pzmaterial.h"
#include "pzelasmat.h"

/**
 * @ingroup material
 */
class  TPZMatGriffith : public TPZElasticityMaterial
{
    
public:
    
    TPZMatGriffith(int id);
    
    /** @brief Default constructor */
    TPZMatGriffith();
    
    /** @brief Creates a material object based on the referred object and inserts it in the vector of material pointers of the mesh. */
    /** Upon return vectorindex contains the index of the material object within the vector */
    TPZMatGriffith(const TPZMatGriffith &mat);
    
    /** @brief Creates a new material from the current object   ??*/
    TPZMaterial * NewMaterial() { return new TPZMatGriffith(*this);}
    
    /** @brief Default destructor */
    virtual ~TPZMatGriffith();
    
    /** @brief Returns the name of the material */
    std::string Name() { return "TPZMatGriffith"; }
    
    /** @brief Returns the integrable dimension of the material */
    int Dimension() const {return 2;}
    
    /** @brief Returns the number of state variables associated with the material */
    int NStateVariables() { return 2;}
    
    
    /**
     * @brief It computes a contribution to the stiffness matrix and load vector at one BC integration point.
     * @param data [in] stores all input data
     * @param weight [in] is the weight of the integration rule
     * @param ek [out] is the stiffness matrix
     * @param ef [out] is the load vector
     * @param bc [in] is the boundary condition material
     * @since October 07, 2011
     */
    void ContributeBC(TPZMaterialData &data, REAL weight, TPZFMatrix<STATE> &ek, TPZFMatrix<STATE> &ef, TPZBndCond &bc);
    
};


#endif /* TPZMatGriffith_h */
