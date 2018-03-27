#include "TPZMatGriffith.h"

#include "pzbndcond.h"
#include "pzmaterial.h"
#include "pzelasmat.h"

TPZMatGriffith::TPZMatGriffith(int id) : TPZElasticityMaterial(id)
{
    
}

/** @brief Default constructor */
TPZMatGriffith::TPZMatGriffith() : TPZElasticityMaterial()
{
    
}


TPZMatGriffith::TPZMatGriffith(const TPZMatGriffith &mat) : TPZElasticityMaterial(mat)
{
    
}

TPZMatGriffith::~TPZMatGriffith()
{
    
}

void TPZMatGriffith::ContributeBC(TPZMaterialData &data, REAL weight, TPZFMatrix<STATE> &ek, TPZFMatrix<STATE> &ef, TPZBndCond &bc)
{
    TPZMaterialData::MShapeFunctionType shapetype = data.fShapeType;
    if(shapetype==data.EVecShape){
        ContributeVecShapeBC(data,weight,ek, ef,bc);
        return;
    }
    
    TPZFMatrix<REAL> &phi = data.phi;
    
    const REAL BIGNUMBER  = TPZMaterial::gBigNumber;
    
    int phr = phi.Rows();
    short in,jn;
    
    if (ef.Cols() != bc.NumLoadCases()) {
        DebugStop();
    }
    
    //		In general when the problem is  needed to stablish any convention for ContributeBC implementations
    
    //     REAL v2[2];
    // 	v2[0] = bc.Val2()(0,0);
    // 	v2[1] = bc.Val2()(1,0);
    
    
    switch (bc.Type()) {
        case 0 :			// Dirichlet condition
        {
            for(in = 0 ; in < phr; in++) {
                for (int il = 0; il<NumLoadCases(); il++)
                {
                    REAL v2[2];
                    v2[0] = bc.Val2(il)(0,0);
                    v2[1] = bc.Val2(il)(1,0);
                    ef(2*in,il)   += BIGNUMBER * v2[0] * phi(in,0) * weight;        // forced v2 displacement
                    ef(2*in+1,il) += BIGNUMBER * v2[1] * phi(in,0) * weight;        // forced v2 displacement
                }
                for (jn = 0 ; jn < phi.Rows(); jn++)
                {
                    ek(2*in,2*jn)     += BIGNUMBER * phi(in,0) *phi(jn,0) * weight;
                    ek(2*in+1,2*jn+1) += BIGNUMBER * phi(in,0) *phi(jn,0) * weight;
                }
            }
        }
            break;
            
        case 1 :		// Neumann condition
        {
            for (in = 0; in < phr; in++)
            {
                for (int il = 0; il <fNumLoadCases; il++)
                {
                    TPZFNMatrix<2,STATE> v2 = bc.Val2(il);
                    ef(2*in,il) += v2(0,0) * phi(in,0) * weight;        // force in x direction
                    ef(2*in+1,il) +=  v2(1,0) * phi(in,0) * weight;      // force in y direction
                }
            }
        }
            break;
            
        case 2 :		// Mixed Condition
        {
            for(in = 0 ; in < phi.Rows(); in++)
            {
                for (int il = 0; il <fNumLoadCases; il++)
                {
                    TPZFNMatrix<2,STATE> v2 = bc.Val2(il);
                    ef(2*in,il) += v2(0,0) * phi(in,0) * weight;        // force in x direction
                    ef(2*in+1,il) += v2(1,0) * phi(in,0) * weight;      // forced in y direction
                }
                
                for (jn = 0 ; jn < phi.Rows(); jn++) {
                    ek(2*in,2*jn) += bc.Val1()(0,0) * phi(in,0) * phi(jn,0) * weight;         // peso de contorno => integral de contorno
                    ek(2*in+1,2*jn) += bc.Val1()(1,0) * phi(in,0) * phi(jn,0) * weight;
                    ek(2*in+1,2*jn+1) += bc.Val1()(1,1) * phi(in,0) * phi(jn,0) * weight;
                    ek(2*in,2*jn+1) += bc.Val1()(0,1) * phi(in,0) * phi(jn,0) * weight;
                }
            }   // este caso pode reproduzir o caso 0 quando o deslocamento
            
            break;
        case 3:
            {
                for(in = 0 ; in < phr; in++) {
                    for (int il = 0; il<NumLoadCases(); il++)
                    {
                        REAL v2[2];
                        v2[0] = bc.Val2(il)(0,0);
                        v2[1] = bc.Val2(il)(1,0);
                        ef(2*in,il)   += BIGNUMBER * v2[0] * phi(in,0) * weight;        // forced v2 displacement
                    }
                    for (jn = 0 ; jn < phi.Rows(); jn++)
                    {
                        ek(2*in,2*jn)     += BIGNUMBER * phi(in,0) *phi(jn,0) * weight;
                    }
                }
            }
            break;
            
            
        case 4:
            {
                for(in = 0 ; in < phr; in++) {
                    for (int il = 0; il<NumLoadCases(); il++)
                    {
                        REAL v2[2];
                        v2[0] = bc.Val2(il)(0,0);
                        v2[1] = bc.Val2(il)(1,0);
                        ef(2*in+1,il) += BIGNUMBER * v2[1] * phi(in,0) * weight;        // forced v2 displacement
                    }
                    for (jn = 0 ; jn < phi.Rows(); jn++)
                    {
                        ek(2*in+1,2*jn+1) += BIGNUMBER * phi(in,0) *phi(jn,0) * weight;
                    }
                }
            }
            break;
            
            
            
        case 5:
            {
                
                for(in = 0 ; in < phr; in++) {
                    for (int il = 0; il<NumLoadCases(); il++)
                    {
                        REAL v2[2];
                        v2[0] = bc.Val2(il)(0,0);
                        v2[1] = bc.Val2(il)(1,0);
                        ef(2*in+1,il) += BIGNUMBER * v2[1] * phi(in,0) * weight;        // forced v2 displacement
                    }
                    for (jn = 0 ; jn < phi.Rows(); jn++)
                    {
                        ek(2*in+1,2*jn+1) += BIGNUMBER * phi(in,0) *phi(jn,0) * weight;
                    }
                }
            }
            break;
        }
    }
}













