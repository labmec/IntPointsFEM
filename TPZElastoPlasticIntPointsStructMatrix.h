/**
 * @file
 * @brief Contains the TPZSolveMatrix class which implements a solution based on a matrix procedure.
 */

#ifndef TPZIntPointsFEM_h
#define TPZIntPointsFEM_h

#include <StrMatrix/TPZSSpStructMatrix.h>
#include "TPZIrregularBlocksMatrix.h"
#include "TPZMyLambdaExpression.h"
#include "TPZCoefToGradSol.h"


class TPZElastoPlasticIntPointsStructMatrix : public TPZSymetricSpStructMatrix {
public:
    /** @brief Default constructor */
    TPZElastoPlasticIntPointsStructMatrix();

    /** @brief Creates the object based on a Compmesh
     * @param Compmesh : Computational mesh */
    TPZElastoPlasticIntPointsStructMatrix(TPZCompMesh *cmesh);

    /** @brief Default destructor */
    ~TPZElastoPlasticIntPointsStructMatrix();

    /** @brief Clone */
    TPZStructMatrix *Clone();

    // need help
    void CreateAssemble();

    void SetUpDataStructure();

    void CalcResidual(TPZFMatrix<REAL> &rhs);

    bool isBuilt() {
        if(fBlockMatrix) return true;
        else return false;
    }

private:
    TPZIrregularBlocksMatrix *fBlockMatrix;

    TPZCoefToGradSol fCoefdoGradSol;

    TPZMyLambdaExpression fLambdaExp;

    TPZStructMatrix fStructMatrix;
};

#endif /* TPZIntPointsFEM_h */
