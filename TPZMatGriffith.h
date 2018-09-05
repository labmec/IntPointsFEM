//
//  TPZMatGriffith.h
//  PZ
//
//  Created by Omar on 10/27/14.
//
//

#ifndef __PZ__TPZMatGriffith__
#define __PZ__TPZMatGriffith__

#include <stdio.h>
#include "TPZMaterial.h"
#include "pzvec.h"
#include <iostream>


/**
 * @ingroup material
 * @author Omar Duran
 * @since 10/27/2014.
 * @brief Material to solve a 2D linear elasticity
 * @brief Here is used CG approximation.
 */




/**
 * @ingroup material
 * @brief Description Linear elastic equations
 */
/**
 **@ingroup Linear Elastic Equation
 * \f$  div(T(u)) + b = 0  ==> Int{Grad(v).T(u)}dx - Int{v.gN}ds  = Int{b.v}dx  \f$ (Eq. 1)
 *
 *\f$ T(u) =  lambda*Trace(E(u)I + 2*mu*(E(u)) - \f$
 *
 *\f$ E(u) =  (1/2)(Grad(u) + Transpose(Grad(u)) \f$
 *
 */

class TPZMatGriffith : public TPZMaterial {

protected:

    /** @brief Forcing vector */
    TPZManVector<STATE,2>  ff;

    /** @brief Elasticity modulus */
    REAL fE;

    /** @brief Poison coeficient */
    REAL fnu;

    /** @brief first Lame Parameter */
    REAL flambda;

    /** @brief Second Lame Parameter */
    REAL fmu;

    /** @brief Initial Stress */
    REAL fPreStressXX;
    REAL fPreStressXY;
    REAL fPreStressYY;
    REAL fPreStressZZ;

    /** @brief Uses plain stress
     * @note \f$fPlaneStress = 1\f$ => Plain stress state
     * @note \f$fPlaneStress != 1\f$ => Plain Strain state
     */
    int fPlaneStress;


public:
    virtual int ClassId() const;

    TPZMatGriffith();

    /**
     * @brief Creates an elastic material with:
     * @param id material id
     * @param E elasticity modulus
     * @param nu poisson coefficient
     * @param fx forcing function \f$ -x = fx \f$
     * @param fy forcing function \f$ -y = fy \f$
     * @param plainstress \f$ plainstress = 1 \f$ indicates use of plainstress
     */
    TPZMatGriffith(int matid, REAL E, REAL nu, REAL fx, REAL fy, int plainstress = 1);

    TPZMatGriffith(int matid);

    TPZMatGriffith &operator=(const TPZMatGriffith &copy);

    virtual ~TPZMatGriffith();

    /** @brief Copy constructor */
    TPZMatGriffith(const TPZMatGriffith &cp);

    virtual TPZMaterial *NewMaterial()
    {
        return new TPZMatGriffith(*this);
    }

    virtual void Print(std::ostream & out);

    virtual std::string Name() { return "TPZMatGriffith"; }

    int Dimension() const {return 2;}

    virtual int NStateVariables();

    /**
     * @brief Set parameters of elastic material:
     * @param First  Lame Parameter Lambda
     * @param Second Lame Parameter Mu -> G
     * @param fx forcing function \f$ -x = fx \f$
     * @param fy forcing function \f$ -y = fy \f$
     * @param plainstress \f$ plainstress = 1 \f$ indicates use of plainstress
     */
    void SetParameters(REAL Lambda, REAL mu, REAL fx, REAL fy)
    {
        fE = (mu*(3.0*Lambda+2.0*mu))/(Lambda+mu);
        fnu = (Lambda)/(2*(Lambda+mu));

        flambda = Lambda;
        fmu = mu;
        ff[0] = fx;
        ff[1] = fy;
    }

    /**
     * @brief Set parameters of elastic material:
     * @param First  Lame Parameter Lambda
     * @param Second Lame Parameter Mu -> G
     * @param fx forcing function \f$ -x = 0 \f$
     * @param fy forcing function \f$ -y = 0 \f$
     */
    void SetElasticParameters(REAL Eyoung, REAL nu)
    {
        this->SetElasticity(Eyoung,nu);
    }

    /**
     * @brief Set parameters of elastic material:
     * @param First  Lame Parameter Lambda
     * @param Second Lame Parameter Mu -> G
     * @param fx forcing function \f$ -x = fx \f$
     * @param fy forcing function \f$ -y = fy \f$
     * @param plainstress \f$ plainstress = 1 \f$ indicates use of plainstress
     */
    void SetElasticity(REAL Ey, REAL nu)
    {
        fE = Ey;
        fnu = nu;
        flambda = (Ey*nu)/((1+nu)*(1-2*nu));
        fmu = Ey/(2*(1+nu));

    }

    /** @brief Set plane problem
     * planestress = 1 => Plain stress state
     * planestress != 1 => Plain Strain state
     */
    void SetfPlaneProblem(int planestress)
    {
        fPlaneStress = planestress;
    }

    /** @brief Set plane problem
     * planestress = 1 => Plain stress state
     * planestress != 1 => Plain Strain state
     */
    void SetPlaneStrain()
    {
        fPlaneStress = 0;
    }

    /** @brief Set Initial Stress */
    void SetPreStress(REAL SigmaXX, REAL SigmaXY, REAL SigmaYY, REAL SigmaZZ)
    {
        fPreStressXX = SigmaXX;
        fPreStressXY = SigmaXY;
        fPreStressYY = SigmaYY;
        fPreStressZZ = SigmaZZ;
    }

    /// compute the stress tensor as a function of the solution gradient
    void ComputeSigma(const TPZFMatrix<STATE> &dudx, TPZFMatrix<STATE> &sigma);

    // Get Elastic Materials Parameters
    void GetElasticParameters(REAL &Ey, REAL &nu, REAL &Lambda, REAL &G)
    {
        Ey = fE;
        nu =  fnu;
        Lambda =  flambda;
        G = fmu;
    }

    /** @brief Get Eyoung and Poisson
     * fE young modulus
     * fnu Poisson ratio
     */
    STATE GetEyoung() {return fE;}
    STATE GetNu() {return fnu;}

    /** @brief Get lame parameters
     * Lambda first lame
     * Mu Second lame
     */
    STATE GetLambda() {return flambda;}
    STATE GetMu() {return fmu;}


    virtual void FillDataRequirements(TPZMaterialData &data);

    virtual void FillBoundaryConditionDataRequirement(int type, TPZMaterialData &data);

    /**
     * @brief It computes a contribution to the stiffness matrix and load vector at one integration point to multiphysics simulation.
     * @param datavec [in] stores all input data
     * @param weight [in] is the weight of the integration rule
     * @param ek [out] is the stiffness matrix
     * @param ef [out] is the load vector
     */
    virtual void Contribute(TPZMaterialData &data, REAL weight, TPZFMatrix<STATE> &ek, TPZFMatrix<STATE> &ef);
    virtual void Contribute(TPZMaterialData &data, REAL weight, TPZFMatrix<STATE> &ef);
    void ContributeVec(TPZMaterialData &data, REAL weight, TPZFMatrix<STATE> &ek, TPZFMatrix<STATE> &ef);
    void ContributeVec(TPZMaterialData &data, REAL weight, TPZFMatrix<STATE> &ef);

    virtual void ContributeBC(TPZMaterialData &data, REAL weight, TPZFMatrix<STATE> &ek,TPZFMatrix<STATE> &ef,TPZBndCond &bc);
    virtual void ContributeBC(TPZMaterialData &data, REAL weight, TPZFMatrix<STATE> &ef,TPZBndCond &bc);

    virtual int VariableIndex(const std::string &name);

    virtual int NSolutionVariables(int var);

    //public:
    virtual void Solution(TPZMaterialData &data, int var, TPZVec<STATE> &Solout);
    virtual void Solution(TPZMaterialData &data, TPZVec<TPZMaterialData> &dataleftvec, TPZVec<TPZMaterialData> &datarightvec, int var, TPZVec<STATE> &Solout, TPZCompEl * Left, TPZCompEl * Right) {
        DebugStop();
    }

    /**
     * @brief Computes the error due to the difference between the interpolated flux \n
     * and the flux computed based on the derivative of the solution
     */
    virtual void Errors(TPZVec<REAL> &x, TPZVec<STATE> &sol, TPZFMatrix<STATE> &dsol,
                        TPZFMatrix<REAL> &axes, TPZVec<STATE> &flux,
                        TPZVec<STATE> &uexact, TPZFMatrix<STATE> &duexact,
                        TPZVec<REAL> &val);


    /**
     * Save the element data to a stream
     */
    virtual void Write(TPZStream &buf, int withclassid) const;

    /**
     * Read the element data from a stream
     */
    void Read(TPZStream &buf, void *context);

};





#endif /* defined(__PZ__TPZMatGriffith__) */
