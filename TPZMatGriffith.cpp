//
//  TPZMatGriffith.cpp
//  PZ
//
//  Created by Omar on 10/27/14.
//
//


#include <iostream>
#include <string>
#include "TPZMatGriffith.h"
#include "pzbndcond.h"
#include "pzaxestools.h"
#include "pzlog.h"

#ifdef LOG4CXX
static LoggerPtr logger(Logger::getLogger("pz.elasticity"));
#endif


TPZMatGriffith::TPZMatGriffith()
        : TPZRegisterClassId(&TPZMatGriffith::ClassId), TPZMaterial()
{
    fE = 0.;
    fnu = 0.;
    flambda = 0.;
    fmu = 0.;
    ff.resize(2);
    ff[0]=0.;
    ff[1]=0.;
    fPlaneStress = 1;
    fPreStressXX = 0.0;
    fPreStressXY = 0.0;
    fPreStressYY = 0.0;
    fPreStressZZ = 0.0;

}

TPZMatGriffith::TPZMatGriffith(int matid)
        : TPZRegisterClassId(&TPZMatGriffith::ClassId), TPZMaterial(matid)
{
    fE = 0.;
    fnu = 0.;
    flambda = 0.;
    fmu = 0.;
    ff.resize(2);
    ff[0]=0.;
    ff[1]=0.;
    fPlaneStress = 1;
    fPreStressXX = 0.0;
    fPreStressXY = 0.0;
    fPreStressYY = 0.0;
    fPreStressZZ = 0.0;
}

TPZMatGriffith::TPZMatGriffith(int matid, REAL E, REAL nu, REAL fx, REAL fy, int plainstress)
        : TPZRegisterClassId(&TPZMatGriffith::ClassId), TPZMaterial(matid)
{
    fE = E;
    fnu = nu;
    flambda = (E*nu)/((1+nu)*(1-2*nu));
    fmu = E/(2*(1+nu));
    ff.resize(2);
    ff[0]=fx;
    ff[1]=fy;
    fPlaneStress = plainstress;
    fPreStressXX = 0.0;
    fPreStressXY = 0.0;
    fPreStressYY = 0.0;
    fPreStressZZ = 0.0;
}

TPZMatGriffith::~TPZMatGriffith()
{
}


TPZMatGriffith::TPZMatGriffith(const TPZMatGriffith &copy)
        : TPZRegisterClassId(&TPZMatGriffith::ClassId),  TPZMaterial(copy)
{
    fE = copy.fE;
    fnu = copy.fnu;
    flambda = copy.flambda;
    fmu = copy.fmu;
    ff.resize(copy.ff.size());
    for (int i = 0; i < copy.ff.size(); i++) {
        ff[i] = copy.ff[i];
    }
    fPlaneStress = copy.fPlaneStress;
    fPreStressXX = copy.fPreStressXX;
    fPreStressXY = copy.fPreStressXY;
    fPreStressYY = copy.fPreStressYY;
    fPreStressZZ = copy.fPreStressZZ;
}

TPZMatGriffith & TPZMatGriffith::operator=(const TPZMatGriffith &copy)
{
    TPZMaterial::operator = (copy);
    fE = copy.fE;
    fnu = copy.fnu;
    flambda = copy.flambda;
    fmu = copy.fmu;
    fPreStressXX = copy.fPreStressXX;
    fPreStressXY = copy.fPreStressXY;
    fPreStressYY = copy.fPreStressYY;
    fPreStressZZ = copy.fPreStressZZ;
    ff.resize(copy.ff.size());
    for (int i = 0; i < copy.ff.size(); i++) {
        ff[i] = copy.ff[i];
    }
    fPlaneStress = copy.fPlaneStress;
    return *this;
}

int TPZMatGriffith::NStateVariables() {
    return 2;
}


void TPZMatGriffith::Contribute(TPZMaterialData &data, REAL weight, TPZFMatrix<STATE>  &ek, TPZFMatrix<STATE> &ef) {


    if (data.fShapeType == TPZMaterialData::EVecShape) {
        ContributeVec(data,weight,ek,ef);
        return;
    }
    // Getting weight functions
    TPZFMatrix<REAL>  &phiU     =  data.phi;
    TPZFMatrix<REAL> &dphiU     =  data.dphix;
    int phrU = phiU.Rows();

    int FirstU  = 0;

    //TPZManVector<STATE,3> sol_u =    data.sol[0];

    //TPZFMatrix<STATE> dsol_u = data.dsol[0];

    REAL LambdaL, MuL;

    // Functions computed at point x_{k} for each integration point
    LambdaL     = flambda;
    MuL         = fmu;

    //  ////////////////////////// Jacobian Matrix ///////////////////////////////////
    //  Contribution of domain integrals for Jacobian matrix
    //  Elasticity Block (Equation for elasticity )
    //	Elastic equation
    //	Linear strain operator
    //	Ke Matrix
    for(int iu = 0; iu < phrU; iu++ )
    {
        TPZManVector<REAL,2> dv(2);
        //	Derivative for Vx
        dv[0] = dphiU(0,iu)*data.axes(0,0)+dphiU(1,iu)*data.axes(1,0);
        //	Derivative for Vy
        dv[1] = dphiU(0,iu)*data.axes(0,1)+dphiU(1,iu)*data.axes(1,1);

        for(int ju = 0; ju < phrU; ju++)
        {
            TPZManVector<REAL,2> du(2);
            //	Derivative for Ux
            du[0] = dphiU(0,ju)*data.axes(0,0)+dphiU(1,ju)*data.axes(1,0);
            //	Derivative for Uy
            du[1] = dphiU(0,ju)*data.axes(0,1)+dphiU(1,ju)*data.axes(1,1);

            if (this->fPlaneStress == 1)
            {
                /* Plain stress state */
                ek(2*iu + FirstU, 2*ju + FirstU)	     += weight*((4*(MuL)*(LambdaL+MuL)/(LambdaL+2*MuL))*dv[0]*du[0]		+ (MuL)*dv[1]*du[1]);

                ek(2*iu + FirstU, 2*ju+1 + FirstU)       += weight*((2*(MuL)*(LambdaL)/(LambdaL+2*MuL))*dv[0]*du[1]			+ (MuL)*dv[1]*du[0]);

                ek(2*iu+1 + FirstU, 2*ju + FirstU)       += weight*((2*(MuL)*(LambdaL)/(LambdaL+2*MuL))*dv[1]*du[0]			+ (MuL)*dv[0]*du[1]);

                ek(2*iu+1 + FirstU, 2*ju+1 + FirstU)     += weight*((4*(MuL)*(LambdaL+MuL)/(LambdaL+2*MuL))*dv[1]*du[1]		+ (MuL)*dv[0]*du[0]);
            }
            else
            {
                /* Plain Strain State */
                ek(2*iu + FirstU,2*ju + FirstU)         += weight*	((LambdaL + 2*MuL)*dv[0]*du[0]	+ (MuL)*dv[1]*du[1]);

                ek(2*iu + FirstU,2*ju+1 + FirstU)       += weight*	(LambdaL*dv[0]*du[1]			+ (MuL)*dv[1]*du[0]);

                ek(2*iu+1 + FirstU,2*ju + FirstU)       += weight*	(LambdaL*dv[1]*du[0]			+ (MuL)*dv[0]*du[1]);

                ek(2*iu+1 + FirstU,2*ju+1 + FirstU)     += weight*	((LambdaL + 2*MuL)*dv[1]*du[1]	+ (MuL)*dv[0]*du[0]);

            }
        }
    }
    //  ////////////////////////// Jacobian Matrix ///////////////////////////////////
    this->Contribute(data,weight,ef);
}

void TPZMatGriffith::ContributeVec(TPZMaterialData &data, REAL weight, TPZFMatrix<STATE> &ek, TPZFMatrix<STATE> &ef)
{
    if (data.fShapeType != TPZMaterialData::EVecShape) {
        DebugStop();
    }

    // Getting weight functions
    TPZFMatrix<REAL> &dphiU     =  data.dphix;
    int phrU = dphiU.Cols();

    TPZFNMatrix<200,REAL> dudaxes(2,dphiU.Cols()), dvdaxes(2,dphiU.Cols()), dudx(3,phrU), dvdx(3,phrU);
    for (int i=0; i<2; i++) {
        for (int j=0; j<phrU; j++) {
            dudaxes(i,j) = dphiU(i,j);
            dvdaxes(i,j) = dphiU(2+i,j);
        }
    }
    TPZAxesTools<REAL>::Axes2XYZ(dudaxes, dudx, data.axes);
    TPZAxesTools<REAL>::Axes2XYZ(dvdaxes, dvdx, data.axes);



    REAL LambdaL, MuL;

    // Functions computed at point x_{k} for each integration point
    LambdaL     = flambda;
    MuL         = fmu;

    //  ////////////////////////// Jacobian Matrix ///////////////////////////////////
    //  Contribution of domain integrals for Jacobian matrix
    //  Elasticity Block (Equation for elasticity )
    //    Elastic equation
    //    Linear strain operator
    //    Ke Matrix
    for(int iu = 0; iu < phrU; iu++ )
    {
        TPZFNMatrix<4,REAL> gradv(2,2);
        gradv(0,0) = dudx(0,iu);
        gradv(0,1) = dudx(1,iu);
        gradv(1,0) = dvdx(0,iu);
        gradv(1,1) = dvdx(1,iu);

        for(int ju = 0; ju < phrU; ju++)
        {
            TPZFNMatrix<4,REAL> gradu(2,2);
            gradu(0,0) = dudx(0,ju);
            gradu(0,1) = dudx(1,ju);
            gradu(1,0) = dvdx(0,ju);
            gradu(1,1) = dvdx(1,ju);

            if (this->fPlaneStress == 1)
            {
                /* Plain stress state
                 \sigma_x = E/(1-\nu\nu) (\epsilon_x + \nu \epsilon_y)
                 \sigma_x = \frac{4\mu(\lambda+\mu)}{\lambda+2\mu)}\epsilon_x + \frac{2\mu\lambda}{\lambda+2\mu} \epsilon_y
                 \sigma_y = E/(1-\nu\nu) (\epsilon_y + \nu \epsilon_x)
                 \sigma_y = \frac{4\mu(\lambda+\mu)}{\lambda+2\mu)}\epsilon_y + \frac{2\mu\lambda}{\lambda+2\mu} \epsilon_x
                 \tau_{xy} = \frac{E}{1+\nu} \epsilon_{xy}
                 \tau_{xy} = \frac{1}{2\mu} \epsilon_{xy}
                 */
                TPZFNMatrix<4,REAL> sigma_u(2,2);
                sigma_u(0,0) = fE/(1-fnu*fnu) *(gradu(0,0)+fnu*gradu(1,1));
                sigma_u(1,1) = fE/(1-fnu*fnu) *(gradu(1,1)+fnu*gradu(0,0));
                sigma_u(0,1) = fE/(2.*(1+fnu))*(gradu(0,1)+gradu(1,0));
                sigma_u(1,0) = sigma_u(0,1);
                ek(iu, ju)         += weight*(sigma_u(0,0)*gradv(0,0)+sigma_u(1,1)*gradv(1,1)+
                                              sigma_u(1,0)*gradv(1,0)+sigma_u(0,1)*gradv(0,1));
            }
            else
            {
                /* Plain Strain State */

                /*
                 \sigma_x = \frac{E}{(1+\nu)(1-2\nu)}((1-\nu)\epsilon_x + \nu\epsilon_y)
                 \sigma_x = (\lambda+2\mu)\epsilon_x + \lambda\epsilon_y
                 \sigma_y = \frac{E}{(1+\nu)(1-2\nu)}((1-\nu)\epsilon_y + \nu\epsilon_x)
                 \sigma_y = (\lambda+2\mu)\epsilon_y + \lambda\epsilon_x
                 \tau_{xy} = \frac{E}{1+\nu} \epsilon_{xy}
                 \tau_{xy} = \frac{1}{2\mu} \epsilon_{xy}
                 */
                TPZFNMatrix<4,REAL> sigma_u(2,2);
                sigma_u(0,0) = (flambda+2.*fmu)*gradu(0,0)+flambda*gradu(1,1);
                sigma_u(1,1) = (flambda+2.*fmu)*gradu(1,1)+flambda*gradu(0,0);
                sigma_u(0,1) = fmu*(gradu(0,1)+gradu(1,0));
                sigma_u(1,0) = sigma_u(0,1);
                STATE energy = (sigma_u(0,0)*gradv(0,0)+sigma_u(1,1)*gradv(1,1)+
                                sigma_u(1,0)*gradv(1,0)+sigma_u(0,1)*gradv(0,1));
                ek(iu, ju)  += weight*energy;

            }
        }
    }
    //  ////////////////////////// Jacobian Matrix ///////////////////////////////////
    this->ContributeVec(data,weight,ef);

}
void TPZMatGriffith::ContributeVec(TPZMaterialData &data, REAL weight, TPZFMatrix<STATE> &ef)
{
    // Getting weight functions
    TPZFMatrix<REAL>  &phiU =  data.phi;
    int phrU = phiU.Rows()/2;
    TPZFMatrix<REAL> &dphiU     =  data.dphix;

    TPZFNMatrix<200,REAL> dudaxes(2,dphiU.Cols()), dvdaxes(2,dphiU.Cols()), dudx(2,phrU), dvdx(2,phrU);
    for (int i=0; i<2; i++) {
        for (int j=0; j<phrU; j++) {
            dudaxes(i,j) = dphiU(i,j);
            dvdaxes(i,j) = dphiU(2+i,j);
        }
    }
    TPZAxesTools<REAL>::Axes2XYZ(dudaxes, dudx, data.axes);
    TPZAxesTools<REAL>::Axes2XYZ(dvdaxes, dvdx, data.axes);

    TPZManVector<STATE,3> sol_u =data.sol[0];
    TPZFNMatrix<4,STATE> dsol_xy(2,2), dsol_u = data.dsol[0];

    TPZAxesTools<STATE>::Axes2XYZ(dsol_u, dsol_xy, data.axes);

    REAL LambdaL, MuL;

    LambdaL = flambda;
    MuL     = fmu;

    TPZManVector<STATE,3> P(ff);
    TPZFNMatrix<4,STATE> GradP(2,2,0.0);

    if(this->HasForcingFunction())
    {
        fForcingFunction->Execute(data.x,P);
        //        REAL Pressure = P[0];
    }

    //  ////////////////////////// Residual Vector ///////////////////////////////////
    //  Contribution of domain integrals for Residual Vector
    //  Elastic equation
    //  Linear strain operator
    //  Ke Matrix


    for(int iu = 0; iu < phrU; iu++ )
    {
        TPZFNMatrix<4,REAL> gradv(2,2);
        gradv(0,0) = dudx(0,iu);
        gradv(0,1) = dudx(1,iu);
        gradv(1,0) = dvdx(0,iu);
        gradv(1,1) = dvdx(1,iu);

        //          Vector Force right hand term
        ef(iu)     +=    weight*(P[0]*phiU(2*iu, 0) + P[1]*phiU(2*iu+1,0)
                                 -fPreStressXX*gradv(0,0)-fPreStressYY*gradv(1,1)-fPreStressXY*(gradv(0,1)+gradv(1,0)));

        if (fPlaneStress == 1)
        {
            /* Plain stress state */
            TPZFNMatrix<4,REAL> sigma_u(2,2);
            sigma_u(0,0) = fE/(1-fnu*fnu) *(dsol_xy(0,0)+fnu*dsol_xy(1,1));
            sigma_u(1,1) = fE/(1-fnu*fnu) *(dsol_xy(1,1)+fnu*dsol_xy(0,0));
            sigma_u(0,1) = fE/(2.*(1+fnu))*(dsol_xy(0,1)+dsol_xy(1,0));
            sigma_u(1,0) = sigma_u(0,1);

            ef(iu) -= weight*(gradv(0,0)*sigma_u(0,0)+gradv(1,1)*sigma_u(1,1)+gradv(1,0)*sigma_u(1,0)+gradv(0,1)*sigma_u(0,1));
        }
        else
        {
            /* Plain Strain State */
            TPZFNMatrix<4,REAL> sigma_u(2,2);
            sigma_u(0,0) = (flambda+2.*fmu)*dsol_xy(0,0)+flambda*dsol_xy(1,1);
            sigma_u(1,1) = (flambda+2.*fmu)*dsol_xy(1,1)+flambda*dsol_xy(0,0);
            sigma_u(0,1) = fmu*(dsol_xy(0,1)+dsol_xy(1,0));
            sigma_u(1,0) = sigma_u(0,1);
            ef(iu) -= weight*(gradv(0,0)*sigma_u(0,0)+gradv(1,1)*sigma_u(1,1)+gradv(1,0)*sigma_u(1,0)+gradv(0,1)*sigma_u(0,1));

        }
    }

    //  ////////////////////////// Residual Vector ///////////////////////////////////


}

void TPZMatGriffith::Contribute(TPZMaterialData &data, REAL weight, TPZFMatrix<STATE> &ef) {

    if (data.fShapeType == TPZMaterialData::EVecShape) {
        ContributeVec(data, weight, ef);
    }
    // Getting weight functions
    TPZFMatrix<REAL>  &phiU =  data.phi;
    int phrU = phiU.Rows();
    int FirstU  = 0;

    TPZFNMatrix<40,REAL> dphidx(3,phrU);
    TPZAxesTools<REAL>::Axes2XYZ(data.dphix, dphidx, data.axes);
    TPZManVector<STATE,3> sol_u =data.sol[0];
    TPZFMatrix<STATE> dsol_u = data.dsol[0];
    REAL LambdaL, MuL;

    LambdaL = flambda;
    MuL     = fmu;

    TPZManVector<STATE,3> P(ff);
    TPZFNMatrix<4,STATE> GradP(2,2,0.0);

    if(this->HasForcingFunction())
    {
        fForcingFunction->Execute(data.x,P);
//        REAL Pressure = P[0];
    }

    //  ////////////////////////// Residual Vector ///////////////////////////////////
    //  Contribution of domain integrals for Residual Vector
    //  Elastic equation
    //  Linear strain operator
    //  Ke Matrix

//    TPZFMatrix<REAL>    du(2,2);
    TPZFMatrix<REAL>    GradU(2,1);
    TPZFMatrix<REAL>    GradV(2,1);
    // Required check out of this implementation
    //  Derivative for Ux
    GradU(0,0) = dsol_u(0,0)*data.axes(0,0)+dsol_u(1,0)*data.axes(1,0); // dUx/dx
    GradU(1,0) = dsol_u(0,0)*data.axes(0,1)+dsol_u(1,0)*data.axes(1,1); // dUx/dy

    //  Derivative for Uy
    GradV(0,0) = dsol_u(0,1)*data.axes(0,0)+dsol_u(1,1)*data.axes(1,0); // dUy/dx
    GradV(1,0) = dsol_u(0,1)*data.axes(0,1)+dsol_u(1,1)*data.axes(1,1); // dUy/dy

    for(int iu = 0; iu < phrU; iu++ )
    {

//          Vector Force right hand term
        ef(2*iu + FirstU)     +=    weight*P[0]*phiU(iu, 0)- (dphidx(0,iu)*fPreStressXX + dphidx(1,iu)*fPreStressXY);    // direcao x
        ef(2*iu+1 + FirstU)   +=    weight*P[1]*phiU(iu, 0)- (dphidx(0,iu)*fPreStressXY + dphidx(1,iu)*fPreStressYY);    // direcao y

        if (fPlaneStress == 1)
        {
            /* Plain stress state */
            ef(2*iu + FirstU)           += weight*((4*(MuL)*(LambdaL+MuL)/(LambdaL+2*MuL))*dphidx(0,iu)*GradU(0,0)      + (2*MuL)*dphidx(1,iu)*GradU(1,0));

            ef(2*iu + FirstU)           += weight*((2*(MuL)*(LambdaL)/(LambdaL+2*MuL))*dphidx(0,iu)*GradV(1,0)         + (2*MuL)*dphidx(1,iu)*GradV(0,0));

            ef(2*iu+1 + FirstU)         += weight*((2*(MuL)*(LambdaL)/(LambdaL+2*MuL))*dphidx(1,iu)*GradU(0,0)         + (2*MuL)*dphidx(0,iu)*GradU(1,0));

            ef(2*iu+1 + FirstU)         += weight*((4*(MuL)*(LambdaL+MuL)/(LambdaL+2*MuL))*dphidx(1,iu)*GradV(1,0)     + (2*MuL)*dphidx(0,iu)*GradV(0,0));
        }
        else
        {
            /* Plain Strain State */
            ef(2*iu + FirstU)           += weight*  ((LambdaL + 2*MuL)*dphidx(0,iu)*GradU(0,0)  + (2*MuL)*dphidx(1,iu)*(GradU(1,0)));

            ef(2*iu + FirstU)           += weight*  (LambdaL*dphidx(0,iu)*GradV(1,0)            + (2*MuL)*dphidx(1,iu)*(GradV(0,0)));

            ef(2*iu+1 + FirstU)         += weight*  (LambdaL*dphidx(1,iu)*GradU(0,0)            + (2*MuL)*dphidx(0,iu)*(GradU(1,0)));

            ef(2*iu+1 + FirstU)         += weight*  ((LambdaL + 2*MuL)*dphidx(1,iu)*GradV(1,0)  + (2*MuL)*dphidx(0,iu)*(GradV(0,0)));
        }
    }

    //  ////////////////////////// Residual Vector ///////////////////////////////////

}

/// compute the stress tensor as a function of the solution gradient
void TPZMatGriffith::ComputeSigma(const TPZFMatrix<STATE> &dudx, TPZFMatrix<STATE> &sigma)
{
#ifdef PZDEBUG
    if (dudx.Rows() < 2 || dudx.Cols() != 2 || sigma.Rows() != 2 || sigma.Cols() != 2) {
        DebugStop();
    }
#endif

    if (fPlaneStress == 1)
    {
        sigma(0,0) = (4*(fmu)*(flambda+fmu)/(flambda+2*fmu))*dudx.g(0,0)+(2*(fmu)*(flambda)/(flambda+2*fmu))*dudx.g(1,1);
        sigma(0,1) = (fmu)*(dudx.g(1,0)+dudx.g(0,1));
        sigma(1,0) = sigma(0,1);
        sigma(1,1) = (2*(fmu)*(flambda)/(flambda+2*fmu))*dudx.g(0,0)+(4*(fmu)*(flambda+fmu)/(flambda+2*fmu))*dudx.g(1,1);
        /* Plain stress state */
    }
    else
    {
        sigma(0,0) = (flambda + 2*fmu)*dudx.g(0,0)+flambda*dudx.g(1,1);
        sigma(1,0) = fmu*(dudx.g(1,0)+dudx.g(0,1));
        sigma(0,1) = sigma(1,0);
        sigma(1,1) = (flambda + 2*fmu)*dudx.g(1,1)+flambda*dudx.g(0,0);
        /* Plain Strain State */
    }
    sigma.Print(std::cout);

}



void TPZMatGriffith::ContributeBC(TPZMaterialData &data,REAL weight, TPZFMatrix<STATE> &ek,TPZFMatrix<STATE> &ef,TPZBndCond &bc)
{
    TPZFMatrix<REAL>  &phiu = data.phi;
    TPZManVector<STATE,3> sol_u = data.sol[0];
    TPZFMatrix<STATE> dsol_u = data.dsol[0];

    REAL ux = sol_u[0];
    REAL uy = sol_u[1];

    TPZFNMatrix<4,STATE> val1loc(bc.Val1()),val2loc(bc.Val2());

    if (bc.HasForcingFunction()) {
        TPZManVector<STATE,2> val2vec(2);

        bc.ForcingFunction()->Execute(data.x, val2vec, val1loc);
        val2loc(0,0) = val2vec[0];
        val2loc(1,0) = val2vec[1];
        // we assume type 2 is displacement value is weakly imposed
        if(bc.Type() == 2)
        {
            val1loc = bc.Val1();
            for (int i=0; i<2; i++) {
                val2loc(i,0) = 0.;
                for (int j=0; j<2; j++) {
                    val2loc(i,0) += val1loc(i,j)*val2vec[j];
                }
            }
        }
        if(bc.Type() == 1)
        {
            for (int i=0; i<2; i++) {
                val2loc(i,0) = 0.;
                for (int j=0; j<2; j++) {
                    val2loc(i,0) += val1loc(i,j)*data.normal[j];
                }
            }
        }
    }
    int phru = phiu.Rows();
    short in,jn;
    STATE v2[3];
    TPZFMatrix<STATE> &v1 = val1loc;
    v2[0] = val2loc(0,0);	//	Ux displacement or Tnx
    v2[1] = val2loc(1,0);	//	Uy displacement or Tny

    //	Here each digit represent an individual boundary condition corresponding to each state variable.
    //	0 means Dirichlet condition on x-y
    //	1 means Neumann condition
    //	7 means Dirichlet condition on x
    //	8 means Dirichlet condition on y

    const REAL BIGNUMBER  = TPZMaterial::gBigNumber;
    switch (bc.Type())
    {
        case 0 :
        {
            //	Dirichlet condition for each state variable
            //	Elasticity Equation
            for(in = 0 ; in < phru; in++)
            {
                //	Contribution for load Vector
                ef(2*in,0)      += BIGNUMBER*(v2[0] - ux)*phiu(in,0)*weight;	// X displacement Value
                ef(2*in+1,0)	+= BIGNUMBER*(v2[1] - uy)*phiu(in,0)*weight;	// y displacement Value

                for (jn = 0 ; jn < phru; jn++)
                {
                    //	Contribution for Stiffness Matrix
                    ek(2*in,2*jn)       += BIGNUMBER*phiu(in,0)*phiu(jn,0)*weight;	// X displacement
                    ek(2*in+1,2*jn+1)	+= BIGNUMBER*phiu(in,0)*phiu(jn,0)*weight;	// Y displacement
                }
            }

            break;
        }
        case 1 :
        {
            //	Neumann condition for each state variable
            //	Elasticity Equation
            for(in = 0 ; in <phru; in++)
            {
                //	Normal Tension Components on neumann boundary
                ef(2*in,0)      += v2[0]*phiu(in,0)*weight;		//	Tnx
                ef(2*in+1,0)	+= v2[1]*phiu(in,0)*weight;		//	Tny
            }
            break;
        }
        case 2 :
        {
            //	Mixed condition for each state variable no used here
            //	Elasticity Equation
            TPZFNMatrix<2,STATE> res(2,1,0.);
            for(int i=0; i<2; i++) for(int j=0; j<2; j++)
                {
                    res(i,0) += v1(i,j)*data.sol[0][j];
                }

            for(in = 0 ; in < phru; in++)
            {
                ef(2*in+0,0) += weight * (v2[0]-res(0,0)) * phiu(in,0);
                ef(2*in+1,0) += weight * (v2[1]-res(1,0)) * phiu(in,0);

                for (jn = 0 ; jn < phru; jn++)
                {
                    for(int idf=0; idf < this->Dimension(); idf++) for(int jdf=0; jdf< this->Dimension(); jdf++)
                        {
                            ek(2*in+idf,2*jn+jdf) += v1(idf,jdf)*phiu(in,0)*phiu(jn,0)*weight;
                            //      Not Complete with val2? HERE! PHIL!!!!
                            //      DebugStop();
                        }
                }
            }

            break;
        }
        case 3 :
        {
            //	Null Dirichlet condition for each state variable
            //	Elasticity Equation
            for(in = 0 ; in < phru; in++)
            {
                //	Contribution for load Vector
                ef(2*in,0)      += BIGNUMBER*( v2[0])*phiu(in,0)*weight;	// X displacement Value
                ef(2*in+1,0)	+= BIGNUMBER*( v2[1])*phiu(in,0)*weight;	// y displacement Value

                for (jn = 0 ; jn < phru; jn++)
                {
                    //	Contribution for Stiffness Matrix
                    ek(2*in,2*jn)       += BIGNUMBER*phiu(in,0)*phiu(jn,0)*weight;	// X displacement
                    ek(2*in+1,2*jn+1)	+= BIGNUMBER*phiu(in,0)*phiu(jn,0)*weight;	// Y displacement
                }
            }

            break;
        }
        case 4 :
        {
            //	Stress Field as Neumann condition for each state variable
            //	Elasticity Equation

            for(in = 0; in < this->Dimension(); in ++){
                v2[in] = ( v1(in,0) * data.normal[0] + v1(in,1) * data.normal[1]);
            }

            for(in = 0 ; in <phru; in++)
            {
                //	Normal Tension Components on neumann boundary
                ef(2*in,0)      += 1.0*v2[0]*phiu(in,0)*weight;        //	Tnx
                ef(2*in+1,0)	+= 1.0*v2[1]*phiu(in,0)*weight;		//	Tny
            }

            break;
        }
        case 5 :
            //	Normal Pressure condition Pressure value Should be inserted in v2[0]
            //	Elasticity Equation
        {
            TPZFNMatrix<2,STATE> res(2,1,0.);
            for(int i=0; i<2; i++) for(int j=0; j<2; j++)
                {
                    res(i,0) += data.normal[i]*bc.Val1()(i,j)*data.sol[0][j]*data.normal[j];
                }
            for(int in = 0 ; in < phru; in++)
            {
                ef(2*in+0,0) += (v2[0]*data.normal[0]-res(0,0)) * phiu(in,0) * weight ;
                ef(2*in+1,0) += (v2[0]*data.normal[1]-res(1,0)) * phiu(in,0) * weight ;
                for(int jn=0; jn< phru; jn++)
                {
                    for(int idf=0; idf < this->Dimension(); idf++) for(int jdf=0; jdf < this->Dimension(); jdf++)
                        {
                            ek(2*in+idf,2*jn+jdf) += v1(idf,jdf)*data.normal[idf]*data.normal[jdf]*phiu(in,0)*phiu(jn,0)*weight;
                            //      Not Complete with val2? HERE! PHIL!!!!
                            //      DebugStop();
                        }
                }
            }
        }
            break;
        case 6 :
            //	Normal Pressure condition Pressure value Should be inserted in v2[0]
            //	Elasticity Equation
        {
            TPZFNMatrix<2,STATE> res(2,1,0.);
            for(int i=0; i<2; i++) for(int j=0; j<2; j++)
                {
                    res(i,0) += data.normal[i]*bc.Val1()(i,j)*data.sol[0][j]*data.normal[j];
                }
            for(int in = 0 ; in < phru; in++)
            {
                ef(2*in+0,0) += (v2[0]*data.normal[0]-res(0,0)) * phiu(in,0) * weight ;
                ef(2*in+1,0) += (v2[0]*data.normal[1]-res(1,0)) * phiu(in,0) * weight ;
                for(int jn=0; jn< phru; jn++)
                {
                    for(int idf=0; idf < this->Dimension(); idf++) for(int jdf=0; jdf < this->Dimension(); jdf++)
                        {
                            ek(2*in+idf,2*jn+jdf) += v1(idf,jdf)*data.normal[idf]*data.normal[jdf]*phiu(in,0)*phiu(jn,0)*weight;
                            //      Not Complete
                            //      DebugStop();
                        }
                }
            }
        }
            break;
        case 7 :
        {
            //	Dirichlet condition for each state variable
            //	Elasticity Equation
            for(in = 0 ; in < phru; in++)
            {
                //	Contribution for load Vector
                ef(2*in,0)		+= -BIGNUMBER*(ux - v2[0])*phiu(in,0)*weight;	// X displacement Value

                for (jn = 0 ; jn < phru; jn++)
                {
                    //	Contribution for Stiffness Matrix
                    ek(2*in,2*jn)		+= BIGNUMBER*phiu(in,0)*phiu(jn,0)*weight;	// X displacement
                }
            }

            break;
        }
        case 8 :
        {
            //	Dirichlet condition for uy
            //	Elasticity Equation
            for(in = 0 ; in < phru; in++)
            {
                //	Contribution for load Vector
                ef(2*in+1,0)	+= -BIGNUMBER*(uy - v2[1])*phiu(in,0)*weight;	// y displacement Value

                for (jn = 0 ; jn < phru; jn++)
                {
                    //	Contribution for Stiffness Matrix
                    ek(2*in+1,2*jn+1)	+= BIGNUMBER*phiu(in,0)*phiu(jn,0)*weight;	// Y displacement
                }
            }

            break;
        }
        default:
        {
            PZError << "TPZMatGriffith::ContributeBC error - Wrong boundary condition type" << std::endl;
            DebugStop();
        }
            break;
    }

}


/*
void TPZMatGriffith::ContributeBC(TPZMaterialData &data,REAL weight,TPZFMatrix<REAL> &ek,TPZFMatrix<REAL> &ef,TPZBndCond &bc)
{
    TPZFMatrix<REAL> &phi = data.phi;
    const REAL BIGNUMBER  = TPZMaterial::gBigNumber;
    int dim = Dimension();
    int nstate = NStateVariables();

    const int phr = phi.Rows();
    int in,jn,idf,jdf;
    REAL v2[2];
    v2[0] = bc.Val2()(0,0);
    v2[1] = bc.Val2()(1,0);

    if (this->fForcingFunction) {

    }

    TPZFMatrix<REAL> &v1 = bc.Val1();
    switch (bc.Type()){
        case 0: // Dirichlet condition
            for(in = 0 ; in < phr; in++){
                ef(nstate*in+0,0) += BIGNUMBER * (v2[0] - data.sol[0][0]) * phi(in,0) * weight;
                ef(nstate*in+1,0) += BIGNUMBER * (v2[1] - data.sol[0][1]) * phi(in,0) * weight;

                for (jn = 0 ; jn < phr; jn++) {
                    ek(nstate*in+0,nstate*jn+0) += BIGNUMBER * phi(in,0) * phi(jn,0) * weight;
                    ek(nstate*in+1,nstate*jn+1) += BIGNUMBER * phi(in,0) * phi(jn,0) * weight;

                }//jn
            }//in
            break;

        case 1: // Neumann condition
            for(in = 0 ; in < phi.Rows(); in++) {
                ef(nstate*in+0,0) += v2[0] * phi(in,0) * weight;
                ef(nstate*in+1,0) += v2[1] * phi(in,0) * weight;
            }
            break;

        case 2: // Mixed condition
        {
            TPZFNMatrix<2,STATE> res(2,1,0.);
            for(int i=0; i<2; i++) for(int j=0; j<2; j++)
            {
                res(i,0) += bc.Val1()(i,j)*data.sol[0][j];
            }

            for(in = 0 ; in < phi.Rows(); in++) {
                ef(nstate*in+0,0) += (v2[0]-res(0,0)) * phi(in,0) * weight;
                ef(nstate*in+1,0) += (v2[1]-res(1,0)) * phi(in,0) * weight;
                for(jn=0; jn<phi.Rows(); jn++)
                {
                    for(idf=0; idf<2; idf++) for(jdf=0; jdf<2; jdf++)
                    {
                        ek(nstate*in+idf,nstate*jn+jdf) += bc.Val1()(idf,jdf)*phi(in,0)*phi(jn,0)*weight;
                        //BUG FALTA COLOCAR VAL2
                        //DebugStop();
                    }
                }
            }//in
        }
            break;

        case 3: // Directional Null Dirichlet - displacement is set to null in the non-null vector component direction
            for(in = 0 ; in < phr; in++) {
                ef(nstate*in+0,0) += BIGNUMBER * (0. - data.sol[0][0]) * v2[0] * phi(in,0) * weight;
                ef(nstate*in+1,0) += BIGNUMBER * (0. - data.sol[0][1]) * v2[1] * phi(in,0) * weight;
                for (jn = 0 ; jn < phr; jn++) {
                    ek(nstate*in+0,nstate*jn+0) += BIGNUMBER * phi(in,0) * phi(jn,0) * weight * v2[0];
                    ek(nstate*in+1,nstate*jn+1) += BIGNUMBER * phi(in,0) * phi(jn,0) * weight * v2[1];
                }//jn
            }//in
            break;

        case 4: // stressField Neumann condition
            for(in = 0; in < dim; in ++)
                v2[in] = ( v1(in,0) * data.normal[0] +
                          v1(in,1) * data.normal[1]);
            // The normal vector points towards the neighbour. The negative sign is there to
            // reflect the outward normal vector.
            for(in = 0 ; in < phi.Rows(); in++) {
                ef(nstate*in+0,0) += v2[0] * phi(in,0) * weight;
                ef(nstate*in+1,0) += v2[1] * phi(in,0) * weight;
                //	cout << "normal:" << data.normal[0] << ' ' << data.normal[1] << endl;
                //	cout << "val2:  " << v2[0]  << endl;
            }
            break;

        case 6://PRESSAO DEVE SER POSTA NA POSICAO 0 DO VETOR v2
        {
            TPZFNMatrix<2,STATE> res(2,1,0.);
            for(int i=0; i<2; i++) for(int j=0; j<2; j++)
            {
                res(i,0) += bc.Val1()(i,j)*data.sol[0][j];
            }
            for(in = 0 ; in < phi.Rows(); in++)
            {
                ef(nstate*in+0,0) += (v2[0]*data.normal[0]-res(0,0)) * phi(in,0) * weight ;
                ef(nstate*in+1,0) += (v2[0]*data.normal[1]-res(1,0)) * phi(in,0) * weight ;
                for(jn=0; jn<phi.Rows(); jn++)
                {
                    for(idf=0; idf<2; idf++) for(jdf=0; jdf<2; jdf++)
                    {
                        ek(nstate*in+idf,nstate*jn+jdf) += bc.Val1()(idf,jdf)*phi(in,0)*phi(jn,0)*weight;
                        //BUG FALTA COLOCAR VAL2
                        //                        DebugStop();
                    }
                }

            }

        }
            break;
        case 5://PRESSAO DEVE SER POSTA NA POSICAO 0 DO VETOR v2
        {
            TPZFNMatrix<2,STATE> res(2,1,0.);
            for(int i=0; i<2; i++) for(int j=0; j<2; j++)
            {
                res(i,0) += data.normal[i]*bc.Val1()(i,j)*data.sol[0][j]*data.normal[j];
            }
            for(in = 0 ; in < phi.Rows(); in++)
            {
                ef(nstate*in+0,0) += (v2[0]*data.normal[0]-res(0,0)) * phi(in,0) * weight ;
                ef(nstate*in+1,0) += (v2[0]*data.normal[1]-res(1,0)) * phi(in,0) * weight ;
                for(jn=0; jn<phi.Rows(); jn++)
                {
                    for(idf=0; idf<2; idf++) for(jdf=0; jdf<2; jdf++)
                    {
                        ek(nstate*in+idf,nstate*jn+jdf) += bc.Val1()(idf,jdf)*data.normal[idf]*data.normal[jdf]*phi(in,0)*phi(jn,0)*weight;
                        //BUG FALTA COLOCAR VAL2
                        //                        DebugStop();
                    }
                }

            }
        }
            break;

        default:
        PZError << "TPZMatElastoPlastic2D::ContributeBC error - Wrong boundary condition type" << std::endl;
    }
    //cout << "normal:" << data.normal[0] << ' ' << data.normal[1] << ' ' << data.normal[2] << endl;
    //cout << "val2:  " << v2[0] << endl;
}
*/


void TPZMatGriffith::ContributeBC(TPZMaterialData &data,REAL weight,TPZFMatrix<STATE> &ef,TPZBndCond &bc)
{



    TPZFMatrix<REAL>  &phiu = data.phi;
    TPZManVector<STATE,3> sol_u = data.sol[0];
    TPZFMatrix<STATE> dsol_u = data.dsol[0];

    REAL ux = sol_u[0];
    REAL uy = sol_u[1];

    int phru = phiu.Rows();
    short in;
    STATE v2[3]; TPZFMatrix<STATE> &v1 = bc.Val1();
    v2[0] = bc.Val2()(0,0);	//	Ux displacement or Tnx
    v2[1] = bc.Val2()(1,0);	//	Uy displacement or Tny

    //	Here each digit represent an individual boundary condition corresponding to each state variable.
    //	0 means Dirichlet condition on x-y
    //	1 means Neumann condition
    //	7 means Dirichlet condition on x
    //	8 means Dirichlet condition on y

    const REAL BIGNUMBER  = TPZMaterial::gBigNumber;
    switch (bc.Type())
    {
        case 0 :
        {
            //	Dirichlet condition for each state variable
            //	Elasticity Equation
            for(in = 0 ; in < phru; in++)
            {
                //	Contribution for load Vector
                ef(2*in,0)      += BIGNUMBER*(ux - v2[0])*phiu(in,0)*weight;	// X displacement Value
                ef(2*in+1,0)	+= BIGNUMBER*(uy - v2[1])*phiu(in,0)*weight;	// y displacement Value

            }

            break;
        }
        case 1 :
        {
            //	Neumann condition for each state variable
            //	Elasticity Equation
            for(in = 0 ; in <phru; in++)
            {
                //	Normal Tension Components on neumann boundary
                ef(2*in,0)      += -1.0*v2[0]*phiu(in,0)*weight;		//	Tnx
                ef(2*in+1,0)	+= -1.0*v2[1]*phiu(in,0)*weight;		//	Tny
            }
            break;
        }
        case 2 :
        {
            //	Mixed condition for each state variable no used here
            //	Elasticity Equation
            TPZFNMatrix<2,STATE> res(2,1,0.);
            for(int i=0; i<2; i++) for(int j=0; j<2; j++)
                {
                    res(i,0) += bc.Val1()(i,j)*data.sol[0][j];
                }

            for(in = 0 ; in < phru; in++)
            {
                ef(2*in+0,0) += weight * (v2[0]-res(0,0)) * phiu(in,0);
                ef(2*in+1,0) += weight * (v2[1]-res(1,0)) * phiu(in,0);

            }

            break;
        }
        case 3 :
        {
            //	Null Dirichlet condition for each state variable
            //	Elasticity Equation
            for(in = 0 ; in < phru; in++)
            {
                //	Contribution for load Vector
                ef(2*in,0)      += BIGNUMBER*(0.0 - v2[0])*phiu(in,0)*weight;	// X displacement Value
                ef(2*in+1,0)	+= BIGNUMBER*(0.0 - v2[1])*phiu(in,0)*weight;	// y displacement Value

            }

            break;
        }
        case 4 :
        {
            //	Stress Field as Neumann condition for each state variable
            //	Elasticity Equation

            for(in = 0; in < this->Dimension(); in ++){ v2[in] = ( v1(in,0) * data.normal[0] + v1(in,1) * data.normal[1]);}

            for(in = 0 ; in <phru; in++)
            {
                //	Normal Tension Components on neumann boundary
                ef(2*in,0)      += -1.0*v2[0]*phiu(in,0)*weight;        //	Tnx
                ef(2*in+1,0)	+= -1.0*v2[1]*phiu(in,0)*weight;		//	Tny
            }

            break;
        }
        case 5 :
            //	Normal Pressure condition Pressure value Should be inserted in v2[0]
            //	Elasticity Equation
        {
            TPZFNMatrix<2,STATE> res(2,1,0.);
            for(int i=0; i<2; i++) for(int j=0; j<2; j++)
                {
                    res(i,0) += data.normal[i]*bc.Val1()(i,j)*data.sol[0][j]*data.normal[j];
                }
            for(int in = 0 ; in < phru; in++)
            {
                ef(2*in+0,0) += (v2[0]*data.normal[0]-res(0,0)) * phiu(in,0) * weight ;
                ef(2*in+1,0) += (v2[0]*data.normal[1]-res(1,0)) * phiu(in,0) * weight ;
            }
        }
            break;
        case 6 :
            //	Normal Pressure condition Pressure value Should be inserted in v2[0]
            //	Elasticity Equation
        {
            TPZFNMatrix<2,STATE> res(2,1,0.);
            for(int i=0; i<2; i++) for(int j=0; j<2; j++)
                {
                    res(i,0) += data.normal[i]*bc.Val1()(i,j)*data.sol[0][j]*data.normal[j];
                }
            for(int in = 0 ; in < phru; in++)
            {
                ef(2*in+0,0) += (v2[0]*data.normal[0]-res(0,0)) * phiu(in,0) * weight ;
                ef(2*in+1,0) += (v2[0]*data.normal[1]-res(1,0)) * phiu(in,0) * weight ;
            }
        }
            break;
        case 7 :
        {
            //	Dirichlet condition for each state variable
            //	Elasticity Equation
            for(in = 0 ; in < phru; in++)
            {
                //	Contribution for load Vector
                ef(2*in,0)		+= BIGNUMBER*(ux - v2[0])*phiu(in,0)*weight;	// X displacement Value
            }

            break;
        }
        case 8 :
        {
            //	Dirichlet condition for each state variable
            //	Elasticity Equation
            for(in = 0 ; in < phru; in++)
            {
                //	Contribution for load Vector
                ef(2*in+1,0)	+= BIGNUMBER*(uy - v2[1])*phiu(in,0)*weight;	// y displacement Value
            }

            break;
        }
        default:
        {
            PZError << "TPZMatGriffith::ContributeBC error - Wrong boundary condition type" << std::endl;
            DebugStop();
        }
            break;
    }

}


void TPZMatGriffith::FillDataRequirements(TPZMaterialData &data)
{
    data.SetAllRequirements(false);
    data.fNeedsSol = true;
    data.fNeedsNeighborSol = true;
    data.fNeedsNeighborCenter = false;
    data.fNeedsNormal = true;
}

void TPZMatGriffith::FillBoundaryConditionDataRequirement(int type, TPZMaterialData &data){
    data.SetAllRequirements(false);
    data.fNeedsSol = true;
    data.fNeedsNormal = true;
}


void TPZMatGriffith::Print(std::ostream &out)
{
    out << "Material Name : " << Name() << "\n";
    out << "Plane Problem (fPlaneStress = 0, for Plane Strain conditions) " << fPlaneStress << std::endl;
    out << "Properties for elasticity: \n";
    out << "\t Young modulus   = "											<< fE		<< std::endl;
    out << "\t Poisson Ratio   = "											<< fnu		<< std::endl;
    out << "\t First Lamé Parameter   = "									<< flambda	<< std::endl;
    out << "\t Second Lamé Parameter   = "									<< fmu		<< std::endl;
    out << "\t Body force vector B {X-direction, Y-direction}   = "			<< ff[0] << ' ' << ff[1]   << std::endl;
    out << "\t fPreStressXX   = "			<< fPreStressXX << std::endl;
    out << "\t fPreStressXY   = "			<< fPreStressXY << std::endl;
    out << "\t fPreStressYY   = "			<< fPreStressYY << std::endl;
    out << "\t fPreStressZZ   = "			<< fPreStressZZ << std::endl;
    out << "Class properties :";
    TPZMaterial::Print(out);
    out << "\n";

}

/** Returns the variable index associated with the name */
int TPZMatGriffith::VariableIndex(const std::string &name)
{
    //	Elasticity Variables
    if(!strcmp("Displacement",name.c_str()))				return	1;
    if(!strcmp("SolidPressure",name.c_str()))				return	2;
    if(!strcmp("SigmaX",name.c_str()))						return	3;
    if(!strcmp("SigmaY",name.c_str()))						return	4;
    if(!strcmp("SigmaZ",name.c_str()))						return	5;
    if(!strcmp("TauXY",name.c_str()))						return	6;
    if(!strcmp("EpsX",name.c_str()))                        return    7;
    if(!strcmp("EpsY",name.c_str()))                        return    8;
    if(!strcmp("EpsZ",name.c_str()))                        return    9;
    if(!strcmp("EpsXY",name.c_str()))                        return    10;
//    PZError << "TPZMatGriffith::VariableIndex Error\n";

    return TPZMaterial::VariableIndex(name);
}

/**
 * Save the element data to a stream
 */
void TPZMatGriffith::Write(TPZStream &buf, int withclassid) const
{
    TPZMaterial::Write(buf,withclassid);
    buf.Write(&fE);
    buf.Write(&fnu);
    buf.Write(&flambda);
    buf.Write(&fmu);
    buf.Write( ff);
    buf.Write(&fPreStressXX);
    buf.Write(&fPreStressXY);
    buf.Write(&fPreStressYY);
    buf.Write(&fPreStressZZ);
    buf.Write(&fPlaneStress);

}

/**
 * Read the element data from a stream
 */
void TPZMatGriffith::Read(TPZStream &buf, void *context)
{
    TPZMaterial::Read(buf,context);
    buf.Read(&fE);
    buf.Read(&fnu);
    buf.Read(&flambda);
    buf.Read(&fmu);
    buf.Read( ff);
    buf.Read(&fPreStressXX);
    buf.Read(&fPreStressXY);
    buf.Read(&fPreStressYY);
    buf.Read(&fPreStressZZ);
    buf.Read(&fPlaneStress);

}

int TPZMatGriffith::NSolutionVariables(int var){
    if(var == 1)	return 3;
    if(var == 2)	return 1;
    if(var == 3)	return 1;
    if(var == 4)	return 1;
    if(var == 5)	return 1;
    if(var == 6)	return 1;
    if(var == 7)    return 1;
    if(var == 8)    return 1;
    if(var == 9)    return 1;
    if(var == 10)    return 1;

    return TPZMaterial::NSolutionVariables(var);
}

//	Calculate Secondary variables based on ux, uy, Pore pressure and their derivatives
void TPZMatGriffith::Solution(TPZMaterialData &data, int var, TPZVec<STATE> &Solout){

    Solout.Resize(this->NSolutionVariables(var));

    TPZManVector<STATE,3> SolU, SolP;
    TPZFNMatrix <6,STATE> DSolU, DSolP;
    TPZFNMatrix <9> axesU, axesP;

    TPZVec<REAL> ptx(3);
    TPZVec<STATE> solExata(3);
    TPZFMatrix<STATE> flux(5,1);

    if (data.sol.size() != 1) {
        DebugStop();
    }

    SolU	=	data.sol[0];
    DSolU	=	data.dsol[0];
    axesU	=	data.axes;


    //	Displacements
    if(var == 1 || var == 0){
        Solout[0] = SolU[0];
        Solout[1] = SolU[1];
        if(var==1) Solout[2] = 0.0;
        return;
    }


    REAL epsx;
    REAL epsy;
    REAL epsxy;
    REAL SigX;
    REAL SigY;
    REAL SigZ;
    REAL Tau, DSolxy[2][2];
    REAL divu;

    DSolxy[0][0] = DSolU(0,0)*axesU(0,0)+DSolU(1,0)*axesU(1,0); // dUx/dx
    DSolxy[1][0] = DSolU(0,0)*axesU(0,1)+DSolU(1,0)*axesU(1,1); // dUx/dy

    DSolxy[0][1] = DSolU(0,1)*axesU(0,0)+DSolU(1,1)*axesU(1,0); // dUy/dx
    DSolxy[1][1] = DSolU(0,1)*axesU(0,1)+DSolU(1,1)*axesU(1,1); // dUy/dy

    divu = DSolxy[0][0]+DSolxy[1][1]+0.0;

    epsx = DSolxy[0][0];// du/dx
    epsy = DSolxy[1][1];// dv/dy
    epsxy = 0.5*(DSolxy[1][0]+DSolxy[0][1]);
    REAL C11 = 4*(fmu)*(flambda+fmu)/(flambda+2*fmu);
    REAL C22 = 2*(fmu)*(flambda)/(flambda+2*fmu);

    if (this->fPlaneStress)
    {
        SigX = C11*epsx+C22*epsy;
        SigY = C11*epsy+C22*epsx;
        SigZ = 0.0;
        Tau = 2.0*fmu*epsxy;
    }
    else
    {
        SigX = ((flambda + 2*fmu)*(epsx) + (flambda)*epsy);
        SigY = ((flambda + 2*fmu)*(epsy) + (flambda)*epsx);
        SigZ = flambda*divu;
        Tau = 2.0*fmu*epsxy;
    }


    //	Hydrostatic stress
    if(var == 2)
    {
        Solout[0] = SigX+SigY+SigZ;
        return;
    }

    //	Effective Stress x-direction
    if(var == 3) {
        Solout[0] = SigX + fPreStressXX;
        return;
    }

    //	Effective Stress y-direction
    if(var == 4) {
        Solout[0] = SigY + fPreStressYY;
        return;
    }

    //	Effective Stress y-direction
    if(var == 5) {
        Solout[0] = SigZ + fPreStressZZ;
        return;
    }

    //	Shear Stress
    if(var == 6) {
        Solout[0] = Tau + fPreStressXY;
        return;
    }

    // epsx
    if (var == 7) {
        Solout[0] = epsx;
    }

    // epsy
    if (var == 8) {
        Solout[0] = epsy;
    }

    // epsz
    if (var == 9) {
        if (fPlaneStress) {
            Solout[0] = -fnu*(epsx+epsy);
        }
        else
        {
            Solout[0] = 0.;
        }
    }

    // epsxy
    if (var == 10) {
        Solout[0] = epsxy;
    }
}

int TPZMatGriffith::ClassId() const{
    return Hash("TPZMatGriffith") ^ TPZMaterial::ClassId() << 1;
}

/**
 * @brief Computes the error due to the difference between the interpolated flux \n
 * and the flux computed based on the derivative of the solution
 */
void TPZMatGriffith::Errors(TPZVec<REAL> &x, TPZVec<STATE> &sol, TPZFMatrix<STATE> &dsol,
                                TPZFMatrix<REAL> &axes, TPZVec<STATE> &flux,
                                TPZVec<STATE> &uexact, TPZFMatrix<STATE> &duexact,
                                TPZVec<REAL> &val)
{
    TPZFNMatrix<9,STATE> dudx(2,2), stress(2,2), stressexact(2,2);
    TPZAxesTools<STATE>::Axes2XYZ(dsol, dudx, axes);
    ComputeSigma(dudx, stress);
    ComputeSigma(duexact, stressexact);
    REAL L2 = 0.;
    L2 = (sol[0]-uexact[0])*(sol[0]-uexact[0])+(sol[1]-uexact[1])*(sol[1]-uexact[1]);
    REAL H1 = 0.;
    REAL energy = 0.;
    for (int i=0; i<2; i++) {
        for (int j=0; j<2; j++) {
            H1 += (dudx(i,j)-duexact(i,j))*(dudx(i,j)-duexact(i,j));
            energy += (stress(i,j)-stressexact(i,j))*(dudx(i,j)-duexact(i,j));
        }
    }
    val[0] = energy;
    val[1] = L2;
    val[2] = H1;
}
