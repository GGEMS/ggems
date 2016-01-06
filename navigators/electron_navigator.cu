// GGEMS Copyright (C) 2015

/*!
 * \file electron_navigator.cu
 * \brief
 * \author Yannick Lemaréchal <yannick.lemarechal@univ-brest.fr>
 * \version 0.1
 * \date 20 novembre 2015
 *
 *
 *
 */

#ifndef ELECTRON_NAVIGATOR_CU
#define ELECTRON_NAVIGATOR_CU

#include "electron.cuh"
#include "electron_navigator.cuh"



#ifdef __CUDA_ARCH__
// Constant values for eSampleSecondariesGamma
__constant__ f32
ah10= 4.67733E+00,ah11=-6.19012E-01,ah12= 2.02225E-02,
ah20=-7.34101E+00,ah21= 1.00462E+00,ah22=-3.20985E-02,
ah30= 2.93119E+00,ah31=-4.03761E-01,ah32= 1.25153E-02;
__constant__ f32
bh10= 4.23071E+00,bh11=-6.10995E-01,bh12= 1.95531E-02,
bh20=-7.12527E+00,bh21= 9.69160E-01,bh22=-2.74255E-02,
bh30= 2.69925E+00,bh31=-3.63283E-01,bh32= 9.55316E-03;
__constant__ f32
al00=-2.05398E+00,al01= 2.38815E-02,al02= 5.25483E-04,
al10=-7.69748E-02,al11=-6.91499E-02,al12= 2.22453E-03,
al20= 4.06463E-02,al21=-1.01281E-02,al22= 3.40919E-04;
__constant__ f32
bl00= 1.04133E+00,bl01=-9.43291E-03,bl02=-4.54758E-04,
bl10= 1.19253E-01,bl11= 4.07467E-02,bl12=-1.30718E-03,
bl20=-1.59391E-02,bl21= 7.27752E-03,bl22=-1.94405E-04;

// Constant parameters for bremstrahlung table
__constant__ f32  ZZ[8]= {2.,4.,6.,14.,26.,50.,82.,92.};

__constant__ f32  coefsig[8][11]= {{.4638,.37748,.32249,-.060362,-.065004,-.033457,-.004583,.011954,.0030404,-.0010077,-.00028131},
    {.50008,.33483,.34364,-.086262,-.055361,-.028168,-.0056172,.011129,.0027528,-.00092265,-.00024348},
    {.51587,.31095,.34996,-.11623,-.056167,-.0087154,.00053943,.0054092,.00077685,-.00039635,-6.7818e-05},
    {.55058,.25629,.35854,-.080656,-.054308,-.049933,-.00064246,.016597,.0021789,-.001327,-.00025983},
    {.5791,.26152,.38953,-.17104,-.099172,.024596,.023718,-.0039205,-.0036658,.00041749,.00023408},
    {.62085,.27045,.39073,-.37916,-.18878,.23905,.095028,-.068744,-.023809,.0062408,.0020407},
    {.66053,.24513,.35404,-.47275,-.22837,.35647,.13203,-.1049,-.034851,.0095046,.0030535},
    {.67143,.23079,.32256,-.46248,-.20013,.3506,.11779,-.1024,-.032013,.0092279,.0028592}
};

#else

f32
ah10= 4.67733E+00,ah11=-6.19012E-01,ah12= 2.02225E-02,
ah20=-7.34101E+00,ah21= 1.00462E+00,ah22=-3.20985E-02,
ah30= 2.93119E+00,ah31=-4.03761E-01,ah32= 1.25153E-02;
f32
bh10= 4.23071E+00,bh11=-6.10995E-01,bh12= 1.95531E-02,
bh20=-7.12527E+00,bh21= 9.69160E-01,bh22=-2.74255E-02,
bh30= 2.69925E+00,bh31=-3.63283E-01,bh32= 9.55316E-03;
f32
al00=-2.05398E+00,al01= 2.38815E-02,al02= 5.25483E-04,
al10=-7.69748E-02,al11=-6.91499E-02,al12= 2.22453E-03,
al20= 4.06463E-02,al21=-1.01281E-02,al22= 3.40919E-04;
f32
bl00= 1.04133E+00,bl01=-9.43291E-03,bl02=-4.54758E-04,
bl10= 1.19253E-01,bl11= 4.07467E-02,bl12=-1.30718E-03,
bl20=-1.59391E-02,bl21= 7.27752E-03,bl22=-1.94405E-04;

// Constant parameters for bremstrahlung table
f32  ZZ[8]= {2.,4.,6.,14.,26.,50.,82.,92.};

f32  coefsig[8][11]= {{.4638,.37748,.32249,-.060362,-.065004,-.033457,-.004583,.011954,.0030404,-.0010077,-.00028131},
    {.50008,.33483,.34364,-.086262,-.055361,-.028168,-.0056172,.011129,.0027528,-.00092265,-.00024348},
    {.51587,.31095,.34996,-.11623,-.056167,-.0087154,.00053943,.0054092,.00077685,-.00039635,-6.7818e-05},
    {.55058,.25629,.35854,-.080656,-.054308,-.049933,-.00064246,.016597,.0021789,-.001327,-.00025983},
    {.5791,.26152,.38953,-.17104,-.099172,.024596,.023718,-.0039205,-.0036658,.00041749,.00023408},
    {.62085,.27045,.39073,-.37916,-.18878,.23905,.095028,-.068744,-.023809,.0062408,.0020407},
    {.66053,.24513,.35404,-.47275,-.22837,.35647,.13203,-.1049,-.034851,.0095046,.0030535},
    {.67143,.23079,.32256,-.46248,-.20013,.3506,.11779,-.1024,-.032013,.0092279,.0028592}
};
#endif

__host__ __device__ void e_read_CS_table (
//                             ParticlesData particles,
    int id,
    int mat, //material
    f32 energy, //energy of particle
    ElectronsCrossSectionTable &d_table,
    unsigned char &next_discrete_process, //next discrete process id
    int &table_index,
    f32 & next_interaction_distance,
    f32 & dedxeIoni,
    f32 & dedxeBrem,
    f32 & erange,
    f32 & lambda,
    f32 randomnumbereBrem,
    f32 randomnumbereIoni,
    GlobalSimulationParametersData parameters )
{
// GGcout<< __FUNCTION__ << "  " << __LINE__ << GGendl;
//     printf("energy %e mat %d id %d  mat %d d_table.nb_bins %u \n",energy,mat,id,mat,d_table.nb_bins);
//     printf("energy %e mat %d id %d  mat %d d_table.nb_bins %u \n",energy,mat,id,mat,d_table.nb_bins);
//     table_index = binary_search(energy,d_table.E,d_table.nb_bins) + mat*d_table.nb_bins;
    table_index = binary_search ( energy,d_table.E, ( mat+1 ) *d_table.nb_bins,mat*d_table.nb_bins );
// GGcout<< __FUNCTION__ << "  " << __LINE__ << GGendl;
    if ( parameters.physics_list[ELECTRON_IONISATION] == ENABLED )
    {
// GGcout<< __FUNCTION__ << "  " << __LINE__ << GGendl;
        f32 distanceeIoni = randomnumbereIoni / linear_interpolation ( d_table.E[table_index-1], d_table.eIonisationCS[table_index-1], d_table.E[table_index], d_table.eIonisationCS[table_index], energy );
// GGcout<< __FUNCTION__ << "  " << __LINE__ << GGendl;
        if ( distanceeIoni<next_interaction_distance )
        {
            next_interaction_distance = distanceeIoni;
            next_discrete_process = ELECTRON_IONISATION;
        }
// GGcout<< __FUNCTION__ << "  " << __LINE__ << GGendl;
        dedxeIoni = linear_interpolation ( d_table.E[table_index-1],d_table.eIonisationdedx[table_index-1], d_table.E[table_index], d_table.eIonisationdedx[table_index], energy );
// GGcout<< __FUNCTION__ << "  " << __LINE__ << GGendl;
    }
// GGcout<< __FUNCTION__ << "  " << __LINE__ << GGendl;
    if ( parameters.physics_list[ELECTRON_BREMSSTRAHLUNG] == ENABLED )
    {

        f32 distanceeBrem = randomnumbereBrem /  linear_interpolation ( d_table.E[table_index-1], d_table.eBremCS[table_index-1], d_table.E[table_index], d_table.eBremCS[table_index], energy ) ;

        if ( distanceeBrem<next_interaction_distance )
        {
            next_interaction_distance = distanceeBrem;
            next_discrete_process = ELECTRON_BREMSSTRAHLUNG;
        }

        dedxeBrem =  linear_interpolation ( d_table.E[table_index-1],d_table.eBremdedx[table_index-1], d_table.E[table_index], d_table.eBremdedx[table_index], energy );
    }

// GGcout<< __FUNCTION__ << "  " << __LINE__ << GGendl;


    erange = linear_interpolation ( d_table.E[table_index-1],d_table.eRange[table_index-1], d_table.E[table_index], d_table.eRange[table_index], energy );


//         printf("d_table.E[table_index-1] %e ,d_table.eRange[table_index-1] %e , d_table.E[table_index] %e, d_table.eRange[table_index] %e table_index %d\n",d_table.E[table_index-1],d_table.eRange[table_index-1], d_table.E[table_index], d_table.eRange[table_index],table_index);


    if ( parameters.physics_list[ELECTRON_MSC] == ENABLED )
    {
        lambda = linear_interpolation ( d_table.E[table_index-1],d_table.eMSC[table_index-1], d_table.E[table_index], d_table.eMSC[table_index], energy );
        lambda=1./lambda;
    }
//     printf("energy %e mat %d id %d  mat %d d_table.nb_bins %u \n",energy,mat,id,mat,d_table.nb_bins);

}

__host__ __device__ f32 StepFunction ( f32 Range )
{
    f32 alpha=0.2;
    f32 rho=1.*mm;
    f32  StepF;
    if ( Range<rho )
        return  Range;
    StepF=alpha*Range+rho* ( 1.-alpha ) * ( 2.-rho/Range );
    if ( StepF<rho )
        StepF=rho;

    return  StepF;
}

__host__ __device__ f32 LossApproximation ( f32 StepLength, f32 Ekine, f32 erange, ElectronsCrossSectionTable d_table, int mat, int id )
{
    f32  range,perteApp = 0;
    range=erange;
    range-=StepLength;
    if ( range >1.*nm )
        perteApp=GetEnergy ( range, d_table, mat );
    else
        perteApp = 0.;

    perteApp=Ekine-perteApp;

    return  perteApp;
}


#define rate .55
#define fw 4.
#define nmaxCont 16
#define minLoss 10.*eV
__host__ __device__ f32 eFluctuation ( f32 meanLoss,f32 cutEnergy, MaterialsTable materials, ParticlesData &particles, int id, int id_mat )
{
    /*if(id==DEBUGID) PRINT_PARTICLE_STATE("");*/
    int nb,k;
    f32  LossFluct=0.,lossc=0.;//,minLoss=10.*eV;
//     f32  rate=.55,fw=4.,nmaxCont=16.;
    f32  tau,gamma,gamma2,beta2;
    f32  F1,F2,E0,E1,E2,E1Log,E2Log,I,ILog;
    f32  e1,e2,esmall,w,w1,w2,C,alfa,alfa1,namean;
    f32  a1=0.,a2=0.,a3=0.,sa1;
    f32  emean=0.,sig2e=0.,sige=0.,p1=0.,p2=0.,p3=0.;
    f32  tmax=min ( cutEnergy,.5*particles.E[id] );

    if ( meanLoss<minLoss )
        return  meanLoss;
    /*if(id==DEBUGID) PRINT_PARTICLE_STATE("");*/
    tau=particles.E[id]/electron_mass_c2;
    gamma=tau+1.;
    gamma2=gamma*gamma;
    beta2=tau* ( tau+2. ) /gamma2;
    F1=materials.fF1[id_mat];
    F2=materials.fF2[id_mat];
    E0=materials.fEnergy0[id_mat];
    E1=materials.fEnergy1[id_mat];
    E2=materials.fEnergy2[id_mat];
    E1Log=materials.fLogEnergy1[id_mat];
    E2Log=materials.fLogEnergy2[id_mat];
    I=materials.electron_mean_excitation_energy[id_mat];
    ILog=logf ( I ); //materials.fLogMeanExcitationEnergy[id_mat];
    esmall=.5*sqrtf ( E0*I );

//     if(id==DEBUGID) printf("%f \n",ILog);
//     return meanLoss;
    if ( tmax<=E0 )
    {
        return  meanLoss;
    }

    if ( tmax>I )
    {
        w2=logf ( 2.*electron_mass_c2*beta2*gamma2 )-beta2;
        if ( w2>ILog )
        {
            C=meanLoss* ( 1.-rate ) / ( w2-ILog );
            a1=C*F1* ( w2-E1Log ) /E1;
            if ( w2>E2Log )
                a2=C*F2* ( w2-E2Log ) /E2;
            if ( a1<nmaxCont )
            {
                sa1=sqrtf ( a1 );
                if ( JKISS32 ( particles, id ) <expf ( -sa1 ) )
                {
                    e1=esmall;
                    a1=meanLoss* ( 1.-rate ) /e1;
                    a2=0.;
                    e2=E2;
                }
                else
                {
                    a1=sa1 ;
                    e1=sa1*E1;
                    e2=E2;
                }
            }
            else
            {
                a1/=fw;
                e1=fw*E1;
                e2=E2;
            }
        }
    }

    w1=tmax/E0;
    if ( tmax>E0 )
        a3=rate*meanLoss* ( tmax-E0 ) / ( E0*tmax*log ( w1 ) );
    if ( a1>nmaxCont )
    {
        emean+=a1*e1;
        sig2e+=a1*e1*e1;
    }
    else if ( a1>0. )
    {
        p1= ( f32 ) G4Poisson ( a1, particles, id );
        LossFluct+=p1*e1;
        if ( p1>0. )
            LossFluct+= ( 1.-2.*JKISS32 ( particles, id ) ) *e1;
    }
    if ( a2>nmaxCont )
    {
        emean+=a2*e2;
        sig2e+=a2*e2*e2;
    }
    else if ( a2>0. )
    {
        p2= ( f32 ) G4Poisson ( a2, particles, id );
        LossFluct+=p2*e2;
        if ( p2>0. )
            LossFluct+= ( 1.-2.*JKISS32 ( particles, id ) ) *e2;
    }
    if ( a3>0. )
    {
        p3=a3;
        alfa=1.;
        if ( a3>nmaxCont )
        {
            alfa=w1* ( nmaxCont+a3 ) / ( w1*nmaxCont+a3 );
            alfa1=alfa*log ( alfa ) / ( alfa-1. );
            namean=a3*w1* ( alfa-1. ) / ( ( w1-1. ) *alfa );
            emean+=namean*E0*alfa1;
            sig2e+=E0*E0*namean* ( alfa-alfa1*alfa1 );
            p3=a3-namean;
        }
        w2=alfa*E0;
        w= ( tmax-w2 ) /tmax;
        nb=G4Poisson ( p3, particles, id );
        if ( nb>0 )
            for ( k=0; k<nb; k++ )
                lossc+=w2/ ( 1.-w*JKISS32 ( particles, id ) );
    }
    if ( emean>0. )
    {
        sige=sqrtf ( sig2e );
//         if(isnan(emean)) emean = 0;
//         if(isnan(sige)) sige = 0;
        LossFluct+=max ( 0.,Gaussian ( emean,sige,particles, id ) );
//         f32 toto = Gaussian(emean,sige,particles, id);
//         if(isnan(toto)||isinf(toto)) toto = 0;
//         if(toto>0) LossFluct += toto;

//         LossFluct+=fmaxf(0.,emean);


    }
    LossFluct+=lossc;

    return  LossFluct;
}
#undef rate
#undef fw
#undef nmaxCont
#undef minLoss


__host__ __device__ f32 eLoss ( f32 LossLength, f32 &Ekine, f32 dedxeIoni, f32 dedxeBrem, f32 erange,ElectronsCrossSectionTable d_table, int mat, MaterialsTable materials, ParticlesData &particles,GlobalSimulationParametersData parameters, int id )
{
    f32  perteTot=0.;//,perteBrem=0.,perteIoni=0.;
    perteTot=LossLength* ( dedxeIoni + dedxeBrem );

    if ( perteTot>Ekine*0.01 ) // 0.01 is xi
        perteTot=LossApproximation ( LossLength, Ekine, erange, d_table, mat, id );

/// \warning ADD for eFluctuation
    if ( dedxeIoni>0. )
        perteTot=eFluctuation ( perteTot,parameters.electron_cut,materials,particles,id,mat );

    if ( ( Ekine-perteTot ) <= ( 1.*eV ) )
    {
        perteTot=Ekine;

    }

    Ekine-=perteTot;

    return  perteTot;
}


#define tausmall 1.E-16
__host__ __device__ f32 gGeomLengthLimit ( f32 gPath,f32 cStep,f32 currentLambda,f32 currentRange,f32 par1,f32 par3 )
{
    f32  tPath;
//     f32  tausmall=1.E-16; //tausmall=1.E-16;

    par3=1.+par3;
    tPath=gPath;
    if ( gPath>currentLambda*tausmall )
    {
        if ( par1<0. )
            tPath=-currentLambda*log ( 1.-gPath/currentLambda );
        else
        {
            if ( par1*par3*gPath<1. )
                tPath= ( 1.-exp ( log ( 1.-par1*par3*gPath ) /par3 ) ) /par1;
            else
                tPath=currentRange;
        }
    }
    if ( tPath<gPath )
        tPath=gPath;
    return  tPath;
}
#undef tausmall


__host__ __device__ f32 eSimpleScattering ( f32 xmeanth,f32 x2meanth, int id, ParticlesData &particles )
{
    f32    a= ( 2.*xmeanth+9.*x2meanth-3. ) / ( 2.*xmeanth-3.*x2meanth+1. );
    f32    prob= ( a+2. ) *xmeanth/a;
    f32  cth=1.;

    if ( JKISS32 ( particles, id ) <prob )
        cth=-1.+2.*expf ( logf ( JKISS32 ( particles, id ) ) / ( a+1. ) );
    else
        cth=-1.+2.*JKISS32 ( particles, id );
    return    cth;
}



__host__ __device__ f32 eCosineTheta ( f32 trueStep,f32 currentRange,f32 currentLambda,f32 currentEnergy,f32 *currentTau,f32 par1,f32 par2, MaterialsTable materials, int id_mat, int id, ParticlesData &particles )
{
    f32 particleEnergy = particles.E[id];
    f32  costh,sinth;
    f32  tau;
    const f32 taubig=8.,tausmall=1.E-16,taulim=1.E-6;
    f32  c,c1,x0,b,bx,b1,ebx,eb1;
    f32  prob=0.,qprob=1.;
    f32  a=1.,ea=0.,eaa=1.;
    f32  xmeanth,xmean1=1.,xmean2=0.,x2meanth;
    f32  dtrl=0.05;//5./100.;
    f32  xsi=3.;
    f32    theta0,theta0max=pi/6.,y,corr,betacp,c_highland=13.6*MeV;
    f32    f1x0,f2x0;


    costh=1.;
    tau=trueStep/currentLambda;

    if ( trueStep>=currentRange*dtrl )
    {
        if ( ( par1*trueStep ) <1. )
        {
            tau=-par2*logf ( 1.-par1*trueStep );
        }
        else if ( ( 1.-particleEnergy/currentEnergy ) >taulim )
            tau=taubig;
    }

    *currentTau=tau;
    if ( tau>=taubig )
    {
        f32 temp;
//         do{
        temp=JKISS32 ( particles, id );
//         }while((1.-temp)<2.e-7); // to avoid 1 due to f32 approximation
        costh=-1.+2.*temp;

    }
    else if ( tau>=tausmall )
    {
        x0=1.;
        b=2.;
        b1=3.;
        bx=1.;
        eb1=3.;
        ebx=1.;
        prob=1.;
        qprob=1.;
        xmeanth=expf ( -tau );
        x2meanth= ( 1.+2.*expf ( -2.5*tau ) ) /3.;
        if ( 1.-particleEnergy/currentEnergy>.5 )
        {
            costh=eSimpleScattering ( xmeanth,x2meanth, id, particles );

            return  costh;
        }

        betacp=sqrtf ( currentEnergy* ( currentEnergy+2.*electron_mass_c2 )
                       *particleEnergy* ( particleEnergy+2.*electron_mass_c2 )
                       / ( ( currentEnergy+electron_mass_c2 ) * ( particleEnergy+electron_mass_c2 ) ) );


        y=trueStep/materials.rad_length[id_mat];
        theta0=c_highland*sqrtf ( y ) /betacp;
        y=logf ( y );

        f32 Zeff = materials.nb_electrons_per_vol[id_mat]/materials.nb_atoms_per_vol[id_mat];

        corr= ( 1.-8.778E-2/Zeff ) * ( .87+.03*logf ( Zeff ) )
              + ( 4.078E-2+1.7315E-4*Zeff ) * ( .87+.03*logf ( Zeff ) ) *y;
        theta0*=corr ;

        if ( theta0*theta0<tausmall )
            return  costh;
        if ( theta0>theta0max )
        {
            costh=eSimpleScattering ( xmeanth,x2meanth, id, particles );
            return  costh;
        }

        sinth=sinf ( .5*theta0 );
        a=.25/ ( sinth*sinth );
        ea=expf ( -xsi );
        eaa=1.-ea ;
        xmean1=1.- ( 1.- ( 1.+xsi ) *ea ) / ( a*eaa );
        x0=1.-xsi/a;
        if ( xmean1<=.999*xmeanth )
        {
            costh=eSimpleScattering ( xmeanth,x2meanth, id, particles );
            return  costh;
        }

        c=2.943-.197*logf ( Zeff+1. )
          + ( .0987-.0143*logf ( Zeff+1. ) ) *y;

        if ( fabsf ( c-3. ) <.001 )
            c=3.001;
        if ( fabsf ( c-2. ) <.001 )
            c=2.001;
        if ( fabsf ( c-1. ) <.001 )
            c=1.001;
        c1=c-1.;

        b=1.+ ( c-xsi ) /a;
        b1=b+1.;
        bx=c/a;
        eb1=expf ( c1*logf ( b1 ) );
        ebx=expf ( c1*logf ( bx ) );
        xmean2= ( x0*eb1+ebx- ( eb1*bx-b1*ebx ) / ( c-2. ) ) / ( eb1-ebx );
        f1x0=a*ea/eaa;
        f2x0=c1*eb1/ ( bx* ( eb1-ebx ) );
        prob=f2x0/ ( f1x0+f2x0 );
        qprob=xmeanth/ ( prob*xmean1+ ( 1.-prob ) *xmean2 );
        if ( JKISS32 ( particles, id ) <qprob )
        {
            if ( JKISS32 ( particles, id ) <prob )
                costh=1.+logf ( ea+JKISS32 ( particles, id ) *eaa ) /a;
            else
                costh=b-b1*bx/expf ( logf ( ebx+ ( eb1-ebx ) *JKISS32 ( particles, id ) ) /c1 );
        }
        else
            costh=-1.+2.*JKISS32 ( particles, id );
    }
    return  costh;
}



__host__ __device__ void gLatCorrection ( f32xyz currentDir,f32 tPath,f32 zPath,f32 currentTau,f32 phi,f32 sinth, ParticlesData &particles, int id, f32 safety )
{
    f32  latcorr,etau,rmean,rmax,Phi,psi,lambdaeff;
    const f32  kappa=2.5,tlimitminfix=1.E-6*mm,taulim=1.E-6,tausmall=1.E-16,taubig=8.,geomMin=1.E-6*mm;
//     struct  Vector  latDir;
    f32xyz  latDir;
    lambdaeff=tPath/currentTau;

    if ( safety>tlimitminfix ) // Safety is distance to near voxel
    {
        rmean=0.;
        if ( ( currentTau>=tausmall ) )
        {
            if ( currentTau<taulim )
                rmean=kappa*currentTau*currentTau*currentTau* ( 1.- ( kappa+1. ) *currentTau*.25 ) /6.; //powf(currentTau,3.)
            else
            {
                etau=0.;
                if ( currentTau<taubig )
                    etau=expf ( -currentTau );
                rmean=-kappa*currentTau;
                rmean=-expf ( rmean ) / ( kappa* ( kappa-1. ) );
                rmean+=currentTau- ( kappa+1. ) /kappa+kappa*etau/ ( kappa-1. );
            }
            if ( rmean>0. )
                rmean=2.*lambdaeff*sqrtf ( rmean/3. );
            else
                rmean=0.;
        }
        rmax= ( tPath-zPath ) * ( tPath+zPath );
        if ( rmax<0. )
            rmax=0.;
        else
            rmax=sqrtf ( rmax );
        if ( rmean>=rmax )
            rmean=rmax;

        if ( rmean<=geomMin )
            return;

        if ( rmean>0. )
        {
            if ( ( currentTau>=tausmall ) )
            {
                if ( currentTau<taulim )
                {
                    latcorr=lambdaeff*kappa*currentTau*currentTau* ( 1.- ( kappa+1. ) *currentTau/3. ) /3.;
                }
                else
                {
                    etau=0.;
                    if ( currentTau<taubig )
                        etau=expf ( -currentTau );
                    latcorr=-kappa*currentTau;
                    latcorr=expf ( latcorr ) / ( kappa-1. );
                    latcorr+=1.-kappa*etau/ ( kappa-1. );
                    latcorr*=2.*lambdaeff/3.;

                }
            }
            if ( latcorr>rmean )
                latcorr=rmean;
            else if ( latcorr<0. )
                latcorr=0.;
            Phi=0.;
            if ( fabsf ( rmean*sinth ) <=latcorr )
            {
                Phi=2.*pi*JKISS32 ( particles, id );
            }
            else
            {
                psi=acosf ( latcorr/ ( rmean*sinth ) );
                if ( JKISS32 ( particles, id ) <.5 )
                    Phi=phi+psi;
                else
                    Phi=phi-psi;
            }
            latDir.x=cos ( Phi );
            latDir.y=sin ( Phi );
            latDir.z=0.;
            latDir=rotateUz ( latDir,currentDir );
            if ( rmean>safety )
                rmean=safety*.99;

            particles.px[id]+=latDir.x*rmean;
            particles.py[id]+=latDir.y*rmean;
            particles.pz[id]+=latDir.z*rmean;


        }
    }
}




__host__ __device__ void eMscScattering ( f32 tPath,f32 zPath,f32 currentRange,f32 currentLambda,f32 currentEnergy,f32 par1,f32 par2, ParticlesData &particles, int id, MaterialsTable materials, int mat, VoxVolumeData phantom,ui16xyzw index_phantom )
{


    f32  costh,sinth,phi,currentTau;
    const f32  tlimitminfix=1.E-10*mm,tausmall=1.E-16; //,taulim=1.E-6
    f32xyz  Dir, currentDir;

    if ( ( particles.E[id]<0. ) || ( tPath<=tlimitminfix ) || ( tPath/tausmall<currentLambda ) )
    {
        return;
    }


    costh=eCosineTheta ( tPath,currentRange,currentLambda,currentEnergy,&currentTau,par1,par2, materials, mat, id,particles );

    if ( fabs ( costh ) >1. )
        return;
    if ( costh< ( 1.-1000.*tPath/currentLambda ) && ( particles.E[id] ) > ( 20.*MeV ) )
    {
        do
        {

            costh=1.+2.*logf ( JKISS32 ( particles, id ) ) *tPath/currentLambda;
        }
        while ( ( costh<-1. ) );
    }


    sinth=sqrtf ( ( 1.-costh ) * ( 1.+costh ) );
    phi=2.*pi*JKISS32 ( particles, id );

    Dir = make_f32xyz ( sinth*cosf ( phi ), sinth*sinf ( phi ), costh );

    particles.px[id]+=particles.dx[id]*zPath;
    particles.py[id]+=particles.dy[id]*zPath;
    particles.pz[id]+=particles.dz[id]*zPath;


    currentDir = make_f32xyz ( particles.dx[id],particles.dy[id],particles.dz[id] );

    Dir=rotateUz ( Dir,currentDir );


    particles.dx[id] = Dir.x;
    particles.dy[id] = Dir.y;
    particles.dz[id] = Dir.z;


    // Read position
    f32xyz position; // mm
    position.x = particles.px[id];
    position.y = particles.py[id];
    position.z = particles.pz[id];

    // Read direction
    f32xyz direction;
    direction.x = particles.dx[id];
    direction.y = particles.dy[id];
    direction.z = particles.dz[id];

    //Get Phantom index
//         int4 index_phantom;
//         const f32xyz ivoxsize = vec3_inverse(phantom.voxel_size);
//         index_phantom.x = 0;//int(position.x * ivoxsize.x);
//         index_phantom.y = 0;//int(position.y * ivoxsize.y);
//         index_phantom.z = 0;//int(position.z * ivoxsize.z);
//         index_phantom.w = index_phantom.z*phantom.nb_voxel_slice
//                           + index_phantom.y*phantom.size_in_vox.x
//                           + index_phantom.x; // linear index

    f32 safety = GetSafety ( position, direction,index_phantom,make_f32xyz ( phantom.spacing_x,phantom.spacing_y,phantom.spacing_z ) ) ;

//     Comment next line to disable lateral correction
    gLatCorrection ( currentDir,tPath,zPath,currentTau,phi,sinth,particles,id,safety );

}


// From Eric's code
__host__ __device__ f32 GlobalMscScattering ( f32 GeomPath,f32 cutstep,f32 CurrentRange,f32 CurrentEnergy, f32 CurrentLambda, f32 dedxeIoni, f32 dedxeBrem, ElectronsCrossSectionTable d_table, int mat, ParticlesData &particles, int id,f32 par1,f32 par2, MaterialsTable materials,DoseData &dosi, ui16xyzw index_phantom, VoxVolumeData phantom,GlobalSimulationParametersData parameters )
{
// GGcout<< __FUNCTION__ << "  " << __LINE__ << GGendl;
// for(int i = 0;i<phantom.number_of_voxels;i++){
// if(phantom.values[i]!=0){ printf("%d %d %d %d\n",__LINE__,id,i,phantom.values[i]); exit(0);}
// }
    f32  edep,TruePath,zPath;//,tausmall=1.E-16;
//     // MSC disabled
    if ( parameters.physics_list[ELECTRON_MSC] != ENABLED )
    {
        if ( GeomPath<cutstep )
        {
            edep = eLoss ( GeomPath, particles.E[id], dedxeIoni, dedxeBrem, CurrentRange, d_table, mat, materials, particles,parameters, id );

            /// TODO WARNING  ACTIVER LA FONCTION DE DOSIMETRIE
            dose_record_standard ( dosi, edep, particles.px[id],particles.py[id],particles.pz[id] );

        }
        particles.px[id] += particles.dx[id] * GeomPath;
        particles.py[id] += particles.dy[id] * GeomPath;
        particles.pz[id] += particles.dz[id] * GeomPath;

        return  GeomPath;
    }
// for(int i = 0;i<phantom.number_of_voxels;i++){
// if(phantom.values[i]!=0){ printf("%d %d %d %d\n",__LINE__,id,i,phantom.values[i]); exit(0);}
// }

    if ( GeomPath==cutstep )
    {
        zPath=gTransformToGeom ( GeomPath,CurrentRange,CurrentLambda,CurrentEnergy,&par1,&par2,d_table, mat );
        /*if(id==DEBUGID) PRINT_PARTICLE_STATE("");*/
    }
    else
    {
        zPath=GeomPath;
        TruePath=gGeomLengthLimit ( GeomPath,cutstep,CurrentLambda,CurrentRange,par1,par2 );
        GeomPath=TruePath;

        edep = eLoss ( TruePath, particles.E[id], dedxeIoni, dedxeBrem, CurrentRange, d_table, mat, materials, particles, parameters, id );

        /// TODO WARNING  ACTIVER LA FONCTION DE DOSIMETRIE
        dose_record_standard ( dosi, edep, particles.px[id],particles.py[id],particles.pz[id] );

    }


    if ( particles.E[id] > 0.0 ) // if not laststep
    {
        eMscScattering ( GeomPath,zPath,CurrentRange,CurrentLambda,CurrentEnergy,par1,par2, particles,  id, materials, mat, phantom, index_phantom );
        /*if(id==DEBUGID) PRINT_PARTICLE_STATE("");*/

    }
    else
    {
        particles.endsimu[id]=PARTICLE_DEAD;
        particles.px[id]+=particles.dx[id]*zPath;
        particles.py[id]+=particles.dy[id]*zPath;
        particles.pz[id]+=particles.dz[id]*zPath;
    }
    /*if(id==DEBUGID) PRINT_PARTICLE_STATE("");*/
    /*for(int i = 0;i<phantom.number_of_voxels;i++){
    if(phantom.values[i]!=0){ printf("%d %d %d %d\n",__LINE__,id,i,phantom.values[i]); exit(0);}
    }    */
    return  TruePath;
}


__host__ __device__ SecParticle eSampleSecondarieElectron ( f32 CutEnergy, ParticlesData &particles, int id, DoseData &dosi,GlobalSimulationParametersData parameters )
{
    f32  totalEnergy,deltaEnergy,totMom,deltaMom;
    f32  xmin,xmax,gamma,gamma2;//,beta2;
    f32  x,z,q,grej,g,y;
    f32  cost,sint,phi;
    f32  tmax=fminf ( 1.*GeV,.5*particles.E[id] );
    f32  tmin=CutEnergy;

    f32xyz  ElecDir;

    SecParticle secondary_part;

    secondary_part.E = 0.;
    secondary_part.dir = make_f32xyz ( 0.,0.,0. );
    secondary_part.pname = ELECTRON;
    secondary_part.endsimu = PARTICLE_DEAD;


    if ( tmin>=tmax )
        return secondary_part;

    totalEnergy=particles.E[id]+electron_mass_c2;
    totMom=sqrtf ( particles.E[id]* ( totalEnergy+ electron_mass_c2 ) );
    xmin=tmin/particles.E[id];
    xmax=tmax/particles.E[id];
    gamma=totalEnergy/electron_mass_c2;
    gamma2=gamma*gamma;
//     beta2=1.-1./gamma2;
    g= ( 2.*gamma-1. ) /gamma2;
    y=1.-xmax;
    grej=1.-g*xmax+xmax*xmax* ( 1.-g+ ( 1.-g*y ) / ( y*y ) );

    do
    {
        q=JKISS32 ( particles,id );
        x=xmin*xmax/ ( xmin* ( 1.-q ) +xmax*q );
        y=1.-x;
        z=1.-g*x+x*x* ( 1.-g+ ( 1.-g*y ) / ( y*y ) );
    }
    while ( ( grej*JKISS32 ( particles,id ) >z ) );

    deltaEnergy=x*particles.E[id];
    deltaMom=sqrtf ( deltaEnergy* ( deltaEnergy+2.*electron_mass_c2 ) );
    cost=deltaEnergy* ( totalEnergy+electron_mass_c2 ) / ( deltaMom*totMom );
    sint=1.-cost*cost;
    if ( sint>0. )
        sint=sqrtf ( sint );
    phi=2.*pi*JKISS32 ( particles,id );

    ElecDir.x=sint*cosf ( phi );
    ElecDir.y=sint*sinf ( phi );
    ElecDir.z=cost;
    f32xyz currentDir;
    currentDir = make_f32xyz ( particles.dx[id],particles.dy[id],particles.dz[id] );

    ElecDir=rotateUz ( ElecDir,currentDir );

//     deltaEnergy = __int2f32_rn(__f322int_rn(deltaEnergy));
    particles.E[id]-=deltaEnergy;
//     if(id==7949384) printf(" delta %f ",deltaEnergy);
    if ( particles.E[id]>0.0 )
        currentDir=CorrUnit ( currentDir,ElecDir,totMom,deltaMom );

    particles.dx[id]=currentDir.x;
    particles.dy[id]=currentDir.y;
    particles.dz[id]=currentDir.z;

//     SecParticle secondary_part;

    secondary_part.E = deltaEnergy;
    secondary_part.dir = ElecDir;
    secondary_part.pname = ELECTRON;

//     if((int)(particles.level[id])<particles.nb_of_secondaries)
//         {
//             secondary_part.endsimu = PARTICLE_ALIVE;
//
// /// \warning \TODO COMMENT FOR NO SECONDARY,
// //         GenerateNewElectronParticle(deltaEnergy,ElecDir);
//
//         }
//     else
//         {
//         /// WARNING TODO ACTIVER DOSIMETRY ICI
//         dose_record_standard(dosi, deltaEnergy, particles.px[id],particles.py[id],particles.pz[id]);
//         secondary_part.endsimu = PARTICLE_DEAD;
//         }

    return secondary_part;
}


__host__ __device__ f32xyz CorrUnit ( f32xyz u, f32xyz v,f32 uMom, f32 vMom )
{
    f32  r;
    f32xyz  Final;

    Final.x=u.x*uMom-v.x*vMom;
    Final.y=u.y*uMom-v.y*vMom;
    Final.z=u.z*uMom-v.z*vMom;
    r=Final.x*Final.x+Final.y*Final.y+Final.z*Final.z;
    if ( r>0. )
    {
        r=sqrt ( Final.x*Final.x+Final.y*Final.y+Final.z*Final.z );
        Final.x=Final.x/r;
        Final.y=Final.y/r;
        Final.z=Final.z/r;
    }

    return  Final;
}


#define tausmall 1.E-20
#define taulim 1.E-6
#define tlimitminfix 1.E-6*mm
#define dtrl 5./100
__host__ __device__ f32 gTransformToGeom ( f32 TPath,f32 currentRange,f32 currentLambda,f32 currentEnergy,f32 *par1,f32 *par2, ElectronsCrossSectionTable electron_CS_table, int mat )
{
    f32  ZPath,zmean;
//     f32  tausmall=1.E-20,taulim=1.E-6,tlimitminfix=1.E-6*mm;
//     f32  dtrl=5./100.;
    f32  tau,t1,lambda1;
    f32  par3;

    *par1=-1.;
    *par2=par3=0.;
    ZPath=TPath;
    if ( TPath<tlimitminfix )
        return  ZPath;
    if ( TPath>currentRange )
        TPath=currentRange;
    tau=TPath/currentLambda;
    if ( ( tau<=tausmall ) /*||insideskin*/ )
    {
        ZPath=TPath;
        if ( ZPath>currentLambda )
            ZPath=currentLambda;
        return  ZPath;
    }
    zmean=TPath;
    if ( TPath<currentRange*dtrl )
    {
        if ( tau<taulim )
            zmean=TPath* ( 1.-0.5*tau );
        else
            zmean=currentLambda* ( 1.-expf ( -tau ) );
    }
    else if ( currentEnergy<electron_mass_c2 )
    {
        *par1=1./currentRange;
        *par2=1./ ( *par1*currentLambda );
        par3=1.+*par2;
        if ( TPath<currentRange )
            zmean= ( 1.-expf ( par3*logf ( 1.-TPath/currentRange ) ) ) / ( *par1*par3 );
        else
            zmean=1./ ( *par1*par3 );
    }
    else
    {
        t1=GetEnergy ( currentRange-TPath, electron_CS_table, mat );
        lambda1=1./GetLambda ( t1,1, electron_CS_table, mat );
        *par1= ( currentLambda-lambda1 ) / ( currentLambda*TPath );
        *par2=1./ ( *par1*currentLambda );
        par3=1.+*par2;

        zmean= ( 1.-expf ( par3*logf ( lambda1/currentLambda ) ) ) / ( *par1*par3 );
    }
    ZPath=zmean;

//     return (fminf(ZPath,currentLambda));
    if ( ZPath>currentLambda )
        ZPath=currentLambda;
    return  ZPath;
}
#undef tausmall
#undef taulim
#undef tlimitminfix
#undef dtrl

__host__ __device__ f32 GetEnergy ( f32 Range, ElectronsCrossSectionTable d_table, int mat )
{

    int index = binary_search ( Range, d_table.eRange, d_table.nb_bins*mat+d_table.nb_bins, d_table.nb_bins*mat );

    f32 newRange = linear_interpolation ( d_table.eRange[index-1], d_table.E[index-1], d_table.eRange[index], d_table.E[index], Range );

    return newRange;
}

__host__ __device__ f32 GetLambda ( f32 Range, unsigned short int flag, ElectronsCrossSectionTable d_table, int mat )
{
    int index = binary_search ( Range, d_table.E, d_table.nb_bins*mat+d_table.nb_bins, d_table.nb_bins*mat );

    if ( flag == 1 ) return linear_interpolation ( d_table.E[index-1],d_table.eMSC[index-1], d_table.E[index], d_table.eMSC[index], Range );

    else if ( flag == 2 ) return linear_interpolation ( d_table.E[index-1],d_table.eIonisationCS[index-1], d_table.E[index], d_table.eIonisationCS[index], Range );

    else /*if (flag == 3)*/ return linear_interpolation ( d_table.E[index-1],d_table.eBremCS[index-1], d_table.E[index], d_table.eBremCS[index], Range );

}


__host__ __device__ f32 eBremCrossSectionPerAtom ( f32 Z,f32 cut, f32 Ekine )
{
    int i,j,iz=0,NZ=8,Nsig=11;
    f32    Cross=0.;
    f32    ksi=2.,alfa=1.;
    f32    csigh=.127,csiglow=.25,asiglow=.02*MeV;
    f32    Tlim=10.*MeV;
    f32    xlim=1.2,delz=1.E6,absdelz;
    f32    xx,fs;

    if ( Ekine<1.*keV||Ekine<cut )
        return  Cross;

    for ( i=0; i<NZ; i++ )
    {
        absdelz=fabs ( Z-ZZ[i] );
        if ( absdelz<delz )
        {
            iz=i;
            delz=absdelz;
        }
    }

    xx=log10 ( Ekine );
    fs=1.;
    if ( xx<=xlim )
    {
        fs=coefsig[iz][Nsig-1];
        for ( j=Nsig-2; j>=0; j-- )
            fs=fs*xx+coefsig[iz][j];
        if ( fs<0. )
            fs=0.;
    }
    Cross=Z* ( Z+ksi ) * ( 1.-csigh*exp ( log ( Z ) /4. ) ) *pow ( log ( Ekine/cut ),alfa );

    if ( Ekine<=Tlim )
        Cross*=exp ( csiglow*log ( Tlim/Ekine ) ) * ( 1.+asiglow/ ( sqrt ( Z ) *Ekine ) );
    Cross*=fs/N_avogadro;
    if ( Cross<0. )
        Cross=0.;

    return  Cross;
}

__host__ __device__ int RandomAtom ( f32 CutEnergyGamma,ParticlesData &particles, int id, MaterialsTable &materials, int id_mat )
{
    int indice,tmp;
    f32 rval;
//     int indexelt= materials.index[id_mat];
    tmp=materials.index[id_mat]+materials.nb_elements[id_mat]-1;
    rval=JKISS32 ( particles, id );
    rval*=materials.atom_num_dens[id_mat];
    rval/=materials.nb_atoms_per_vol[id_mat];
    rval*=eBremCrossSectionPerAtom ( materials.mixture[tmp],CutEnergyGamma, particles.E[id] );

    for ( int i=0; i<materials.nb_elements[id_mat]; ++i )
    {
        int indexelt= i+ materials.index[id_mat];
        f32 U =materials.atom_num_dens[id_mat];
        U/=materials.nb_atoms_per_vol[id_mat];
        U*eBremCrossSectionPerAtom ( materials.mixture[indexelt],CutEnergyGamma, particles.E[id] );

        if ( rval<=U )
        {
            indice=indexelt;
            break;
        }
    }
    return  indice;
}


__host__ __device__ f32 ScreenFunction1 ( f32 ScreenVariable )
{
    f32 screenVal;
    if ( ScreenVariable>1. )
        screenVal=42.24-8.368*logf ( ScreenVariable+.952 );
    else
        screenVal=42.392-ScreenVariable* ( 7.796-1.961*ScreenVariable );
    return  screenVal;
}

__host__ __device__ f32 ScreenFunction2 ( f32 ScreenVariable )
{
    f32 screenVal;
    if ( ScreenVariable>1. )
        screenVal=42.24-8.368*logf ( ScreenVariable+.952 );
    else
        screenVal=41.734-ScreenVariable* ( 6.484-1.25*ScreenVariable );
    return  screenVal;
}

__host__ __device__ f32 RejectionFunction ( f32 value,f32 rej1,f32 rej2,f32 rej3,f32 ratio,f32 z )
{
    f32  argument= ( 1.+value ) * ( 1.+value );
    return ( 4.+logf ( rej3+ ( z/argument ) ) ) * ( ( 4.*ratio*value/argument )-rej1 ) +rej2;
}


__host__ __device__ f32 AngleDistribution ( f32 initial_energy,f32 final_energy,f32 Z, ParticlesData &particles, int id )
{
    f32  Theta=0.;
    f32  initialTotalEnergy= ( initial_energy+electron_mass_c2 ) /electron_mass_c2;
    f32  finalTotalEnergy= ( final_energy+electron_mass_c2 ) /electron_mass_c2;
    f32  EnergyRatio=finalTotalEnergy/initialTotalEnergy;
    f32  gMaxEnergy= ( pi*initialTotalEnergy ) * ( pi*initialTotalEnergy );
    f32  z,rejection_argument1,rejection_argument2,rejection_argument3;
    f32  gfunction0,gfunction1,gfunctionEmax,gMaximum;
    f32  rand,gfunctionTest,randTest;

    z=.00008116224* ( powf ( Z,1./3. ) +powf ( Z+1,1./3. ) );
    rejection_argument1= ( 1.+EnergyRatio*EnergyRatio );
    rejection_argument2=-2.*EnergyRatio+3.*rejection_argument1;
    rejection_argument3= ( ( 1.-EnergyRatio ) / ( 2.*initialTotalEnergy*EnergyRatio ) ) * ( ( 1.-EnergyRatio ) / ( 2.*initialTotalEnergy*EnergyRatio ) );
    gfunction0=RejectionFunction ( 0.,rejection_argument1,rejection_argument2,rejection_argument3,EnergyRatio,z );
    gfunction1=RejectionFunction ( 1.,rejection_argument1,rejection_argument2,rejection_argument3,EnergyRatio,z );
    gfunctionEmax=RejectionFunction ( gMaxEnergy,rejection_argument1,rejection_argument2,rejection_argument3,EnergyRatio,z );
    gMaximum=fmaxf ( gfunction0,gfunction1 );
    gMaximum=fmaxf ( gMaximum,gfunctionEmax );

    do
    {
        rand=JKISS32 ( particles, id );
        rand/= ( 1.-rand+1./gMaxEnergy );
        gfunctionTest=RejectionFunction ( rand,rejection_argument1,rejection_argument2,rejection_argument3,EnergyRatio,z );
        randTest=JKISS32 ( particles,id );
    }
    while ( randTest*gMaximum>gfunctionTest );
    Theta=sqrtf ( rand ) /initialTotalEnergy;

    return  Theta;
}



__host__ __device__
void eSampleSecondarieGamma ( f32 cutEnergy, ParticlesData &particles, int id, MaterialsTable materials, int id_mat, GlobalSimulationParametersData parameters )
{
    f32 MaxKinEnergy = parameters.cs_table_max_E;
    int ind;
    f32  gammaEnergy,totalEnergy;
    f32  xmin,xmax,kappa,epsilmin,epsilmax;
    f32  lnZ,FZ,Z3,ZZ,F1,F2,theta,sint,phi;
    f32  tmin=cutEnergy;
    f32  tmax=fminf ( MaxKinEnergy,particles.E[id] ); // MaxKinEnergy = 250* MeV
    f32  MigdalFactor,MigdalConstant=elec_radius*hbarc*hbarc*4.*pi/ ( electron_mass_c2*electron_mass_c2 );
    f32  x,xm,epsil,greject,migdal,grejmax,q,U,U2;
    f32  ah,bh,screenvar,screenmin,screenfac=0.;
    f32  ah1,ah2,ah3,bh1,bh2,bh3;
    f32  al1,al2,al0,bl1,bl2,bl0;
    f32  tlow = 1.*MeV;
    f32  totMom;
    f32xyz GamDir;

    if ( tmin>=tmax )
    {
        return;
    }

    ind=RandomAtom ( cutEnergy,particles,id, materials, id_mat );


    Z3=powf ( materials.mixture[ind],1./3. );
    lnZ=3.*logf ( Z3 );
    FZ=lnZ* ( 4.-.55*lnZ );
    ZZ=powf ( materials.mixture[ind]* ( materials.mixture[ind]+1. ),1./3. );

    totalEnergy=particles.E[id]+electron_mass_c2;
    xmin=tmin/particles.E[id];
    xmax=tmax/particles.E[id];
    kappa=0.;
    if ( xmax>=1. )
        xmax=1.;
    else
        kappa=log ( xmax ) /log ( xmin );
    epsilmin=tmin/totalEnergy;
    epsilmax=tmax/totalEnergy;
    MigdalFactor=materials.nb_electrons_per_vol[id_mat]*MigdalConstant/ ( epsilmax*epsilmax );
    U=log ( particles.E[id]/electron_mass_c2 );
    U2=U*U;
//
    if ( particles.E[id]>tlow )
    {
        ah1=ah10+ZZ* ( ah11+ZZ*ah12 );
        ah2=ah20+ZZ* ( ah21+ZZ*ah22 );
        ah3=ah30+ZZ* ( ah31+ZZ*ah32 );
        bh1=bh10+ZZ* ( bh11+ZZ*bh12 );
        bh2=bh20+ZZ* ( bh21+ZZ*bh22 );
        bh3=bh30+ZZ* ( bh31+ZZ*bh32 );
        ah=1.+ ( ah1*U2+ah2*U+ah3 ) / ( U2*U );
        bh=.75+ ( bh1*U2+bh2*U+bh3 ) / ( U2*U );
        screenfac=136.*electron_mass_c2/ ( Z3*totalEnergy );
        screenmin=screenfac*epsilmin/ ( 1.-epsilmin );
        F1=fmaxf ( ScreenFunction1 ( screenmin )-FZ,0. );
        F2=fmaxf ( ScreenFunction2 ( screenmin )-FZ,0. );
        grejmax= ( F1-epsilmin* ( F1*ah-bh*epsilmin*F2 ) ) / ( 42.392-FZ );
    }
    else
    {
        al0=al00+ZZ* ( al01+ZZ*al02 );
        al1=al10+ZZ* ( al11+ZZ*al12 );
        al2=al20+ZZ* ( al21+ZZ*al22 );
        bl0=bl00+ZZ* ( bl01+ZZ*bl02 );
        bl1=bl10+ZZ* ( bl11+ZZ*bl12 );
        bl2=bl20+ZZ* ( bl21+ZZ*bl22 );
        ah=al0+al1*U+al2*U2;
        bh=bl0+bl1*U+bl2*U2;
        grejmax=fmaxf ( 1.+xmin* ( ah+bh*xmin ),1.+ah+bh );
        xm=-ah/ ( 2.*bh );
        if ( xmin<xm&&xm<xmax )
            grejmax=fmaxf ( grejmax,1.+xm* ( ah+bh*xm ) );
    }

    if ( particles.E[id]>tlow )
    {
        do
        {
            q=JKISS32 ( particles,id );
            x=powf ( xmin,q+kappa* ( 1.-q ) );
            epsil=x*particles.E[id]/totalEnergy;
            screenvar=screenfac*epsil/ ( 1.-epsil );
            F1=fmaxf ( ScreenFunction1 ( screenvar )-FZ,0. );
            F2=fmaxf ( ScreenFunction2 ( screenvar )-FZ,0. );
            migdal= ( 1.+MigdalFactor ) / ( 1.+MigdalFactor/ ( x*x ) );
            greject=migdal* ( F1-epsil* ( ah*F1-bh*epsil*F2 ) ) / ( 42.392-FZ );
        }
        while ( greject<JKISS32 ( particles,id ) *grejmax );
    }
    else
    {
        do
        {
            q=JKISS32 ( particles,id );
            x=powf ( xmin,q+kappa* ( 1.-q ) );
            migdal= ( 1.+MigdalFactor ) / ( 1.+MigdalFactor/ ( x*x ) );
            greject=migdal* ( 1.+x* ( ah+bh*x ) );
        }
        while ( greject<JKISS32 ( particles,id ) *grejmax );
    }
    gammaEnergy=x*particles.E[id];

    theta=AngleDistribution ( totalEnergy,totalEnergy-gammaEnergy,materials.mixture[ind],particles,id );
    sint=sin ( theta );
    phi=2.*pi*JKISS32 ( particles,id );
    GamDir.x=sint*cos ( phi );
    GamDir.y=sint*sin ( phi );
    GamDir.z=cos ( theta );
    f32xyz currentDir;
    currentDir = make_f32xyz ( particles.dx[id],particles.dy[id],particles.dz[id] );
    GamDir=rotateUz ( GamDir,currentDir );
    totMom=sqrtf ( particles.E[id]* ( totalEnergy+electron_mass_c2 ) );

    currentDir=CorrUnit ( currentDir,GamDir,totMom,gammaEnergy );

    particles.dx[id]=currentDir.x;
    particles.dy[id]=currentDir.y;
    particles.dz[id]=currentDir.z;

    particles.E[id]=particles.E[id]-gammaEnergy;

}
#endif