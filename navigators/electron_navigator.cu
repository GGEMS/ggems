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

__host__ __device__ f32 compute_lambda_for_scaled_energy( f32 CS, f32 e, ElectronsCrossSectionTable table, ui16 mat_id )
{

    f32 E_CS_max = table.eIonisation_E_CS_max[ mat_id ];



    if (e <= E_CS_max)
    {
        return CS;

    }
    else
    {
        f32 e1 = e*0.8f;  // lambdaFactor = 0.8

        if ( e1 > E_CS_max )
        {

            /// Get Lambda for scale Energy e1 ////////////////

            // Find energy index
            ui32 energy_index;
            if ( e1 <= table.E_min )
            {
                energy_index = 0;
            }
            else if ( e1 >= table.E_max )
            {
                energy_index = table.nb_bins-1;
            }
            else
            {
                energy_index = binary_search ( e1, table.E, table.nb_bins );
            }

            // Get absolute index table (considering mat id)
            ui32 table_index = mat_id*table.nb_bins + energy_index;

            f32 preStepLambda1;

            // Get CS for e1
            if ( energy_index == 0 )
            {
                preStepLambda1 = table.eIonisationCS[ table_index ];
            }
            else
            {
                preStepLambda1 = linear_interpolation ( table.E[ energy_index-1 ], table.eIonisationCS[ table_index-1 ],
                                                        table.E[ energy_index ], table.eIonisationCS[ table_index ], e1 );
            }

            /////////////////////////////////////////////////////

            if ( preStepLambda1 > CS )
            {
                CS = preStepLambda1;
            }
        }
        else
        {
            CS = table.eIonisation_CS_max[ mat_id ];  // fFactor = 1.0  - JB
        }
    }

    return CS;

}

__host__ __device__ void e_read_CS_table (
                                            ui16 mat, //material
                                            f32 energy, //energy of particle
                                            ElectronsCrossSectionTable d_table,
                                            ui8 &next_discrete_process, //next discrete process id
                                            ui32 &table_index,
                                            f32 & next_interaction_distance,
                                            f32 & dedxeIoni,
                                            f32 & dedxeBrem,
                                            f32 & erange,
                                            f32 & lambda,
                                            f32 randomnumbereBrem,
                                            f32 randomnumbereIoni,
                                            GlobalSimulationParametersData parameters )
{
    // Find energy index
    ui32 energy_index;
    if ( energy <= d_table.E_min )
    {
        energy_index = 0;       
    }
    else if ( energy >= d_table.E_max )
    {
        energy_index = d_table.nb_bins-1;       
    }
    else
    {
        energy_index = binary_search ( energy, d_table.E, d_table.nb_bins );        
    }    

#ifdef DEBUG
    assert( energy_index < d_table.nb_bins );
#endif

    // Get absolute index table (considering mat id)
    table_index = mat*d_table.nb_bins + energy_index;   

    // Vars
    f32 CS, interaction_distance;

    // Electron ionisation
    if ( parameters.physics_list[ELECTRON_IONISATION] == ENABLED )
    {

        // Get CS and dE/dx
        if ( energy_index==0 )
        {
            CS = d_table.eIonisationCS[ table_index ];
        }
        else
        {
            CS = linear_interpolation ( d_table.E[ energy_index-1 ], d_table.eIonisationCS[ table_index-1 ],
                                        d_table.E[ energy_index ], d_table.eIonisationCS[ table_index ], energy );

            // TODO, adding this correction create some looping on some particles (too small CS for small E) - JB
            //CS = compute_lambda_for_scaled_energy( CS, energy, d_table, mat );

            dedxeIoni = linear_interpolation ( d_table.E[ energy_index-1 ], d_table.eIonisationdedx[ table_index-1 ],
                                               d_table.E[ energy_index ], d_table.eIonisationdedx[ table_index ], energy );


        }

        // Get interaction distance
        if ( CS != 0.0 )
        {
            interaction_distance = randomnumbereIoni / CS;
        }
        else
        {
            interaction_distance = FLT_MAX;
        }

        if ( interaction_distance < next_interaction_distance )
        {
            next_interaction_distance = interaction_distance;
            next_discrete_process = ELECTRON_IONISATION;
        }


    } // eIoni

    // Bremsstrahlung
    if ( parameters.physics_list[ELECTRON_BREMSSTRAHLUNG] == ENABLED )
    {
        // Get CS and dE/dx
        if ( energy_index==0 )
        {
            CS = d_table.eBremCS[ table_index ];
        }
        else
        {
            CS = linear_interpolation ( d_table.E[ energy_index-1 ], d_table.eBremCS[ table_index-1 ],
                                        d_table.E[ energy_index ], d_table.eBremCS[ table_index ], energy );

            dedxeBrem = linear_interpolation ( d_table.E[ energy_index-1 ], d_table.eBremdedx[ table_index-1 ],
                                               d_table.E[ energy_index ], d_table.eBremdedx[ table_index ], energy );
        }

        // Get interaction distance
        if ( CS != 0.0 )
        {
            interaction_distance = randomnumbereBrem / CS;
        }
        else
        {
            interaction_distance = FLT_MAX;
        }

        if ( interaction_distance < next_interaction_distance )
        {
            next_interaction_distance = interaction_distance;
            next_discrete_process = ELECTRON_BREMSSTRAHLUNG;
        }
    } // eBrem

    // Multiple scattering
    if ( parameters.physics_list[ELECTRON_MSC] == ENABLED )
    {
        // Get CS
        if ( energy_index==0 )
        {
            lambda = d_table.eMSC[ table_index ];
        }
        else
        {
            lambda = linear_interpolation ( d_table.E[ energy_index-1 ], d_table.eMSC[ table_index-1 ],
                                            d_table.E[ energy_index ], d_table.eMSC[ table_index ], energy );
        }

        if ( lambda != 0.0 )
        {
            lambda = 1. / lambda;
        }
        else
        {
            lambda = FLT_MAX;
        }
    } // eMSC

    // Electron range
    if ( energy_index==0 )
    {
        erange = d_table.eRange[ table_index ];
    }
    else
    {
        erange = linear_interpolation ( d_table.E[ energy_index-1 ], d_table.eRange[ table_index-1 ],
                                        d_table.E[ energy_index ], d_table.eRange[ table_index ], energy );
    }    

}

__host__ __device__ f32 StepFunction ( f32 Range )
{
    //f32 alpha = 0.2f;
    //f32 rho = 1.*mm;
    f32  StepF;

    if ( Range < 1.0f ) return  Range;

    // StepF = alpha*Range + rho* ( 1.-alpha ) * ( 2.-rho/Range );
    StepF = 0.2f*Range + 1.0f* ( 1.-0.2f ) * ( 2.-1.0f/Range );

    if ( StepF < 1.0f ) StepF = 1.0f;

    return  StepF;
}

/* Not used anymore, insert directly into the code - JB
__host__ __device__ f32 LossApproximation ( f32 StepLength, f32 Ekine, f32 erange, ElectronsCrossSectionTable d_table, int mat, int id )
{
    f32  range,perteApp = 0;
    range=erange;
    range-=StepLength;
    
//     printf("StepLength %g erange %g Ekine %g\n",StepLength, erange, Ekine);
    
    if ( range >1.*nm )
        perteApp=GetEnergy ( range, d_table, mat );
    else
        perteApp = 0.;

    perteApp=Ekine-perteApp;

    return  perteApp;
}
*/


#define rate .55
#define fw 4.
#define nmaxCont 16
#define minLoss 10.*eV
__host__ __device__ f32 eFluctuation (f32 meanLoss, f32 Ekine, MaterialsTable materials, ParticlesData &particles, ui32 id, ui8 id_mat )
{

    f32 cutEnergy = materials.electron_energy_cut[ id_mat ];

    i32 nb, k;
    f32 LossFluct = 0., lossc = 0.;//,minLoss=10.*eV;
    // f32  rate=.55,fw=4.,nmaxCont=16.;
    f32 tau, /*gamma,*/ gamma2, beta2;
    f32 F1, F2, E0, E1, E2, E1Log, E2Log, I, ILog;
    f32 e1, e2, esmall, w, w1, w2, C, alfa, alfa1, namean;
    f32 a1 = 0., a2 = 0., a3 = 0., sa1;
    f32 emean = 0., sig2e = 0., sige = 0., p1 = 0., p2 = 0., p3 = 0.;
    f32 tmax = min( cutEnergy, 0.5f*Ekine );

    if ( meanLoss < minLoss ) return meanLoss;

    tau = Ekine / electron_mass_c2;
    //         gamma       *   gamma
    gamma2 = ( tau + 1.0 ) * ( tau + 1.0 );
    beta2 = tau * ( tau+2. ) / gamma2;

    F1 = materials.fF1[ id_mat ];
    F2 = materials.fF2[ id_mat ];
    E0 = materials.fEnergy0[ id_mat ];
    E1 = materials.fEnergy1[ id_mat ];
    E2 = materials.fEnergy2[ id_mat ];
    E1Log = materials.fLogEnergy1[ id_mat ];
    E2Log = materials.fLogEnergy2[ id_mat ];
    I = materials.electron_mean_excitation_energy[ id_mat ];
    ILog = logf ( I ); //materials.fLogMeanExcitationEnergy[id_mat];
    esmall = .5*sqrtf ( E0*I );




    if ( tmax <= E0 )
    {
        return meanLoss;
    }

    if ( tmax > I )
    {
        w2 = logf ( 2.*electron_mass_c2*beta2*gamma2 ) - beta2;
        if ( w2 > ILog )
        {
            C = meanLoss * ( 1.-rate ) / ( w2-ILog );
            a1 = C*F1* ( w2-E1Log ) /E1;
            if ( w2 > E2Log ) a2 = C*F2* ( w2-E2Log ) /E2;

            if ( a1 < nmaxCont )
            {
                sa1 = sqrtf ( a1 );
                if ( prng_uniform ( &(particles.prng[id]) ) < expf ( -sa1 ) )
                {
                    e1 = esmall;
                    a1 = meanLoss * ( 1.-rate ) /e1;
                    a2 = 0.;
                    e2 = E2;
                }
                else
                {
                    a1 = sa1 ;
                    e1 = sa1*E1;
                    e2 = E2;
                }
            }
            else
            {
                a1 /= fw;
                e1 = fw*E1;
                e2 = E2;
            }
        }
    }

    w1 = tmax/E0;
    if ( tmax > E0 ) a3 = rate*meanLoss* ( tmax-E0 ) / ( E0*tmax*logf ( w1 ) );

    if ( a1 > nmaxCont )
    {
        emean += ( a1*e1 );
        sig2e += ( a1*e1*e1 );
    }
    else if ( a1 > 0. )
    {
        p1 = ( f32 ) prng_poisson( &(particles.prng[id]), a1 );  /// HERE
        LossFluct += p1*e1;
        if ( p1 > 0. ) LossFluct += ( 1. - 2.* prng_uniform( &(particles.prng[id]) ) ) *e1;
    }

//#ifdef DEBUG_TRACK_ID
//        if ( id == DEBUG_TRACK_ID )
//        {
////            printf("Ekin=%e tmax= %e E0= %e I=%e\n  lossc %e a1 %e a2 %e a3 %e LossFluct %e\n \
////                      emean %e sige %e p1 %e e1 %e\n", meanLoss, tmax, E0, I,
////                   lossc, a1, a2, a3, LossFluct, emean, sige, p1, e1);

//            printf("a1 %e a2 %e a3 %e LossFluct %e rndpois %i rnduni %f\n", a1, a2, a3, LossFluct,
//                   prng_poisson( &(particles.prng[id]), 1 ),
//                   prng_uniform( &(particles.prng[id]) ));
//        }
//#endif


    if ( a2 > nmaxCont )
    {
        emean += ( a2*e2 );
        sig2e += ( a2*e2*e2 );
    }
    else if ( a2 > 0. )
    {
        p2 = ( f32 ) prng_poisson( &(particles.prng[id]), a2 );
        LossFluct += ( p2*e2 );
        if ( p2 > 0. ) LossFluct += ( 1. - 2.*prng_uniform( &(particles.prng[id]) ) ) *e2;
    }




    if ( a3 > 0. )
    {
        p3 = a3;
        alfa = 1.0;

        if ( a3 > nmaxCont )
        {
            alfa = w1* ( nmaxCont+a3 ) / ( w1*nmaxCont+a3 );
            alfa1 = alfa*logf ( alfa ) / ( alfa-1. );
            namean = a3*w1* ( alfa-1. ) / ( ( w1-1. ) *alfa );
            emean += namean*E0*alfa1;
            sig2e += E0*E0*namean* ( alfa-alfa1*alfa1 );
            p3 = a3-namean;
        }
        w2 = alfa*E0;
        w = ( tmax-w2 ) /tmax;
        nb = prng_poisson( &(particles.prng[id]), p3 );

        if ( nb>0 )
        {
            for ( k=0; k<nb; k++ )
            {
                lossc += w2/ ( 1.-w*prng_uniform( &(particles.prng[id]) ) );
            }
        }
    }

    if ( emean > 0. )
    {
        sige = sqrtf ( sig2e );
        LossFluct += max ( 0., Gaussian ( emean, sige, particles, id ) );
    }



    LossFluct += lossc;

    return  LossFluct;
}
#undef rate
#undef fw
#undef nmaxCont
#undef minLoss


__host__ __device__ f32 eLoss ( f32 LossLength, f32 Ekine, f32 dedxeIoni, f32 dedxeBrem, f32 erange,
                                ElectronsCrossSectionTable d_table, ui8 mat, MaterialsTable materials,
                                ParticlesData &particles, ui32 id )
{    
    // DEBUG
    //LossLength = 0.09;

    f32 perteTot = LossLength * ( dedxeIoni + dedxeBrem );   

//#ifdef DEBUG_TRACK_ID
//        if ( id == DEBUG_TRACK_ID )
//        {
//            printf("ID %i dedx= %e  Ekin= %e  erange= %e  eLoss= %e length= %e\n",
//                   id, dedxeIoni + dedxeBrem, Ekine, erange, perteTot, LossLength);
//        }
//#endif


    // Long step
    if ( perteTot > Ekine * 0.01 ) // linLossLimit = 0.01
    {
        // Here, I directly insert the LossApproximation function into the code - JB
        perteTot = 0.0;
        erange -= LossLength; // / reduceFactor (reduceFactor=1);

#ifdef DEBUG
    assert( erange >= 0.0f );
#endif

        perteTot = GetEnergy( erange, d_table, mat);
        perteTot = Ekine - perteTot;

//#ifdef DEBUG_TRACK_ID
//        if ( id == DEBUG_TRACK_ID )
//        {


//            printf("ID %i Ekin= %e  erange= %e  GetE= %e Pertot=%e eFluc=%e\n",
//                   id, Ekine, erange, GetEnergy( erange, d_table, mat), perteTot,
//                   eFluctuation ( perteTot, Ekine, materials, particles, id, mat ));
//        }
//#endif

        //printf("   long step: EkinforLoss= %e  ::: eLoss= %e\n", GetEnergy( erange, d_table, mat), perteTot);
    }

    /// Warning ADD for eFluctuation
    if ( dedxeIoni > 0. ) {
        perteTot = eFluctuation ( perteTot, Ekine, materials, particles, id, mat );


//        #ifdef DEBUG_TRACK_ID
//                if ( id == DEBUG_TRACK_ID )
//                {


//                    printf("rndpois %i rnduni %f\n",
//                           prng_poisson( &(particles.prng[id]), 1 ),
//                           prng_uniform( &(particles.prng[id]) ));
//                }
//        #endif




        //printf("   Fluc ::: eloss= %e\n", perteTot);
    }
/*
    if ( ( Ekine-perteTot ) <= ( 1.*eV ) )
    {
        perteTot = Ekine;
    }
*/
    perteTot = fminf( Ekine, perteTot );

    particles.E[ id ] -= perteTot;

    //printf("eloss= %e  Ekin= %e\n", perteTot, particles.E[ id ]);

    return  perteTot;
}


#define tausmall 1.E-16
__host__ __device__ f32 gGeomLengthLimit ( f32 gPath, f32 currentLambda, f32 currentRange, f32 par1, f32 par3 )
{
    f32  tPath;
    //f32  tausmall=1.E-16; //tausmall=1.E-16;

    par3 = 1. + par3;
    tPath = gPath;

    if ( gPath > currentLambda*tausmall )
    {
        if ( par1 < 0. )
        {
            tPath = -currentLambda*log ( 1. - gPath/currentLambda );
        }
        else
        {
            if ( par1*par3*gPath < 1.0 )
            {
                tPath = ( 1.-exp ( log ( 1.-par1*par3*gPath ) /par3 ) ) /par1;
            }
            else
            {
                tPath = currentRange;
            }
        }
    }

    if ( tPath < gPath ) tPath = gPath;

    return  tPath;
}
#undef tausmall


__host__ __device__ f32 eSimpleScattering ( f32 xmeanth, f32 x2meanth, ui32 id, ParticlesData &particles )
{
    f32 a = ( 2.*xmeanth + 9.*x2meanth - 3. ) / ( 2.*xmeanth - 3.*x2meanth + 1. );
    f32 prob = ( a + 2. ) * xmeanth/a;

    if ( prng_uniform( &(particles.prng[id]) ) < prob )
    {
        return -1. + 2.*expf ( logf ( prng_uniform( &(particles.prng[id]) ) ) / ( a + 1. ) );
    }
    else
    {
        return -1. + 2.*prng_uniform( &(particles.prng[id]) );
    }

}



__host__ __device__ f32 eCosineTheta (f32 trueStep, f32 currentRange, f32 currentLambda, f32 currentEnergy, f32 *currentTau,
                                       f32 par1, f32 par2, MaterialsTable materials, ui8 id_mat, ui32 id, ParticlesData &particles )
{
    f32 particleEnergy = particles.E[id];
    f32  costh,sinth;
    f32  tau;
    const f32 taubig=8., tausmall=1.E-16, taulim=1.E-6;
    f32  c, c1, x0, b, bx, b1, ebx, eb1;
    f32  prob = 0., qprob = 1.;
    f32  a = 1., ea = 0., eaa = 1.;
    f32  xmeanth, xmean1 = 1., xmean2 = 0., x2meanth;
    f32  dtrl = 0.05;//5./100.;
    f32  xsi = 3.;
    f32  theta0, theta0max = pi/6., y, corr, betacp, c_highland = 13.6*MeV;
    f32  f1x0, f2x0;

    costh = 1.;
    tau = trueStep/currentLambda;

    if ( trueStep >= currentRange*dtrl )
    {
        if ( ( par1*trueStep ) <1. )
        {
            tau = -par2*logf ( 1. - par1*trueStep );
        }
        else if ( ( 1. - particleEnergy/currentEnergy ) > taulim )
        {
            tau = taubig;
        }
    }

    ( *currentTau ) = tau;

    if ( tau >= taubig )
    {
        costh = -1. + 2. * prng_uniform( &(particles.prng[id]) );
    }
    else if ( tau >= tausmall )
    {
        x0 = 1.;
        b = 2.;
        b1 = 3.;
        bx = 1.;
        eb1 = 3.;
        ebx = 1.;
        prob = 1.;
        qprob = 1.;
        xmeanth = expf ( -tau );
        x2meanth = ( 1. + 2.*expf ( -2.5*tau ) ) /3.;

        if ( 1. - particleEnergy/currentEnergy >.5 )
        {
            return eSimpleScattering ( xmeanth, x2meanth, id, particles );
        }

        betacp = sqrtf ( currentEnergy * ( currentEnergy + 2.*electron_mass_c2 )
                        * particleEnergy * ( particleEnergy + 2.*electron_mass_c2 )
                       / ( ( currentEnergy + electron_mass_c2 ) * ( particleEnergy + electron_mass_c2 ) ) );

        y = trueStep / materials.rad_length[ id_mat ];
        theta0 = c_highland*sqrtf ( y ) / betacp;
        y = logf ( y );

        f32 Zeff = materials.nb_electrons_per_vol[ id_mat ] / materials.nb_atoms_per_vol[ id_mat ];

        corr = ( 1.-8.778E-2 / Zeff ) * ( .87 + .03*logf ( Zeff ) )
              + ( 4.078E-2 + 1.7315E-4*Zeff ) * ( .87 + .03*logf ( Zeff ) ) *y;
        theta0 *= corr ;

        if ( theta0*theta0 < tausmall )
        {
            return costh;
        }
        if ( theta0 > theta0max )
        {            
            return  eSimpleScattering ( xmeanth, x2meanth, id, particles );
        }

        sinth = sinf ( .5*theta0 );
        a = .25/ ( sinth*sinth );
        ea = expf ( -xsi );
        eaa = 1. - ea ;
        xmean1 = 1. - ( 1. - ( 1. + xsi ) * ea ) / ( a*eaa );
        x0 = 1. - xsi/a;
        if ( xmean1 <= .999*xmeanth )
        {
            return  eSimpleScattering ( xmeanth, x2meanth, id, particles );
        }

        c = 2.943 - .197*logf ( Zeff + 1. ) + ( .0987 - .0143*logf ( Zeff + 1. ) ) * y;

        if ( fabsf ( c-3. ) <.001 ) c = 3.001;
        if ( fabsf ( c-2. ) <.001 ) c = 2.001;
        if ( fabsf ( c-1. ) <.001 ) c = 1.001;
        c1 = c - 1.;

        b = 1. + ( c - xsi ) /a;
        b1 = b + 1.;
        bx = c/a;
        eb1 = expf ( c1*logf ( b1 ) );
        ebx = expf ( c1*logf ( bx ) );
        xmean2 = ( x0*eb1 + ebx - ( eb1*bx - b1*ebx ) / ( c - 2. ) ) / ( eb1 - ebx );
        f1x0 = a*ea/eaa;
        f2x0 = c1*eb1 / ( bx * ( eb1 - ebx ) );
        prob = f2x0 / ( f1x0 + f2x0 );
        qprob = xmeanth/ ( prob*xmean1 + ( 1. - prob ) * xmean2 );

        if ( prng_uniform( &(particles.prng[id]) ) < qprob )
        {
            if ( prng_uniform( &(particles.prng[id]) ) < prob )
                costh = 1. + logf ( ea + prng_uniform( &(particles.prng[id]) ) * eaa ) / a;
            else
                costh = b - b1*bx / expf ( logf ( ebx + ( eb1 - ebx ) * prng_uniform( &(particles.prng[id]) ) ) /c1 );
        }
        else
            costh = -1. + 2.*prng_uniform( &(particles.prng[id]) );
    }

    return  costh;
}



__host__ __device__ void gLatCorrection ( f32xyz currentDir, f32 tPath, f32 zPath, f32 currentTau, f32 phi, f32 sinth,
                                          ParticlesData &particles, ui32 id, f32 safety )
{
    f32 latcorr, etau, rmean, rmax, Phi, psi, lambdaeff;
    const f32 kappa = 2.5, tlimitminfix = 1.E-6*mm, taulim = 1.E-6, tausmall = 1.E-16, taubig = 8., geomMin = 1.E-6*mm;

    // struct  Vector  latDir;
    f32xyz latDir;
    lambdaeff = tPath/currentTau;

    if ( safety > tlimitminfix ) // Safety is distance to near voxel
    {
        rmean = 0.;

        if ( ( currentTau >= tausmall ) )
        {
            if ( currentTau < taulim )
            {
                rmean = kappa*currentTau*currentTau*currentTau* ( 1. - ( kappa+1. ) *currentTau*.25 ) / 6.; //powf(currentTau,3.)
            }
            else
            {
                etau = 0.;
                if ( currentTau < taubig )
                    etau = expf ( -currentTau );
                rmean = -kappa*currentTau;
                rmean = -expf ( rmean ) / ( kappa* ( kappa - 1. ) );
                rmean += currentTau - ( kappa + 1. ) / kappa + kappa*etau / ( kappa - 1. );
            }
            if ( rmean > 0. )
            {
                rmean = 2.*lambdaeff*sqrtf ( rmean/3. );
            }
            else
            {
                rmean = 0.;
            }
        }

        rmax = ( tPath - zPath ) * ( tPath + zPath );

        if ( rmax < 0. )
        {
            rmax = 0.;
        }
        else
        {
            rmax = sqrtf ( rmax );
        }

        if ( rmean >= rmax ) rmean = rmax;

        if ( rmean <= geomMin ) return;

        if ( rmean > 0. )
        {
            if ( ( currentTau >= tausmall ) )
            {
                if ( currentTau < taulim )
                {
                    latcorr = lambdaeff*kappa*currentTau*currentTau* ( 1. - ( kappa + 1. ) * currentTau/3. ) /3.;
                }
                else
                {
                    etau = 0.;
                    if ( currentTau < taubig ) etau = expf ( -currentTau );
                    latcorr = -kappa*currentTau;
                    latcorr = expf ( latcorr ) / ( kappa - 1. );
                    latcorr += 1. - kappa*etau/ ( kappa - 1. );
                    latcorr *= 2.*lambdaeff /3.;
                }
            }

            if ( latcorr > rmean )
            {
                latcorr = rmean;
            }
            else if ( latcorr < 0. )
            {
                latcorr = 0.;
            }

            Phi = 0.;
            if ( fabsf ( rmean*sinth ) <= latcorr )
            {
                Phi = gpu_twopi * prng_uniform( &(particles.prng[id]) );
            }
            else
            {
                psi = acosf ( latcorr/ ( rmean*sinth ) );
                if ( prng_uniform( &(particles.prng[id]) ) < .5 )
                {
                    Phi = phi + psi;
                }
                else
                {
                    Phi = phi - psi;
                }
            }

            latDir.x = cos ( Phi );
            latDir.y = sin ( Phi );
            latDir.z = 0.;
            latDir = rotateUz ( latDir, currentDir );

            if ( rmean > safety ) rmean = safety*.99;

            particles.px[ id ] += latDir.x*rmean;
            particles.py[ id ] += latDir.y*rmean;
            particles.pz[ id ] += latDir.z*rmean;
        }

    } // if safety


}


__host__ __device__ void eMscScattering ( f32 tPath, f32 zPath, f32 currentRange, f32 currentLambda,
                                          f32 currentEnergy, f32 par1, f32 par2, ParticlesData &particles,
                                          ui32 id, MaterialsTable materials, ui8 mat, VoxVolumeData phantom, ui32xyzw index_phantom )
{
    f32  costh, sinth, phi, currentTau;
    const f32 tlimitminfix = 1.E-10*mm, tausmall = 1.E-16; //,taulim=1.E-6
    f32xyz Dir, currentDir;

    if ( ( particles.E[id] < 0. ) || ( tPath <= tlimitminfix ) || ( tPath/tausmall < currentLambda ) )
    {
        return;
    }

    costh = eCosineTheta ( tPath, currentRange, currentLambda, currentEnergy, &currentTau, par1, par2, materials, mat, id,particles );

    if ( fabs ( costh ) > 1. )
    {
        return;
    }
    if ( costh < ( 1. - 1000.*tPath/currentLambda ) && ( particles.E[id] ) > ( 20.*MeV ) )
    {
        do
        {
            costh = 1. + 2.*logf ( prng_uniform( &(particles.prng[id]) ) ) * tPath/currentLambda;
        }
        while ( ( costh < -1. ) );
    }

    sinth = sqrtf ( ( 1. - costh ) * ( 1. + costh ) );
    phi = gpu_twopi * prng_uniform( &(particles.prng[id]) );

    Dir = make_f32xyz ( sinth*cosf ( phi ), sinth*sinf ( phi ), costh );

    particles.px[ id ] += particles.dx[ id ] * zPath;
    particles.py[ id ] += particles.dy[ id ] * zPath;
    particles.pz[ id ] += particles.dz[ id ] * zPath;

    currentDir = make_f32xyz ( particles.dx[ id ], particles.dy[ id ], particles.dz[ id ] );

    Dir = rotateUz ( Dir,currentDir );

    particles.dx[ id ] = Dir.x;
    particles.dy[ id ] = Dir.y;
    particles.dz[ id ] = Dir.z;

    // Read position
    f32xyz position; // mm
    position.x = particles.px[id];
    position.y = particles.py[id];
    position.z = particles.pz[id];

    // get voxel params
    f32 vox_xmin = index_phantom.x*phantom.spacing_x + phantom.off_x;
    f32 vox_ymin = index_phantom.y*phantom.spacing_y + phantom.off_y;
    f32 vox_zmin = index_phantom.z*phantom.spacing_z + phantom.off_z;
    f32 vox_xmax = vox_xmin + phantom.spacing_x;
    f32 vox_ymax = vox_ymin + phantom.spacing_y;
    f32 vox_zmax = vox_zmin + phantom.spacing_z;

    // Get safety within the voxel
    f32 safety = transport_compute_safety_AABB( position, vox_xmin, vox_xmax, vox_ymin, vox_ymax,
                                                vox_zmin, vox_zmax );

    // Lateral correction
    gLatCorrection ( currentDir, tPath, zPath, currentTau, phi, sinth, particles, id, safety );

}


// From Eric's code
__host__ __device__ f32 GlobalMscScattering ( f32 GeomPath,f32 cutstep,f32 CurrentRange,f32 CurrentEnergy, f32 CurrentLambda,
                                              f32 dedxeIoni, f32 dedxeBrem, ElectronsCrossSectionTable d_table, ui8 mat,
                                              ParticlesData &particles, ui32 id,f32 par1,f32 par2, MaterialsTable materials,
                                              DoseData &dosi, ui32xyzw index_phantom, VoxVolumeData phantom,
                                              GlobalSimulationParametersData parameters )
{

    f32  edep, TruePath, zPath;//,tausmall=1.E-16;

    if ( parameters.physics_list[ELECTRON_MSC] != ENABLED )
    {

        particles.px[id] += particles.dx[id] * GeomPath;
        particles.py[id] += particles.dy[id] * GeomPath;
        particles.pz[id] += particles.dz[id] * GeomPath;

        if ( GeomPath < cutstep )
        {
            edep = eLoss ( GeomPath, particles.E[ id ], dedxeIoni, dedxeBrem, CurrentRange, d_table,
                           mat, materials, particles, id );

            // Drop dose
            dose_record_standard ( dosi, edep, particles.px[id], particles.py[id], particles.pz[id] );
            //printf("Edep %e  - pos %e %e %e - MscProc\n", edep, particles.px[id], particles.py[id], particles.pz[id]);

        }

        return  GeomPath;
    }

    if ( GeomPath == cutstep )
    {
        zPath = gTransformToGeom ( GeomPath, CurrentRange, CurrentLambda,
                                   CurrentEnergy, par1, par2, d_table, mat );
    }
    else
    {
        zPath = GeomPath;
        TruePath = gGeomLengthLimit ( GeomPath, CurrentLambda, CurrentRange, par1, par2 );
        GeomPath = TruePath;

        edep = eLoss ( TruePath, particles.E[ id ], dedxeIoni, dedxeBrem, CurrentRange,
                       d_table, mat, materials, particles, id );

        dose_record_standard ( dosi, edep, particles.px[id], particles.py[id], particles.pz[id] );
    }


    if ( particles.E[id] > 0.0 ) // if not laststep
    {
        eMscScattering ( GeomPath, zPath, CurrentRange, CurrentLambda, CurrentEnergy, par1, par2, particles,
                         id, materials, mat, phantom, index_phantom );
    }
    else
    {
        particles.endsimu[ id ] = PARTICLE_DEAD;
        particles.px[ id ] += particles.dx[ id ]*zPath;
        particles.py[ id ] += particles.dy[ id ]*zPath;
        particles.pz[ id ] += particles.dz[ id ]*zPath;
    }

    return  TruePath;
}


__host__ __device__ SecParticle eSampleSecondarieElectron ( f32 CutEnergy, ParticlesData &particles, ui32 id )
{
    f32  totalEnergy, deltaEnergy, totMom, deltaMom;
    f32  xmin, xmax, gamma; //, gamma2;//,beta2;
    f32  x, z, q, grej, g, y;
    f32  cost, sint, phi;
    f32  Ekine = particles.E[ id ];
    f32  tmax = fmin ( 1.*GeV, .5 * Ekine );

    f32xyz  ElecDir;

    SecParticle secondary_part;
    secondary_part.E = 0.;
    secondary_part.dir = make_f32xyz ( 0.,0.,0. );
    secondary_part.pname = ELECTRON;
    secondary_part.endsimu = PARTICLE_DEAD;

    if ( CutEnergy >= tmax ) return secondary_part;

    totalEnergy = Ekine + electron_mass_c2;
    totMom = sqrtf ( Ekine * ( totalEnergy + electron_mass_c2 ) );
    xmin = CutEnergy / Ekine;
    xmax = tmax / Ekine;
    gamma = totalEnergy/electron_mass_c2;
    // beta2=1.-1./gamma2;
    // gamma2 = gamma*gamma
    g = ( 2.*gamma - 1. ) / ( gamma*gamma );
    y = 1. - xmax;
    grej = 1. - g*xmax + xmax*xmax* ( 1. - g + ( 1. - g*y ) / ( y*y ) );

    do
    {
        q = prng_uniform( &(particles.prng[id]) );
        x = xmin*xmax/ ( xmin* ( 1. - q ) + xmax*q );
        y = 1. - x;
        z = 1. - g*x + x*x* ( 1. - g + ( 1. - g*y ) / ( y*y ) );
    }
    while ( ( grej*prng_uniform( &(particles.prng[id]) ) > z ) );

    deltaEnergy = x*Ekine;
    deltaMom = sqrtf ( deltaEnergy * ( deltaEnergy + 2.*electron_mass_c2 ) );
    cost = deltaEnergy * ( totalEnergy + electron_mass_c2 ) / ( deltaMom*totMom );
    sint = 1. - cost*cost;
    if ( sint > 0. ) sint = sqrtf ( sint );
    phi = gpu_twopi * prng_uniform( &(particles.prng[id]) );

    ElecDir.x = sint*cosf ( phi );
    ElecDir.y = sint*sinf ( phi );
    ElecDir.z = cost;

    f32xyz currentDir = make_f32xyz ( particles.dx[id], particles.dy[id], particles.dz[id] );

    ElecDir = rotateUz ( ElecDir, currentDir );

    particles.E[id]= Ekine - deltaEnergy;

    //printf("Ekin= %e   deltaEnergy= %e\n", Ekine, deltaEnergy);

    if ( particles.E[id] > 0.0 ) currentDir = CorrUnit ( currentDir, ElecDir, totMom, deltaMom );

    particles.dx[id] = currentDir.x;
    particles.dy[id] = currentDir.y;
    particles.dz[id] = currentDir.z;

    // SecParticle secondary_part;
    secondary_part.E = deltaEnergy;
    secondary_part.dir = ElecDir;
    secondary_part.pname = ELECTRON;
    secondary_part.endsimu = PARTICLE_ALIVE;

    return secondary_part;
}


__host__ __device__ f32xyz CorrUnit ( f32xyz u, f32xyz v, f32 uMom, f32 vMom )
{
    f32 r;
    f32xyz Final;

    Final.x = u.x*uMom - v.x*vMom;
    Final.y = u.y*uMom - v.y*vMom;
    Final.z = u.z*uMom - v.z*vMom;
    r = Final.x*Final.x + Final.y*Final.y + Final.z*Final.z;
    if ( r > 0. )
    {
        r = sqrt ( Final.x*Final.x + Final.y*Final.y + Final.z*Final.z );
        Final.x = Final.x/r;
        Final.y = Final.y/r;
        Final.z = Final.z/r;
    }

    return  Final;
}


#define tausmall 1.E-20
#define taulim 1.E-6
#define tlimitminfix 1.E-6*mm
#define dtrl 5./100
__host__ __device__ f32 gTransformToGeom (f32 TPath, f32 currentRange, f32 currentLambda, f32 currentEnergy,
                                          f32 &par1, f32 &par2, ElectronsCrossSectionTable electron_CS_table, ui8 mat )
{
    f32  ZPath, zmean;
//     f32  tausmall=1.E-20,taulim=1.E-6,tlimitminfix=1.E-6*mm;
//     f32  dtrl=5./100.;
    f32  tau, t1, lambda1;
    f32  par3;

    par1 = -1.0f;
    par2 = 0.0f;
    par3 = 0.0f;
    ZPath = TPath;
    if ( TPath < tlimitminfix ) return ZPath;
    if ( TPath > currentRange ) TPath = currentRange;

    tau= TPath / currentLambda;
    if ( tau <= tausmall ) /*||insideskin*/
    {
        ZPath = TPath;
        if ( ZPath > currentLambda ) ZPath = currentLambda;
        return  ZPath;
    }

    zmean = TPath;
    if ( TPath < currentRange*dtrl )
    {
        if ( tau < taulim )
        {
            zmean = TPath* ( 1.-0.5*tau );
        }
        else
        {
            zmean = currentLambda * ( 1.-expf ( -tau ) );
        }
    }
    else if ( currentEnergy < electron_mass_c2 )
    {
        par1 = 1. / currentRange;
        par2 = 1. / ( par1*currentLambda );
        par3 = 1. + par2;

        if ( TPath < currentRange )
        {
            zmean = ( 1.-expf ( par3*logf ( 1.-TPath/currentRange ) ) ) / ( par1*par3 );
        }
        else
        {
            zmean = 1./ ( par1*par3 );
        }
    }
    else
    {
        t1 = GetEnergy ( currentRange-TPath, electron_CS_table, mat );
        lambda1 = 1. / GetLambda ( t1, electron_CS_table, mat );
        par1 = ( currentLambda-lambda1 ) / ( currentLambda*TPath );
        par2 = 1./ ( par1*currentLambda );
        par3 = 1. + par2;

        zmean = ( 1.-expf ( par3*logf ( lambda1/currentLambda ) ) ) / ( par1*par3 );
    }
    ZPath = zmean;

    //     return (fminf(ZPath,currentLambda));
    if ( ZPath > currentLambda ) ZPath = currentLambda;

    return ZPath;
}
#undef tausmall
#undef taulim
#undef tlimitminfix
#undef dtrl

__host__ __device__ f32 GetEnergy ( f32 Range, ElectronsCrossSectionTable d_table, ui8 mat )
{
    ui32 index = binary_search ( Range, d_table.eRange, d_table.nb_bins*mat+d_table.nb_bins, d_table.nb_bins*mat );
    ui32 E_index = index - d_table.nb_bins*mat;

    if ( E_index == 0 )
    {
        return d_table.E[ E_index ];
    }
    else
    {
        return linear_interpolation ( d_table.eRange[ index-1 ], d_table.E[ E_index-1 ],
                                      d_table.eRange[ index ], d_table.E[ E_index ], Range );
    }
}

// Get Lambda with flag 1, in the code only flag 1 is used, so I fixed the function to flag 1 - JB
__host__ __device__ f32 GetLambda ( f32 energy, ElectronsCrossSectionTable d_table, ui8 mat )
{
    ui32 E_index = binary_search ( energy, d_table.E, d_table.nb_bins );
    ui32 index = d_table.nb_bins*mat + E_index;

    if ( E_index == 0)
    {
        return d_table.eMSC[ index ];
    }
    else
    {
        return linear_interpolation ( d_table.E[ E_index-1 ], d_table.eMSC[ index-1 ],
                                      d_table.E[ E_index ], d_table.eMSC[ index ], energy );
    }
}

//// Old GetLambda function
//__host__ __device__ f32 GetLambda ( f32 Range, unsigned short int flag, ElectronsCrossSectionTable d_table, int mat )
//{
//    int index = binary_search ( Range, d_table.E, d_table.nb_bins*mat+d_table.nb_bins, d_table.nb_bins*mat );

//    if ( flag == 1 ) return linear_interpolation ( d_table.E[index-1],d_table.eMSC[index-1], d_table.E[index], d_table.eMSC[index], Range );

//    else if ( flag == 2 ) return linear_interpolation ( d_table.E[index-1],d_table.eIonisationCS[index-1], d_table.E[index], d_table.eIonisationCS[index], Range );

//    else /*if (flag == 3)*/ return linear_interpolation ( d_table.E[index-1],d_table.eBremCS[index-1], d_table.E[index], d_table.eBremCS[index], Range );

//}
//// Old RandomAtom function
/*
__host__ __device__ int RandomAtom ( f32 CutEnergyGamma, ParticlesData &particles, ui32 id, MaterialsTable materials, ui8 id_mat )
{                 
    ui32 indice, last_index;
    f32 rval;

    last_index = materials.index[ id_mat ] + materials.nb_elements[ id_mat ] - 1;

    rval = JKISS32 ( particles, id );
    rval *= materials.atom_num_dens[ id_mat ];
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
*/

// The old function from GGEMS V1 was wrong, I extracted this one from G4.10  - JB
__host__ __device__ ui16 RandomAtom ( f32 CutEnergyGamma, f32 min_E, ParticlesData &particles, ui32 id,
                                      MaterialsTable materials, ui8 mat_id )
{

    ui32 n = materials.nb_elements[ mat_id ] - 1;
    ui32 index = materials.index[ mat_id ];
    ui16 cur_Z = materials.mixture[ index + n ];
    f32 Ekine = particles.E[ id ];

    f32 x = prng_uniform( &(particles.prng[id]) ) * ElectronBremmsstrahlung_CS( materials, Ekine, min_E, mat_id );

    f32 xsec = 0.0;
    for ( ui16 i = 0; i < n; ++i )
    {
        xsec += materials.atom_num_dens[ index + i ] *
                ElectronBremmsstrahlung_CSPA( materials.mixture[ index + i ], CutEnergyGamma, Ekine );

        if ( x <= xsec )
        {
            cur_Z = materials.mixture[ index + i ];
            break;
        }
    }

    return cur_Z;
}

__host__ __device__ f32 ScreenFunction1 ( f32 ScreenVariable )
{
    f32 screenVal;
    if ( ScreenVariable > 1. )
    {
        screenVal = 42.24 - 8.368*logf ( ScreenVariable + .952 );
    }
    else
    {
        screenVal = 42.392 - ScreenVariable * ( 7.796 - 1.961*ScreenVariable );
    }
    return  screenVal;
}

__host__ __device__ f32 ScreenFunction2 ( f32 ScreenVariable )
{
    f32 screenVal;
    if ( ScreenVariable > 1. )
    {
        screenVal = 42.24 - 8.368*logf ( ScreenVariable + .952 );
    }
    else
    {
        screenVal = 41.734 - ScreenVariable * ( 6.484 - 1.25*ScreenVariable );
    }
    return  screenVal;
}

__host__ __device__ f32 RejectionFunction ( f32 value, f32 rej1, f32 rej2, f32 rej3, f32 ratio, f32 z )
{
    f32  argument = ( 1. + value ) * ( 1. + value );
    return ( 4. + logf ( rej3 + ( z/argument ) ) ) * ( ( 4.*ratio*value/argument ) - rej1 ) + rej2;
}


__host__ __device__ f32 AngleDistribution ( f32 initial_energy, f32 final_energy, f32 Z, ParticlesData &particles, ui32 id )
{
    f32  initialTotalEnergy = ( initial_energy + electron_mass_c2 ) /electron_mass_c2;
    f32  finalTotalEnergy = ( final_energy + electron_mass_c2 ) /electron_mass_c2;
    f32  EnergyRatio = finalTotalEnergy / initialTotalEnergy;
    f32  gMaxEnergy = ( gpu_pi *initialTotalEnergy ) * ( gpu_pi *initialTotalEnergy );
    f32  z, rejection_argument1, rejection_argument2, rejection_argument3;
    f32  gfunction0, gfunction1, gfunctionEmax, gMaximum;
    f32  rand, gfunctionTest, randTest;

    z = .00008116224* ( powf ( Z, 1./3. ) + powf ( Z + 1, 1./3. ) );
    rejection_argument1 = ( 1. + EnergyRatio*EnergyRatio );
    rejection_argument2 = -2.*EnergyRatio + 3.*rejection_argument1;
    rejection_argument3 = ( ( 1. - EnergyRatio ) / ( 2.*initialTotalEnergy*EnergyRatio ) ) *
                          ( ( 1. - EnergyRatio ) / ( 2.*initialTotalEnergy*EnergyRatio ) );
    gfunction0 = RejectionFunction ( 0., rejection_argument1, rejection_argument2, rejection_argument3, EnergyRatio, z );
    gfunction1 = RejectionFunction ( 1., rejection_argument1, rejection_argument2, rejection_argument3, EnergyRatio, z );
    gfunctionEmax = RejectionFunction ( gMaxEnergy, rejection_argument1, rejection_argument2, rejection_argument3, EnergyRatio, z );
    gMaximum = fmax ( gfunction0, gfunction1 );
    gMaximum = fmax ( gMaximum, gfunctionEmax );

    do
    {
        rand = prng_uniform( &(particles.prng[id]) );
        rand /= ( 1. - rand + 1./gMaxEnergy );
        gfunctionTest = RejectionFunction ( rand, rejection_argument1, rejection_argument2, rejection_argument3, EnergyRatio, z );
        randTest = prng_uniform( &(particles.prng[id]) );
    }
    while ( randTest*gMaximum > gfunctionTest );

    // return Theta
    return sqrtf ( rand ) / initialTotalEnergy;
}



__host__ __device__
void eSampleSecondarieGamma ( f32 minEnergy, f32 maxEnergy, ParticlesData &particles, ui32 id, MaterialsTable materials, ui8 id_mat )
{

    // DEBUG
    //printf("eBrem sample second\n");

    ui32 ind;
    f32  gammaEnergy, totalEnergy;
    f32  xmin, xmax, kappa, epsilmin, epsilmax;
    f32  lnZ, FZ, Z3, ZZ, F1, F2, theta, sint, phi;
    f32  Ekine = particles.E[ id ];
    f32  tmin = materials.photon_energy_cut[ id_mat ];
    f32  tmax = fminf ( maxEnergy, Ekine ); // MaxKinEnergy = 250* MeV
    f32  MigdalFactor, MigdalConstant = elec_radius*hbarc*hbarc*4.*pi/ ( electron_mass_c2*electron_mass_c2 );
    f32  x, xm, epsil, greject, migdal, grejmax, q, U, U2;
    f32  ah, bh, screenvar, screenmin, screenfac = 0.;
    f32  ah1, ah2, ah3, bh1, bh2, bh3;
    f32  al1, al2, al0, bl1, bl2, bl0;
    f32  tlow = 1.*MeV;
    f32  totMom;

    if ( tmin >= tmax ) return;

    ind = materials.index[ id_mat ] + RandomAtom ( tmin, minEnergy, particles, id, materials, id_mat );

    Z3 = powf ( materials.mixture[ ind ], 1./3. );
    lnZ = 3.*logf ( Z3 );
    FZ = lnZ* ( 4. - .55*lnZ );
    ZZ = powf ( materials.mixture[ ind ] * ( materials.mixture[ ind ] + 1. ),1./3. );

    totalEnergy = Ekine + electron_mass_c2;
    xmin = tmin / Ekine;
    xmax = tmax / Ekine;
    kappa = 0.;

    if ( xmax >= 1. )
    {
        xmax = 1.;
    }
    else
    {
        kappa = logf ( xmax ) / logf ( xmin );
    }

    epsilmin = tmin / totalEnergy;
    epsilmax = tmax / totalEnergy;
    MigdalFactor = materials.nb_electrons_per_vol[ id_mat ] * MigdalConstant/ ( epsilmax*epsilmax );
    U = logf ( Ekine / electron_mass_c2 );
    U2 = U*U;

    if ( Ekine > tlow )
    {
        ah1 = ah10 + ZZ* ( ah11 + ZZ*ah12 );
        ah2 = ah20 + ZZ* ( ah21 + ZZ*ah22 );
        ah3 = ah30 + ZZ* ( ah31 + ZZ*ah32 );
        bh1 = bh10 + ZZ* ( bh11 + ZZ*bh12 );
        bh2 = bh20 + ZZ* ( bh21 + ZZ*bh22 );
        bh3 = bh30 + ZZ* ( bh31 + ZZ*bh32 );
        ah = 1. + ( ah1*U2 + ah2*U + ah3 ) / ( U2*U );
        bh = .75 + ( bh1*U2 + bh2*U + bh3 ) / ( U2*U );
        screenfac = 136.*electron_mass_c2/ ( Z3*totalEnergy );
        screenmin = screenfac*epsilmin/ ( 1. - epsilmin );
        F1 = fmaxf ( ScreenFunction1 ( screenmin ) - FZ, 0. );
        F2 = fmaxf ( ScreenFunction2 ( screenmin ) - FZ, 0. );
        grejmax = ( F1 - epsilmin * ( F1*ah - bh*epsilmin*F2 ) ) / ( 42.392 - FZ );
    }
    else
    {
        al0 = al00 + ZZ* ( al01 + ZZ*al02 );
        al1 = al10 + ZZ* ( al11 + ZZ*al12 );
        al2 = al20 + ZZ* ( al21 + ZZ*al22 );
        bl0 = bl00 + ZZ* ( bl01 + ZZ*bl02 );
        bl1 = bl10 + ZZ* ( bl11 + ZZ*bl12 );
        bl2 = bl20 + ZZ* ( bl21 + ZZ*bl22 );
        ah = al0 + al1*U + al2*U2;
        bh = bl0 + bl1*U + bl2*U2;
        grejmax = fmaxf ( 1. + xmin* ( ah + bh*xmin ), 1. + ah + bh );
        xm = -ah/ ( 2.*bh );
        if ( xmin < xm && xm < xmax ) grejmax = fmaxf ( grejmax, 1. + xm * ( ah + bh*xm ) );
    }

    if ( Ekine > tlow )
    {
        do
        {
            q = prng_uniform( &(particles.prng[id]) );
            x = powf ( xmin, q + kappa * ( 1. - q ) );
            epsil = x*Ekine / totalEnergy;
            screenvar = screenfac*epsil / ( 1. - epsil );
            F1 = fmaxf ( ScreenFunction1 ( screenvar ) - FZ,0. );
            F2 = fmaxf ( ScreenFunction2 ( screenvar ) - FZ,0. );
            migdal = ( 1. + MigdalFactor ) / ( 1. + MigdalFactor / ( x*x ) );
            greject = migdal* ( F1 - epsil* ( ah*F1 - bh*epsil*F2 ) ) / ( 42.392 - FZ );
        }
        while ( greject < prng_uniform( &(particles.prng[id]) ) * grejmax );
    }
    else
    {
        do
        {
            q = prng_uniform( &(particles.prng[id]) );
            x = powf ( xmin, q + kappa* ( 1. - q ) );
            migdal = ( 1. + MigdalFactor ) / ( 1. + MigdalFactor / ( x*x ) );
            greject = migdal * ( 1. + x* ( ah + bh*x ) );
        }
        while ( greject < prng_uniform( &(particles.prng[id]) ) * grejmax );
    }
    gammaEnergy = x*Ekine;

    theta = AngleDistribution ( totalEnergy, totalEnergy-gammaEnergy, materials.mixture[ind], particles, id );
    sint = sin ( theta );
    phi = gpu_twopi * prng_uniform( &(particles.prng[id]) );

    f32xyz GamDir;
    GamDir.x = sint * cos ( phi );
    GamDir.y = sint * sin ( phi );
    GamDir.z = cos ( theta );

    f32xyz currentDir;
    currentDir = make_f32xyz ( particles.dx[id], particles.dy[id], particles.dz[id] );
    GamDir = rotateUz ( GamDir, currentDir );
    totMom = sqrtf ( Ekine * ( totalEnergy + electron_mass_c2 ) );

    currentDir = CorrUnit ( currentDir, GamDir, totMom, gammaEnergy );

    // Update electron
    particles.dx[ id ] = currentDir.x;
    particles.dy[ id ] = currentDir.y;
    particles.dz[ id ] = currentDir.z;
    particles.E[ id ] = Ekine - gammaEnergy;

    // Photon not produced
    //printf("Gamme energy %e\n", gammaEnergy);

}
#endif
