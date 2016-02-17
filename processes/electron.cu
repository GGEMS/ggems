// GGEMS Copyright (C) 2015

#ifndef ELECTRON_CU
#define ELECTRON_CU

#include "electron.cuh"

/// ElectronIonisation ///////////////////////////////////////////////////

// eIon de/dx
f32 ElectronIonisation_DEDX( MaterialsTable materials, f32 Ekine, ui8 mat_id )
{
    f32 Dedx = 0.;
    f32 th = .25*sqrt ( materials.nb_electrons_per_vol[mat_id] / materials.nb_atoms_per_vol[mat_id] ) *keV;
    f32 lowLimit = .2*keV;
    f32 tmax, tkin;
    f32 eexc, eexc2, d, x;
    f32 tau, gamma, gamma2, beta2, bg2;

    tkin = Ekine;
    if ( Ekine < th ) tkin = th;
    tmax = tkin * .5;
    tau = tkin / electron_mass_c2;
    gamma = tau + 1.;
    gamma2 = gamma * gamma;
    beta2 = 1. - 1. / gamma2;
    bg2 = beta2 * gamma2;
    eexc = materials.electron_mean_excitation_energy[mat_id] / electron_mass_c2;
    eexc2 = eexc * eexc;
    d = std::min ( materials.electron_energy_cut[mat_id], tmax );
    d /= electron_mass_c2;

    Dedx = log ( 2.* ( tau+2. ) /eexc2 )-1.-beta2+log ( ( tau-d ) *d ) +tau/ ( tau-d )
            + ( .5*d*d+ ( 2.*tau+1. ) *log ( 1.-d/tau ) ) /gamma2;

    x = log ( bg2 ) / ( 2.*log ( 10. ) );

    //// DensCorrection
    f32 dens_correction = 0.0;

    if ( x < materials.fX0[mat_id] )
    {
        if ( materials.fD0[mat_id]>0. )
        {
            dens_correction = materials.fD0[mat_id] * pow ( 10.,2.* ( x-materials.fX0[mat_id] ) );
        }
    }
    else if ( x >= materials.fX1[mat_id] )
    {
        dens_correction = 2.*log ( 10. ) *x-materials.fC[mat_id];
    }
    else
    {
        dens_correction = 2.*log ( 10. ) * x-materials.fC[mat_id] + materials.fA[mat_id]
                *pow ( materials.fX1[mat_id]-x, materials.fM[mat_id] );
    }

    Dedx -= dens_correction;
    ////

    Dedx *= twopi_mc2_rcl2 * materials.nb_electrons_per_vol[mat_id]/beta2;

    if ( Dedx < 0. ) Dedx = 0.;

    if ( Ekine < th )
    {
        if ( Ekine >= lowLimit )
            Dedx *= sqrt ( tkin/Ekine );
        else
            Dedx *= sqrt ( tkin*Ekine ) /lowLimit;
    }

    //printf("E %e dE/dx %e     cut %e\n", Ekine, Dedx, materials.electron_energy_cut[mat_id]);

    return Dedx;

}

// Cross Section Per Electron
f32 ElectronIonisation_CSPE(f32 Ekine, f32 Ecut, ui16 Z)
{
    f32  Cross=0.;
    f32  tmax=std::min ( 1.*GeV,Ekine*.5 );
    f32  xmin,xmax,gamma,gamma2,beta2,g;

    if ( Ecut < tmax )
    {
        xmin = Ecut/Ekine;
        xmax = tmax/Ekine;
        gamma = Ekine/electron_mass_c2+1.;
        gamma2 = gamma*gamma;
        beta2 = 1.-1./gamma2;
        g = ( 2.*gamma-1. ) /gamma2;
        Cross = ( ( xmax-xmin ) * ( 1.-g+1./ ( xmin*xmax ) +1./ ( ( 1.-xmin ) * ( 1.-xmax ) ) )
                 -g*std::log ( xmax* ( 1.-xmin ) / ( xmin* ( 1.-xmax ) ) ) ) /beta2;

        Cross *= twopi_mc2_rcl2/Ekine;
    }

    return  Cross * ( f32 ) Z;
}

// Total cross section
f32 ElectronIonisation_CS(MaterialsTable materials, f32 Ekine, ui16 mat_id)
{
    f32  CrossTotale=0.;
    ui32 index = materials.index[mat_id];

    for ( ui32 i=0; i<materials.nb_elements[mat_id]; ++i )
    {
        CrossTotale += ( materials.atom_num_dens[index+i] *
                ElectronIonisation_CSPE( Ekine, materials.electron_energy_cut[mat_id], materials.mixture[index+i] ) );
    }

    return  CrossTotale;
}

/////////////////////////////////////////////////////////////////////////////////////////////


/// Bremsstrahlung //////////////////////////////////////////////////////////////////////////



/// GPU
// Constant parameters for bremstrahlung table
__constant__ f32 gpu_ZZ[ 8 ] = { 2., 4., 6., 14., 26., 50., 82., 92. };
__constant__ f32 gpu_coefloss[ 8 ][ 11 ] =
{
    { .98916, .47564, -.2505, -.45186, .14462, .21307, -.013738, -.045689, -.0042914, .0034429, .00064189 },
    { 1.0626, .37662, -.23646, -.45188, .14295, .22906, -.011041, -.051398, -.0055123, .0039919, .00078003 },
    { 1.0954, .315, -.24011, -.43849, .15017, .23001, -.012846, -.052555, -.0055114, .0041283, .00080318 },
    { 1.1649, .18976, -.24972, -.30124, .1555, .13565, -.024765, -.027047, -.00059821, .0019373, .00027647 },
    { 1.2261, .14272, -.25672, -.28407, .13874, .13586, -.020562, -.026722, -.00089557, .0018665, .00026981 },
    { 1.3147, .020049, -.35543, -.13927, .17666, .073746, -.036076, -.013407, .0025727, .00084005, -1.4082e-05 },
    { 1.3986, -.10586, -.49187, -.0048846, .23621, .031652, -.052938, -.0076639, .0048181, .00056486, -.00011995 },
    { 1.4217, -.116, -.55497, -.044075, .27506, .081364, -.058143, -.023402, .0031322, .0020201, .00017519 }
};

__constant__ f32 gpu_coefsig[ 8 ][ 11 ] =
{
    { .4638, .37748, .32249, -.060362, -.065004, -.033457, -.004583, .011954, .0030404, -.0010077, -.00028131},
    { .50008, .33483, .34364, -.086262, -.055361, -.028168, -.0056172, .011129, .0027528, -.00092265, -.00024348},
    { .51587, .31095, .34996, -.11623, -.056167, -.0087154, .00053943, .0054092, .00077685, -.00039635, -6.7818e-05},
    { .55058, .25629, .35854, -.080656, -.054308, -.049933, -.00064246, .016597, .0021789, -.001327, -.00025983},
    { .5791, .26152, .38953, -.17104, -.099172, .024596, .023718, -.0039205, -.0036658, .00041749, .00023408},
    { .62085, .27045, .39073, -.37916, -.18878, .23905, .095028, -.068744, -.023809, .0062408, .0020407},
    { .66053, .24513, .35404, -.47275, -.22837, .35647, .13203, -.1049, -.034851, .0095046, .0030535},
    { .67143, .23079, .32256, -.46248, -.20013, .3506, .11779, -.1024, -.032013, .0092279, .0028592}
};


/// CPU
// Constant parameters for bremstrahlung table
const f32 cpu_ZZ[ 8 ] = { 2., 4., 6., 14., 26., 50., 82., 92. };
const f32 cpu_coefloss[ 8 ][ 11 ] =
{
    { .98916, .47564, -.2505, -.45186, .14462, .21307, -.013738, -.045689, -.0042914, .0034429, .00064189 },
    { 1.0626, .37662, -.23646, -.45188, .14295, .22906, -.011041, -.051398, -.0055123, .0039919, .00078003 },
    { 1.0954, .315, -.24011, -.43849, .15017, .23001, -.012846, -.052555, -.0055114, .0041283, .00080318 },
    { 1.1649, .18976, -.24972, -.30124, .1555, .13565, -.024765, -.027047, -.00059821, .0019373, .00027647 },
    { 1.2261, .14272, -.25672, -.28407, .13874, .13586, -.020562, -.026722, -.00089557, .0018665, .00026981 },
    { 1.3147, .020049, -.35543, -.13927, .17666, .073746, -.036076, -.013407, .0025727, .00084005, -1.4082e-05 },
    { 1.3986, -.10586, -.49187, -.0048846, .23621, .031652, -.052938, -.0076639, .0048181, .00056486, -.00011995 },
    { 1.4217, -.116, -.55497, -.044075, .27506, .081364, -.058143, -.023402, .0031322, .0020201, .00017519 }
};

#ifndef __CUDA_ARCH__
const f32 cpu_coefsig[ 8 ][ 11 ] =
{
    { .4638, .37748, .32249, -.060362, -.065004, -.033457, -.004583, .011954, .0030404, -.0010077, -.00028131},
    { .50008, .33483, .34364, -.086262, -.055361, -.028168, -.0056172, .011129, .0027528, -.00092265, -.00024348},
    { .51587, .31095, .34996, -.11623, -.056167, -.0087154, .00053943, .0054092, .00077685, -.00039635, -6.7818e-05},
    { .55058, .25629, .35854, -.080656, -.054308, -.049933, -.00064246, .016597, .0021789, -.001327, -.00025983},
    { .5791, .26152, .38953, -.17104, -.099172, .024596, .023718, -.0039205, -.0036658, .00041749, .00023408},
    { .62085, .27045, .39073, -.37916, -.18878, .23905, .095028, -.068744, -.023809, .0062408, .0020407},
    { .66053, .24513, .35404, -.47275, -.22837, .35647, .13203, -.1049, -.034851, .0095046, .0030535},
    { .67143, .23079, .32256, -.46248, -.20013, .3506, .11779, -.1024, -.032013, .0092279, .0028592}
};
#endif

__host__ __device__ f32 ZZ(ui8 pos)
{
#ifdef __CUDA_ARCH__
    return gpu_ZZ[ pos ];
#else
    return cpu_ZZ[ pos ];
#endif
}

__host__ __device__ f32 coefsig( ui8 i, ui8 j )
{
#ifdef __CUDA_ARCH__
    return gpu_coefsig[ i ][ j ];
#else
    return cpu_coefsig[ i ][ j ];
#endif
}

f32 ElectronBremmsstrahlung_loss ( f32 Z, f32 T, f32 Cut )
{
    ui32   i, j;
    ui32   NZ=8, Nloss=11, iz=0;
    f32    Loss;
    f32    dz, xx, yy, fl, E;
    f32    aaa=.414, bbb=.345, ccc=.460, delz=1.e6;
    f32    beta=1.0, ksi=2.0, clossh=.254, closslow=1./3., alosslow=1.;
    f32    Tlim=10.*MeV, xlim=1.2;

    for ( i=0; i<NZ; i++ )
    {
        dz =fabs ( Z-cpu_ZZ[i] );
        if ( dz<delz )
        {
            iz=i;
            delz=dz;
        }
    }
    xx = log10 ( T );
    fl = 1.;
    if ( xx<=xlim )
    {
        xx /= xlim;
        yy = 1.;
        fl = 0.;
        for ( j=0; j<Nloss; j++ )
        {
            fl += yy+cpu_coefloss[iz][j];
            yy *= xx;
        }
        if ( fl < .00001 )
            fl = .00001;
        else if ( fl > 1. )
            fl = 1.;
    }

    E = T+electron_mass_c2;
    Loss = Z* ( Z+ksi ) *E*E/ ( T+E ) *exp ( beta*log ( Cut/T ) ) * ( 2.-clossh*exp ( log ( Z ) /4. ) );
    if ( T<=Tlim )
        Loss /= exp ( closslow*log ( Tlim/T ) );
    if ( T<=Cut )
        Loss *= exp ( alosslow*log ( T/Cut ) );
    Loss *= ( aaa+bbb*T/Tlim ) / ( 1.+ccc*T/Tlim );
    Loss *=fl;
    Loss /=N_avogadro;

    return  Loss;
}

f32 ElectronBremsstrahlung_DEDX(MaterialsTable materials, f32 Ekine, ui8 mat_id)
{

    ui32 i, n, nn, nmax;
    f32 Dedx;
    f32 totalEnergy, Z, natom, kp2, kmin, kmax, floss;
    f32 vmin,vmax,u,fac,c,v,dv;
    f32 thigh = 100.*GeV;
    f32 cut = std::min ( materials.photon_energy_cut[mat_id], Ekine );
    f32  /*rate,*/loss;
    //     f32  factorHigh=36./ ( 1450.*GeV );
    //     f32  coef1=-.5;
    //     f32  coef2=2./9.;
    f32 lowKinEnergy = 0.*eV;
    f32 highKinEnergy = 1.*GeV;
    f32 probsup = 1.;
    f32 MigdalConstant = elec_radius*hbarc*hbarc*4.*pi/ ( electron_mass_c2*electron_mass_c2 );

    totalEnergy = Ekine+electron_mass_c2;
    Dedx = 0.;

    if ( Ekine<lowKinEnergy ) return  0.;

    for ( i=0; i<materials.nb_elements[mat_id]; ++i ) // Check in each elt
    {
        int indexelt = i + materials.index[mat_id];
        Z = materials.mixture[indexelt];
        natom = materials.atom_num_dens[indexelt] / materials.nb_atoms_per_vol[mat_id];

        if ( Ekine<=thigh ) loss = ElectronBremmsstrahlung_loss ( Z, Ekine, cut );

        loss *= natom;
        kp2 = MigdalConstant * totalEnergy*totalEnergy * materials.nb_electrons_per_vol[mat_id];

        kmin = 1.*eV;
        kmax = cut;
        if ( kmax>kmin )
        {
            floss = 0.;
            nmax = 100;
            vmin =log ( kmin );
            vmax =log ( kmax );
            nn= ( int ) ( nmax* ( vmax-vmin ) / ( log ( highKinEnergy )-vmin ) ) ;
            if ( nn>0 )
            {
                dv = ( vmax-vmin ) /nn;
                v = vmin-dv;
                for ( n=0; n<=nn; n++ )
                {
                    v += dv;
                    u =exp ( v );
                    //fac=u*SupressionFunction(material,Ekine,u);   //LPM flag off
                    fac = u*1.;
                    fac *= probsup* ( u*u/ ( u*u+kp2 ) ) +1.-probsup;
                    if ( ( n==0 ) || ( n==nn ) )
                        c=.5;
                    else
                        c=1.;
                    fac*=c;
                    floss+=fac ;
                }
                floss *= dv/ ( kmax-kmin );
            }
            else
                floss=1.;
            if ( floss>1. )
                floss=1.;
            loss*=floss;
        }
        Dedx += loss;
    }

    if ( Dedx<0. ) Dedx=0.;

    Dedx *= materials.nb_atoms_per_vol[mat_id];

    return Dedx;  // I removed *mm2 => *1 - JB

}


__host__ __device__ f32 ElectronBremmsstrahlung_CSPA( f32 Z, f32 cut, f32 Ekine )
{
    i32 i,j,iz = 0, NZ = 8, Nsig = 11;
    f32 Cross = 0.;
    f32 ksi = 2., alfa = 1.;
    f32 csigh = .127, csiglow = .25, asiglow = .02*MeV;
    f32 Tlim = 10.*MeV;
    f32 xlim = 1.2, delz = 1.E6, absdelz;
    f32 xx, fs;

    if ( Ekine<1.*keV || Ekine<cut )
        return  Cross;

    for ( i=0; i<NZ; i++ )
    {
        absdelz = fabs ( Z-ZZ( i ) );
        if ( absdelz < delz )
        {
            iz = i;
            delz = absdelz;
        }
    }

    xx = log10f ( Ekine );
    fs = 1.;
    if ( xx <= xlim )
    {
        fs = coefsig( iz, Nsig-1 );
        for ( j = Nsig-2; j>=0; j-- )
            fs = fs*xx + coefsig( iz, j );
        if ( fs < 0. )
            fs = 0.;
    }
    Cross = Z* ( Z+ksi ) * ( 1.-csigh*expf ( logf ( Z ) /4. ) ) *powf ( logf ( Ekine/cut ), alfa );

    if ( Ekine <= Tlim )
        Cross *= expf ( csiglow*logf ( Tlim/Ekine ) ) * ( 1.+asiglow/ ( sqrtf ( Z ) *Ekine ) );
    Cross *= fs/N_avogadro;
    if ( Cross<0. )
        Cross=0.;

    return  Cross;
}


//                                                                         v-- CS table max energy - JB
__host__ __device__ f32 ElectronBremmsstrahlung_CS( MaterialsTable materials, f32 Ekine, f32 max_E, ui8 mat_id )
{
    i32 i, n, nn, nmax = 100;
    f32 Cross = 0.;
    f32 kmax, kmin, vmin, vmax, totalEnergy, kp2;
    f32 u, fac, c, v, dv, y;
    f32 tmax = fmin ( max_E, Ekine );
    f32 cut = fmax ( materials.photon_energy_cut[mat_id], 0.0001f ); // 0.1 keV
    if ( cut >= tmax ) return Cross;

    f32 fsig = 0.;
    f32 highKinEnergy = 1.*GeV;
    f32 probsup = 1.;
    f32 MigdalConstant = elec_radius*hbarc*hbarc*4.*pi / ( electron_mass_c2*electron_mass_c2 );

    ui32 index = materials.index[mat_id];

    for ( i=0; i<materials.nb_elements[mat_id]; i++ )
    {
        Cross += materials.atom_num_dens[index+i]
                 * ElectronBremmsstrahlung_CSPA(materials.mixture[index+i], cut, Ekine);

        if ( tmax < Ekine )
        {
            Cross -= materials.atom_num_dens[index+i]
                     * ElectronBremmsstrahlung_CSPA(materials.mixture[index+i], tmax, Ekine);
        }
    }

    kmax = tmax;
    kmin = cut;
    totalEnergy = Ekine+electron_mass_c2;
    kp2 = MigdalConstant * totalEnergy*totalEnergy * materials.nb_electrons_per_vol[mat_id];
    vmin = logf ( kmin );
    vmax = logf ( kmax );
    nn= ( i32 ) ( nmax* ( vmax-vmin ) / ( logf ( highKinEnergy )-vmin ) );

    if ( nn>0 )
    {
        dv = ( vmax-vmin ) /nn;
        v = vmin-dv;
        for ( n=0; n<=nn; n++ )
        {
            v += dv;
            u = expf ( v );
            //fac=SupressionFunction(material,Ekine,u);     //LPM flag is off
            fac = 1.;
            y = u/kmax;
            fac *= ( 4.-4.*y+3.*y*y ) /3.;
            fac *= probsup* ( u*u/ ( u*u+kp2 ) ) +1.-probsup;
            if ( ( n==0 ) || ( n==nn ) )
                c=.5;
            else
                c=1.;
            fac *= c;
            fsig += fac;
        }
        y = kmin/kmax;
        fsig *= dv/ ( -4.*logf ( y ) /3.-4.* ( 1.-y ) /3.+0.5* ( 1.-y*y ) );
    }
    else
        fsig = 1.;
    if ( fsig > 1. )
        fsig = 1.;
    Cross *= fsig;

    return Cross;  // I removed *mm2 => *1 - JB
}

/////////////////////////////////////////////////////////////////////////////////////////////

/// Electron Multiple Scattering ////////////////////////////////////////////////////////////

// constants for eMSC
const f32 Zdat[ 15 ] = { 4., 6., 13., 20., 26., 29., 32., 38., 47., 50., 56., 64., 74., 79., 82. };

const f32 Tdat[ 22 ] =
{
    100.*eV, 200.*eV, 400.*eV, 700.*eV, 1.*keV, 2.*keV, 4.*keV, 7.*keV,
    10.*keV, 20.*keV, 40.*keV, 70.*keV, 100.*keV, 200.*keV, 400.*keV, 700.*keV,
    1.*MeV, 2.*MeV, 4.*MeV, 7.*MeV, 10.*MeV, 20.*MeV
};

const f32 celectron[ 15 ][ 22 ] =
{
    { 1.125, 1.072, 1.051, 1.047, 1.047, 1.050, 1.052, 1.054, 1.054, 1.057, 1.062, 1.069, 1.075, 1.090, 1.105, 1.111, 1.112, 1.108, 1.100, 1.093, 1.089, 1.087 },
    { 1.408, 1.246, 1.143, 1.096, 1.077, 1.059, 1.053, 1.051, 1.052, 1.053, 1.058, 1.065, 1.072, 1.087, 1.101, 1.108, 1.109, 1.105, 1.097, 1.090, 1.086, 1.082 },
    { 2.833, 2.268, 1.861, 1.612, 1.486, 1.309, 1.204, 1.156, 1.136, 1.114, 1.106, 1.106, 1.109, 1.119, 1.129, 1.132, 1.131, 1.124, 1.113, 1.104, 1.099, 1.098 },
    { 3.879, 3.016, 2.380, 2.007, 1.818, 1.535, 1.340, 1.236, 1.190, 1.133, 1.107, 1.099, 1.098, 1.103, 1.110, 1.113, 1.112, 1.105, 1.096, 1.089, 1.085, 1.098 },
    { 6.937, 4.330, 2.886, 2.256, 1.987, 1.628, 1.395, 1.265, 1.203, 1.122, 1.080, 1.065, 1.061, 1.063, 1.070, 1.073, 1.073, 1.070, 1.064, 1.059, 1.056, 1.056 },
    { 9.616, 5.708, 3.424, 2.551, 2.204, 1.762, 1.485, 1.330, 1.256, 1.155, 1.099, 1.077, 1.070, 1.068, 1.072, 1.074, 1.074, 1.070, 1.063, 1.059, 1.056, 1.052 },
    { 11.72, 6.364, 3.811, 2.806, 2.401, 1.884, 1.564, 1.386, 1.300, 1.180, 1.112, 1.082, 1.073, 1.066, 1.068, 1.069, 1.068, 1.064, 1.059, 1.054, 1.051, 1.050 },
    { 18.08, 8.601, 4.569, 3.183, 2.662, 2.025, 1.646, 1.439, 1.339, 1.195, 1.108, 1.068, 1.053, 1.040, 1.039, 1.039, 1.039, 1.037, 1.034, 1.031, 1.030, 1.036 },
    { 18.22, 1.48, 5.333, 3.713, 3.115, 2.367, 1.898, 1.631, 1.498, 1.301, 1.171, 1.105, 1.077, 1.048, 1.036, 1.033, 1.031, 1.028, 1.024, 1.022, 1.021, 1.024 },
    { 14.14, 10.65, 5.710, 3.929, 3.266, 2.453, 1.951, 1.669, 1.528, 1.319, 1.178, 1.106, 1.075, 1.040, 1.027, 1.022, 1.020, 1.017, 1.015, 1.013, 1.013, 1.020 },
    { 14.11, 11.73, 6.312, 4.240, 3.478, 2.566, 2.022, 1.720, 1.569, 1.342, 1.186, 1.102, 1.065, 1.022, 1.003, 0.997, 0.995, 0.993, 0.993, 0.993, 0.993, 1.011 },
    { 22.76, 20.01, 8.835, 5.287, 4.144, 2.901, 2.219, 1.855, 1.677, 1.410, 1.224, 1.121, 1.073, 1.014, 0.986, 0.976, 0.974, 0.972, 0.973, 0.974, 0.975, 0.987 },
    { 50.77, 40.85, 14.13, 7.184, 5.284, 3.435, 2.520, 2.059, 1.837, 1.512, 1.283, 1.153, 1.091, 1.010, 0.969, 0.954, 0.950, 0.947, 0.949, 0.952, 0.954, 0.963 },
    { 65.87, 59.06, 15.87, 7.570, 5.567, 3.650, 2.682, 2.182, 1.939, 1.579, 1.325, 1.178, 1.108, 1.014, 0.965, 0.947, 0.941, 0.938, 0.940, 0.944, 0.946, 0.954 },
    { 55.60, 47.34, 15.92, 7.810, 5.755, 3.767, 2.760, 2.239, 1.985, 1.609, 1.343, 1.188, 1.113, 1.013, 0.960, 0.939, 0.933, 0.930, 0.933, 0.936, 0.939, 0.949 }
};

const f32 sig0[ 15 ] =
{
    .2672*barn, .5922*barn, 2.653*barn, 6.235*barn, 11.69*barn, 13.24*barn, 16.12*barn, 23.00*barn,
    35.13*barn, 39.95*barn, 50.85*barn, 67.19*barn, 91.15*barn, 104.4*barn, 113.1*barn
};

const f32 hecorr[ 15 ] =
{
    120.70, 117.50, 105.00, 92.92, 79.23, 74.510, 68.29, 57.39, 41.97, 36.14, 24.53, 10.21, -7.855, -16.84, -22.30
};

f32 ElectronMultipleScattering_CSPA( f32 Ekine,  ui8 Z )
{
    f32 AtomicNumber = ( f32 ) Z;

    i32 iZ = 14, iT = 21;
    f32  Cross = 0.;
    f32  eKin, eTot, T, E;
    f32  beta2, bg2, b2big, b2small, ratb2, Z23, tau, w;
    f32  Z1, Z2, ratZ;
    f32  c, c1, c2, cc1, cc2, corr;
    f32  Tlim = 10.*MeV;
    f32  sigmafactor = 2.*pi*elec_radius*elec_radius;
    f32  epsfactor = 2.*electron_mass_c2*electron_mass_c2*Bohr_radius*Bohr_radius/ ( hbarc*hbarc );
    f32  eps, epsmin = 1.e-4, epsmax = 1.e10;
    f32  beta2lim = Tlim* ( Tlim+2.*electron_mass_c2 ) / ( ( Tlim+electron_mass_c2 ) * ( Tlim+electron_mass_c2 ) );
    f32  bg2lim = Tlim* ( Tlim+2.*electron_mass_c2 ) / ( electron_mass_c2*electron_mass_c2 );

    Z23 = 2.*log ( AtomicNumber ) /3.;
    Z23 = exp ( Z23 );

    tau = Ekine/electron_mass_c2;
    c = electron_mass_c2*tau* ( tau+2. ) / ( electron_mass_c2* ( tau+1. ) ); // a simplifier
    w = c-2.;
    tau = .5* ( w+sqrt ( w*w+4.*c ) );
    eKin = electron_mass_c2*tau;

    eTot = eKin + electron_mass_c2;
    beta2 = eKin* ( eTot+electron_mass_c2 ) / ( eTot*eTot );
    bg2 = eKin* ( eTot+electron_mass_c2 ) / ( electron_mass_c2*electron_mass_c2 );
    eps = epsfactor*bg2/Z23;
    if ( eps<epsmin )
        Cross = 2.*eps*eps;
    else if ( eps<epsmax )
        Cross = log ( 1.+2.*eps )-2.*eps/ ( 1.+2.*eps );
    else
        Cross = log ( 2.*eps )-1.+1./eps;
    Cross *= AtomicNumber*AtomicNumber / ( beta2*bg2 );

    while ( ( iZ>=0 ) && ( Zdat[iZ] >= AtomicNumber ) )
        iZ -= 1;
    if ( iZ == 14 )
        iZ = 13;
    if ( iZ == -1 )
        iZ = 0;
    Z1 = Zdat[iZ];
    Z2 = Zdat[iZ+1];
    ratZ = ( AtomicNumber-Z1 ) * ( AtomicNumber+Z1 ) / ( ( Z2-Z1 ) * ( Z2+Z1 ) );

    if ( eKin<=Tlim )
    {
        while ( ( iT>=0 ) && ( Tdat[iT]>=eKin ) )
            iT -= 1;
        if ( iT == 21 )
            iT = 20;
        if ( iT == -1 )
            iT = 0;
        T = Tdat[iT];
        E = T+electron_mass_c2;
        b2small = T* ( E+electron_mass_c2 ) / ( E*E );
        T = Tdat[iT+1];
        E = T+electron_mass_c2;
        b2big = T* ( E+electron_mass_c2 ) / ( E*E );
        ratb2 = ( beta2-b2small ) / ( b2big-b2small );

        c1 = celectron[iZ][iT];
        c2 = celectron[iZ+1][iT];
        cc1 = c1+ratZ* ( c2-c1 );
        c1 = celectron[iZ][iT+1];
        c2 = celectron[iZ+1][iT+1];
        cc2 = c1+ratZ* ( c2-c1 );
        corr = cc1+ratb2* ( cc2-cc1 );
        Cross *= sigmafactor/corr;
    }
    else
    {
        c1 = bg2lim*sig0[iZ]* ( 1.+hecorr[iZ]* ( beta2-beta2lim ) ) /bg2;
        c2 = bg2lim*sig0[iZ+1]* ( 1.+hecorr[iZ+1]* ( beta2-beta2lim ) ) /bg2;
        if ( ( AtomicNumber>=Z1 ) && ( AtomicNumber<=Z2 ) )
            Cross = c1+ratZ* ( c2-c1 );
        else if ( AtomicNumber<Z1 )
            Cross = AtomicNumber*AtomicNumber*c1/ ( Z1*Z1 );
        else if ( AtomicNumber>Z2 )
            Cross = AtomicNumber*AtomicNumber*c2/ ( Z2*Z2 );
    }
    return  Cross;
}

f32 ElectronMultipleScattering_CS( MaterialsTable material, f32 Ekine, ui8 mat_id)
{
    ui32 i;
    f32 CrossTotale = 0.;
    ui32 index = material.index[mat_id];

    for ( i=0; i<material.nb_elements[mat_id]; i++ )
    {
        CrossTotale += material.atom_num_dens[index+i]
                       *ElectronMultipleScattering_CSPA ( Ekine, material.mixture[index+i] );
    }

    return  CrossTotale;
}

/////////////////////////////////////////////////////////////////////////////////////////////


// OLD FUNCKING CODE FROM YANNICK

/*
void ElectronCrossSection::initialize ( GlobalSimulationParameters params,MaterialsTable materials )
{
    nb_bins = params.data_h.cs_table_nbins;
    nb_mat = materials.nb_materials;
//     params = parameters;
    parameters = params;
    myMaterials = materials;
    MaxKinEnergy = parameters.data_h.cs_table_max_E;
    MinKinEnergy = parameters.data_h.cs_table_min_E;
    cutEnergyElectron = parameters.data_h.electron_cut;
    cutEnergyGamma = parameters.data_h.photon_cut;
}
*/

/*
void ElectronCrossSection::generateTable()
{
    data_h.nb_bins = nb_bins;
    data_h.nb_mat = nb_mat;
    // Memory allocation for tables
    data_h.E = new f32[nb_mat *nb_bins];  //Mandatories tables
    data_h.eRange = new f32[nb_mat *nb_bins];

    data_h.eIonisationCS = new f32[nb_mat * nb_bins];
    data_h.eIonisationdedx= new f32[nb_mat * nb_bins];

    data_h.eBremCS = new f32[nb_mat * nb_bins];
    data_h.eBremdedx= new f32[nb_mat * nb_bins];

    data_h.eMSC = new f32[nb_mat * nb_bins];

    data_h.pAnni_CS_tab = new f32[nb_mat * nb_bins];


    for ( int i=0; i<nb_mat*nb_bins; i++ ) // Create Energy table between Emin and Emax
    {
        data_h.E[i]=0.;
    }


    Energy_table();


    for ( int id_mat = 0; id_mat< myMaterials.nb_materials; ++id_mat )
    {

//         int i = 0;
        for ( int i=0; i<nb_bins; i++ ) // Initialize tables
        {

            //                 if(m_physics_list[ELECTRON_IONISATION] == 1)
            //                 {
            data_h.eIonisationCS[i + nb_bins * id_mat]=0.;
            data_h.eIonisationdedx[i + nb_bins * id_mat]=0.;
            //                 }
            //                 if(m_physics_list[ELECTRON_BREMSSTRAHLUNG] == 1)
            //                 {
            data_h.eBremCS[i + nb_bins * id_mat]=0.;
            data_h.eBremdedx[i + nb_bins * id_mat]=0.;
            //                 }
            //                 if(m_physics_list[ELECTRON_MSC] == 1)
            //                 {
            data_h.eMSC[i + nb_bins * id_mat]=0.;
            //                 }

            data_h.eRange[i + nb_bins * id_mat]=0.;

            data_h.pAnni_CS_tab[i + nb_bins * id_mat]=0.;

        }


        // Create tables if physic is activated
        if ( parameters.data_h.physics_list[ELECTRON_IONISATION] == true )
        {
            eIoni_DEDX_table ( id_mat );
            eIoni_CrossSection_table ( id_mat );
        }

        if ( parameters.data_h.physics_list[ELECTRON_BREMSSTRAHLUNG] == true )
        {
            eBrem_DEDX_table ( id_mat );
            eBrem_CrossSection_table ( id_mat );
        }

        if ( parameters.data_h.physics_list[ELECTRON_MSC] == true )
        {
            eMSC_CrossSection_table ( id_mat );
        }


        Range_table ( id_mat );


    }

}
*/
/*
void ElectronCrossSection::Range_table ( int id_mat )
{

    int i,j,n;
    f32  energy,de,esum,eDXDE=0.;

    i=0;
    n=100;


    eDXDE=data_h.eIonisationdedx[i+id_mat*nb_bins]+data_h.eBremdedx[i+id_mat*nb_bins];
    if ( eDXDE>0. )
        eDXDE=2.*data_h.E[i+id_mat*nb_bins]/eDXDE;
    data_h.eRange[i+id_mat*nb_bins]=eDXDE;

    for ( i=1; i<nb_bins; i++ )
    {
        de= ( data_h.E[i+id_mat*nb_bins]-data_h.E[i-1+id_mat*nb_bins] ) /n;
        energy=data_h.E[i+id_mat*nb_bins]+de*.5;

        esum=0.;
        for ( j=0; j<n; j++ )
        {
            energy-=de;
            eDXDE=GetDedx ( energy,id_mat ); //+GetDedx(energy,2);
            if ( eDXDE>0. )
                esum+=de/eDXDE;
        }
        data_h.eRange[i+id_mat*nb_bins]=data_h.eRange[i-1+id_mat*nb_bins]+esum;
    }



}
*/

/*
f32 ElectronCrossSection::GetDedx ( f32 Energy,int material ) // get dedx eioni and ebrem sum for energy Energy in material material
{
    int index = 0;
    index = binary_search ( Energy,data_h.E, ( material+1 ) *nb_bins, ( material ) *nb_bins );

    f32 DeDxeIoni = linear_interpolation ( data_h.E[index]  ,data_h.eIonisationdedx[index],
                                           data_h.E[index+1],data_h.eIonisationdedx[index+1],
                                           Energy
                                         );

    f32 DeDxeBrem = linear_interpolation ( data_h.E[index]  ,data_h.eBremdedx[index],
                                           data_h.E[index+1],data_h.eBremdedx[index+1],
                                           Energy
                                         );

    return DeDxeIoni + DeDxeBrem;

}
*/

/*
//Print a table file per material
void ElectronCrossSection::printElectronTables ( std::string dirname )
{

    for ( int i = 0; i< nb_mat; ++i )
    {
        std::string tmp = dirname + to_string ( i ) + ".txt";
        ImageReader::recordTables ( tmp.c_str(),i*nb_bins, ( i+1 ) *nb_bins,
                                    data_h.E,
                                    data_h.eIonisationdedx,
                                    data_h.eIonisationCS,
                                    data_h.eBremdedx,
                                    data_h.eBremCS,
                                    data_h.eMSC,
                                    data_h.eRange );
    }

}
*/

/*
void ElectronCrossSection::Energy_table() // Create energy table between emin and emax
{
//     int id_mat = myMaterials.nb_materials;
    ui32 i;
    f32  constant,slope,x;//,energy;

    constant = parameters.data_h.cs_table_min_E;
    slope = log ( parameters.data_h.cs_table_max_E / parameters.data_h.cs_table_min_E );
    for ( int id_mat = 0; id_mat < myMaterials.nb_materials; ++id_mat )
    {
        for ( i = 0; i < nb_bins; i++ )
        {
            x  = ( f32 ) i;
            x /= ( nb_bins-1 );
            data_h.E[i+id_mat*nb_bins] = constant*exp ( slope*x ) *MeV;
        }

    }


}
*/

/*
void ElectronCrossSection:: eIoni_DEDX_table ( int id_mat )
{
    int i; // Index to increment energy

    for ( i=0; i<nb_bins; i++ )
    {

        f32 Ekine=data_h.E[i];
        data_h.eIonisationdedx[i + nb_bins * id_mat]=eIoniDEDX ( Ekine,id_mat );

    }

}
*/

/*
f32 ElectronCrossSection::eIoniDEDX ( f32 Ekine, int id_mat )
{

    f32  Dedx=0.;
    f32  th=.25*sqrt ( myMaterials.nb_electrons_per_vol[id_mat]/myMaterials.nb_atoms_per_vol[id_mat] ) *keV;
    f32  lowLimit=.2*keV;
    f32  tmax,tkin;
    f32  eexc,eexc2,d,x;//,y;
    f32  tau,gamma,gamma2,beta2,bg2;

    tkin=Ekine;
    if ( Ekine<th )
        tkin=th;
    tmax=tkin*.5;
    tau=tkin/electron_mass_c2;
    gamma=tau+1.;
    gamma2=gamma*gamma;
    beta2=1.-1./gamma2;
    bg2=beta2*gamma2;
    eexc=myMaterials.electron_mean_excitation_energy[id_mat]/electron_mass_c2;
    eexc2=eexc*eexc;
    d=std::min ( cutEnergyElectron,tmax );
    d/=electron_mass_c2;

    Dedx=log ( 2.* ( tau+2. ) /eexc2 )-1.-beta2+log ( ( tau-d ) *d ) +tau/ ( tau-d )
         + ( .5*d*d+ ( 2.*tau+1. ) *log ( 1.-d/tau ) ) /gamma2;

    x=log ( bg2 ) / ( 2.*log ( 10. ) );
    Dedx-=DensCorrection ( x,id_mat );
    Dedx*=twopi_mc2_rcl2*myMaterials.nb_electrons_per_vol[id_mat]/beta2;

    if ( Dedx<0. )
        Dedx=0.;
    if ( Ekine<th )
    {
        if ( Ekine>=lowLimit )
            Dedx*=sqrt ( tkin/Ekine );
        else
            Dedx*=sqrt ( tkin*Ekine ) /lowLimit;
    }

    return    Dedx;
}
*/

/*
f32 ElectronCrossSection::DensCorrection ( f32 x, int id_mat )
{
    f32  y=0.;

    if ( x<myMaterials.fX0[id_mat] )
    {
        if ( myMaterials.fD0[id_mat]>0. )
            y=myMaterials.fD0[id_mat]*pow ( 10.,2.* ( x-myMaterials.fX0[id_mat] ) );
    }
    else if ( x>=myMaterials.fX1[id_mat] )
        y=2.*log ( 10. ) *x-myMaterials.fC[id_mat];
    else
        y=2.*log ( 10. ) *x-myMaterials.fC[id_mat]+myMaterials.fA[id_mat]
          *pow ( myMaterials.fX1[id_mat]-x,myMaterials.fM[id_mat] );
    return  y;
}
*/

/*
void ElectronCrossSection::eIoni_CrossSection_table ( int id_mat )
{
    int i;

    for ( i=0; i<nb_bins; i++ )
    {
        f32 Ekine=data_h.E[i];
        data_h.eIonisationCS[i + nb_bins * id_mat]=eIoniCrossSection ( id_mat, Ekine );
    }
}

f32  ElectronCrossSection::eIoniCrossSection ( int id_mat, f32 Ekine )
{
    int i;
    f32  CrossTotale=0.;
    int index = myMaterials.index[id_mat]; // Get index of 1st element of the mixture

    for ( i=0; i<myMaterials.nb_elements[id_mat]; ++i ) // Get the sum of each element cross section
    {
        f32 tempo = myMaterials.atom_num_dens[index+i] * eIoniCrossSectionPerAtom ( index+i, Ekine );
//         tempo *=myMaterials.nb_atoms_per_vol[id_mat]; // Tempo value to avoid overflow
        CrossTotale+=tempo;//myMaterials.atom_num_dens[index+i]*myMaterials.nb_atoms_per_vol[id_mat]*eIoniCrossSectionPerAtom(index+i, Ekine);
    }

    return  CrossTotale;
}

f32 ElectronCrossSection:: eIoniCrossSectionPerAtom ( int index, f32 Ekine )
{
    f32  Cross=0.;
    f32  tmax=std::min ( 1.*GeV,Ekine*.5 );
    f32  xmin,xmax,gamma,gamma2,beta2,g;
//     GGcout << cutEnergyElectron << "  " << tmax << GGendl;
    if ( cutEnergyElectron<tmax )
    {
        xmin=cutEnergyElectron/Ekine;
        xmax=tmax/Ekine;
        gamma=Ekine/electron_mass_c2+1.;
        gamma2=gamma*gamma;
        beta2=1.-1./gamma2;
        g= ( 2.*gamma-1. ) /gamma2;
        Cross= ( ( xmax-xmin ) * ( 1.-g+1./ ( xmin*xmax ) +1./ ( ( 1.-xmin ) * ( 1.-xmax ) ) )
                 -g*std::log ( xmax* ( 1.-xmin ) / ( xmin* ( 1.-xmax ) ) ) ) /beta2;

        Cross*=twopi_mc2_rcl2/Ekine;
    }
    Cross*=myMaterials.mixture[index];

    return  Cross;
}
*/

/*
void ElectronCrossSection::eBrem_DEDX_table ( int id_mat )
{
    int i;

    for ( i=0; i<nb_bins; i++ )
    {
        f32 Ekine=data_h.E[i];
        data_h.eBremdedx[i + nb_bins * id_mat]=eBremDEDX ( Ekine,id_mat ) *mm2; //G4 internal unit
    }
}

f32 ElectronCrossSection::eBremDEDX ( f32 Ekine, int id_mat ) //id_mat = index material
{
    int i,n,nn,nmax;
    f32  Dedx;
    f32  totalEnergy,Z,natom,kp2,kmin,kmax,floss;
    f32  vmin,vmax,u,fac,c,v,dv;
    f32  thigh=100.*GeV;
    f32  cut=std::min ( cutEnergyGamma,Ekine );
    f32  loss; // rate
//     f32  factorHigh=36./ ( 1450.*GeV );
//     f32  coef1=-.5;
//     f32  coef2=2./9.;
    f32  lowKinEnergy=0.*eV;
    f32  highKinEnergy=1.*GeV;
    f32  probsup=1.;
    f32  MigdalConstant=elec_radius*hbarc*hbarc*4.*pi/ ( electron_mass_c2*electron_mass_c2 );

    totalEnergy=Ekine+electron_mass_c2;
    Dedx=0.;

    if ( Ekine<lowKinEnergy )
        return  0.;

    for ( i=0; i<myMaterials.nb_elements[id_mat]; ++i ) // Check in each elt
    {
        int indexelt= i + myMaterials.index[id_mat];
        Z=myMaterials.mixture[indexelt];
        natom=myMaterials.atom_num_dens[indexelt]/myMaterials.nb_atoms_per_vol[id_mat];
        if ( Ekine<=thigh )
            loss=eBremLoss ( Z,Ekine,cut );
        loss*=natom;
        kp2=MigdalConstant*totalEnergy*totalEnergy*myMaterials.nb_electrons_per_vol[id_mat];

        kmin=1.*eV;
        kmax=cut;
        if ( kmax>kmin )
        {
            floss=0.;
            nmax=100;
            vmin=log ( kmin );
            vmax=log ( kmax );
            nn= ( int ) ( nmax* ( vmax-vmin ) / ( log ( highKinEnergy )-vmin ) ) ;
            if ( nn>0 )
            {
                dv= ( vmax-vmin ) /nn;
                v=vmin-dv;
                for ( n=0; n<=nn; n++ )
                {
                    v+=dv;
                    u=exp ( v );
                    //fac=u*SupressionFunction(material,Ekine,u);   //LPM flag off
                    fac=u*1.;
                    fac*=probsup* ( u*u/ ( u*u+kp2 ) ) +1.-probsup;
                    if ( ( n==0 ) || ( n==nn ) )
                        c=.5;
                    else
                        c=1.;
                    fac*=c;
                    floss+=fac ;
                }
                floss*=dv/ ( kmax-kmin );
            }
            else
                floss=1.;
            if ( floss>1. )
                floss=1.;
            loss*=floss;
        }
        Dedx+=loss;
    }
    if ( Dedx<0. )
        Dedx=0.;
    Dedx*=myMaterials.nb_atoms_per_vol[id_mat];
    return  Dedx;
}
    */

/*
f32 ElectronCrossSection::eBremLoss ( f32 Z,f32 T,f32 Cut )
{

    int   i,j;
    int   NZ=8,Nloss=11,iz=0;
    f32    Loss;
    f32    dz,xx,yy,fl,E;
    f32    aaa=.414,bbb=.345,ccc=.460,delz=1.e6;
    f32  beta=1.0,ksi=2.0,clossh=.254,closslow=1./3.,alosslow=1.;
    f32    Tlim=10.*MeV,xlim=1.2;

    for ( i=0; i<NZ; i++ )
    {
        dz=fabs ( Z-ZZ[i] );
        if ( dz<delz )
        {
            iz=i;
            delz=dz;
        }
    }
    xx=log10 ( T );
    fl=1.;
    if ( xx<=xlim )
    {
        xx/=xlim;
        yy=1.;
        fl=0.;
        for ( j=0; j<Nloss; j++ )
        {
            fl+=yy+coefloss[iz][j];
            yy*=xx;
        }
        if ( fl<.00001 )
            fl=.00001;
        else if ( fl>1. )
            fl=1.;
    }

    E=T+electron_mass_c2;
    Loss=Z* ( Z+ksi ) *E*E/ ( T+E ) *exp ( beta*log ( Cut/T ) ) * ( 2.-clossh*exp ( log ( Z ) /4. ) );
    if ( T<=Tlim )
        Loss/=exp ( closslow*log ( Tlim/T ) );
    if ( T<=Cut )
        Loss*=exp ( alosslow*log ( T/Cut ) );
    Loss*= ( aaa+bbb*T/Tlim ) / ( 1.+ccc*T/Tlim );
    Loss*=fl;
    Loss/=N_avogadro;

    return  Loss;
}
*/
/*
void ElectronCrossSection::eBrem_CrossSection_table ( int id_mat )
{

    for ( int i=0; i<nb_bins; i++ )
    {
        f32 Ekine=data_h.E[i];
        data_h.eBremCS[i + nb_bins * id_mat]=eBremCrossSection ( Ekine,id_mat ) *mm2; //G4 internal unit;
    }
}
*/
/*
f32 ElectronCrossSection::eBremCrossSection ( f32 Ekine,int id_mat )
{
    f32 CrossTotale=0.;
    CrossTotale=eBremCrossSectionPerVolume ( Ekine, id_mat );
    return  CrossTotale;
}

f32 ElectronCrossSection::eBremCrossSectionPerVolume ( f32 Ekine, int id_mat )
{
    int i,n,nn,nmax=100;
    f32  Cross=0.;
    f32  kmax,kmin,vmin,vmax,totalEnergy,kp2;
    f32  u,fac,c,v,dv,y;
    f32  tmax=std::min ( MaxKinEnergy,Ekine );
    f32  cut=std::max ( cutEnergyGamma, ( f32 ).1* ( f32 ) keV );
    f32  fsig=0.;
    f32  highKinEnergy=1.*GeV;
    f32  probsup=1.;
    f32  MigdalConstant=elec_radius*hbarc*hbarc*4.*pi/ ( electron_mass_c2*electron_mass_c2 );

    if ( cut>=tmax )
        return Cross;

    for ( i=0; i<myMaterials.nb_elements[id_mat]; i++ )
    {
        int indexelt= i + myMaterials.index[id_mat];

        Cross+=myMaterials.atom_num_dens[indexelt]
               *eBremCrossSectionPerAtom ( myMaterials.mixture[indexelt],cut, Ekine );
        if ( tmax<Ekine )
            Cross-=myMaterials.atom_num_dens[indexelt]
                   *eBremCrossSectionPerAtom ( myMaterials.mixture[indexelt],tmax,Ekine );
    }

    kmax=tmax;
    kmin=cut;
    totalEnergy=Ekine+electron_mass_c2;
    kp2=MigdalConstant*totalEnergy*totalEnergy*myMaterials.nb_electrons_per_vol[id_mat];
    vmin=log ( kmin );
    vmax=log ( kmax ) ;
    nn= ( int ) ( nmax* ( vmax-vmin ) / ( log ( highKinEnergy )-vmin ) );
    if ( nn>0 )
    {
        dv= ( vmax-vmin ) /nn;
        v=vmin-dv;
        for ( n=0; n<=nn; n++ )
        {
            v+=dv;
            u=exp ( v );
            //fac=SupressionFunction(material,Ekine,u);     //LPM flag is off
            fac=1.;
            y=u/kmax;
            fac*= ( 4.-4.*y+3.*y*y ) /3.;
            fac*=probsup* ( u*u/ ( u*u+kp2 ) ) +1.-probsup;
            if ( ( n==0 ) || ( n==nn ) )
                c=.5;
            else
                c=1.;
            fac*=c;
            fsig+=fac;
        }
        y=kmin/kmax;
        fsig*=dv/ ( -4.*log ( y ) /3.-4.* ( 1.-y ) /3.+0.5* ( 1.-y*y ) );
    }
    else
        fsig=1.;
    if ( fsig>1. )
        fsig=1.;
    Cross*=fsig;

    return Cross;
}
*/

/*
f32 ElectronCrossSection::eBremCrossSectionPerAtom ( f32 Z,f32 cut, f32 Ekine )
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

void ElectronCrossSection::eMSC_CrossSection_table ( int id_mat )
{
    int i;
    for ( i=0; i<nb_bins; i++ )
    {
        f32 Ekine=data_h.E[i];
        data_h.eMSC[i + nb_bins * id_mat]=eMscCrossSection ( Ekine,id_mat );
    }
}
*/

/*
f32 ElectronCrossSection::eMscCrossSection ( f32 Ekine, int id_mat )
{
    int i;
    f32 CrossTotale=0.;

    for ( i=0; i<myMaterials.nb_elements[id_mat]; i++ )
    {
        int indexelt= i + myMaterials.index[id_mat];
        CrossTotale+=myMaterials.atom_num_dens[indexelt]
                     *eMscCrossSectionPerAtom ( Ekine, myMaterials.mixture[indexelt] );
    }

    return  CrossTotale;
}
*/

/*
f32 ElectronCrossSection::eMscCrossSectionPerAtom ( f32 Ekine,  unsigned short int AtomNumber )
{
    f32 AtomicNumber = ( f32 ) AtomNumber;

    int iZ=14,iT=21;
    f32  Cross=0.;
    f32  eKin,eTot,T,E;
    f32  beta2,bg2,b2big,b2small,ratb2,Z23,tau,w;
    f32  Z1,Z2,ratZ;
    f32  c,c1,c2,cc1,cc2,corr;
    f32  Tlim=10.*MeV;
    f32  sigmafactor=2.*pi*elec_radius*elec_radius;
    f32  epsfactor=2.*electron_mass_c2*electron_mass_c2*Bohr_radius*Bohr_radius/ ( hbarc*hbarc );
    f32  eps,epsmin=1.e-4,epsmax=1.e10;
    f32  beta2lim=Tlim* ( Tlim+2.*electron_mass_c2 ) / ( ( Tlim+electron_mass_c2 ) * ( Tlim+electron_mass_c2 ) );
    f32  bg2lim=Tlim* ( Tlim+2.*electron_mass_c2 ) / ( electron_mass_c2*electron_mass_c2 );

    Z23=2.*log ( AtomicNumber ) /3.;
    Z23=exp ( Z23 );



    tau=Ekine/electron_mass_c2;
    c=electron_mass_c2*tau* ( tau+2. ) / ( electron_mass_c2* ( tau+1. ) ); // a simplifier
    w=c-2.;
    tau=.5* ( w+sqrt ( w*w+4.*c ) );
    eKin=electron_mass_c2*tau;


    eTot=eKin+electron_mass_c2;
    beta2=eKin* ( eTot+electron_mass_c2 ) / ( eTot*eTot );
    bg2=eKin* ( eTot+electron_mass_c2 ) / ( electron_mass_c2*electron_mass_c2 );
    eps=epsfactor*bg2/Z23;
    if ( eps<epsmin )
        Cross=2.*eps*eps;
    else if ( eps<epsmax )
        Cross=log ( 1.+2.*eps )-2.*eps/ ( 1.+2.*eps );
    else
        Cross=log ( 2.*eps )-1.+1./eps;
    Cross*=AtomicNumber*AtomicNumber/ ( beta2*bg2 );

    while ( ( iZ>=0 ) && ( Zdat[iZ]>=AtomicNumber ) )
        iZ-=1;
    if ( iZ==14 )
        iZ=13;
    if ( iZ==-1 )
        iZ=0;
    Z1=Zdat[iZ];
    Z2=Zdat[iZ+1];
    ratZ= ( AtomicNumber-Z1 ) * ( AtomicNumber+Z1 ) / ( ( Z2-Z1 ) * ( Z2+Z1 ) );

    if ( eKin<=Tlim )
    {
        while ( ( iT>=0 ) && ( Tdat[iT]>=eKin ) )
            iT-=1;
        if ( iT==21 )
            iT=20;
        if ( iT==-1 )
            iT=0;
        T=Tdat[iT];
        E=T+electron_mass_c2;
        b2small=T* ( E+electron_mass_c2 ) / ( E*E );
        T=Tdat[iT+1];
        E=T+electron_mass_c2;
        b2big=T* ( E+electron_mass_c2 ) / ( E*E );
        ratb2= ( beta2-b2small ) / ( b2big-b2small );

        c1=celectron[iZ][iT];
        c2=celectron[iZ+1][iT];
        cc1=c1+ratZ* ( c2-c1 );
        c1=celectron[iZ][iT+1];
        c2=celectron[iZ+1][iT+1];
        cc2=c1+ratZ* ( c2-c1 );
        corr=cc1+ratb2* ( cc2-cc1 );
        Cross*=sigmafactor/corr;


    }
    else
    {
        c1=bg2lim*sig0[iZ]* ( 1.+hecorr[iZ]* ( beta2-beta2lim ) ) /bg2;
        c2=bg2lim*sig0[iZ+1]* ( 1.+hecorr[iZ+1]* ( beta2-beta2lim ) ) /bg2;
        if ( ( AtomicNumber>=Z1 ) && ( AtomicNumber<=Z2 ) )
            Cross=c1+ratZ* ( c2-c1 );
        else if ( AtomicNumber<Z1 )
            Cross=AtomicNumber*AtomicNumber*c1/ ( Z1*Z1 );
        else if ( AtomicNumber>Z2 )
            Cross=AtomicNumber*AtomicNumber*c2/ ( Z2*Z2 );
    }
    return  Cross;
}
*/

/*
void ElectronCrossSection::m_copy_cs_table_cpu2gpu()
{

        ui32 n = data_h.nb_bins;
        ui32 k = data_h.nb_mat;
// 
//         // Allocate GPU mem
        HANDLE_ERROR( cudaMalloc((void**) &data_d.E              , k*n*sizeof(f32)) );
        HANDLE_ERROR( cudaMalloc((void**) &data_d.eIonisationdedx, k*n*sizeof(f32)) );
        HANDLE_ERROR( cudaMalloc((void**) &data_d.eIonisationCS  , k*n*sizeof(f32)) );
        HANDLE_ERROR( cudaMalloc((void**) &data_d.eBremdedx      , k*n*sizeof(f32)) );
        HANDLE_ERROR( cudaMalloc((void**) &data_d.eBremCS        , k*n*sizeof(f32)) );
        HANDLE_ERROR( cudaMalloc((void**) &data_d.eMSC           , k*n*sizeof(f32)) );
        HANDLE_ERROR( cudaMalloc((void**) &data_d.eRange         , k*n*sizeof(f32)) );
//         HANDLE_ERROR( cudaMalloc((void**) &data_d.pIonisationdedx, k*n*sizeof(f32)) );
//         HANDLE_ERROR( cudaMalloc((void**) &data_d.pIonisationCS  , k*n*sizeof(f32)) );
//         HANDLE_ERROR( cudaMalloc((void**) &data_d.pBremdedx      , k*n*sizeof(f32)) );
//         HANDLE_ERROR( cudaMalloc((void**) &data_d.pBremCS        , k*n*sizeof(f32)) );
//         HANDLE_ERROR( cudaMalloc((void**) &data_d.pMSC           , k*n*sizeof(f32)) );
//         HANDLE_ERROR( cudaMalloc((void**) &data_d.pRange         , k*n*sizeof(f32)) );
//         HANDLE_ERROR( cudaMalloc((void**) &data_d.pAnni_CS_tab   , k*n*sizeof(f32)) );

// 
//         // Copy data to GPU
        data_d.nb_bins = n;
        data_d.nb_mat = k;
        data_d.E_min = data_h.E_min;
        data_d.E_max = data_h.E_max;
        data_d.cutEnergyElectron = data_h.cutEnergyElectron;
        data_d.cutEnergyGamma    = data_h.cutEnergyGamma;
        
        
        HANDLE_ERROR( cudaMemcpy(data_d.E              , data_h.E               , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(data_d.eIonisationdedx, data_h.eIonisationdedx , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(data_d.eIonisationCS  , data_h.eIonisationCS   , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(data_d.eBremdedx      , data_h.eBremdedx       , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(data_d.eBremCS        , data_h.eBremCS         , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(data_d.eMSC           , data_h.eMSC            , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(data_d.eRange         , data_h.eRange          , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
//         HANDLE_ERROR( cudaMemcpy(data_d.pIonisationdedx, data_h.pIonisationdedx , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
//         HANDLE_ERROR( cudaMemcpy(data_d.pIonisationCS  , data_h.pIonisationCS   , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
//         HANDLE_ERROR( cudaMemcpy(data_d.pBremdedx      , data_h.pBremdedx       , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
//         HANDLE_ERROR( cudaMemcpy(data_d.pBremCS        , data_h.pBremCS         , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
//         HANDLE_ERROR( cudaMemcpy(data_d.pMSC           , data_h.pMSC            , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
//         HANDLE_ERROR( cudaMemcpy(data_d.pRange         , data_h.pRange          , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
//         HANDLE_ERROR( cudaMemcpy(data_d.pAnni_CS_tab   , data_h.pAnni_CS_tab    , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );


}
*/

#endif
