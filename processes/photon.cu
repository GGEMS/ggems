// GGEMS Copyright (C) 2015

/*!
 * \file photon.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 18 novembre 2015
 *
 *
 *
 */

#ifndef PHOTON_CU
#define PHOTON_CU
#include "photon.cuh"


__host__ __device__ f32 get_CS_from_table(f32 *E_bins, f32 *CSTable, f32 energy,
                                          ui32 E_index, ui32 CS_index) {


    // xa, ya, xb, yb
    // xa = E_bins[E_index-1]
    // ya = CS[CS_index-1]
    // xb = E_bins[E_index]
    // yb = CS[CS_index]
    // x = energy

    if ( E_index == 0 )
    {
        return CSTable[CS_index];
    }
    else
    {
        return linear_interpolation(E_bins[E_index-1],  CSTable[CS_index-1],
                                    E_bins[E_index],    CSTable[CS_index],
                                    energy);
    }

}

//////// Compton /////////////////////////////////////////////
// Model standard G4
//////////////////////////////////////////////////////////////

// Compton Cross Section Per Atom (Standard - Klein-Nishina)
__host__ __device__ f32 Compton_CSPA_standard(f32 E, ui16 Z) {
    f32 CrossSection = 0.0;
    if (Z<1 || E < 1e-4f) {return CrossSection;}

    f32 p1Z = Z*(2.7965e-23f + 1.9756e-27f*Z + -3.9178e-29f*Z*Z);
    f32 p2Z = Z*(-1.8300e-23f + -1.0205e-24f*Z + 6.8241e-27f*Z*Z);
    f32 p3Z = Z*(6.7527e-22f + -7.3913e-24f*Z + 6.0480e-27f*Z*Z);
    f32 p4Z = Z*(-1.9798e-21f + 2.7079e-24f*Z + 3.0274e-26f*Z*Z);
    f32 T0 = (Z < 1.5f)? 40.0e-3f : 15.0e-3f;
    f32 d1, d2, d3, d4, d5;

    d1 = fmaxf(E, T0) / 0.510998910f; 
    CrossSection = p1Z*logf(1.0f+2.0f*d1)/d1+(p2Z+p3Z*d1+p4Z*d1*d1)/(1.0f+20.0f*d1+230.0f*d1*d1+440.0f*d1*d1*d1);

    if (E < T0) {
        d1 = (T0+1.0e-3f) / 0.510998910f;
        d2 = p1Z*logf(1.0f+2.0f*d1)/d1+(p2Z+p3Z*d1+p4Z*d1*d1)/(1.0f+20.0f*d1+230.0f*d1*d1+440.0f*d1*d1*d1);
        d3 = (-T0 * (d2 - CrossSection)) / (CrossSection*1.0e-3f);
        d4 = (Z > 1.5f)? 0.375f-0.0556f*logf(Z) : 0.15f;
        d5 = logf(E / T0);
        CrossSection *= expf(-d5 * (d3 + d4*d5));
    }
    
    return CrossSection;
}

// Compute the total Compton cross section for a given material
__host__ __device__ f32 Compton_CS_standard(const MaterialsTable materials, ui16 mat, f32 E) {
    f32 CS = 0.0f;
    i32 i;
    i32 index = materials.index[mat];
    // Model standard
    for (i = 0; i < materials.nb_elements[mat]; ++i) {
        CS += (materials.atom_num_dens[index+i] * 
               Compton_CSPA_standard(E, materials.mixture[index+i]));
    }
    return CS;
}

// Compton Scatter (Standard - Klein-Nishina) with secondary (e-)
__host__ __device__ SecParticle Compton_SampleSecondaries_standard(ParticlesData &particles,
                                                                   f32 cutE,
                                                                   ui32 id,
                                                                   const GlobalSimulationParametersData parameters) {

    f32 gamE0 = particles.E[id];
    f32 E0 = gamE0 / 0.510998910f;
    f32xyz gamDir0 = make_f32xyz(particles.dx[id], particles.dy[id], particles.dz[id]);

    // sample the energy rate pf the scattered gamma

    f32 epszero = 1.0f / (1.0f + 2.0f * E0);
    f32 eps02 = epszero*epszero;
    f32 a1 = -logf(epszero);
    f32 a2 = a1 / (a1 + 0.5f*(1.0f-eps02));

    f32 greject, onecost, eps, eps2, sint2, cosTheta, sinTheta, phi;
    do {
        if (a2 > prng_uniform( particles, id )) {
            eps = expf(-a1 * prng_uniform( particles, id ));
            eps2 = eps*eps;
        } else {
            eps2 = eps02 + (1.0f - eps02) * prng_uniform( particles, id );
            eps = sqrt(eps2);
        }
        onecost = (1.0f - eps) / (eps * E0);
        sint2 = onecost * (2.0f - onecost);
        greject = 1.0f - eps * sint2 / (1.0f + eps2);
    } while (greject < prng_uniform( particles, id ));

    // scattered gamma angles

    if (sint2 < 0.0f) {sint2 = 0.0f;}
    cosTheta = 1.0f - onecost;
    sinTheta = sqrt(sint2);
    phi = prng_uniform( particles, id ) * gpu_twopi;

    // update the scattered gamma

    f32xyz gamDir1 = make_f32xyz(sinTheta*cosf(phi), sinTheta*sinf(phi), cosTheta);
    gamDir1 = rotateUz(gamDir1, gamDir0);
    gamDir1 = fxyz_unit( gamDir1 );

    particles.dx[id] = gamDir1.x;
    particles.dy[id] = gamDir1.y;
    particles.dz[id] = gamDir1.z;

    f32 gamE1  = gamE0 * eps;
    if (gamE1 > 1.0e-06f) {particles.E[id] = gamE1;}
    else {
        particles.endsimu[id] = PARTICLE_DEAD;  // absorbed this particle
        particles.E[id] = gamE1;                // Local energy deposit
    }

    // kinematic of the scattered electron

    SecParticle electron;
    electron.pname = ELECTRON;
    electron.E = gamE0 - gamE1; // eKinE
    electron.dir = make_f32xyz(0.0, 0.0, 0.0);
    electron.endsimu = PARTICLE_DEAD;


    //               DBL_MIN                  cut production
    if (electron.E > 1.0e-38f && electron.E > cutE && parameters.secondaries_list[ELECTRON]) {
        electron.dir = fxyz_sub(fxyz_scale(gamDir0, gamE0), fxyz_scale(gamDir1, gamE1));
        electron.dir = fxyz_unit(electron.dir);
        electron.endsimu = PARTICLE_ALIVE;
        //printf("Compton: new e-\n");
    }

    //return e-
    return electron;

}

// Compton Scatter (Standard - Klein-Nishina) without secondary (e-)
__host__ __device__ void Compton_standard(ParticlesData &particles,
                                          f32 cutE,
                                          ui32 id,
                                          const GlobalSimulationParametersData parameters) {

    f32 gamE0 = particles.E[id];
    f32 E0 = gamE0 / 0.510998910f;
    f32xyz gamDir0 = make_f32xyz(particles.dx[id], particles.dy[id], particles.dz[id]);

    // sample the energy rate pf the scattered gamma

    f32 epszero = 1.0f / (1.0f + 2.0f * E0);
    f32 eps02 = epszero*epszero;
    f32 a1 = -logf(epszero);
    f32 a2 = a1 / (a1 + 0.5f*(1.0f-eps02));

    f32 greject, onecost, eps, eps2, sint2, cosTheta, sinTheta, phi;
    do {
        if (a2 > prng_uniform( particles, id )) {
            eps = expf(-a1 * prng_uniform( particles, id ));
            eps2 = eps*eps;
        } else {
            eps2 = eps02 + (1.0f - eps02) * prng_uniform( particles, id );
            eps = sqrt(eps2);
        }
        onecost = (1.0f - eps) / (eps * E0);
        sint2 = onecost * (2.0f - onecost);
        greject = 1.0f - eps * sint2 / (1.0f + eps2);
    } while (greject < prng_uniform( particles, id ));

    // scattered gamma angles

    if (sint2 < 0.0f) {sint2 = 0.0f;}
    cosTheta = 1.0f - onecost;
    sinTheta = sqrt(sint2);
    phi = prng_uniform( particles, id ) * gpu_twopi;

    // update the scattered gamma

    f32xyz gamDir1 = make_f32xyz(sinTheta*cosf(phi), sinTheta*sinf(phi), cosTheta);
    gamDir1 = rotateUz(gamDir1, gamDir0);
    gamDir1 = fxyz_unit( gamDir1 );

    particles.dx[id] = gamDir1.x;
    particles.dy[id] = gamDir1.y;
    particles.dz[id] = gamDir1.z;

    f32 gamE1  = gamE0 * eps;
    if (gamE1 > 1.0e-06f) {particles.E[id] = gamE1;}
    else {
        particles.endsimu[id] = PARTICLE_DEAD;  // absorbed this particle
        particles.E[id] = gamE1;                // Local energy deposit
    }

}

//////// Photoelectric ////////////////////////////////////////
// Model standard G4
//////////////////////////////////////////////////////////////

// PhotoElectric Cross Section Per Atom (Standard)
__host__ __device__ f32 Photoelec_CSPA_standard(f32 E, ui16 Z) {
    // from Sandia data, the same for all Z
    f32 Emin = fmax(PhotoElec_std_IonizationPotentials(Z)*1e-6f, 0.01e-3f);
    if (E < Emin) {return 0.0f;}
    
    i32 start = PhotoElec_std_CumulIntervals(Z-1);
    i32 stop = start + PhotoElec_std_NbIntervals(Z);
    i32 pos=stop;
    while (E < PhotoElec_std_SandiaTable(pos, 0)*1.0e-3f){--pos;}
    f32 AoverAvo = 0.0103642688246f * ( (f32)Z / PhotoElec_std_ZtoAratio(Z) );
    f32 rE = 1.0f / E;
    f32 rE2 = rE*rE;

    return rE * PhotoElec_std_SandiaTable(pos, 1) * AoverAvo * 0.160217648e-22f
        + rE2 * PhotoElec_std_SandiaTable(pos, 2) * AoverAvo * 0.160217648e-25f
        + rE * rE2 * PhotoElec_std_SandiaTable(pos, 3) * AoverAvo * 0.160217648e-28f
        + rE2 * rE2 * PhotoElec_std_SandiaTable(pos, 4) * AoverAvo * 0.160217648e-31f;
}

// Compute the total Compton cross section for a given material
__host__ __device__ f32 Photoelec_CS_standard(const MaterialsTable materials,
                                              ui16 mat, f32 E) {
    f32 CS = 0.0f;
    i32 i;
    i32 index = materials.index[mat];
    // Model standard
    for (i = 0; i < materials.nb_elements[mat]; ++i) {
        CS += (materials.atom_num_dens[index+i] * 
               Photoelec_CSPA_standard(E, materials.mixture[index+i]));
    }
    return CS;
}

// Compute Theta distribution of the emitted electron, with respect to the incident Gamma
// The Sauter-Gavrila distribution for the K-shell is used
__host__ __device__ f32 Photoelec_ElecCosThetaDistribution(ParticlesData &particles,
                                                           ui32 id,
                                                           f32 kineEnergy) {
    f32 costeta = 1.0f;
    f32 gamma = kineEnergy * 1.9569513367f + 1.0f;  // 1/electron_mass_c2
    if (gamma > 5.0f) {return costeta;}
    f32 beta = sqrtf(gamma*gamma - 1.0f) / gamma;
    f32 b    = 0.5f*gamma*(gamma - 1.0f)*(gamma - 2.0f);

    f32 rndm, term, greject, grejsup;
    if (gamma < 2.0f) {grejsup = gamma*gamma*(1.0f + b - beta*b);}
    else              {grejsup = gamma*gamma*(1.0f + b + beta*b);}

    do {
        rndm = 1.0f - 2.0f*prng_uniform( particles, id );
        costeta = (rndm + beta) / (rndm*beta + 1.0f);
        term = 1.0f - beta*costeta;
        greject = ((1.0f - costeta*costeta)*(1.0f + b*term)) / (term*term);
    } while(greject < prng_uniform( particles, id )*grejsup);

    return costeta;
}

// PhotoElectric effect (standard) with secondary (e-)
__host__ __device__ SecParticle Photoelec_SampleSecondaries_standard(ParticlesData &particles,
                                                                     const MaterialsTable mat,
                                                                     const PhotonCrossSectionTable photon_CS_table,
                                                                     ui32 E_index,
                                                                     f32 cutE,
                                                                     ui16 matindex,
                                                                     ui32 id,
                                                                     const GlobalSimulationParametersData parameters) {

    // Kill the photon without mercy
    particles.endsimu[id] = PARTICLE_DEAD;

    // Electron allocation
    SecParticle electron;
    electron.pname = ELECTRON;
    electron.endsimu = PARTICLE_DEAD;
    electron.E = 0.0f;

    // If no secondary required return a stillborn electron
    if (parameters.secondaries_list[ELECTRON] == DISABLED) return electron;

    // Get index CS table (considering mat id)
    ui32 CS_index = matindex*photon_CS_table.nb_bins + E_index;

    //// Photo electron

    f32 energy = particles.E[id];
    f32xyz PhotonDirection = make_f32xyz(particles.dx[id], particles.dy[id], particles.dz[id]);

    // Select randomly one element that composed the material
    ui32 n = mat.nb_elements[matindex]-1;
    ui32 mixture_index = mat.index[matindex];
    ui32 Z = mat.mixture[mixture_index];
    ui32 i = 0;
    if (n > 0) {                
        f32 x = prng_uniform( particles, id ) * get_CS_from_table(photon_CS_table.E_bins,
                                                            photon_CS_table.Photoelectric_Std_CS,
                                                            particles.E[id], E_index, CS_index);
        f32 xsec = 0.0f;
        while (i < n) {
            Z = mat.mixture[mixture_index+i];
            xsec += photon_CS_table.Photoelectric_Std_xCS[Z*photon_CS_table.nb_bins + E_index];
            if (x <= xsec) break;
            ++i;
        }
    }

    // Select atomic shell
    ui16 nShells = atom_NumberOfShells(Z);
    mixture_index = atom_IndexOfShells(Z);
    f32 bindingEnergy = atom_BindingEnergies(mixture_index) * eV; //1.0e-06f; // in eV
    i=0; while (i < nShells && energy < bindingEnergy) {       
        bindingEnergy = atom_BindingEnergies(mixture_index + i)* eV; //1.0e-06f; // in ev
        ++i;
    }
        
    // no shell available return stillborn electron
    if (i == nShells) {return electron;}
    f32 ElecKineEnergy = energy - bindingEnergy;

    f32 cosTeta = 0.0f;
    //                   1 eV                         cut production
    if (ElecKineEnergy > 1.0e-06f && ElecKineEnergy > cutE) {
        // direction of the photo electron
        cosTeta = Photoelec_ElecCosThetaDistribution(particles, id, ElecKineEnergy);
        f32 sinTeta = sqrtf(1.0f - cosTeta*cosTeta);
        f32 Phi = gpu_twopi * prng_uniform( particles, id );
        f32xyz ElecDirection = make_f32xyz(sinTeta*cos(Phi), sinTeta*sin(Phi), cosTeta);
        ElecDirection = rotateUz(ElecDirection, PhotonDirection);
        // Configure the new electron
        electron.dir.x = ElecDirection.x;
        electron.dir.y = ElecDirection.y;
        electron.dir.z = ElecDirection.z;
        electron.E = ElecKineEnergy;
        electron.endsimu = PARTICLE_ALIVE;
        // gamma will depose energy given by the binding energy
        particles.E[id] = bindingEnergy;        
    }
    
    // Return electron (dead or alive)
    return electron;

}


//////// Rayleigh scattering ////////////////////////////////////////
// Model Livermore G4
/////////////////////////////////////////////////////////////////////

// * Usefull to find cross section from Rayleigh table.
// * The table contains [E, cross section, E, cross section, ...] for all Z
// * JB - 2011-02-17 08:46:56

// GPU

__constant__ i32 GPU_Rayleigh_LV_CS_CumulIntervals [101] =
{
        0, // nonexisting 'zero' element

//      H,                                                              He,          (2)
        0,                                                             724,

//     Li,      Be,      B,      C,      N,      O,      F,             Ne,         (10)
     1384,    2392,   3352,   4564,   5832,   7144,   8442,           9584,

//     Na,      Mg,     Al,     Si,      P,      S,     Cl,             Ar,         (18)
    10464,   11652,  12748,  14528,  16162,  17626,  18996,          20302,

//      K,      Ca,     Sc,     Ti,      V,     Cr,     Mn,     Fe,     Co,     Ni, (28)
    21470,   22964,  24530,  26242,  27954,  29606,  31232,  33034,  34834,  36610,

//     Cu,      Zn,     Ga,     Ge,     As,     Se,     Br,             Kr,         (36)
    38360,   40022,  41598,  43976,  46176,  48170,  50000,          51762,

//     Rb,      Sr,      Y,     Zr,     Nb,     Mo,     Tc,     Ru,     Rh,     Pd, (46)
    53174,   54932,  56688,  58474,  60308,  62118,  64228,  66372,  68516,  70738,

//     Ag,      Cd,     In,     Sn,     Sb,     Te,      J,             Xe,         (54)
    72418,   74566,  76542,  79532,  82246,  84688,  86938,          89036,

//     Cs,      Ba,     La,     Ce,     Pr,     Nd,     Pm,     Sm,     Eu,     Gd, (64)
    90860,   93094,  95420,  97738, 100198, 102636, 105000, 107432, 110000, 112622,

//      Tb,     Dy,     Ho,     Er,     Tm,     Yb,     Lu,     Hf,     Ta,      W, (74)
    115364, 118050, 120826, 123548, 126050, 128678, 131138, 133880, 136530, 138866,

//      Re,     Os,     Ir,     Pt,     Au,     Hg,     Tl,     Pb,     Bi,     Po, (84)
    140858, 143108, 145294, 147348, 149674, 151894, 154120, 157430, 160686, 163556,

//      At,     Rn,     Fr,     Ra,     Ac,     Th,     Pa,      U,     Np,     Pu, (94)
    166206, 168652, 170824, 173534, 176404, 179562, 182544, 185612, 188852, 192076,

//      Am,     Cm,     Bk,     Cf,     Es,     Fm                                  (100)
    194904, 197808, 201370, 204620, 207702, 210732
};

__constant__ ui16 GPU_Rayleigh_LV_CS_NbIntervals [101] =
{
       0, // nonexisting 'zero' element

//     H,                                             He,                (2)
     362,                                            330,

//    Li,   Be,    B,    C,    N,    O,    F,         Ne,               (10)
     504,  480,  606,  634,  656,  649,  571,        440,

//    Na,   Mg,   Al,   Si,    P,    S,   Cl,         Ar,               (18)
     594,  548,  890,  817,  732,  685,  653,        584,

//     K,   Ca,   Sc,   Ti,    V,   Cr,   Mn,   Fe,   Co,   Ni,         (28)
     747,  783,  856,  856,  826,  813,  901,  900,  888,  875,

//    Cu,   Zn,   Ga,   Ge,   As,   Se,   Br,         Kr,               (36)
     831,  788, 1189, 1100,  997,  915,  881,        706,

//    Rb,   Sr,    Y,   Zr,   Nb,   Mo,   Tc,   Ru,   Rh,   Pd,         (46)
     879,  878,  893,  917,  905, 1055, 1072, 1072, 1111,  840,

//    Ag,   Cd,   In,   Sn,   Sb,   Te,    J,         Xe,               (54)
    1074,  988, 1495, 1357, 1221, 1125, 1049,        912,

//    Cs,   Ba,   La,   Ce,   Pr,   Nd,   Pm,   Sm,   Eu,   Gd,         (64)
    1117, 1163, 1159, 1230, 1219, 1182, 1216, 1284, 1311, 1371,

//    Tb,   Dy,   Ho,   Er,   Tm,   Yb,   Lu,   Hf,   Ta,    W,         (74)
    1343, 1388, 1361, 1251, 1314, 1230, 1371, 1325, 1168,  996,

//    Re,   Os,   Ir,   Pt,   Au,   Hg,   Tl,   Pb,   Bi,   Po,         (84)
    1125, 1093, 1027, 1163, 1110, 1113, 1655, 1628, 1435, 1325,

//    At,   Rn,   Fr,   Ra,   Ac,   Th,   Pa,    U,   Np,   Pu,         (94)
    1223, 1086, 1355, 1435, 1579, 1491, 1534, 1620, 1612, 1414,

//    Am,   Cm,   Bk,   Cf,   Es,   Fm                                  (100)
    1452, 1781, 1625, 1541, 1515, 1542
};

__constant__ ui16 GPU_Rayleigh_LV_SF_CumulIntervals [101] =
{
        0, // nonexisting 'zero' element

//      H,                                                              He,          (2)
        0,                                                             180,

//     Li,      Be,      B,      C,      N,      O,      F,             Ne,         (10)
      434,     698,    966,   1228,   1504,   1818,   2136,           2470,

//     Na,      Mg,     Al,     Si,      P,      S,     Cl,             Ar,         (18)
     2794,    3134,   3460,   3758,   4064,   4362,   4658,           4962,

//      K,      Ca,     Sc,     Ti,      V,     Cr,     Mn,     Fe,     Co,     Ni, (28)
     5250,    5550,   5852,   6142,   6434,   6724,   7028,   7322,   7622,   7918,

//     Cu,      Zn,     Ga,     Ge,     As,     Se,     Br,             Kr,         (36)
     8230,    8530,   8832,   9130,   9432,   9732,  10036,          10336,

//     Rb,      Sr,      Y,     Zr,     Nb,     Mo,     Tc,     Ru,     Rh,     Pd, (46)
    10618,   10904,  11192,  11480,  11786,  12096,  12396,  12690,  12976,  13262,

//     Ag,      Cd,     In,     Sn,     Sb,     Te,      J,             Xe,         (54)
    13552,   13842,  14136,  14432,  14714,  14992,  15274,          15566,

//     Cs,      Ba,     La,     Ce,     Pr,     Nd,     Pm,     Sm,     Eu,     Gd, (64)
    15858,   16140,  16428,  16730,  17030,  17320,  17600,  17904,  18208,  18494,

//      Tb,     Dy,     Ho,     Er,     Tm,     Yb,     Lu,     Hf,     Ta,      W, (74)
     18792,  19084,  19366,  19668,  19952,  20246,  20550,  20840,  21132,  21428,

//      Re,     Os,     Ir,     Pt,     Au,     Hg,     Tl,     Pb,     Bi,     Po, (84)
     21722,  22010,  22286,  22556,  22822,  23084,  23356,  23638,  23912,  24196,

//      At,     Rn,     Fr,     Ra,     Ac,     Th,     Pa,      U,     Np,     Pu, (94)
     24474,  24764,  25038,  25312,  25584,  25852,  26124,  26392,  26672,  26942,

//      Am,     Cm,     Bk,     Cf,     Es,     Fm                                  (100)
     27216,  27484,  27752,  28018,  28288,  28558
};

__constant__ ui16 GPU_Rayleigh_LV_SF_NbIntervals [101] =
{
       0, // nonexisting 'zero' element

//     H,                                             He,                (2)
      90,                                            127,

//    Li,   Be,    B,    C,    N,    O,    F,         Ne,               (10)
     132,  134,  131,  138,  157,  159,  167,        162,

//    Na,   Mg,   Al,   Si,    P,    S,   Cl,         Ar,               (18)
     170,  163,  149,  153,  149,  148,  152,        144,

//     K,   Ca,   Sc,   Ti,    V,   Cr,   Mn,   Fe,   Co,   Ni,         (28)
     150,  151,  145,  146,  145,  152,  147,  150,  148,  156,

//    Cu,   Zn,   Ga,   Ge,   As,   Se,   Br,         Kr,               (36)
     150,  151,  149,  151,  150,  152,  150,        141,

//    Rb,   Sr,    Y,   Zr,   Nb,   Mo,   Tc,   Ru,   Rh,   Pd,         (46)
     143,  144,  144,  153,  155,  150,  147,  143,  143,  145,

//    Ag,   Cd,   In,   Sn,   Sb,   Te,    J,         Xe,               (54)
     145,  147,  148,  141,  139,  141,  146,        146,

//    Cs,   Ba,   La,   Ce,   Pr,   Nd,   Pm,   Sm,   Eu,   Gd,         (64)
     141,  144,  151,  150,  145,  140,  152,  152,  143,  149,

//    Tb,   Dy,   Ho,   Er,   Tm,   Yb,   Lu,   Hf,   Ta,    W,         (74)
     146,  141,  151,  142,  147,  152,  145,  146,  148,  147,

//    Re,   Os,   Ir,   Pt,   Au,   Hg,   Tl,   Pb,   Bi,   Po,         (84)
     144,  138,  135,  133,  131,  136,  141,  137,  142,  139,

//    At,   Rn,   Fr,   Ra,   Ac,   Th,   Pa,    U,   Np,   Pu,         (94)
     145,  137,  137,  136,  134,  136,  134,  140,  135,  137,

//    Am,   Cm,   Bk,   Cf,   Es,   Fm                                  (100)
     134,  134,  133,  135,  135,  133
};

// CPU

#ifndef __CUDA_ARCH__
const i32 CPU_Rayleigh_LV_CS_CumulIntervals [101] =
{
        0, // nonexisting 'zero' element

//      H,                                                              He,          (2)
        0,                                                             724,

//     Li,      Be,      B,      C,      N,      O,      F,             Ne,         (10)
     1384,    2392,   3352,   4564,   5832,   7144,   8442,           9584,

//     Na,      Mg,     Al,     Si,      P,      S,     Cl,             Ar,         (18)
    10464,   11652,  12748,  14528,  16162,  17626,  18996,          20302,

//      K,      Ca,     Sc,     Ti,      V,     Cr,     Mn,     Fe,     Co,     Ni, (28)
    21470,   22964,  24530,  26242,  27954,  29606,  31232,  33034,  34834,  36610,

//     Cu,      Zn,     Ga,     Ge,     As,     Se,     Br,             Kr,         (36)
    38360,   40022,  41598,  43976,  46176,  48170,  50000,          51762,

//     Rb,      Sr,      Y,     Zr,     Nb,     Mo,     Tc,     Ru,     Rh,     Pd, (46)
    53174,   54932,  56688,  58474,  60308,  62118,  64228,  66372,  68516,  70738,

//     Ag,      Cd,     In,     Sn,     Sb,     Te,      J,             Xe,         (54)
    72418,   74566,  76542,  79532,  82246,  84688,  86938,          89036,

//     Cs,      Ba,     La,     Ce,     Pr,     Nd,     Pm,     Sm,     Eu,     Gd, (64)
    90860,   93094,  95420,  97738, 100198, 102636, 105000, 107432, 110000, 112622,

//      Tb,     Dy,     Ho,     Er,     Tm,     Yb,     Lu,     Hf,     Ta,      W, (74)
    115364, 118050, 120826, 123548, 126050, 128678, 131138, 133880, 136530, 138866,

//      Re,     Os,     Ir,     Pt,     Au,     Hg,     Tl,     Pb,     Bi,     Po, (84)
    140858, 143108, 145294, 147348, 149674, 151894, 154120, 157430, 160686, 163556,

//      At,     Rn,     Fr,     Ra,     Ac,     Th,     Pa,      U,     Np,     Pu, (94)
    166206, 168652, 170824, 173534, 176404, 179562, 182544, 185612, 188852, 192076,

//      Am,     Cm,     Bk,     Cf,     Es,     Fm                                  (100)
    194904, 197808, 201370, 204620, 207702, 210732
};

const ui16 CPU_Rayleigh_LV_CS_NbIntervals [101] =
{
       0, // nonexisting 'zero' element

//     H,                                             He,                (2)
     362,                                            330,

//    Li,   Be,    B,    C,    N,    O,    F,         Ne,               (10)
     504,  480,  606,  634,  656,  649,  571,        440,

//    Na,   Mg,   Al,   Si,    P,    S,   Cl,         Ar,               (18)
     594,  548,  890,  817,  732,  685,  653,        584,

//     K,   Ca,   Sc,   Ti,    V,   Cr,   Mn,   Fe,   Co,   Ni,         (28)
     747,  783,  856,  856,  826,  813,  901,  900,  888,  875,

//    Cu,   Zn,   Ga,   Ge,   As,   Se,   Br,         Kr,               (36)
     831,  788, 1189, 1100,  997,  915,  881,        706,

//    Rb,   Sr,    Y,   Zr,   Nb,   Mo,   Tc,   Ru,   Rh,   Pd,         (46)
     879,  878,  893,  917,  905, 1055, 1072, 1072, 1111,  840,

//    Ag,   Cd,   In,   Sn,   Sb,   Te,    J,         Xe,               (54)
    1074,  988, 1495, 1357, 1221, 1125, 1049,        912,

//    Cs,   Ba,   La,   Ce,   Pr,   Nd,   Pm,   Sm,   Eu,   Gd,         (64)
    1117, 1163, 1159, 1230, 1219, 1182, 1216, 1284, 1311, 1371,

//    Tb,   Dy,   Ho,   Er,   Tm,   Yb,   Lu,   Hf,   Ta,    W,         (74)
    1343, 1388, 1361, 1251, 1314, 1230, 1371, 1325, 1168,  996,

//    Re,   Os,   Ir,   Pt,   Au,   Hg,   Tl,   Pb,   Bi,   Po,         (84)
    1125, 1093, 1027, 1163, 1110, 1113, 1655, 1628, 1435, 1325,

//    At,   Rn,   Fr,   Ra,   Ac,   Th,   Pa,    U,   Np,   Pu,         (94)
    1223, 1086, 1355, 1435, 1579, 1491, 1534, 1620, 1612, 1414,

//    Am,   Cm,   Bk,   Cf,   Es,   Fm                                  (100)
    1452, 1781, 1625, 1541, 1515, 1542
};

const ui16 CPU_Rayleigh_LV_SF_CumulIntervals [101] =
{
        0, // nonexisting 'zero' element

//      H,                                                              He,          (2)
        0,                                                             180,

//     Li,      Be,      B,      C,      N,      O,      F,             Ne,         (10)
      434,     698,    966,   1228,   1504,   1818,   2136,           2470,

//     Na,      Mg,     Al,     Si,      P,      S,     Cl,             Ar,         (18)
     2794,    3134,   3460,   3758,   4064,   4362,   4658,           4962,

//      K,      Ca,     Sc,     Ti,      V,     Cr,     Mn,     Fe,     Co,     Ni, (28)
     5250,    5550,   5852,   6142,   6434,   6724,   7028,   7322,   7622,   7918,

//     Cu,      Zn,     Ga,     Ge,     As,     Se,     Br,             Kr,         (36)
     8230,    8530,   8832,   9130,   9432,   9732,  10036,          10336,

//     Rb,      Sr,      Y,     Zr,     Nb,     Mo,     Tc,     Ru,     Rh,     Pd, (46)
    10618,   10904,  11192,  11480,  11786,  12096,  12396,  12690,  12976,  13262,

//     Ag,      Cd,     In,     Sn,     Sb,     Te,      J,             Xe,         (54)
    13552,   13842,  14136,  14432,  14714,  14992,  15274,          15566,

//     Cs,      Ba,     La,     Ce,     Pr,     Nd,     Pm,     Sm,     Eu,     Gd, (64)
    15858,   16140,  16428,  16730,  17030,  17320,  17600,  17904,  18208,  18494,

//      Tb,     Dy,     Ho,     Er,     Tm,     Yb,     Lu,     Hf,     Ta,      W, (74)
     18792,  19084,  19366,  19668,  19952,  20246,  20550,  20840,  21132,  21428,

//      Re,     Os,     Ir,     Pt,     Au,     Hg,     Tl,     Pb,     Bi,     Po, (84)
     21722,  22010,  22286,  22556,  22822,  23084,  23356,  23638,  23912,  24196,

//      At,     Rn,     Fr,     Ra,     Ac,     Th,     Pa,      U,     Np,     Pu, (94)
     24474,  24764,  25038,  25312,  25584,  25852,  26124,  26392,  26672,  26942,

//      Am,     Cm,     Bk,     Cf,     Es,     Fm                                  (100)
     27216,  27484,  27752,  28018,  28288,  28558
};

const ui16 CPU_Rayleigh_LV_SF_NbIntervals [101] =
{
       0, // nonexisting 'zero' element

//     H,                                             He,                (2)
      90,                                            127,

//    Li,   Be,    B,    C,    N,    O,    F,         Ne,               (10)
     132,  134,  131,  138,  157,  159,  167,        162,

//    Na,   Mg,   Al,   Si,    P,    S,   Cl,         Ar,               (18)
     170,  163,  149,  153,  149,  148,  152,        144,

//     K,   Ca,   Sc,   Ti,    V,   Cr,   Mn,   Fe,   Co,   Ni,         (28)
     150,  151,  145,  146,  145,  152,  147,  150,  148,  156,

//    Cu,   Zn,   Ga,   Ge,   As,   Se,   Br,         Kr,               (36)
     150,  151,  149,  151,  150,  152,  150,        141,

//    Rb,   Sr,    Y,   Zr,   Nb,   Mo,   Tc,   Ru,   Rh,   Pd,         (46)
     143,  144,  144,  153,  155,  150,  147,  143,  143,  145,

//    Ag,   Cd,   In,   Sn,   Sb,   Te,    J,         Xe,               (54)
     145,  147,  148,  141,  139,  141,  146,        146,

//    Cs,   Ba,   La,   Ce,   Pr,   Nd,   Pm,   Sm,   Eu,   Gd,         (64)
     141,  144,  151,  150,  145,  140,  152,  152,  143,  149,

//    Tb,   Dy,   Ho,   Er,   Tm,   Yb,   Lu,   Hf,   Ta,    W,         (74)
     146,  141,  151,  142,  147,  152,  145,  146,  148,  147,

//    Re,   Os,   Ir,   Pt,   Au,   Hg,   Tl,   Pb,   Bi,   Po,         (84)
     144,  138,  135,  133,  131,  136,  141,  137,  142,  139,

//    At,   Rn,   Fr,   Ra,   Ac,   Th,   Pa,    U,   Np,   Pu,         (94)
     145,  137,  137,  136,  134,  136,  134,  140,  135,  137,

//    Am,   Cm,   Bk,   Cf,   Es,   Fm                                  (100)
     134,  134,  133,  135,  135,  133
};
#endif

// Functions allowing to fetch value according if GPU or CPU code is used.

__host__ __device__ ui16 Rayleigh_LV_CS_CumulIntervals(ui32 pos) {

#ifdef __CUDA_ARCH__
    return GPU_Rayleigh_LV_CS_CumulIntervals[pos];
#else
    return CPU_Rayleigh_LV_CS_CumulIntervals[pos];
#endif

}

__host__ __device__ ui16 Rayleigh_LV_CS_NbIntervals(ui32 pos) {

#ifdef __CUDA_ARCH__
    return GPU_Rayleigh_LV_CS_NbIntervals[pos];
#else
    return CPU_Rayleigh_LV_CS_NbIntervals[pos];
#endif

}

__host__ __device__ ui16 Rayleigh_LV_SF_CumulIntervals(ui32 pos) {

#ifdef __CUDA_ARCH__
    return GPU_Rayleigh_LV_SF_CumulIntervals[pos];
#else
    return CPU_Rayleigh_LV_SF_CumulIntervals[pos];
#endif

}

__host__ __device__ ui16 Rayleigh_LV_SF_NbIntervals(ui32 pos) {

#ifdef __CUDA_ARCH__
    return GPU_Rayleigh_LV_SF_NbIntervals[pos];
#else
    return CPU_Rayleigh_LV_SF_NbIntervals[pos];
#endif

}


// Load CS information from G4 Em data
f32* Rayleigh_CS_Livermore_load_data() {
    const i32 ncs = 213816;  // CS file contains 213,816 floats
    ui32 mem_cs = ncs * sizeof(f32);
    f32* raylcs = (f32*)malloc(mem_cs);

    std::string filename = std::string(getenv("GGEMSHOME"));
    filename += "/data/rayleigh_cs.bin";

    FILE * pfile = fopen(filename.c_str(), "rb");

    if( !pfile )
    {
        GGcerr << "Error to open the file '" << filename << "'!" << GGendl;
        exit_simulation();
    }

    fread(raylcs, sizeof(f32), ncs, pfile);
    fclose(pfile);

    return raylcs;
}

// Load SF information from G4 Em data
f32* Rayleigh_SF_Livermore_load_data() {
    const i32 nsf = 28824;   // SF file contains  28,824 floats
    ui32 mem_sf = nsf * sizeof(f32);
    f32* raylsf = (f32*)malloc(mem_sf);

    std::string filename = std::string(getenv("GGEMSHOME"));
    filename += "/data/rayleigh_sf.bin";

    FILE * pfile = fopen(filename.c_str(), "rb");

    if( !pfile )
    {
           GGcerr << "Error to open the file '" << filename << "'!" << GGendl;
           exit_simulation();
    }

    fread(raylsf, sizeof(f32), nsf, pfile);
    fclose(pfile);

    return raylsf;
}

// Rayleigh Cross Section Per Atom (Livermore)
__host__ __device__ f32 Rayleigh_CSPA_Livermore(f32* rayl_cs, f32 E, ui16 Z) {
    if (E < 250e-6f || E > 100e3f) {return 0.0f;} // 250 eV < E < 100 GeV

    i32 start = Rayleigh_LV_CS_CumulIntervals(Z);
    i32 stop  = start + 2 * (Rayleigh_LV_CS_NbIntervals(Z) - 1);

    i32 pos;
    for (pos=start; pos<stop; pos+=2) {
        if (rayl_cs[pos] >= E) break;
    }

    if (E < 1e3f) { // 1 Gev
        return 1.0e-22f * loglog_interpolation(E, rayl_cs[pos-2], rayl_cs[pos-1],
                                               rayl_cs[pos], rayl_cs[pos+1]);
    }
    else {return 1.0e-22f * rayl_cs[pos-1];}
}

// Compute the total Compton cross section for a given material
__host__ __device__ f32 Rayleigh_CS_Livermore(const MaterialsTable materials,
                                              f32* rayl_cs, ui16 mat, f32 E) {
    f32 CS = 0.0f;
    i32 i;
    i32 index = materials.index[mat];
    // Model Livermore
    for (i = 0; i < materials.nb_elements[mat]; ++i) {
        CS += (materials.atom_num_dens[index+i] *
               Rayleigh_CSPA_Livermore(rayl_cs, E, materials.mixture[index+i]));
    }
    return CS;
}


// Rayleigh Scatter Factor (Livermore)
__host__ __device__ f32 Rayleigh_SF_Livermore(f32* rayl_sf, f32 E, i32 Z) {
    i32 start = Rayleigh_LV_SF_CumulIntervals(Z);
    i32 stop = start + 2 * (Rayleigh_LV_SF_NbIntervals(Z) - 1);

    // check boundary
    if (E==0.0f) return rayl_sf[start+1];

    i32 pos;
    for (pos=start; pos<stop; pos+=2) {
        if (rayl_sf[pos]*eV >= E) break;  // SF data are in eV
    }

    // Return loglog interpolation

    // If the min bin of energy is equal to 0, loglog is not possible (return Inf)
    if (rayl_sf[pos-2] == 0.0f) {
        return rayl_sf[pos-1];
    } else {
        return loglog_interpolation(E, rayl_sf[pos-2]*eV, rayl_sf[pos-1],
                                        rayl_sf[pos]*eV, rayl_sf[pos+1]);
    }

}

// Rayleigh Scattering (Livermore)
__host__ __device__ void Rayleigh_SampleSecondaries_Livermore(ParticlesData &particles,
                                                              const MaterialsTable mat,
                                                              const PhotonCrossSectionTable photon_CS_table,
                                                              ui32 E_index,
                                                              ui16 matindex,
                                                              ui32 id) {

    if (particles.E[id] <= 250.0e-6f) { // 250 eV
        // Kill the photon without mercy
        particles.endsimu[id] = PARTICLE_DEAD;
        return;
    }   

    // Select randomly one element that composed the material
    ui32 n = mat.nb_elements[matindex]-1;
    ui32 mixture_index = mat.index[matindex];
    ui32 Z = mat.mixture[mixture_index];
    ui32 i = 0;
    if (n > 0) {
        f32 x = prng_uniform( particles, id ) * linear_interpolation(photon_CS_table.E_bins[E_index-1],
                                                               photon_CS_table.Rayleigh_Lv_CS[E_index-1],
                                                               photon_CS_table.E_bins[E_index],
                                                               photon_CS_table.Rayleigh_Lv_CS[E_index],
                                                               particles.E[id]);
        f32 xsec = 0.0f;
        while (i < n) {
            Z = mat.mixture[mixture_index+i];
            xsec += photon_CS_table.Rayleigh_Lv_xCS[Z*photon_CS_table.nb_bins + E_index];
            if (x <= xsec) break;
            ++i;
        }
    }

    // Scattering
    f32 wphot = 1.23984187539e-10f / particles.E[id];
    f32 costheta, SF, x, sintheta, phi;
    do {
        do {costheta = 2.0f * prng_uniform( particles, id ) - 1.0f;
        } while ((1.0f + costheta*costheta)*0.5f < prng_uniform( particles, id ));
        if (particles.E[id] > 5.0f) {costheta = 1.0f;}
        x = sqrt((1.0f - costheta) * 0.5f) / wphot;

        if (x > 1.0e+05f) {
            SF = linear_interpolation(photon_CS_table.E_bins[E_index-1],
                                      photon_CS_table.Rayleigh_Lv_SF[Z*photon_CS_table.nb_bins + E_index-1],
                                      photon_CS_table.E_bins[E_index],
                                      photon_CS_table.Rayleigh_Lv_SF[Z*photon_CS_table.nb_bins + E_index],
                                      particles.E[id]);
        } else {
            SF = photon_CS_table.Rayleigh_Lv_SF[Z*photon_CS_table.nb_bins]; // for energy E=0.0f
        }

    } while (SF*SF < prng_uniform( particles, id ) * Z*Z);

    sintheta = sqrt(1.0f - costheta*costheta);
    phi = prng_uniform( particles, id ) * gpu_twopi;

    // Apply deflection
    f32xyz gamDir0 = make_f32xyz(particles.dx[id], particles.dy[id], particles.dz[id]);
    f32xyz gamDir1 = make_f32xyz(sintheta*cosf(phi), sintheta*sinf(phi), costheta);
    gamDir1 = rotateUz(gamDir1, gamDir0);
    gamDir1 = fxyz_unit( gamDir1 );
    particles.dx[id] = gamDir1.x;
    particles.dy[id] = gamDir1.y;
    particles.dz[id] = gamDir1.z;

}



#endif

