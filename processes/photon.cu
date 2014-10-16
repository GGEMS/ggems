// This file is part of GGEMS
//
// GGEMS is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// GGEMS is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with GGEMS.  If not, see <http://www.gnu.org/licenses/>.
//
// GGEMS Copyright (C) 2013-2014 Julien Bert

#ifndef PHOTON_CU
#define PHOTON_CU
#include "photon.cuh"


//////// Compton /////////////////////////////////////////////
// Model standard G4
//////////////////////////////////////////////////////////////

// Compton Cross Section Per Atom (Standard - Klein-Nishina)
__host__ __device__ float Compton_CSPA_standard(float E, unsigned short int Z) {
    float CrossSection = 0.0;
    if (Z<1 || E < 1e-4f) {return CrossSection;}

    float p1Z = Z*(2.7965e-23f + 1.9756e-27f*Z + -3.9178e-29f*Z*Z);
    float p2Z = Z*(-1.8300e-23f + -1.0205e-24f*Z + 6.8241e-27f*Z*Z);
    float p3Z = Z*(6.7527e-22f + -7.3913e-24f*Z + 6.0480e-27f*Z*Z);
    float p4Z = Z*(-1.9798e-21f + 2.7079e-24f*Z + 3.0274e-26f*Z*Z);
    float T0 = (Z < 1.5f)? 40.0e-3f : 15.0e-3f;
    float d1, d2, d3, d4, d5;

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
__host__ __device__ float Compton_CS_standard(MaterialsTable materials, unsigned short int mat, float E) {
    float CS = 0.0f;
    int i;
    int index = materials.index[mat];
    // Model standard
    for (i = 0; i < materials.nb_elements[mat]; ++i) {
        CS += (materials.atom_num_dens[index+i] * 
               Compton_CSPA_standard(E, materials.mixture[index+i]));
    }
    return CS;
}


// Compton Scatter (Standard - Klein-Nishina) with secondary (e-)
__host__ __device__ SecParticle Compton_SampleSecondaries_standard(ParticleStack particles,
                                                                   float cutE,
                                                                   unsigned int id,
                                                                   GlobalSimulationParameters parameters) {

    float gamE0 = particles.E[id];
    float E0 = gamE0 / 0.510998910f;
    float3 gamDir0 = make_float3(particles.dx[id], particles.dy[id], particles.dz[id]);

    // sample the energy rate pf the scattered gamma

    float epszero = 1.0f / (1.0f + 2.0f * E0);
    float eps02 = epszero*epszero;
    float a1 = -logf(epszero);
    float a2 = a1 / (a1 + 0.5f*(1.0f-eps02));

    float greject, onecost, eps, eps2, sint2, cosTheta, sinTheta, phi;
    do {
        if (a2 > JKISS32(particles, id)) {
            eps = expf(-a1 * JKISS32(particles, id));
            eps2 = eps*eps;
        } else {
            eps2 = eps02 + (1.0f - eps02) * JKISS32(particles, id);
            eps = sqrt(eps2);
        }
        onecost = (1.0f - eps) / (eps * E0);
        sint2 = onecost * (2.0f - onecost);
        greject = 1.0f - eps * sint2 / (1.0f + eps2);
    } while (greject < JKISS32(particles, id));

    // scattered gamma angles

    if (sint2 < 0.0f) {sint2 = 0.0f;}
    cosTheta = 1.0f - onecost;
    sinTheta = sqrt(sint2);
    phi = JKISS32(particles, id) * gpu_twopi;

    // update the scattered gamma

    float3 gamDir1 = make_float3(sinTheta*cosf(phi), sinTheta*sinf(phi), cosTheta);
    gamDir1 = rotateUz(gamDir1, gamDir0);

    particles.dx[id] = gamDir1.x;
    particles.dy[id] = gamDir1.y;
    particles.dz[id] = gamDir1.z;
    float gamE1  = gamE0 * eps;
    if (gamE1 > 1.0e-06f) {particles.E[id] = gamE1;}
    else {
        particles.endsimu[id] = PARTICLE_DEAD;  // absorbed this particle
        particles.E[id] = gamE1;                // Local energy deposit
    }

    // kinematic of the scattered electron

    SecParticle electron;
    electron.pname = ELECTRON;
    electron.E = gamE0 - gamE1; // eKinE
    electron.dir = make_float3(0.0, 0.0, 0.0);
    electron.endsimu = PARTICLE_DEAD;

    //          DBL_MIN             cut production
    if (electron.E > 1.0e-38f && electron.E > cutE && parameters.secondaries_list[ELECTRON]) {
        electron.dir = f3_sub(f3_scale(gamDir0, gamE0), f3_scale(gamDir1, gamE1));
        electron.dir = f3_unit(electron.dir);
        electron.endsimu = PARTICLE_ALIVE;
    }

    //return e-
    return electron;
}

//////// Photoelectric ////////////////////////////////////////
// Model standard G4
//////////////////////////////////////////////////////////////

// PhotoElectric Cross Section Per Atom (Standard)
__host__ __device__ float PhotoElec_CSPA_standard(float E, unsigned short int Z) {
    // from Sandia data, the same for all Z
    float Emin = fmax(PhotoElec_std_IonizationPotentials(Z)*1e-6f, 0.01e-3f);
    if (E < Emin) {return 0.0f;}
    
    int start = PhotoElec_std_CumulIntervals(Z-1);
    int stop = start + PhotoElec_std_NbIntervals(Z);
    int pos=stop;
    while (E < PhotoElec_std_SandiaTable(pos, 0)*1.0e-3f){--pos;}
    float AoverAvo = 0.0103642688246f * ( (float)Z / PhotoElec_std_ZtoAratio(Z) );
    float rE = 1.0f / E;
    float rE2 = rE*rE;

    return rE * PhotoElec_std_SandiaTable(pos, 1) * AoverAvo * 0.160217648e-22f
        + rE2 * PhotoElec_std_SandiaTable(pos, 2) * AoverAvo * 0.160217648e-25f
        + rE * rE2 * PhotoElec_std_SandiaTable(pos, 3) * AoverAvo * 0.160217648e-28f
        + rE2 * rE2 * PhotoElec_std_SandiaTable(pos, 4) * AoverAvo * 0.160217648e-31f;
}

// Compute the total Compton cross section for a given material
__host__ __device__ float PhotoElec_CS_standard(MaterialsTable materials,
                                                unsigned short int mat, float E) {
    float CS = 0.0f;
    int i;
    int index = materials.index[mat];
    // Model standard
    for (i = 0; i < materials.nb_elements[mat]; ++i) {
        CS += (materials.atom_num_dens[index+i] * 
               PhotoElec_CSPA_standard(E, materials.mixture[index+i]));
    }
    return CS;
}

// Compute Theta distribution of the emitted electron, with respect to the incident Gamma
// The Sauter-Gavrila distribution for the K-shell is used
__host__ __device__ float PhotoElec_ElecCosThetaDistribution(ParticleStack part,
                                                             unsigned int id,
                                                             float kineEnergy) {
    float costeta = 1.0f;
    float gamma = kineEnergy * 1.9569513367f + 1.0f;  // 1/electron_mass_c2
    if (gamma > 5.0f) {return costeta;}
    float beta = sqrtf(gamma*gamma - 1.0f) / gamma;
    float b    = 0.5f*gamma*(gamma - 1.0f)*(gamma - 2.0f);

    float rndm, term, greject, grejsup;
    if (gamma < 2.0f) {grejsup = gamma*gamma*(1.0f + b - beta*b);}
    else              {grejsup = gamma*gamma*(1.0f + b + beta*b);}

    do {
        rndm = 1.0f - 2.0f*JKISS32(part,id);
        costeta = (rndm + beta) / (rndm*beta + 1.0f);
        term = 1.0f - beta*costeta;
        greject = ((1.0f - costeta*costeta)*(1.0f + b*term)) / (term*term);
    } while(greject < JKISS32(part,id)*grejsup);

    return costeta;
}

// PhotoElectric effect (standard) with secondary (e-)
__host__ __device__ SecParticle PhotoElec_SampleSecondaries_standard(ParticleStack particles,
                                                                     MaterialsTable mat,
                                                                     float cutE,
                                                                     unsigned short int matindex,
                                                                     unsigned int id,
                                                                     GlobalSimulationParameters parameters) {

    // Kill the photon without mercy
    particles.endsimu[id] = PARTICLE_DEAD;

    // Electron allocation
    SecParticle electron;
    electron.pname = ELECTRON;
    electron.endsimu = PARTICLE_DEAD;
    electron.E = 0.0f;

    // If no secondary required return a stillborn electron
    if (parameters.secondaries_list[ELECTRON] == DISABLED) return electron;

    //// Photo electron

    float energy = particles.E[id];
    //float cutE = mat.electron_cut_energy[matindex]; // TODO - JB
    float3 PhotonDirection = make_float3(particles.dx[id], particles.dy[id], particles.dz[id]);

    //*******************************************************************************
    // TODO - build a table of xSection into MaterialTable - JB

    // Select randomly one element constituing the material
    unsigned int n = mat.nb_elements[matindex]-1;
    unsigned int index = mat.index[matindex];
    unsigned int Z = mat.mixture[index+n];
    unsigned int i = 0;
    if (n > 0) {
        float x = JKISS32(particles,id) * PhotoElec_CS_standard(mat, matindex, energy);
        float xsec = 0.0f;
        while (i < n) {
            xsec += mat.atom_num_dens[index+i] * PhotoElec_CSPA_standard(energy, mat.mixture[index+i]);
            if (x <= xsec) {
                Z = mat.mixture[index+i];
                break;
            }
            ++i;
        }
    }
    //*******************************************************************************

    // Select atomic shell
    unsigned short int nShells = atom_NumberOfShells(Z);
    index = atom_IndexOfShells(Z);
    float bindingEnergy = atom_BindingEnergies(index) * eV; //1.0e-06f; // in eV
    i=0; while (i < nShells && energy < bindingEnergy) {       
        bindingEnergy = atom_BindingEnergies(index + i)* eV; //1.0e-06f; // in ev
        ++i;
    }
        
    // no shell available return stillborn electron
    if (i == nShells) {return electron;}
    float ElecKineEnergy = energy - bindingEnergy;

    float cosTeta = 0.0f;
    //                   1 eV                         cut production
    if (ElecKineEnergy > 1.0e-06f && ElecKineEnergy > cutE) {
        // direction of the photo electron
        cosTeta = PhotoElec_ElecCosThetaDistribution(particles, id, ElecKineEnergy);
        float sinTeta = sqrtf(1.0f - cosTeta*cosTeta);
        float Phi = gpu_twopi * JKISS32(particles, id);
        float3 ElecDirection = make_float3(sinTeta*cos(Phi), sinTeta*sin(Phi), cosTeta);
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

__constant__ int GPU_Rayleigh_LV_CS_NbIntervals [101] =
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

__constant__ unsigned short int GPU_Rayleigh_LV_CS_NbIntervals [101] =
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

__constant__ unsigned short int GPU_Rayleigh_LV_SF_CumulIntervals [101] =
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

__constant__ unsigned short int GPU_Rayleigh_LV_SF_NbIntervals [101] =
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

const int CPU_Rayleigh_LV_CS_CumulIntervals [101] =
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

const unsigned short int CPU_Rayleigh_LV_CS_NbIntervals [101] =
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

const unsigned short int CPU_Rayleigh_LV_SF_CumulIntervals [101] =
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

const unsigned short int CPU_Rayleigh_LV_SF_NbIntervals [101] =
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


// Functions allowing to fetch value according if GPU or CPU code is used.

__host__ __device__ unsigned short int Rayleigh_LV_CS_CumulIntervals(unsigned int pos) {

#ifdef __CUDA_ARCH__
    return GPU_Rayleigh_LV_CS_CumulIntervals[pos];
#else
    return CPU_Rayleigh_LV_CS_CumulIntervals[pos];
#endif

}

__host__ __device__ unsigned short int Rayleigh_LV_CS_NbIntervals(unsigned int pos) {

#ifdef __CUDA_ARCH__
    return GPU_Rayleigh_LV_CS_NbIntervals[pos];
#else
    return CPU_Rayleigh_LV_CS_NbIntervals[pos];
#endif

}

__host__ __device__ unsigned short int Rayleigh_LV_SF_CumulIntervals(unsigned int pos) {

#ifdef __CUDA_ARCH__
    return GPU_Rayleigh_LV_SF_CumulIntervals[pos];
#else
    return CPU_Rayleigh_LV_SF_CumulIntervals[pos];
#endif

}

__host__ __device__ unsigned short int Rayleigh_LV_SF_NbIntervals(unsigned int pos) {

#ifdef __CUDA_ARCH__
    return GPU_Rayleigh_LV_SF_NbIntervals[pos];
#else
    return CPU_Rayleigh_LV_SF_NbIntervals[pos];
#endif

}


// Load CS information from G4 Em data
float* Rayleigh_CS_Livermore_load_data() {
    const int ncs = 213816;  // CS file contains 213,816 floats
    unsigned int mem_cs = ncs * sizeof(float);
    float* raylcs = (float*)malloc(mem_cs);
    FILE * pfile = fopen("../data/rayleigh_cs.bin", "rb");
    fread(raylcs, sizeof(float), ncs, pfile);
    fclose(pfile);
    return raylcs;
}

// Load SF information from G4 Em data
float* Rayleigh_SF_Livermore_load_data() {
    const int nsf = 28824;   // SF file contains  28,824 floats
    unsigned int mem_sf = nsf * sizeof(float);
    float* raylsf = (float*)malloc(mem_sf);
    FILE * pfile = fopen("../data/rayleigh_sf.bin", "rb");
    fread(raylsf, sizeof(float), nsf, pfile);
    fclose(pfile);
    return raylsf;
}

// Rayleigh Cross Section Per Atom (Livermore)
__host__ __device__ float Rayleigh_CSPA_Livermore(float* rayl_cs, float E, unsigned short int Z) {
    if (E < 250e-6f || E > 100e3f) {return 0.0f;} // 250 eV < E < 100 GeV

    int start = Rayleigh_LV_CS_CumulIntervals[Z];
    int stop  = start + 2 * (Rayleigh_LV_CS_NbIntervals[Z] - 1);

    int pos;
    for (pos=start; pos<stop; pos+=2) {
        if (rayl_cs[pos] >= E) {break;}
    }

    if (E < 1e3f) { // 1 Gev
        return 1.0e-22f * loglog_interpolation(E, rayl_cs[pos-2], rayl_cs[pos-1],
                                               rayl_cs[pos], rayl_cs[pos+1]);
    }
    else {return 1.0e-22f * rayl_cs[pos-1];}
}

// Compute the total Compton cross section for a given material
__host__ __device__ float Rayleigh_CS_Livermore(MaterialsTable materials,
                                                float* rayl_cs, unsigned short int mat, float E) {
    float CS = 0.0f;
    int i;
    int index = materials.index[mat];
    // Model Livermore
    for (i = 0; i < materials.nb_elements[mat]; ++i) {
        CS += (materials.atom_num_dens[index+i] *
               Rayleigh_CSPA_Livermore(rayl_cs, E, materials.mixture[index+i]));
    }
    return CS;
}


// Rayleigh Scatter Factor (Livermore)
__host__ __device__ float Rayleigh_SF_Livermore(float* rayl_sf, float E, int Z) {
    int start = Rayleigh_LV_SF_CumulIntervals[Z];
    int stop = start + 2 * (Rayleigh_LV_SF_NbIntervals[Z] - 1);

    // check boundary
    if (E==0.0f) return rayl_sf[start+1];

    int pos;
    for (pos=start; pos<stop; pos+=2) {
        if (rayl_sf[pos] >= E) {break;}
    }

    return loglog_interpolation(E, rayl_sf[pos-2], rayl_sf[pos-1],
                                rayl_sf[pos], rayl_sf[pos+1]);
}

/*
// Rayleigh Scatter (Livermore)
__device__ float3 Rayleigh_scatter(StackGamma stack, unsigned int id, int Z) {
    float E = stack.E[id];

    if (E <= 250.0e-6f) { // 250 eV
        stack.live[id] = 0;
        stack.endsimu[id] = 1;
        return make_float3(0.0f, 0.0f, 1.0f);
    }

    float wphot = __fdividef(1.23984187539e-10f, E);
    float costheta, SF, x, sintheta, phi;
    do {
        do {costheta = 2.0f * Brent_real(id, stack.table_x_brent, 0) - 1.0f;
        } while ((1.0f + costheta*costheta)*0.5f < Brent_real(id, stack.table_x_brent, 0));
        if (E > 5.0f) {costheta = 1.0f;}
        x = __fdividef(sqrt((1.0f - costheta) * 0.5f), wphot);
        SF = (x > 1.0e+05f)? Rayleigh_SF(x, Z) : Rayleigh_SF(0.0f, Z);
    } while (SF*SF < Brent_real(id, stack.table_x_brent, 0) * Z*Z);

    sintheta = sqrt(1.0f - costheta*costheta);
    phi = Brent_real(id, stack.table_x_brent, 0) * twopi;

    return make_float3(sintheta*__cosf(phi), sintheta*__sinf(phi), costheta);
}

// Compute the total Compton cross section for a given material
__device__ float Rayleigh_CS(int mat, float E) {
    float CS = 0.0f;
    int i;
    int index = mat_index[mat];
    for (i = 0; i < mat_nb_elements[mat]; ++i) {
        CS += (mat_atom_num_dens[index+i] * Rayleigh_CSPA(E, mat_mixture[index+i]));
    }
    return CS;
}

// Rayleigh element selector
__device__ unsigned short int Rayleigh_rnd_Z(float rnd, int mat, float E) {
    unsigned short int nb_elements_minus_one = mat_nb_elements[mat] - 1;
    unsigned short int index_mat = mat_index[mat];
    unsigned short int index_element = index_mat * 22;
    unsigned short int i, start, stop, pos;
    float xSection;

    // Search E in xSection table
    start = index_element;
    stop = start + 20;
    for (pos=start; pos<stop; pos+=2) {
        if (Rayleigh_element_selector_CS[pos] >= E) {break;};
    }

    unsigned short int Z = mat_mixture[index_mat + nb_elements_minus_one];
    for (i=0; i<nb_elements_minus_one; ++i) {
        // Get xSection value
        if (E==0.0f) {xSection = Rayleigh_element_selector_CS[pos+1];}
        else {
            xSection = loglog_interpolation(E, Rayleigh_element_selector_CS[pos-2],
                                            Rayleigh_element_selector_CS[pos-1],
                                            Rayleigh_element_selector_CS[pos],
                                            Rayleigh_element_selector_CS[pos+1]);
        }
        // Move inside the table for each element
        pos += 22;
        // Select the element
        if (rnd <= xSection) {
            Z = mat_mixture[index_mat + i];
            break;
        }
    } // for
    return Z;
}
*/

#endif
