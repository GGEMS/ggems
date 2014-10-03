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


#endif
