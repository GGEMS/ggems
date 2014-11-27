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

#ifndef PHOTON_CUH
#define PHOTON_CUH

#include "particles.cuh"
#include "materials.cuh"
#include "global.cuh"
#include "prng.cuh"
#include "fun.cuh"
#include "constants.cuh"
#include "sandia_table.cuh"
#include "shell_data.cuh"

// Cross section table for photon particle
struct PhotonCrossSectionTable{
    f32* E_bins;                // n

    f32* Compton_Std_CS;        // n*k

    f32* Photoelectric_Std_CS;  // n*k
    f32* Photoelectric_Std_xCS; // n*101 (Nb of Z)

    f32* Rayleigh_Lv_CS;        // n*k
    f32* Rayleigh_Lv_SF;        // n*101 (Nb of Z)
    f32* Rayleigh_Lv_xCS;       // n*101 (Nb of Z)

    f32 E_min;
    f32 E_max;
    unsigned int nb_bins;         // n
    unsigned int nb_mat;          // k
};

// Utils
__host__ __device__ f32 get_CS_from_table(f32 *E_bins, f32 *CSTable, f32 energy,
                                            unsigned int E_index, unsigned int mat_index, unsigned int nb_bins);

// Compton - model standard G4
__host__ __device__ f32 Compton_CSPA_standard(f32 E, unsigned short int Z);
__host__ __device__ f32 Compton_CS_standard(MaterialsTable materials, unsigned short int mat, f32 E);

__host__ __device__ SecParticle Compton_SampleSecondaries_standard(ParticleStack particles,
                                                                   f32 cutE,
                                                                   unsigned int id,
                                                                   GlobalSimulationParameters parameters);
//

// PhotoElectric - model standard G4
__host__ __device__ f32 Photoelec_CSPA_standard(f32 E, unsigned short int Z);
__host__ __device__ f32 Photoelec_CS_standard(MaterialsTable materials,
                                                unsigned short int mat, f32 E);
__host__ __device__ SecParticle Photoelec_SampleSecondaries_standard(ParticleStack particles,
                                                                     MaterialsTable mat,
                                                                     PhotonCrossSectionTable photon_CS_table,
                                                                     unsigned int E_index,
                                                                     f32 cutE,
                                                                     unsigned short int matindex,
                                                                     unsigned int id,
                                                                     GlobalSimulationParameters parameters);

//

// Rayleigh scattering - model Livermore G4
__host__ __device__ unsigned short int Rayleigh_LV_CS_CumulIntervals(unsigned int pos);
__host__ __device__ unsigned short int Rayleigh_LV_CS_NbIntervals(unsigned int pos);
__host__ __device__ unsigned short int Rayleigh_LV_SF_CumulIntervals(unsigned int pos);
__host__ __device__ unsigned short int Rayleigh_LV_SF_NbIntervals(unsigned int pos);

f32* Rayleigh_CS_Livermore_load_data();
f32* Rayleigh_SF_Livermore_load_data();
__host__ __device__ f32 Rayleigh_CSPA_Livermore(f32* rayl_cs, f32 E, unsigned short int Z);
__host__ __device__ f32 Rayleigh_CS_Livermore(MaterialsTable materials,
                                                f32* rayl_cs, unsigned short int mat, f32 E);
__host__ __device__ f32 Rayleigh_SF_Livermore(f32* rayl_sf, f32 E, int Z);
__host__ __device__ void Rayleigh_SampleSecondaries_Livermore(ParticleStack particles,
                                                              MaterialsTable mat,
                                                              PhotonCrossSectionTable photon_CS_table,
                                                              unsigned int E_index,
                                                              unsigned short int matindex,
                                                              unsigned int id);


/*
__host__ __device__ f32 Compton_CSPA (f32 E, unsigned short int Z);

// Compute the total Compton cross section for a given material
__host__ __device__ f32 Compton_CS(GPUPhantomMaterials materials, unsigned short int mat, f32 E);
// Compton Scatter (Standard - Klein-Nishina) with secondary (e-)
__host__ __device__ f32 Compton_SampleSecondaries(ParticleStack particles, f32 cutE, unsigned int id,unsigned char flag_secondary);

///// PhotoElectric /////

// PhotoElectric Cross Section Per Atom (Standard)
__host__ __device__ f32 PhotoElec_CSPA(f32 E, unsigned short int Z);

// Compute the total Compton cross section for a given material
__host__ __device__ f32 PhotoElec_CS(GPUPhantomMaterials materials,
                              unsigned short int mat, f32 E);

// Compute Theta distribution of the emitted electron, with respect to the incident Gamma
// The Sauter-Gavrila distribution for the K-shell is used
__host__ __device__ f32 PhotoElec_ElecCosThetaDistribution(ParticleStack part,unsigned int id, f32 kineEnergy);

// PhotoElectric effect
__host__ __device__ f32 PhotoElec_SampleSecondaries(ParticleStack particles, GPUPhantomMaterials mat, unsigned short int matindex, unsigned int id, unsigned char flag_secondary,f32 cutEnergyElectron=990*eV );

*/

#endif
