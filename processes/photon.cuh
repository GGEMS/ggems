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
#include "structures.cuh"
#include "../maths/prng.cuh"
#include "../maths/fun.cuh"
#include "constants.cuh"
#include "sandia_table.cuh"
#include "shell_data.cuh"

__device__ float Compton_CSPA(float E, unsigned short int Z);

// Compute the total Compton cross section for a given material
__device__ float Compton_CS(GPUPhantomMaterials materials, unsigned short int mat, float E);
// Compton Scatter (Standard - Klein-Nishina) with secondary (e-)
__device__ float Compton_SampleSecondaries(ParticleStack particles, float cutE, unsigned int id,unsigned char flag_secondary);

///// PhotoElectric /////

// PhotoElectric Cross Section Per Atom (Standard)
__device__ float PhotoElec_CSPA(float E, unsigned short int Z);

// Compute the total Compton cross section for a given material
__device__ float PhotoElec_CS(GPUPhantomMaterials materials, 
                              unsigned short int mat, float E);

// Compute Theta distribution of the emitted electron, with respect to the incident Gamma
// The Sauter-Gavrila distribution for the K-shell is used
__device__ float PhotoElec_ElecCosThetaDistribution(ParticleStack part,unsigned int id, float kineEnergy);

// PhotoElectric effect
__device__ float PhotoElec_SampleSecondaries(ParticleStack particles, GPUPhantomMaterials mat, unsigned short int matindex, unsigned int id, unsigned char flag_secondary,float cutEnergyElectron=990*eV );


#endif
