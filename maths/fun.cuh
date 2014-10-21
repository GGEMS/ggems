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

#ifndef FUN_H
#define FUN_H

__host__ __device__ float3 rotateUz(float3 vector, float3 newUz);

// Loglog interpolation
__host__ __device__ float loglog_interpolation(float x, float x0, float y0, float x1, float y1);


// Binary search
__host__ __device__ int binary_search(float key, float* tab, int size, int min=0);

// Linear interpolation
__host__ __device__ float linear_interpolation(float xa,float ya, float xb,  float yb, float x);

/*
// Poisson distribution from Geant4 using JKISS32 Generator
inline __device__ int G4Poisson(float mean,ParticleStack &particles, int id);

// Gaussian distribution using JKISS32 Generator
inline __device__ float Gaussian(float mean,float rms,ParticleStack &particles, int id);

*/

#endif
