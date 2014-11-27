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

#include "global.cuh"

__host__ __device__ float3 rotateUz(float3 vector, float3 newUz);

// Loglog interpolation
__host__ __device__ f32 loglog_interpolation(f32 x, f32 x0, f32 y0, f32 x1, f32 y1);


// Binary search
__host__ __device__ int binary_search(f32 key, f32* tab, int size, int min=0);

// Linear interpolation
__host__ __device__ f32 linear_interpolation(f32 xa,f32 ya, f32 xb,  f32 yb, f32 x);

/*
// Poisson distribution from Geant4 using JKISS32 Generator
inline __device__ int G4Poisson(f32 mean,ParticleStack &particles, int id);

// Gaussian distribution using JKISS32 Generator
inline __device__ f32 Gaussian(f32 mean,f32 rms,ParticleStack &particles, int id);

*/

#endif
