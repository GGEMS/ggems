#ifndef SHELL_DATA_CUH
#define SHELL_DATA_CUH

#include "global.cuh"

__host__ __device__ ui16 atom_NumberOfShells(ui32);
__host__ __device__ ui16 atom_IndexOfShells(ui32);
__host__ __device__ f32 atom_BindingEnergies(ui32);


#endif
