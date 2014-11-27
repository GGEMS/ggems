#ifndef SHELL_DATA_CUH
#define SHELL_DATA_CUH

#include "global.cuh"

__host__ __device__ unsigned short int atom_NumberOfShells(unsigned int);
__host__ __device__ unsigned short int atom_IndexOfShells(unsigned int);
__host__ __device__ f32 atom_BindingEnergies(unsigned int);


#endif
