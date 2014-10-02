// This file is part of GGEMS
//
// FIREwork is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// FIREwork is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with FIREwork.  If not, see <http://www.gnu.org/licenses/>.
//
// GGEMS Copyright (C) 2013-2014 Julien Bert

#include "sandia_table.cuh"

#ifndef SANDIA_TABLE_CU
#define SANDIA_TABLE_CU

__host__ __device__ unsigned short int PhotoElec_std_NbIntervals(unsigned int pos) {

#ifdef __CUDA_ARCH__
    return GPU_PhotoElec_std_NbIntervals[pos];
#else
    return CPU_PhotoElec_std_NbIntervals[pos];
#endif

}

__host__ __device__ unsigned short int PhotoElec_std_CumulIntervals(unsigned int pos) {

#ifdef __CUDA_ARCH__
    return GPU_PhotoElec_std_CumulIntervals[pos];
#else
    return CPU_PhotoElec_std_CumulIntervals[pos];
#endif

}

__host__ __device__ float PhotoElec_std_ZtoAratio(unsigned int pos) {

#ifdef __CUDA_ARCH__
    return GPU_PhotoElec_std_ZtoAratio[pos];
#else
    return CPU_PhotoElec_std_ZtoAratio[pos];
#endif

}

__host__ __device__ float PhotoElec_std_IonizationPotentials(unsigned int pos) {

#ifdef __CUDA_ARCH__
    return GPU_PhotoElec_std_IonizationPotentials[pos];
#else
    return CPU_PhotoElec_std_IonizationPotentials[pos];
#endif

}

__host__ __device__ float PhotoElec_std_SandiaTable(unsigned int pos, unsigned int id) {

#ifdef __CUDA_ARCH__
    return GPU_PhotoElec_std_SandiaTable[pos][id];
#else
    return CPU_PhotoElec_std_SandiaTable[pos][id];
#endif

}




















#endif
