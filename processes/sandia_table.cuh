// GGEMS Copyright (C) 2017

#ifndef SANDIA_TABLE_CUH
#define SANDIA_TABLE_CUH

#include "global.cuh"

// Function that drive the tables reading between CPU and GPU
__host__ __device__ ui16 PhotoElec_std_NbIntervals(ui32 pos);
__host__ __device__ ui16 PhotoElec_std_CumulIntervals(ui32 pos);
__host__ __device__ f32 PhotoElec_std_ZtoAratio(ui32 pos);
__host__ __device__ f32 PhotoElec_std_IonizationPotentials(ui32 pos);
__host__ __device__ f32 PhotoElec_std_SandiaTable(ui32 pos, ui32 id);

#endif
