// GGEMS Copyright (C) 2015

#ifndef FUN_H
#define FUN_H

#include "global.cuh"

__host__ __device__ f32xyz rotateUz(f32xyz vector, f32xyz newUz);

// Loglog interpolation
__host__ __device__ f32 loglog_interpolation(f32 x, f32 x0, f32 y0, f32 x1, f32 y1);

// Binary search
__host__ __device__ ui32 binary_search(f32 key, f32* tab, ui32 size, ui32 min=0);
__host__ __device__ ui32 binary_search(f64 key, f64* tab, ui32 size, ui32 min=0);

// Linear interpolation
__host__ __device__ f32 linear_interpolation(f32 xa,f32 ya, f32 xb,  f32 yb, f32 x);

#endif
