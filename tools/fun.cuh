// GGEMS Copyright (C) 2015

/*!
 * \file fun.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 *
 *
 */

#ifndef FUN_H
#define FUN_H

#include "global.cuh"
#include "vector.cuh"
#include "particles.cuh"
#include "prng.cuh"

inline __device__ ui32 get_id(){return blockIdx.x * blockDim.x + threadIdx.x;};

__host__ __device__ f32xyz rotateUz(f32xyz vector, f32xyz newUz);

// Loglog interpolation
__host__ __device__ f32 loglog_interpolation(f32 x, f32 x0, f32 y0, f32 x1, f32 y1);

// Binary search
// __host__ __device__ ui32 binary_search(f32 key, f32* tab, ui32 size, ui32 min=0);
// __host__ __device__ ui32 binary_search(f64 key, f64* tab, ui32 size, ui32 min=0);

template < typename T, typename U >
inline __host__ __device__ ui32 binary_search(T key, U* tab, ui32 size, ui32 min=0) {
    ui32 max=size, mid;
    while ((min < max)) {
        mid = (min + max) >> 1;
        if (key > tab[mid]) {
            min = mid + 1;
        } else {
            max = mid;
        }
    }
    return min;
}


// Linear interpolation
__host__ __device__ f32 linear_interpolation(f32 xa,f32 ya, f32 xb,  f32 yb, f32 x);

__host__ __device__ int G4Poisson(f32 mean,ParticlesData &particles, int id);

__host__ __device__ f32 Gaussian(f32 mean,f32 rms,ParticlesData &particles, int id);

    
template < typename T > std::string to_string( const T& n )
{
    std::ostringstream stm ;
    stm << n ;
    return stm.str() ;
}

__host__ __device__ i32xyz get_bin_xyz(i32 bin, i32xyz size);



#endif
