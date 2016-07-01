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

__host__ __device__ f32xyz rotateUz ( f32xyz vector, f32xyz newUz );

// Loglog interpolation
__host__ __device__ f32 loglog_interpolation ( f32 x, f32 x0, f32 y0, f32 x1, f32 y1 );

// Binary search
// __host__ __device__ ui32 binary_search(f32 key, f32* tab, ui32 size, ui32 min=0);
// __host__ __device__ ui32 binary_search(f64 key, f64* tab, ui32 size, ui32 min=0);

template < typename T, typename U >
inline __host__ __device__ ui32 binary_search ( T key, U* tab, ui32 size, ui32 min=0 )
{
    ui32 max=size, mid;
    while ( ( min < max ) )
    {
        mid = ( min + max ) >> 1;
        if ( key > tab[mid] )
        {
            min = mid + 1;
        }
        else
        {
            max = mid;
        }
    }
    return min;
}


// Linear interpolation
__host__ __device__ f32 linear_interpolation ( f32 xa,f32 ya, f32 xb,  f32 yb, f32 x );

__host__ __device__ i32 G4Poisson(f32 mean, ParticlesData &particles, ui32 id );

__host__ __device__ f32 Gaussian ( f32 mean,f32 rms,ParticlesData &particles, ui32 id );


// Filtering
namespace Filter
{
    f32 *mean( f32* input,  ui32 nx, ui32 ny, ui32 nz, ui32 w_size );
    f32 *median( f32* input,  ui32 nx, ui32 ny, ui32 nz, ui32 w_size );
    f32 *adaptive_median( f32* input, ui32 nx, ui32 ny, ui32 nz, ui32 w_size, ui32 w_size_max );
    f32 *resampling_lanczos3( f32* input, ui32 nx, ui32 ny, ui32 nz, ui32 new_nx, ui32 new_ny, ui32 new_nz );
    f32 *cropping_vox_around_center( f32* input, ui32 nx, ui32 ny, ui32 nz,
                                     i32 xmin, i32 xmax, i32 ymin, i32 ymax, i32 zmin, i32 zmax );
    void capping_values( f32* input, ui32 nx, ui32 ny, ui32 nz, f32 val_min, f32 val_max );
}

// Atomic add

template < typename T,typename U >
__host__ __device__ void ggems_atomic_add(T* array, ui32 pos, U value)
{
#ifdef __CUDA_ARCH__
    atomicAdd(&array[pos], value);
#else
    array[pos] += value;
#endif
}

// Atomic add double

__host__ __device__ void ggems_atomic_add_f64(f64* array, ui32 pos, f64 val);


/*
template < typename T>
__host__ __device__ void ggems_atomic_add(double* array, ui32 pos, T value)
{
#ifdef __CUDA_ARCH__
    atomicAddDouble(&array[pos], value);
#else
    array[pos] += value;
#endif
}
*/












#endif
