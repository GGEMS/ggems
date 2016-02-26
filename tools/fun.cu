// GGEMS Copyright (C) 2015

/*!
 * \file fun.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 *
 *
 */

#ifndef FUN_CU
#define FUN_CU
#include "fun.cuh"
#include "prng.cuh"



// rotateUz, function from CLHEP
__host__ __device__ f32xyz rotateUz ( f32xyz vector, f32xyz newUz )
{
    f32 u1 = newUz.x;
    f32 u2 = newUz.y;
    f32 u3 = newUz.z;
    f32 up = u1*u1 + u2*u2;

    if ( up>0 )
    {
        up = sqrtf ( up );
        f32 px = vector.x,  py = vector.y, pz = vector.z;
        vector.x = ( u1*u3*px - u2*py ) /up + u1*pz;
        vector.y = ( u2*u3*px + u1*py ) /up + u2*pz;
        vector.z =    -up*px +             u3*pz;
    }
    else if ( u3 < 0. )
    {
        vector.x = -vector.x;    // phi=0  theta=gpu_pi
        vector.z = -vector.z;
    }

    return make_f32xyz ( vector.x, vector.y, vector.z );
}

// Loglog interpolation
__host__ __device__ f32 loglog_interpolation ( f32 x, f32 x0, f32 y0, f32 x1, f32 y1 )
{
    if ( x < x0 ) return y0;
    if ( x > x1 ) return y1;
    x0 = 1.0f / x0;
    return powf ( 10.0f, log10f ( y0 ) + log10f ( y1 / y0 ) * ( log10f ( x * x0 ) / log10f ( x1 * x0 ) ) );
}

// // Binary search f32
// __host__ __device__ ui32 binary_search(f32 key, f32* tab, ui32 size, ui32 min=0) {
//     ui32 max=size, mid;
//     while ((min < max)) {
//         mid = (min + max) >> 1;
//         if (key > tab[mid]) {
//             min = mid + 1;
//         } else {
//             max = mid;
//         }
//     }
//     return min;
// }
//
// // Binary search f64
// __host__ __device__ ui32 binary_search(f64 key, f64* tab, ui32 size, ui32 min=0) {
//     ui32 max=size, mid;
//     while ((min < max)) {
//         mid = (min + max) >> 1;
//         if (key > tab[mid]) {
//             min = mid + 1;
//         } else {
//             max = mid;
//         }
//     }
//     return min;
// }

// Linear interpolation
__host__ __device__ f32 linear_interpolation ( f32 xa, f32 ya, f32 xb, f32 yb, f32 x )
{
    // Taylor young 1st order
//     if ( xa > x ) return ya;
//     if ( xb < x ) return yb;
    
    if (xa > xb) return yb;
    if (xa >= x) return ya;
    if (xb <= x) return yb;
    
    return ya + ( x-xa ) * ( yb-ya ) / ( xb-xa );
}


__host__ __device__ i32 G4Poisson ( f32 mean, ParticlesData &particles, ui32 id )
{
    f32 number = 0.;

    f32 position, poissonValue, poissonSum;
    f32 value, y, t;
    if ( mean <= 16. ) // border == 16
    {
        // to avoid 1 due to f32 approximation
        do
        {
            position = prng_uniform( &(particles.prng[id]) );
        }
        while ( ( 1. - position ) < 2.e-7 );

        poissonValue = expf ( -mean );
        poissonSum = poissonValue;
        //                                                 v---- Why ? It's not in G4Poisson - JB
        while ( ( poissonSum <= position ) && ( number < 40000. ) )
        {
            number++;
            poissonValue *= mean/number;
            if ( ( poissonSum + poissonValue ) == poissonSum ) break;   // Not in G4, is it to manage f32 ?  - JB
            poissonSum += poissonValue;
        }

        return  ( i32 ) number;
    }   

    t = sqrtf ( -2.*logf ( prng_uniform( &(particles.prng[id]) ) ) );
    y = 2.*gpu_pi* prng_uniform( &(particles.prng[id]) );
    t *= cosf ( y );
    value = mean + t*sqrtf ( mean ) + 0.5;

    if ( value <= 0. )
    {
        return  0;
    }

    return ( value >= 2.e9 ) ? ( i32 ) 2.e9 : ( i32 ) value;
}

__host__ __device__ f32 Gaussian (f32 mean, f32 rms, ParticlesData &particles, ui32 id )
{
    f32  data;
    f32  U1,U2,Disp,Fx;

    do
    {
        U1 = 2.*prng_uniform( &(particles.prng[id]) )-1.;
        U2 = 2.*prng_uniform( &(particles.prng[id]) )-1.;
        Fx = U1*U1 + U2*U2;

    }
    while ( ( Fx >= 1. ) );

    Fx = sqrtf ( ( -2.*logf ( Fx ) ) /Fx );
    Disp = U1*Fx;
    data = mean + Disp*rms;

    return  data;
}



/// Atomic functions

/*   OLD VERSION
        __device__ double ggems_atomic_add_f64(f64* array, ui32 pos, f64 val)
        {
            f64 *address = &array[pos];
            unsigned long long int* address_as_ull = (unsigned long long int*)address;
            unsigned long long int old = *address_as_ull, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed,
                                __double_as_longlong(val +
                                __longlong_as_double(assumed)));
            } while (assumed != old);
            return __longlong_as_double(old);
        }
*/

__host__ __device__ void ggems_atomic_add_f64(f64* array, ui32 pos, f64 val)
{

    #ifdef __CUDA_ARCH__

        // Single precision f64 is in fact a f32
        #ifdef SINGLE_PRECISION

            ggems_atomic_add(array, pos, val);

        // Double precision
        #else

            f64 *address = &array[pos];
            unsigned long long oldval, newval, readback;

            oldval = __double_as_longlong(*address);
            newval = __double_as_longlong(__longlong_as_double(oldval) + val);
            while ((readback=atomicCAS((unsigned long long *)address, oldval, newval)) != oldval)
            {
                oldval = readback;
                newval = __double_as_longlong(__longlong_as_double(oldval) + val);
            }

        #endif

    #else
        array[pos] += val;
    #endif

}

#endif
