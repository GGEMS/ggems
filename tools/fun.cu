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
#include "prng.cuh"

// rotateUz, function from CLHEP
 __host__ __device__ f32xyz rotateUz(f32xyz vector, f32xyz newUz) {
    f32 u1 = newUz.x;
    f32 u2 = newUz.y;
    f32 u3 = newUz.z;
    f32 up = u1*u1 + u2*u2;

    if (up>0) {
        up = sqrtf(up);
        f32 px = vector.x,  py = vector.y, pz = vector.z;
        vector.x = (u1*u3*px - u2*py)/up + u1*pz;
        vector.y = (u2*u3*px + u1*py)/up + u2*pz;
        vector.z =    -up*px +             u3*pz;
    }
    else if (u3 < 0.) { vector.x = -vector.x; vector.z = -vector.z; } // phi=0  theta=gpu_pi

    return make_f32xyz(vector.x, vector.y, vector.z);
}

// Loglog interpolation
__host__ __device__ f32 loglog_interpolation(f32 x, f32 x0, f32 y0, f32 x1, f32 y1) {
    if (x < x0) return y0;
    if (x > x1) return y1;
    x0 = 1.0f / x0;
    return powf(10.0f, log10f(y0) + log10f(y1 / y0) * (log10f(x * x0) / log10f(x1 * x0)));
}

// Binary search f32
__host__ __device__ ui32 binary_search(f32 key, f32* tab, ui32 size, ui32 min=0) {
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

// Binary search f64
__host__ __device__ ui32 binary_search(f64 key, f64* tab, ui32 size, ui32 min=0) {
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
__host__ __device__ f32 linear_interpolation(f32 xa,f32 ya, f32 xb, f32 yb, f32 x) {
    // Taylor young 1st order
    if (xa > x) return ya;
    if (xb < x) return yb;
    return ya + (x-xa) * (yb-ya) / (xb-xa);
}

#endif
