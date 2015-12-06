// GGEMS Copyright (C) 2015

/*!
 * \file prng.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 *
 *
 */

#ifndef PRNG_CU
#define PRNG_CU

#include "prng.cuh"

// JKISS 32-bit (period ~2^121=2.6x10^36), passes all of the Dieharder tests and the BigCrunch tests in TestU01
__host__ __device__ f32 JKISS32(ParticlesData &particles, ui32 id) {
    i32 t;


//    y ^= (y<<5);
//    y ^= (y>>7);
//    y ^= (y<<22);
//    t = z+w+c;
//    z = w;
//    c = t < 0;
//    w = t & 2147483647;
//    x += 1411392427;


    particles.prng_state_2[id] ^= (particles.prng_state_2[id] << 5);
    particles.prng_state_2[id] ^= (particles.prng_state_2[id] >> 7);
    particles.prng_state_2[id] ^= (particles.prng_state_2[id] << 22);
    t = particles.prng_state_3[id] + particles.prng_state_4[id] + particles.prng_state_5[id];
    particles.prng_state_3[id] = particles.prng_state_4[id];
    particles.prng_state_5[id] = t < 0;
    particles.prng_state_4[id] = t & 2147483647;
    particles.prng_state_1[id] += 1411392427;

    // Instead to return value between [0, 1] we return value between [0, 1[

    // For the double version use that
    // return (double)(x+y+w) / 4294967296.0;  // UINT_MAX+1

    // For the f32 version is more tricky
    f32 temp= ((f32)(particles.prng_state_1[id]+particles.prng_state_2[id]+particles.prng_state_4[id])
    //           UINT_MAX         1.0  - float32_precision
                 / 4294967295.0) * (1.0f - 1.0f/(1<<23));

    return temp;
}

#endif
