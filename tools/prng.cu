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

/*
// JKISS 32-bit (period ~2^121=2.6x10^36), passes all of the Dieharder tests and the BigCrunch tests in TestU01
__device__ f32 JKISS32(prng_states *state) {



    //    y ^= (y<<5);
    //    y ^= (y>>7);
    //    y ^= (y<<22);
    //    t = z+w+c;
    //    z = w;
    //    c = t < 0;
    //    w = t & 2147483647;
    //    x += 1411392427;

        state->state_2 ^= ( state->state_2 << 5 );
        state->state_2 ^= ( state->state_2 >> 7 );
        state->state_2 ^= ( state->state_2 << 22 );
        i32 t = state->state_3 + state->state_4 + state->state_5;
        state->state_3 = state->state_4;
        state->state_5 = t < 0;
        state->state_4 = t & 2147483647;
        state->state_1 += 1411392427;

        // Instead to return value between [0, 1] we return value between [0, 1[

        // For the double version use that
        // return (double)(x+y+w) / 4294967296.0;  // UINT_MAX+1

        // For the f32 version is more tricky
        return ( ( f32 ) ( state->state_1 + state->state_2 + state->state_4 )
        //           UINT_MAX         1.0  - float32_precision
                     / 4294967295.0) * (1.0f - 1.0f/(1<<23));
}
*/

__global__ void gpu_prng_init(prng_states *states, ui32 seed)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence number, no offset */
    curand_init(seed, id, 0, &states[id]);
}

QUALIFIER f32 prng_uniform(prng_states *state)
{

#ifdef __CUDA_ARCH__

    #ifdef DEBUG
    f32 x = curand(state) * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2.0f) - CURAND_2POW32_INV;
    if ( x <= 0.0f )
    {
        printf("[GGEMS error] PRNG NUMBER <= 0.0\n");
        x = CURAND_2POW32_INV;
    }
    if ( x > 1.0f )
    {
        printf("[GGEMS error] PRNG NUMBER > 1.0\n");
        x = 1.0f;
    }
    return x;
    #else
    // Return float between 0 to 1 (0 include, 1 exclude)
    return curand(state) * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2.0f) - CURAND_2POW32_INV;
    //return JKISS32(state);
    #endif

#else
    // CPU code - FIXME - seed not used!!!  - JB
    std::mt19937 generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0-CURAND_2POW32_INV);

    #ifdef DEBUG
    f32 x = distribution(generator);
    if ( x <= 0.0f )
    {
        printf("[GGEMS error] PRNG NUMBER <= 0.0\n");
        x = CURAND_2POW32_INV;
    }
    if ( x > 1.0f )
    {
        printf("[GGEMS error] PRNG NUMBER > 1.0\n");
        x = 1.0f;
    }
    return x;
    #else
    return distribution(generator);
    #endif
#endif
}








#endif
