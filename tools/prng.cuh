// GGEMS Copyright (C) 2015

/*!
 * \file prng.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 *
 *
 */

#ifndef PRNG_H
#define PRNG_H

#include "global.cuh"

#ifdef __CUDA_ARCH__
    #define QUALIFIER __device__
#else
    #define QUALIFIER __host__
#endif

#define DEBUG 1

#define CURAND_2POW32_INV (2.3283064e-10f)

struct randStateJKISS
{
    ui32 state_1;
    ui32 state_2;
    ui32 state_3;
    ui32 state_4;
    ui32 state_5;
};

//typedef curandStateXORWOW_t prng_states;         // 48 B per state  - slower but equivalant to G4
typedef curandStateMRG32k3a prng_states;           // 72 B per state  - slow and different to G4
//typedef curandStatePhilox4_32_10_t prng_states;  // 64 B per state  - fast and the closer to G4 - Default uses this one - JB
//typedef randStateJKISS prng_states;              // 20 B per state  - equivalent to Philox


/////////////////////////////////////////////////////////////////////////////
// Prng
/////////////////////////////////////////////////////////////////////////////

//__host__  __device__ f32 JKISS32(ParticlesData &particles, ui32 id);

QUALIFIER f32 prng_uniform(prng_states *state);
__global__ void gpu_prng_init(prng_states *states, ui32 seed);

#endif
