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
#include "particles.cuh"

//#ifdef __CUDA_ARCH__
//    #define QUALIFIER __device__
//#else
//    #define QUALIFIER __host__
//#endif

//#define CURAND_2POW32_INV (2.3283064e-10f)

//struct randStateJKISS
//{
//    ui32 state_1;
//    ui32 state_2;
//    ui32 state_3;
//    ui32 state_4;
//    ui32 state_5;
//};

//struct randStateCPU
//{
//    ui32 seed;
//};

//#ifdef __CUDA_ARCH__
//    //typedef curandStateXORWOW_t prng_states;         // 48 B per state  - slower but equivalant to G4
//    //typedef curandStateMRG32k3a prng_states;         // 72 B per state  - slow and different to G4
//    //typedef curandStatePhilox4_32_10_t prng_states;  // 64 B per state  - fast and the closer to G4 - Default uses this one - JB
//    typedef randStateJKISS prng_states;              // 20 B per state  - equivalent to Philox
//#else
//    typedef randStateCPU prng_states;
//#endif

/////////////////////////////////////////////////////////////////////////////
// Prng
/////////////////////////////////////////////////////////////////////////////

//__host__  __device__ f32 JKISS32(ParticlesData &particles, ui32 id);

__host__ __device__ f32 prng_uniform( ParticlesData &particles, ui32 id );
__host__ __device__ ui32 prng_poisson( ParticlesData &particles, ui32 id, f32 lambda );

//QUALIFIER f32 prng_uniform(prng_states *state);
//QUALIFIER ui32 prng_poisson(prng_states *state, f32 lambda);
//__global__ void gpu_prng_init(prng_states *states, ui32 seed);
//__host__ void cpu_prng_init( prng_states *states, ui32 size, ui32 seed );
//__host__ void gpu_prng_init( prng_states *states, ui32 size, ui32 seed, ui32 block_size );
#endif
