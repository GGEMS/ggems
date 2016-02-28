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

/// JKISS //////////////////////////////////////////////////////////

// JKISS 32-bit (period ~2^121=2.6x10^36), passes all of the Dieharder tests and the BigCrunch tests in TestU01
__device__ f32 JKISS32(randStateJKISS *state) {



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

__device__ void curand_init( ui32 seed, ui32 id, ui32 offset, randStateJKISS *state )
{
    state->state_1 = seed + offset + id;   // Warning overflow is possible - JB
    state->state_2 = seed + offset + 2*id;
    state->state_3 = seed + offset + 3*id;
    state->state_4 = seed + offset + 4*id;
    state->state_5 = 0;
}

__device__ f32 curand_uniform( randStateJKISS *state )
{
    return JKISS32( state );
}

__device__ ui32 curand_poisson( randStateJKISS *state, f32 lambda )
{
    f32 number = 0.;

    f32 position, poissonValue, poissonSum;
    f32 value, y, t;
    if ( lambda <= 16. ) // border == 16
    {
        // to avoid 1 due to f32 approximation
        do
        {
            position = JKISS32( state );
        }
        while ( ( 1. - position ) < 2.e-7 );

        poissonValue = expf ( -lambda );
        poissonSum = poissonValue;
        //                                                 v---- Why ? It's not in G4Poisson - JB
        while ( ( poissonSum <= position ) && ( number < 40000. ) )
        {
            number++;
            poissonValue *= lambda/number;
            if ( ( poissonSum + poissonValue ) == poissonSum ) break;   // Not in G4, is it to manage f32 ?  - JB
            poissonSum += poissonValue;
        }

        return  ( i32 ) number;
    }

    t = sqrtf ( -2.*logf ( JKISS32( state ) ) );
    y = 2.*gpu_pi* JKISS32( state );
    t *= cosf ( y );
    value = lambda + t*sqrtf ( lambda ) + 0.5;

    if ( value <= 0. )
    {
        return  0;
    }

    return ( value >= 2.e9 ) ? ( ui32 ) 2.e9 : ( ui32 ) value;
}

/// INIT /////////////////////////////////////////////////////////////////

__global__ void kernel_init_seeds( prng_states *states, ui32 *seed, ui32 size )
{    
    #ifdef __CUDA_ARCH__
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if( id >= size ) return;

    /* Each thread gets same seed, a different sequence number, no offset */
    curand_init(seed[id], id, 0, &states[id]);

    #endif
}

__host__ void gpu_prng_init( prng_states *states, ui32 size, ui32 seed, ui32 block_size )
{
    srand(seed);

    ui32 *hseeds = new ui32[ size ];
    for ( ui32 i=0; i<size; i++)
    {
        hseeds[ i ] = rand();
    }

    ui32 *dseeds;
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &dseeds, size*sizeof ( ui32 ) ) );
    HANDLE_ERROR ( cudaMemcpy ( dseeds, hseeds, sizeof ( ui32 ) * size, cudaMemcpyHostToDevice ) );

    dim3 threads, grid;
    threads.x = block_size;
    grid.x = ( size + block_size - 1 ) / block_size;

    kernel_init_seeds<<<grid, threads>>>(states, dseeds, size);
    cuda_error_check ( "Error ", " Kernel_gpu_prng_init" );
}

__host__ void cpu_prng_init(prng_states *states, ui32 size, ui32 seed )
{
    srand(seed);
    #ifndef __CUDA_ARCH__
    prng_states aState;
    for ( ui32 i=0; i<size; i++)
    {
        aState.seed = rand();
        states[i] = aState;
    }
    #endif
}

/// PRNG UNIFORM /////////////////////////////////////////////////////////////////::

QUALIFIER f32 prng_uniform(prng_states *state)
{

#ifdef __CUDA_ARCH__

    #ifdef DEBUG
    //f32 x = curand(state) * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2.0f);
    f32 x = curand_uniform(state);

    if ( x < 0.0f )
    {
        printf("[GGEMS error] PRNG NUMBER < 0.0\n");
        x = 0;
    }
    if ( x >= 1.0f )
    {
        printf("[GGEMS error] PRNG NUMBER >= 1.0\n");
        x = 1.0f - CURAND_2POW32_INV;
    }
    return x;
    #else
    // Return float between 0 to 1 (0 include, 1 exclude)
    //return curand(state) * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2.0f);
    //return JKISS32(state);
    return curand_uniform(state);
    #endif

#else

    /*
    // CPU code - FIXME - the use of suh prng requried a class - JB
    ui32 seed = state->seed;
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(0.0, 1.0-CURAND_2POW32_INV);
    seed += 10;
    if (seed >= ULONG_MAX) {
        seed = seed / LONG_MAX;

        #ifdef DEBUG
        printf("[GGEMS error] PRNG NUMBER reach MAX\n");
        #endif

    }
    state->seed = seed;
    */

    #ifdef DEBUG
        //f32 x = distribution(generator);

        f32 x = rand() / (f32)RAND_MAX;

        if ( x < 0.0f )
        {
            printf("[GGEMS error] PRNG NUMBER < 0.0\n");
            x = 0.0f;
        }
        if ( x >= 1.0f )
        {
            printf("[GGEMS error] PRNG NUMBER >= 1.0\n");
            x = 1.0f-CURAND_2POW32_INV;
        }

        //printf("x %e   seed %i\n", x, seed);

        return x;
    #else
        //return distribution(generator);
        return rand() / (f32)RAND_MAX;
    #endif
#endif
}

////////////////////////// POISSON ///////////////////////////////////

__host__ ui32 cpu_prng_poisson ( prng_states *state, f32 mean )
{
    f32 number = 0.;

    f32 position, poissonValue, poissonSum;
    f32 value, y, t;
    if ( mean <= 16. ) // border == 16
    {
        // to avoid 1 due to f32 approximation
        do
        {
            position = prng_uniform( state );
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

    t = sqrtf ( -2.*logf ( prng_uniform( state ) ) );
    y = 2.*gpu_pi* prng_uniform( state );
    t *= cosf ( y );
    value = mean + t*sqrtf ( mean ) + 0.5;

    if ( value <= 0. )
    {
        return  0;
    }

    return ( value >= 2.e9 ) ? ( ui32 ) 2.e9 : ( ui32 ) value;
}

/* Original function
__host__ i32 cpu_prng_poisson ( f32 mean, ParticlesData &particles, ui32 id )
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
*/


QUALIFIER ui32 prng_poisson(prng_states *state, f32 lambda)
{

#ifdef __CUDA_ARCH__
    return curand_poisson( state, lambda );
#else
    return cpu_prng_poisson( state, lambda );
#endif

}







#endif
