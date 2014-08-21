// This file is part of GGEMS
//
// GGEMS is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// GGEMS is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with GGEMS.  If not, see <http://www.gnu.org/licenses/>.
//
// GGEMS Copyright (C) 2013-2014 Julien Bert

#ifndef PRNG_CU
#define PRNG_CU
#include "prng.cuh"
// JKISS 32-bit (period ~2^121=2.6x10^36), passes all of the Dieharder tests and the BigCrunch tests in TestU01
__host__ __device__ float JKISS32(ParticleStack &stack, unsigned int id) {
    int t;

    /*
    y ^= (y<<5);
    y ^= (y>>7);
    y ^= (y<<22);
    t = z+w+c;
    z = w;
    c = t < 0;
    w = t & 2147483647;
    x += 1411392427;
    */

    stack.prng_state_2[id] ^= (stack.prng_state_2[id] << 5);
    stack.prng_state_2[id] ^= (stack.prng_state_2[id] >> 7);
    stack.prng_state_2[id] ^= (stack.prng_state_2[id] << 22);
    t = stack.prng_state_3[id] + stack.prng_state_4[id] + stack.prng_state_5[id];
    stack.prng_state_3[id] = stack.prng_state_4[id];
    stack.prng_state_5[id] = t < 0;
    stack.prng_state_4[id] = t & 2147483647;
    stack.prng_state_1[id] += 1411392427;

    // Instead to return value between [0, 1] we return value between [0, 1[

    // For the double version use that
    // return (double)(x+y+w) / 4294967296.0;  // UINT_MAX+1

    // For the float version is more tricky
    float temp= ((float)(stack.prng_state_1[id]+stack.prng_state_2[id]+stack.prng_state_4[id]) 
    //        UINT_MAX         1.0  - float32_precision
            / 4294967295.0) * (1.0f - 1.0f/(1<<23));

    return temp;
}



/***********************************************************
 * PRNG Brent xor256
 ***********************************************************/
/*
// Brent PRNG integer version
__device__ unsigned long weyl;
__device__ unsigned long brent_int(unsigned int index, unsigned long *device_x_brent, unsigned long seed)

    {

#define UINT64 (sizeof(unsigned long)>>3)
#define UINT32 (1 - UINT64)
#define wlen (64*UINT64 +  32*UINT32)
#define r    (4*UINT64 + 8*UINT32)
#define s    (3*UINT64 +  3*UINT32)
#define a    (37*UINT64 +  18*UINT32)
#define b    (27*UINT64 +  13*UINT32)
#define c    (29*UINT64 +  14*UINT32)
#define d    (33*UINT64 +  15*UINT32)
#define ws   (27*UINT64 +  16*UINT32)

    int z, z_w, z_i_brent;
    if (r==4) {
        z=6;
        z_w=4;
        z_i_brent=5;
        }
    else {
        z=10;
        z_w=8;
        z_i_brent=9;
        }

    unsigned long w = device_x_brent[z*index + z_w];
    unsigned long i_brent = device_x_brent[z*index + z_i_brent];
    unsigned long zero = 0;
    unsigned long t, v;
    int k;

    if (seed != zero) { // Initialisation necessary
        // weyl = odd approximation to 2**wlen*(3-sqrt(5))/2.
        if (UINT32)
            weyl = 0x61c88647;
        else
            weyl = ((((unsigned long)0x61c88646)<<16)<<16) + (unsigned long)0x80b583eb;

        v = (seed!=zero)? seed:~seed;  // v must be nonzero

        for (k = wlen; k > 0; k--) {   // Avoid correlations for close seeds
            v ^= v<<10;
            v ^= v>>15;    // Recurrence has period 2**wlen-1
            v ^= v<<4;
            v ^= v>>13;    // for wlen = 32 or 64
            }
        for (w = v, k = 0; k < r; k++) { // Initialise circular array
            v ^= v<<10;
            v ^= v>>15;
            v ^= v<<4;
            v ^= v>>13;
            device_x_brent[k + z*index] = v + (w+=weyl);
            }
        for (i_brent = r-1, k = 4*r; k > 0; k--) { // Discard first 4*r results
            t = device_x_brent[(i_brent = (i_brent+1)&(r-1)) + z*index];
            t ^= t<<a;
            t ^= t>>b;
            v = device_x_brent[((i_brent+(r-s))&(r-1)) + z*index];
            v ^= v<<c;
            v ^= v>>d;
            device_x_brent[i_brent + z*index] = t^v;
            }
        }

    // Apart from initialisation (above), this is the generator
    t = device_x_brent[(i_brent = (i_brent+1)&(r-1)) + z*index]; // Assumes that r is a power of two
    v = device_x_brent[((i_brent+(r-s))&(r-1)) + z*index];       // Index is (i-s) mod r
    t ^= t<<a;
    t ^= t>>b;                                       // (I + L^a)(I + R^b)
    v ^= v<<c;
    v ^= v>>d;                                       // (I + L^c)(I + R^d)
    device_x_brent[i_brent + z*index] = (v ^= t);                // Update circular array
    w += weyl;                                                   // Update Weyl generator

    device_x_brent[z*index + z_w] = w;
    device_x_brent[z*index + z_i_brent] = i_brent;

    return (v + (w^(w>>ws)));  // Return combination

#undef UINT64
#undef UINT32
#undef wlen
#undef r
#undef s
#undef a
#undef b
#undef c
#undef d
#undef ws
    }

// Brent PRNG real version
__device__ double Brent_real(int index, unsigned long *device_x_brent, unsigned long seed)

    {

#define UINT64 (sizeof(unsigned long)>>3)
#define UINT32 (1 - UINT64)
#define UREAL64 (sizeof(double)>>3)
#define UREAL32 (1 - UREAL64)

    // sr = number of bits discarded = 11 for double, 40 or 8 for float

#define sr (11*UREAL64 +(40*UINT64 + 8*UINT32)*UREAL32)

    // ss (used for scaling) is 53 or 21 for double, 24 for float

#define ss ((53*UINT64 + 21*UINT32)*UREAL64 + 24*UREAL32)

    // SCALE is 0.5**ss, SC32 is 0.5**32
#define SCALETEMPO ((double)((unsigned long)1<<ss) )
// 16777216 = 2**24
#define SCALE ((double)1/ (SCALETEMPO ))
#define SC32  ((double)1/((double)65536*(double)65536))
#define SCALEVALUE  ((unsigned long)1<<ss)

    double res;

    res = (double)0;
    while (res == (double)0) { // Loop until nonzero result.
        // Usually only one iteration.
        res = (double)(brent_int(index, device_x_brent, seed)>>sr);     // Discard sr random bits.
        seed = (unsigned long)0;                                        // Zero seed for next time.
        if (UINT32 && UREAL64)                                          // Need another call to xor4096i.
            res += SC32*(double)brent_int(index, device_x_brent, seed); // Add low-order 32 bits.
        }

//     unsigned int SS = ((53*UINT64 + 21*UINT32)*UREAL64 + 24*UREAL32);
//     unsigned int scale = ((unsigned long)1<<SS);
//      printf("SCALE %u res %u \n",scale, SS);
    double d = (res * SCALE);
    float c= (res * SCALE);
    if (c==1.0f) {
        d -= SCALE;
        }
    return d; // Return result in (0.0, 1.0).

#undef UINT64
#undef UINT32
#undef UREAL64
#undef UREAL32
#undef SCALE
#undef SC32
#undef sr
#undef ss
    }

*/

#endif
