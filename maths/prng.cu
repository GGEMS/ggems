// This file is part of GGEMS
//
// FIREwork is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// FIREwork is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with FIREwork.  If not, see <http://www.gnu.org/licenses/>.
//
// GGEMS Copyright (C) 2013-2014 Julien Bert

#ifndef PRNG_CU
#define PRNG_CU

// JKISS 32-bit (period ~2^121=2.6x10^36), passes all of the Dieharder tests and the BigCrunch tests in TestU01
__device__ float JKISS32(ParticleStack &stack, unsigned int id) {
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

#endif
