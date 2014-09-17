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

#ifndef PARTICLES_CUH
#define PARTICLES_CUH

#include "../global/global.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>

// Stack of particles, format data is defined as SoA
struct ParticleStack{
    // property
    float* E;
    float* dx;
    float* dy;
    float* dz;
    float* px;
    float* py;
    float* pz;
    float* tof;
    // PRNG
    unsigned int* prng_state_1;
    unsigned int* prng_state_2;
    unsigned int* prng_state_3;
    unsigned int* prng_state_4;
    unsigned int* prng_state_5;
    // Navigation
    unsigned int* geometry_id; // current geometry crossed by the particle
    // simulation
    unsigned char* endsimu;
    unsigned char* level;
    unsigned char* pname; // particle name (photon, electron, etc)
    // stack size
    unsigned int size;
}; //

// Helper to handle secondaries particles
struct SecParticle{
    float3 dir;
    float E;
    unsigned char pname;
    unsigned char endsimu;
};

// Particles class
class ParticleBuilder {
    public:
        ParticleBuilder();
        void set_stack_size(unsigned int nb);
        void set_seed(unsigned int val_seed);
        void cpu_malloc_stack();
        void init_stack_seed();
        void cpu_print_stack(unsigned int nlim);

        ParticleStack stack;
        unsigned int seed;

    private:


};

#endif
