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

#ifndef PARTICLES_CU
#define PARTICLES_CU
#include "particles.cuh"

ParticleBuilder::ParticleBuilder() {
    stack.size = 0;
    seed = 0;
}

// Set the size of the stack buffer
void ParticleBuilder::set_stack_size(unsigned int nb) {
    stack.size = nb;
}

// Set the seed for this stack
void ParticleBuilder::set_seed(unsigned int val_seed) {
    seed = val_seed;
}

// Memory allocation for this stack
void ParticleBuilder::cpu_malloc_stack() {
    if (stack.size == 0) {
        print_warning("Stack allocation, stack size is set to zero?!");
        exit_simulation();
    }

    stack.E = (float*)malloc(stack.size * sizeof(float));
    stack.dx = (float*)malloc(stack.size * sizeof(float));
    stack.dy = (float*)malloc(stack.size * sizeof(float));
    stack.dz = (float*)malloc(stack.size * sizeof(float));
    stack.px = (float*)malloc(stack.size * sizeof(float));
    stack.py = (float*)malloc(stack.size * sizeof(float));
    stack.pz = (float*)malloc(stack.size * sizeof(float));
    stack.tof = (float*)malloc(stack.size * sizeof(float));

    stack.prng_state_1 = (unsigned int*)malloc(stack.size * sizeof(unsigned int));
    stack.prng_state_2 = (unsigned int*)malloc(stack.size * sizeof(unsigned int));
    stack.prng_state_3 = (unsigned int*)malloc(stack.size * sizeof(unsigned int));
    stack.prng_state_4 = (unsigned int*)malloc(stack.size * sizeof(unsigned int));
    stack.prng_state_5 = (unsigned int*)malloc(stack.size * sizeof(unsigned int));

    stack.endsimu = (unsigned char*)malloc(stack.size * sizeof(unsigned char));
    stack.level = (unsigned char*)malloc(stack.size * sizeof(unsigned char));
    stack.pname = (unsigned char*)malloc(stack.size * sizeof(unsigned char));
}

// Init particle seeds with the main seed
void ParticleBuilder::init_stack_seed() {

    if (seed == 0) {
        print_warning("The seed to init the particle stack is equal to zero!!");
    }

    srand(seed);
    unsigned int i=0;
    while (i<stack.size) {
        // init random seed
        stack.prng_state_1[i] = rand();
        stack.prng_state_2[i] = rand();
        stack.prng_state_3[i] = rand();
        stack.prng_state_4[i] = rand();
        stack.prng_state_5[i] = 0;      // carry
        ++i;
    }

}

#endif
