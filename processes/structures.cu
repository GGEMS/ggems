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

#ifndef STRUCTURES_CU
#define STRUCTURES_CU
#include "stuctures.h"

// Some error "checkers"
// comes from "cuda by example" book
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#ifndef HANDLE_ERROR
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#endif


// comes from "cuda programming" book
__host__ void cuda_error_check (const char * prefix, const char * postfix) {
    if(cudaPeekAtLastError() != cudaSuccess ) {
        printf("\n%s%s%s\n",prefix, cudaGetErrorString(cudaGetLastError()),postfix);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

}


// Stack device allocation
void _stack_device_malloc(ParticleStack &stackpart, int stack_size) {
    stackpart.size = stack_size;
    unsigned int mem_stackpart_float = stack_size * sizeof(float);
    unsigned int mem_stackpart_uint = stack_size * sizeof(unsigned int);
    unsigned int mem_stackpart_char = stack_size * sizeof(char);

    // property
    HANDLE_ERROR(cudaMalloc((void**) &stackpart.E, mem_stackpart_float));
    HANDLE_ERROR(cudaMalloc((void**) &stackpart.dx, mem_stackpart_float));
    HANDLE_ERROR(cudaMalloc((void**) &stackpart.dy, mem_stackpart_float));
    HANDLE_ERROR(cudaMalloc((void**) &stackpart.dz, mem_stackpart_float));
    HANDLE_ERROR(cudaMalloc((void**) &stackpart.px, mem_stackpart_float));
    HANDLE_ERROR(cudaMalloc((void**) &stackpart.py, mem_stackpart_float));
    HANDLE_ERROR(cudaMalloc((void**) &stackpart.pz, mem_stackpart_float));
    HANDLE_ERROR(cudaMalloc((void**) &stackpart.tof, mem_stackpart_float));
    // PRNG
    HANDLE_ERROR(cudaMalloc((void**) &stackpart.prng_state_1, mem_stackpart_uint));
    HANDLE_ERROR(cudaMalloc((void**) &stackpart.prng_state_2, mem_stackpart_uint));
    HANDLE_ERROR(cudaMalloc((void**) &stackpart.prng_state_3, mem_stackpart_uint));
    HANDLE_ERROR(cudaMalloc((void**) &stackpart.prng_state_4, mem_stackpart_uint));
    HANDLE_ERROR(cudaMalloc((void**) &stackpart.prng_state_5, mem_stackpart_uint));
    // simulation
    HANDLE_ERROR(cudaMalloc((void**) &stackpart.endsimu, mem_stackpart_char));
    HANDLE_ERROR(cudaMalloc((void**) &stackpart.level, mem_stackpart_char));
    HANDLE_ERROR(cudaMalloc((void**) &stackpart.pname, mem_stackpart_char));
}

// Init particle seeds with the main seed
void wrap_init_particle_seeds(ParticleStack &d_p, int seed) {
    unsigned int *state1 = (unsigned int*)malloc(sizeof(unsigned int)*d_p.size);
    unsigned int *state2 = (unsigned int*)malloc(sizeof(unsigned int)*d_p.size);
    unsigned int *state3 = (unsigned int*)malloc(sizeof(unsigned int)*d_p.size);
    unsigned int *state4 = (unsigned int*)malloc(sizeof(unsigned int)*d_p.size);
    unsigned int *state5 = (unsigned int*)malloc(sizeof(unsigned int)*d_p.size);

    srand(seed);
    int i=0;
    while (i<d_p.size) {
        // init random seed
        state1[i] = rand();
        state2[i] = rand();
        state3[i] = rand();
        state4[i] = rand();
        state5[i] = 0;      // carry
        ++i;
    }
// printf("%f %f %f %f %");
    HANDLE_ERROR(cudaMemcpy(d_p.prng_state_1, state1, sizeof(unsigned int)*d_p.size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_p.prng_state_2, state2, sizeof(unsigned int)*d_p.size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_p.prng_state_3, state3, sizeof(unsigned int)*d_p.size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_p.prng_state_4, state4, sizeof(unsigned int)*d_p.size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_p.prng_state_5, state5, sizeof(unsigned int)*d_p.size, cudaMemcpyHostToDevice));
}


#endif
