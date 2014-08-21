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
#include "structures.cuh"

// Some error "checkers"
// comes from "cuda by example" book
static void HandleError( cudaError_t err, const char *file,int line ) {
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

//copy host to device
void stack_copy_host_to_device(ParticleStack &stackpart_h,ParticleStack &stackpart_d){
    unsigned int stack_size = stackpart_h.size;
    unsigned int mem_stackpart_float = stack_size * sizeof(float);
    unsigned int mem_stackpart_uint = stack_size * sizeof(unsigned int);
    unsigned int mem_stackpart_uchar = stack_size * sizeof(unsigned char);
    
    HANDLE_ERROR(cudaMemcpy(stackpart_h.E, stackpart_h.E, mem_stackpart_float, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(stackpart_h.px, stackpart_h.px, mem_stackpart_float, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(stackpart_h.py, stackpart_h.py, mem_stackpart_float, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(stackpart_h.pz, stackpart_h.pz, mem_stackpart_float, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(stackpart_h.dx, stackpart_h.dx, mem_stackpart_float, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(stackpart_h.dy, stackpart_h.dy, mem_stackpart_float, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(stackpart_h.dz, stackpart_h.dz, mem_stackpart_float, cudaMemcpyHostToDevice));
    
    HANDLE_ERROR(cudaMemcpy(stackpart_h.tof, stackpart_h.tof, mem_stackpart_float, cudaMemcpyHostToDevice));
    
    HANDLE_ERROR(cudaMemcpy(stackpart_h.prng_state_1, stackpart_h.prng_state_1, mem_stackpart_uint, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(stackpart_h.prng_state_2, stackpart_h.prng_state_2, mem_stackpart_uint, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(stackpart_h.prng_state_3, stackpart_h.prng_state_3, mem_stackpart_uint, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(stackpart_h.prng_state_4, stackpart_h.prng_state_4, mem_stackpart_uint, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(stackpart_h.prng_state_5, stackpart_h.prng_state_5, mem_stackpart_uint, cudaMemcpyHostToDevice));
    
    HANDLE_ERROR(cudaMemcpy(stackpart_h.endsimu, stackpart_h.tof, mem_stackpart_uchar, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(stackpart_h.level, stackpart_h.tof, mem_stackpart_uchar, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(stackpart_h.pname, stackpart_h.tof, mem_stackpart_uchar, cudaMemcpyHostToDevice));

}

// stack copy device to host
void stack_copy_device_to_host(ParticleStack &stackpart_d,ParticleStack &stackpart_h){
    unsigned int stack_size = stackpart_h.size;
    unsigned int mem_stackpart_float = stack_size * sizeof(float);
    unsigned int mem_stackpart_uint = stack_size * sizeof(unsigned int);
    unsigned int mem_stackpart_uchar = stack_size * sizeof(unsigned char);
    
    HANDLE_ERROR(cudaMemcpy(stackpart_h.E, stackpart_h.E, mem_stackpart_float, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(stackpart_h.px, stackpart_h.px, mem_stackpart_float, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(stackpart_h.py, stackpart_h.py, mem_stackpart_float, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(stackpart_h.pz, stackpart_h.pz, mem_stackpart_float, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(stackpart_h.dx, stackpart_h.dx, mem_stackpart_float, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(stackpart_h.dy, stackpart_h.dy, mem_stackpart_float, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(stackpart_h.dz, stackpart_h.dz, mem_stackpart_float, cudaMemcpyDeviceToHost));
    
    HANDLE_ERROR(cudaMemcpy(stackpart_h.tof, stackpart_h.tof, mem_stackpart_float, cudaMemcpyDeviceToHost));
    
    HANDLE_ERROR(cudaMemcpy(stackpart_h.prng_state_1, stackpart_h.prng_state_1, mem_stackpart_uint, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(stackpart_h.prng_state_2, stackpart_h.prng_state_2, mem_stackpart_uint, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(stackpart_h.prng_state_3, stackpart_h.prng_state_3, mem_stackpart_uint, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(stackpart_h.prng_state_4, stackpart_h.prng_state_4, mem_stackpart_uint, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(stackpart_h.prng_state_5, stackpart_h.prng_state_5, mem_stackpart_uint, cudaMemcpyDeviceToHost));
    
    HANDLE_ERROR(cudaMemcpy(stackpart_h.endsimu, stackpart_h.tof, mem_stackpart_uchar, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(stackpart_h.level, stackpart_h.tof, mem_stackpart_uchar, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(stackpart_h.pname, stackpart_h.tof, mem_stackpart_uchar, cudaMemcpyDeviceToHost));
    
PRINTFUNCTION
}

// Stack host allocation
void stack_host_malloc(ParticleStack &phasespace, int stack_size)
    {
    phasespace.size = stack_size;
    unsigned int mem_stackpart_float = stack_size * sizeof(float);
    unsigned int mem_stackpart_uint = stack_size * sizeof(unsigned int);
    unsigned int mem_stackpart_uchar = stack_size * sizeof(unsigned char);

    phasespace.E = (float*)malloc(mem_stackpart_float);
    phasespace.dx = (float*)malloc(mem_stackpart_float);
    phasespace.dy = (float*)malloc(mem_stackpart_float);
    phasespace.dz = (float*)malloc(mem_stackpart_float);
    phasespace.px = (float*)malloc(mem_stackpart_float);
    phasespace.py = (float*)malloc(mem_stackpart_float);
    phasespace.pz = (float*)malloc(mem_stackpart_float);
    
    phasespace.tof = (float*)malloc(mem_stackpart_float);
    
    phasespace.prng_state_1 = (unsigned int*)malloc(mem_stackpart_uint);
    phasespace.prng_state_2 = (unsigned int*)malloc(mem_stackpart_uint);
    phasespace.prng_state_3 = (unsigned int*)malloc(mem_stackpart_uint);
    phasespace.prng_state_4 = (unsigned int*)malloc(mem_stackpart_uint);
    phasespace.prng_state_5 = (unsigned int*)malloc(mem_stackpart_uint);
    phasespace.endsimu = (unsigned char*)malloc(mem_stackpart_uchar);
    phasespace.level = (unsigned char*)malloc(mem_stackpart_uchar);
    phasespace.pname = (unsigned char*)malloc(mem_stackpart_uchar);
   
    }


// Stack device allocation
void stack_device_malloc(ParticleStack &stackpart, int stack_size) {
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

// Copy electron cross section table to device
void  wrap_copy_crosssection_to_device (CrossSectionTableElectrons &h_etables,
                                        CrossSectionTableElectrons &d_etables,
                                        char *m_physics_list) {

    unsigned int mem_mat_bins_flt = h_etables.nb_mat*h_etables.nb_bins * sizeof(float);

    d_etables.E_min = h_etables.E_min;
    d_etables.E_max = h_etables.E_max;
    d_etables.nb_bins = h_etables.nb_bins;
    d_etables.nb_mat = h_etables.nb_mat;
    d_etables.cutEnergyElectron = h_etables.cutEnergyElectron;
    d_etables.cutEnergyGamma = h_etables.cutEnergyGamma;

    HANDLE_ERROR(cudaMalloc((void**) &d_etables.E, mem_mat_bins_flt));
    HANDLE_ERROR(cudaMalloc((void**) &d_etables.eRange, mem_mat_bins_flt));

    HANDLE_ERROR(cudaMemcpy(d_etables.E, h_etables.E, mem_mat_bins_flt, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_etables.eRange, h_etables.eRange, mem_mat_bins_flt, cudaMemcpyHostToDevice));

    if(m_physics_list[ELECTRON_MSC] == 1) {
        HANDLE_ERROR(cudaMalloc((void**) &d_etables.eMSC, mem_mat_bins_flt));

        HANDLE_ERROR(cudaMemcpy(d_etables.eMSC, h_etables.eMSC, mem_mat_bins_flt, cudaMemcpyHostToDevice));
    }

    if(m_physics_list[ELECTRON_BREMSSTRAHLUNG] == 1) {
        HANDLE_ERROR(cudaMalloc((void**) &d_etables.eBremdedx, mem_mat_bins_flt));
        HANDLE_ERROR(cudaMalloc((void**) &d_etables.eBremCS, mem_mat_bins_flt));

        HANDLE_ERROR(cudaMemcpy(d_etables.eBremdedx, h_etables.eBremdedx, mem_mat_bins_flt, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_etables.eBremCS, h_etables.eBremCS, mem_mat_bins_flt, cudaMemcpyHostToDevice));

    }

    if(m_physics_list[ELECTRON_IONISATION] == 1) {
        HANDLE_ERROR(cudaMalloc((void**) &d_etables.eIonisationdedx, mem_mat_bins_flt));
        HANDLE_ERROR(cudaMalloc((void**) &d_etables.eIonisationCS, mem_mat_bins_flt));

        HANDLE_ERROR(cudaMemcpy(d_etables.eIonisationdedx, h_etables.eIonisationdedx, mem_mat_bins_flt, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_etables.eIonisationCS, h_etables.eIonisationCS, mem_mat_bins_flt, cudaMemcpyHostToDevice));
    }


}

#endif
