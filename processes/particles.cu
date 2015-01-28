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

//// HistoryBuilder class ////////////////////////////////////////////////////

HistoryBuilder::HistoryBuilder() {
    current_particle_id = 0;
}

// Create a new particle track in the history
void HistoryBuilder::cpu_new_particle_track(ui32 a_pname) {

    // If need record the first position for the tracking history
    if (current_particle_id < max_nb_particles) {

        // new particle
        pname.push_back(a_pname);
        nb_steps.push_back(0);

        std::vector<OneParticleStep> NewParticleTrack;
        history_data.push_back(NewParticleTrack);

        current_particle_id++;

    }
}

// Reacord a step in a history track
void HistoryBuilder::cpu_record_a_step(ParticleStack particles, ui32 id_part) {

    // Absolute index is need to store particle history over different iteration
    ui32 abs_id_part = cur_iter*stack_size + id_part;

    OneParticleStep astep;

    astep.pos.x = particles.px[id_part];
    astep.pos.y = particles.py[id_part];
    astep.pos.z = particles.pz[id_part];
    astep.dir.x = particles.dx[id_part];
    astep.dir.y = particles.dy[id_part];
    astep.dir.z = particles.dz[id_part];
    astep.E = particles.E[id_part];

    // Add this step
    history_data[abs_id_part].push_back(astep);
    nb_steps[abs_id_part]++;

}


//// ParticleBuilder class ///////////////////////////////////////////////////

ParticleBuilder::ParticleBuilder() {
    stack.size = 0;
    seed = 0;
}

// Set the size of the stack buffer
void ParticleBuilder::set_stack_size(ui32 nb) {
    stack.size = nb;
    dstack.size = nb;
}

// Set the seed for this stack
void ParticleBuilder::set_seed(ui32 val_seed) {
    seed = val_seed;
}

// Memory allocation for this stack
void ParticleBuilder::cpu_malloc_stack() {
    if (stack.size == 0) {
        print_warning("Stack allocation, stack size is set to zero?!");
        exit_simulation();
    }

    stack.E = (f64*)malloc(stack.size * sizeof(f64));
    stack.dx = (f64*)malloc(stack.size * sizeof(f64));
    stack.dy = (f64*)malloc(stack.size * sizeof(f64));
    stack.dz = (f64*)malloc(stack.size * sizeof(f64));
    stack.px = (f64*)malloc(stack.size * sizeof(f64));
    stack.py = (f64*)malloc(stack.size * sizeof(f64));
    stack.pz = (f64*)malloc(stack.size * sizeof(f64));
    stack.tof = (f64*)malloc(stack.size * sizeof(f64));

    stack.prng_state_1 = (ui32*)malloc(stack.size * sizeof(ui32));
    stack.prng_state_2 = (ui32*)malloc(stack.size * sizeof(ui32));
    stack.prng_state_3 = (ui32*)malloc(stack.size * sizeof(ui32));
    stack.prng_state_4 = (ui32*)malloc(stack.size * sizeof(ui32));
    stack.prng_state_5 = (ui32*)malloc(stack.size * sizeof(ui32));

    stack.geometry_id = (ui32*)malloc(stack.size * sizeof(ui32));

    stack.endsimu = (ui8*)malloc(stack.size * sizeof(ui8));
    stack.level = (ui8*)malloc(stack.size * sizeof(ui8));
    stack.pname = (ui8*)malloc(stack.size * sizeof(ui8));
}

void ParticleBuilder::gpu_malloc_stack() {

    if (dstack.size == 0) {
        print_warning("Stack allocation, stack size is set to zero?!");
        exit_simulation();
    }

    HANDLE_ERROR( cudaMalloc((void**) &dstack.E, dstack.size*sizeof(f64)) );
    HANDLE_ERROR( cudaMalloc((void**) &dstack.dx, dstack.size*sizeof(f64)) );
    HANDLE_ERROR( cudaMalloc((void**) &dstack.dy, dstack.size*sizeof(f64)) );
    HANDLE_ERROR( cudaMalloc((void**) &dstack.dz, dstack.size*sizeof(f64)) );
    HANDLE_ERROR( cudaMalloc((void**) &dstack.px, dstack.size*sizeof(f64)) );
    HANDLE_ERROR( cudaMalloc((void**) &dstack.py, dstack.size*sizeof(f64)) );
    HANDLE_ERROR( cudaMalloc((void**) &dstack.pz, dstack.size*sizeof(f64)) );
    HANDLE_ERROR( cudaMalloc((void**) &dstack.tof, dstack.size*sizeof(f64)) );

    HANDLE_ERROR( cudaMalloc((void**) &dstack.prng_state_1, dstack.size*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dstack.prng_state_2, dstack.size*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dstack.prng_state_3, dstack.size*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dstack.prng_state_4, dstack.size*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &dstack.prng_state_5, dstack.size*sizeof(ui32)) );

    HANDLE_ERROR( cudaMalloc((void**) &dstack.geometry_id, dstack.size*sizeof(ui32)) );

    HANDLE_ERROR( cudaMalloc((void**) &dstack.endsimu, dstack.size*sizeof(ui8)) );
    HANDLE_ERROR( cudaMalloc((void**) &dstack.level, dstack.size*sizeof(ui8)) );
    HANDLE_ERROR( cudaMalloc((void**) &dstack.pname, dstack.size*sizeof(ui8)) );

}

// Init particle seeds with the main seed
void ParticleBuilder::cpu_init_stack_seed() {

    if (seed == 0) {
        print_warning("The seed to init the particle stack is equal to zero!!");
    }

    srand(seed);
    ui32 i=0;
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

void ParticleBuilder::copy_seed_cpu2gpu() {

    // We consider that the CPU stack was previously initialized with seed
    // cpu_init_stack_seed();

    // Then copy data to GPU
    HANDLE_ERROR( cudaMemcpy(dstack.prng_state_1, stack.prng_state_1,
                             sizeof(ui32)*dstack.size, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dstack.prng_state_2, stack.prng_state_2,
                             sizeof(ui32)*dstack.size, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dstack.prng_state_3, stack.prng_state_3,
                             sizeof(ui32)*dstack.size, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dstack.prng_state_4, stack.prng_state_4,
                             sizeof(ui32)*dstack.size, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dstack.prng_state_5, stack.prng_state_5,
                             sizeof(ui32)*dstack.size, cudaMemcpyHostToDevice) );

}

// Print particles on a CPU stack
void ParticleBuilder::cpu_print_stack(ui32 nlim) {

    nlim = std::min(nlim, stack.size);
    ui32 i = 0;
    while (i < nlim) {
        printf("%i - p %f %f %f - d %f %f %f - E %f\n", i, stack.px[i], stack.py[i], stack.pz[i],
               stack.dx[i], stack.dy[i], stack.dz[i], stack.E[i]);
        ++i;
    }

}








#endif
