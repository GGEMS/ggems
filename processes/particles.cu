// GGEMS Copyright (C) 2015

#ifndef PARTICLES_CU
#define PARTICLES_CU
#include "particles.cuh"

/*
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
*/

//// ParticleManager class ///////////////////////////////////////////////////

ParticleManager::ParticleManager() {
    stack_h.size = 0;
}

// Init stack
void ParticleManager::initialize(GlobalSimulationParameters params) {
    stack_h.size = params.size_of_particles_batch;

    // Check if everything was set properly
    if ( !m_check_mandatory() ) {
        print_error("Stack allocation, stack size is set to zero?!");
        exit_simulation();
    }

    // CPU allocation
    m_cpu_malloc_stack();
    // Init seeds
    //m_cpu_init_stack_seed(params.seed);
    if (params.device_target == GPU_DEVICE) {
        // GPU allocation
        m_gpu_malloc_stack();
        // Copy data to the GPU
        m_copy_seed_cpu2gpu();
    }

}

// Print particles on a CPU stack
void ParticleManager::cpu_print_stack(ui32 nlim) {

    nlim = std::min(nlim, stack_h.size);
    ui32 i = 0;
    while (i < nlim) {
        printf("%i - p %f %f %f - d %f %f %f - E %f\n", i, stack_h.px[i], stack_h.py[i], stack_h.pz[i],
               stack_h.dx[i], stack_h.dy[i], stack_h.dz[i], stack_h.E[i]);
        ++i;
    }

}

// Check mandatory
bool ParticleManager::m_check_mandatory() {
    if (stack_h.size == 0) return false;
    else return true;
}

// Memory allocation for this stack
void ParticleManager::m_cpu_malloc_stack() {

    stack_h.E = (f64*)malloc(stack_h.size * sizeof(f64));
    stack_h.dx = (f64*)malloc(stack_h.size * sizeof(f64));
    stack_h.dy = (f64*)malloc(stack_h.size * sizeof(f64));
    stack_h.dz = (f64*)malloc(stack_h.size * sizeof(f64));
    stack_h.px = (f64*)malloc(stack_h.size * sizeof(f64));
    stack_h.py = (f64*)malloc(stack_h.size * sizeof(f64));
    stack_h.pz = (f64*)malloc(stack_h.size * sizeof(f64));
    stack_h.tof = (f64*)malloc(stack_h.size * sizeof(f64));

    stack_h.prng_state_1 = (ui32*)malloc(stack_h.size * sizeof(ui32));
    stack_h.prng_state_2 = (ui32*)malloc(stack_h.size * sizeof(ui32));
    stack_h.prng_state_3 = (ui32*)malloc(stack_h.size * sizeof(ui32));
    stack_h.prng_state_4 = (ui32*)malloc(stack_h.size * sizeof(ui32));
    stack_h.prng_state_5 = (ui32*)malloc(stack_h.size * sizeof(ui32));

    stack_h.geometry_id = (ui32*)malloc(stack_h.size * sizeof(ui32));

    stack_h.endsimu = (ui8*)malloc(stack_h.size * sizeof(ui8));
    stack_h.level = (ui8*)malloc(stack_h.size * sizeof(ui8));
    stack_h.pname = (ui8*)malloc(stack_h.size * sizeof(ui8));
}

/*
void ParticleManager::m_cpu_free_stack() {

    free(stack_h.E);
    free(stack_h.dx);
    free(stack_h.dy);
    free(stack_h.dz);
    free(stack_h.px);
    free(stack_h.py);
    free(stack_h.pz);
    free(stack_h.tof);

    free(stack_h.prng_state_1);
    free(stack_h.prng_state_2);
    free(stack_h.prng_state_3);
    free(stack_h.prng_state_4);
    free(stack_h.prng_state_5);

    free(stack_h.geometry_id);

    free(stack_h.endsimu);
    free(stack_h.level);
    free(stack_h.pname);
}
*/

void ParticleManager::m_gpu_malloc_stack() {

    HANDLE_ERROR( cudaMalloc((void**) &stack_d.E, stack_d.size*sizeof(f64)) );
    HANDLE_ERROR( cudaMalloc((void**) &stack_d.dx, stack_d.size*sizeof(f64)) );
    HANDLE_ERROR( cudaMalloc((void**) &stack_d.dy, stack_d.size*sizeof(f64)) );
    HANDLE_ERROR( cudaMalloc((void**) &stack_d.dz, stack_d.size*sizeof(f64)) );
    HANDLE_ERROR( cudaMalloc((void**) &stack_d.px, stack_d.size*sizeof(f64)) );
    HANDLE_ERROR( cudaMalloc((void**) &stack_d.py, stack_d.size*sizeof(f64)) );
    HANDLE_ERROR( cudaMalloc((void**) &stack_d.pz, stack_d.size*sizeof(f64)) );
    HANDLE_ERROR( cudaMalloc((void**) &stack_d.tof, stack_d.size*sizeof(f64)) );

    HANDLE_ERROR( cudaMalloc((void**) &stack_d.prng_state_1, stack_d.size*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &stack_d.prng_state_2, stack_d.size*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &stack_d.prng_state_3, stack_d.size*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &stack_d.prng_state_4, stack_d.size*sizeof(ui32)) );
    HANDLE_ERROR( cudaMalloc((void**) &stack_d.prng_state_5, stack_d.size*sizeof(ui32)) );

    HANDLE_ERROR( cudaMalloc((void**) &stack_d.geometry_id, stack_d.size*sizeof(ui32)) );

    HANDLE_ERROR( cudaMalloc((void**) &stack_d.endsimu, stack_d.size*sizeof(ui8)) );
    HANDLE_ERROR( cudaMalloc((void**) &stack_d.level, stack_d.size*sizeof(ui8)) );
    HANDLE_ERROR( cudaMalloc((void**) &stack_d.pname, stack_d.size*sizeof(ui8)) );

}

// Init particle seeds with the main seed
void ParticleManager::m_cpu_init_stack_seed(ui32 seed) {

    srand(seed);
    ui32 i=0;
    while (i<stack_h.size) {
        // init random seed
        stack_h.prng_state_1[i] = rand();
        stack_h.prng_state_2[i] = rand();
        stack_h.prng_state_3[i] = rand();
        stack_h.prng_state_4[i] = rand();
        stack_h.prng_state_5[i] = 0;      // carry
        ++i;
    }
}

void ParticleManager::m_copy_seed_cpu2gpu() {

    // We consider that the CPU stack was previously initialized with seed
    // cpu_init_stack_seed();

    // Then copy data to GPU
    stack_d.size = stack_h.size;

    HANDLE_ERROR( cudaMemcpy(stack_d.prng_state_1, stack_h.prng_state_1,
                             sizeof(ui32)*stack_d.size, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(stack_d.prng_state_2, stack_h.prng_state_2,
                             sizeof(ui32)*stack_d.size, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(stack_d.prng_state_3, stack_h.prng_state_3,
                             sizeof(ui32)*stack_d.size, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(stack_d.prng_state_4, stack_h.prng_state_4,
                             sizeof(ui32)*stack_d.size, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(stack_d.prng_state_5, stack_h.prng_state_5,
                             sizeof(ui32)*stack_d.size, cudaMemcpyHostToDevice) );

}










#endif
