// GGEMS Copyright (C) 2015

/*!
 * \file particles.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 18 novembre 2015
 *
 *
 *
 */

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

ParticleManager::ParticleManager()
{
    particles.size = 0;
}

// Init stack
void ParticleManager::initialize ( GlobalSimulationParameters params )
{
    particles.size = params.data_h.size_of_particles_batch;
    particles.data_h.size = params.data_h.size_of_particles_batch;    

    m_params = params;

    // Check if everything was set properly
    if ( !m_check_mandatory() )
    {
        print_error ( "Stack allocation, is stack size set to zero?!" );
        exit_simulation();
    }

    // CPU allocation
    m_cpu_malloc_stack();

//    // Init seed
//    cpu_prng_init( particles.data_h.prng, particles.size, params.data_h.seed );

    m_cpu_init_stack_seed ( params.data_h.seed );

    // Init seeds
    if ( params.data_h.device_target == GPU_DEVICE )
    {
        // GPU allocation
        m_gpu_malloc_stack();

//        // Init seed on GPU side
//        gpu_prng_init( particles.data_d.prng, particles.size,
//                       params.data_h.seed, params.data_h.gpu_block_size );
/*
        srand(m_params.data_h.seed);
        prng_states *hostStates;
        hostStates = (prng_states*)malloc(particles.size * sizeof(prng_states));
        for (ui32 i=0; i<particles.size; i++)
        {
            prng_states aState;
            aState.state_1 = rand();
            aState.state_2 = rand();
            aState.state_3 = rand();
            aState.state_4 = rand();
            aState.state_5 = 0;

            hostStates[i] = aState;
        }
        cudaMemcpy(particles.data_d.prng, hostStates, particles.size * sizeof(prng_states), cudaMemcpyHostToDevice);
*/
        // Copy data to the GPU    - FIXME - seed on CPU side is not used - JB
        m_copy_seed_cpu2gpu();
    }

}

// Check mandatory
bool ParticleManager::m_check_mandatory()
{
    if ( particles.size == 0 ) return false;
    else return true;
}

// Memory allocation for this stack
void ParticleManager::m_cpu_malloc_stack()
{

    particles.data_h.E = ( f32* ) malloc ( particles.size * sizeof ( f32 ) );
    particles.data_h.dx = ( f32* ) malloc ( particles.size * sizeof ( f32 ) );
    particles.data_h.dy = ( f32* ) malloc ( particles.size * sizeof ( f32 ) );
    particles.data_h.dz = ( f32* ) malloc ( particles.size * sizeof ( f32 ) );
    particles.data_h.px = ( f32* ) malloc ( particles.size * sizeof ( f32 ) );
    particles.data_h.py = ( f32* ) malloc ( particles.size * sizeof ( f32 ) );
    particles.data_h.pz = ( f32* ) malloc ( particles.size * sizeof ( f32 ) );
    particles.data_h.tof = ( f32* ) malloc ( particles.size * sizeof ( f32 ) );

    // scatter_order
    particles.data_h.scatter_order = (ui32*)malloc( particles.size * sizeof( ui32 ) );

    particles.data_h.prng_state_1 = ( ui32* ) malloc ( particles.size * sizeof ( ui32 ) );
    particles.data_h.prng_state_2 = ( ui32* ) malloc ( particles.size * sizeof ( ui32 ) );
    particles.data_h.prng_state_3 = ( ui32* ) malloc ( particles.size * sizeof ( ui32 ) );
    particles.data_h.prng_state_4 = ( ui32* ) malloc ( particles.size * sizeof ( ui32 ) );
    particles.data_h.prng_state_5 = ( ui32* ) malloc ( particles.size * sizeof ( ui32 ) );

    //particles.data_h.prng = ( prng_states* ) malloc ( particles.size * sizeof ( prng_states ) );

    particles.data_h.geometry_id = ( ui32* ) malloc ( particles.size * sizeof ( ui32 ) );
    particles.data_h.E_index = ( ui32* ) malloc ( particles.size * sizeof ( ui32 ) );

    particles.data_h.next_interaction_distance = ( f32* ) malloc ( particles.size * sizeof ( f32 ) );
    particles.data_h.next_discrete_process = ( ui8* ) malloc ( particles.size * sizeof ( ui8 ) );

    particles.data_h.endsimu = ( ui8* ) malloc ( particles.size * sizeof ( ui8 ) );
    particles.data_h.level = ( ui8* ) malloc ( particles.size * sizeof ( ui8 ) );
    particles.data_h.pname = ( ui8* ) malloc ( particles.size * sizeof ( ui8 ) );


    particles.data_h.sec_E =     ( f32* ) malloc ( particles.size * m_params.data_h.nb_of_secondaries * sizeof ( f32 ) );
    particles.data_h.sec_dx =    ( f32* ) malloc ( particles.size * m_params.data_h.nb_of_secondaries * sizeof ( f32 ) );
    particles.data_h.sec_dy =    ( f32* ) malloc ( particles.size * m_params.data_h.nb_of_secondaries * sizeof ( f32 ) );
    particles.data_h.sec_dz =    ( f32* ) malloc ( particles.size * m_params.data_h.nb_of_secondaries * sizeof ( f32 ) );
    particles.data_h.sec_px =    ( f32* ) malloc ( particles.size * m_params.data_h.nb_of_secondaries * sizeof ( f32 ) );
    particles.data_h.sec_py =    ( f32* ) malloc ( particles.size * m_params.data_h.nb_of_secondaries * sizeof ( f32 ) );
    particles.data_h.sec_pz =    ( f32* ) malloc ( particles.size * m_params.data_h.nb_of_secondaries * sizeof ( f32 ) );
    particles.data_h.sec_tof =   ( f32* ) malloc ( particles.size * m_params.data_h.nb_of_secondaries * sizeof ( f32 ) );
    particles.data_h.sec_pname = ( ui8* ) malloc ( particles.size * m_params.data_h.nb_of_secondaries * sizeof ( ui8 ) );

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

void ParticleManager::m_gpu_malloc_stack()
{    
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.E, particles.size*sizeof ( f32 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.dx, particles.size*sizeof ( f32 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.dy, particles.size*sizeof ( f32 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.dz, particles.size*sizeof ( f32 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.px, particles.size*sizeof ( f32 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.py, particles.size*sizeof ( f32 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.pz, particles.size*sizeof ( f32 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.tof, particles.size*sizeof ( f32 ) ) );

    // scatter_order
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.scatter_order, particles.size*sizeof ( ui32 ) ) );

    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.prng_state_1, particles.size*sizeof ( ui32 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.prng_state_2, particles.size*sizeof ( ui32 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.prng_state_3, particles.size*sizeof ( ui32 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.prng_state_4, particles.size*sizeof ( ui32 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.prng_state_5, particles.size*sizeof ( ui32 ) ) );


    //HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.prng, particles.size*sizeof ( prng_states ) ) );

    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.geometry_id, particles.size*sizeof ( ui32 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.E_index, particles.size*sizeof ( ui32 ) ) );

    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.next_interaction_distance, particles.size*sizeof ( f32 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.next_discrete_process, particles.size*sizeof ( ui8 ) ) );

    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.endsimu, particles.size*sizeof ( ui8 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.level, particles.size*sizeof ( ui8 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.pname, particles.size*sizeof ( ui8 ) ) );

    particles.data_d.size = particles.data_h.size;

    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.sec_E,     particles.size * m_params.data_h.nb_of_secondaries *sizeof ( f32 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.sec_dx,    particles.size * m_params.data_h.nb_of_secondaries *sizeof ( f32 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.sec_dy,    particles.size * m_params.data_h.nb_of_secondaries *sizeof ( f32 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.sec_dz,    particles.size * m_params.data_h.nb_of_secondaries *sizeof ( f32 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.sec_px,    particles.size * m_params.data_h.nb_of_secondaries *sizeof ( f32 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.sec_py,    particles.size * m_params.data_h.nb_of_secondaries *sizeof ( f32 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.sec_pz,    particles.size * m_params.data_h.nb_of_secondaries *sizeof ( f32 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.sec_tof,   particles.size * m_params.data_h.nb_of_secondaries *sizeof ( f32 ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &particles.data_d.sec_pname, particles.size * m_params.data_h.nb_of_secondaries *sizeof ( ui8 ) ) );

}


// Init particle seeds with the main seed
void ParticleManager::m_cpu_init_stack_seed ( ui32 seed )
{

    srand ( seed );
    ui32 i=0;
    while ( i<particles.size )
    {
        // init random seed
        particles.data_h.prng_state_1[i] = rand();
        particles.data_h.prng_state_2[i] = rand();
        particles.data_h.prng_state_3[i] = rand();
        particles.data_h.prng_state_4[i] = rand();
        particles.data_h.prng_state_5[i] = 0;      // carry

//         printf("%d %d %d %d %d \n",particles.data_h.prng_state_1[i],particles.data_h.prng_state_2[i],particles.data_h.prng_state_3[i],particles.data_h.prng_state_4[i],particles.data_h.prng_state_5[i]);

        ++i;
    }
}



void ParticleManager::m_copy_seed_cpu2gpu()
{

    // We consider that the CPU stack was previously initialized with seed
    HANDLE_ERROR ( cudaMemcpy ( particles.data_d.prng_state_1, particles.data_h.prng_state_1,
                                sizeof ( ui32 ) *particles.size, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR ( cudaMemcpy ( particles.data_d.prng_state_2, particles.data_h.prng_state_2,
                                sizeof ( ui32 ) *particles.size, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR ( cudaMemcpy ( particles.data_d.prng_state_3, particles.data_h.prng_state_3,
                                sizeof ( ui32 ) *particles.size, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR ( cudaMemcpy ( particles.data_d.prng_state_4, particles.data_h.prng_state_4,
                                sizeof ( ui32 ) *particles.size, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR ( cudaMemcpy ( particles.data_d.prng_state_5, particles.data_h.prng_state_5,
                                sizeof ( ui32 ) *particles.size, cudaMemcpyHostToDevice ) );

}


void ParticleManager::copy_gpu2cpu( Particles part )
{
    HANDLE_ERROR ( cudaMemcpy ( part.data_h.E, part.data_d.E, sizeof ( f32 ) *part.size, cudaMemcpyDeviceToHost ) );

    HANDLE_ERROR ( cudaMemcpy ( part.data_h.px, part.data_d.px, sizeof ( f32 ) *part.size, cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR ( cudaMemcpy ( part.data_h.py, part.data_d.py, sizeof ( f32 ) *part.size, cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR ( cudaMemcpy ( part.data_h.pz, part.data_d.pz, sizeof ( f32 ) *part.size, cudaMemcpyDeviceToHost ) );

    HANDLE_ERROR ( cudaMemcpy ( part.data_h.dx, part.data_d.dx, sizeof ( f32 ) *part.size, cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR ( cudaMemcpy ( part.data_h.dy, part.data_d.dy, sizeof ( f32 ) *part.size, cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR ( cudaMemcpy ( part.data_h.dz, part.data_d.dz, sizeof ( f32 ) *part.size, cudaMemcpyDeviceToHost ) );

    HANDLE_ERROR ( cudaMemcpy ( part.data_h.tof, part.data_d.tof, sizeof ( f32 ) *part.size, cudaMemcpyDeviceToHost ) );

    HANDLE_ERROR ( cudaMemcpy ( part.data_h.endsimu, part.data_d.endsimu, sizeof ( ui8 ) *part.size, cudaMemcpyDeviceToHost ) );
}

void ParticleManager::print_stack( Particles part )
{
    std::vector< std::string > status;
    status.push_back("Alive");
    status.push_back("Dead");
    status.push_back("Freeze");

    ui32 i = 0; while ( i < part.size ) {
        printf("%i - E %f - p %f %f %f - d %f %f %f - tof %f - Status %s\n", i, part.data_h.E[i], part.data_h.px[i],
               part.data_h.py[i], part.data_h.pz[i], part.data_h.dx[i], part.data_h.dy[i], part.data_h.dz[i], part.data_h.tof[i],
               status[ part.data_h.endsimu[ i ] ].c_str() );
        ++i;
    }

}
















#endif
