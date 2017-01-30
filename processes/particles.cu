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


//// ParticleManager class ///////////////////////////////////////////////////

ParticleManager::ParticleManager()
{
    h_particles = nullptr;
    d_particles = nullptr;

    m_particles_size = 0;
}

// Init stack
void ParticleManager::initialize(GlobalSimulationParametersData *h_params )
{
    m_particles_size = h_params->size_of_particles_batch;
    mh_params = h_params;

    // Check if everything was set properly
    if ( !m_check_mandatory() )
    {
        print_error ( "Stack allocation, is stack size set to zero?!" );
        exit_simulation();
    }

    // CPU allocation
    m_cpu_malloc_stack();
    m_cpu_init_stack_seed( h_params->seed );

    // GPU allocation
    m_gpu_alloc_and_seed_copy();

}

// Check mandatory
bool ParticleManager::m_check_mandatory()
{
    if ( m_particles_size == 0 ) return false;
    else return true;
}

// Memory allocation for this stack
void ParticleManager::m_cpu_malloc_stack()
{
    // Struct allocation
    h_particles = (ParticlesData*)malloc( sizeof(ParticlesData) );

    h_particles->E = ( f32* ) malloc ( m_particles_size * sizeof ( f32 ) );
    h_particles->dx = ( f32* ) malloc ( m_particles_size * sizeof ( f32 ) );
    h_particles->dy = ( f32* ) malloc ( m_particles_size * sizeof ( f32 ) );
    h_particles->dz = ( f32* ) malloc ( m_particles_size * sizeof ( f32 ) );
    h_particles->px = ( f32* ) malloc ( m_particles_size * sizeof ( f32 ) );
    h_particles->py = ( f32* ) malloc ( m_particles_size * sizeof ( f32 ) );
    h_particles->pz = ( f32* ) malloc ( m_particles_size * sizeof ( f32 ) );
    h_particles->tof = ( f32* ) malloc ( m_particles_size * sizeof ( f32 ) );

    h_particles->prng_state_1 = ( ui32* ) malloc ( m_particles_size * sizeof ( ui32 ) );
    h_particles->prng_state_2 = ( ui32* ) malloc ( m_particles_size * sizeof ( ui32 ) );
    h_particles->prng_state_3 = ( ui32* ) malloc ( m_particles_size * sizeof ( ui32 ) );
    h_particles->prng_state_4 = ( ui32* ) malloc ( m_particles_size * sizeof ( ui32 ) );
    h_particles->prng_state_5 = ( ui32* ) malloc ( m_particles_size * sizeof ( ui32 ) );

    h_particles->geometry_id = ( ui32* ) malloc ( m_particles_size * sizeof ( ui32 ) );
    h_particles->E_index = ( ui16* ) malloc ( m_particles_size * sizeof ( ui16 ) );
    h_particles->scatter_order = (ui16*)malloc( m_particles_size * sizeof( ui16 ) );

    h_particles->next_interaction_distance = ( f32* ) malloc ( m_particles_size * sizeof ( f32 ) );
    h_particles->next_discrete_process = ( ui8* ) malloc ( m_particles_size * sizeof ( ui8 ) );

    h_particles->status = ( ui8* ) malloc ( m_particles_size * sizeof ( ui8 ) );
    h_particles->level = ( ui8* ) malloc ( m_particles_size * sizeof ( ui8 ) );
    h_particles->pname = ( ui8* ) malloc ( m_particles_size * sizeof ( ui8 ) );

    h_particles->size = m_particles_size;

/*
    h_particles->sec_E =     ( f32* ) malloc ( m_particles_size * mh_params->nb_of_secondaries * sizeof ( f32 ) );
    h_particles->sec_dx =    ( f32* ) malloc ( m_particles_size * mh_params->nb_of_secondaries * sizeof ( f32 ) );
    h_particles->sec_dy =    ( f32* ) malloc ( m_particles_size * mh_params->nb_of_secondaries * sizeof ( f32 ) );
    h_particles->sec_dz =    ( f32* ) malloc ( m_particles_size * mh_params->nb_of_secondaries * sizeof ( f32 ) );
    h_particles->sec_px =    ( f32* ) malloc ( m_particles_size * mh_params->nb_of_secondaries * sizeof ( f32 ) );
    h_particles->sec_py =    ( f32* ) malloc ( m_particles_size * mh_params->nb_of_secondaries * sizeof ( f32 ) );
    h_particles->sec_pz =    ( f32* ) malloc ( m_particles_size * mh_params->nb_of_secondaries * sizeof ( f32 ) );
    h_particles->sec_tof =   ( f32* ) malloc ( m_particles_size * mh_params->nb_of_secondaries * sizeof ( f32 ) );
    h_particles->sec_pname = ( ui8* ) malloc ( m_particles_size * mh_params->nb_of_secondaries * sizeof ( ui8 ) );
*/
}

void ParticleManager::m_gpu_alloc_and_seed_copy()
{    

    ui32 n = m_particles_size;

    /// First, struct allocation
    HANDLE_ERROR( cudaMalloc( (void**) &d_particles, sizeof( ParticlesData ) ) );

    /// Device pointers allocation

    // property
    f32* E;
    HANDLE_ERROR( cudaMalloc((void**) &E, n*sizeof(f32)) );
    f32* dx;
    HANDLE_ERROR( cudaMalloc((void**) &dx, n*sizeof(f32)) );
    f32* dy;
    HANDLE_ERROR( cudaMalloc((void**) &dy, n*sizeof(f32)) );
    f32* dz;
    HANDLE_ERROR( cudaMalloc((void**) &dz, n*sizeof(f32)) );
    f32* px;
    HANDLE_ERROR( cudaMalloc((void**) &px, n*sizeof(f32)) );
    f32* py;
    HANDLE_ERROR( cudaMalloc((void**) &py, n*sizeof(f32)) );
    f32* pz;
    HANDLE_ERROR( cudaMalloc((void**) &pz, n*sizeof(f32)) );
    f32* tof;
    HANDLE_ERROR( cudaMalloc((void**) &tof, n*sizeof(f32)) );

    // PRNG
    ui32* prng_state_1;
    HANDLE_ERROR( cudaMalloc((void**) &prng_state_1, n*sizeof(ui32)) );
    ui32* prng_state_2;
    HANDLE_ERROR( cudaMalloc((void**) &prng_state_2, n*sizeof(ui32)) );
    ui32* prng_state_3;
    HANDLE_ERROR( cudaMalloc((void**) &prng_state_3, n*sizeof(ui32)) );
    ui32* prng_state_4;
    HANDLE_ERROR( cudaMalloc((void**) &prng_state_4, n*sizeof(ui32)) );
    ui32* prng_state_5;
    HANDLE_ERROR( cudaMalloc((void**) &prng_state_5, n*sizeof(ui32)) );

    // Navigation
    ui32* geometry_id;  // current geometry crossed by the particle
    HANDLE_ERROR( cudaMalloc((void**) &geometry_id, n*sizeof(ui32)) );
    ui16* E_index;      // Energy index within CS and Mat tables
    HANDLE_ERROR( cudaMalloc((void**) &E_index, n*sizeof(ui16)) );
    ui16* scatter_order; // Scatter for imaging
    HANDLE_ERROR( cudaMalloc((void**) &scatter_order, n*sizeof(ui16)) );

    // Interactions
    f32* next_interaction_distance;
    HANDLE_ERROR( cudaMalloc((void**) &next_interaction_distance, n*sizeof(f32)) );
    ui8* next_discrete_process;
    HANDLE_ERROR( cudaMalloc((void**) &next_discrete_process, n*sizeof(ui8)) );

    // simulation
    ui8* status;
    HANDLE_ERROR( cudaMalloc((void**) &status, n*sizeof(ui8)) );
    ui8* level;
    HANDLE_ERROR( cudaMalloc((void**) &level, n*sizeof(ui8)) );
    ui8* pname; // particle name (photon, electron, etc)
    HANDLE_ERROR( cudaMalloc((void**) &pname, n*sizeof(ui8)) );

    /// Copy host data to device (only prng values)
    HANDLE_ERROR( cudaMemcpy( prng_state_1, h_particles->prng_state_1,
                              n*sizeof(ui32), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( prng_state_2, h_particles->prng_state_2,
                              n*sizeof(ui32), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( prng_state_3, h_particles->prng_state_3,
                              n*sizeof(ui32), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( prng_state_4, h_particles->prng_state_4,
                              n*sizeof(ui32), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( prng_state_5, h_particles->prng_state_5,
                              n*sizeof(ui32), cudaMemcpyHostToDevice ) );

    /// Bind data to the struct
    HANDLE_ERROR( cudaMemcpy( &(d_particles->E), &E,
                              sizeof(d_particles->E), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_particles->dx), &dx,
                              sizeof(d_particles->dx), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_particles->dy), &dy,
                              sizeof(d_particles->dy), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_particles->dz), &dz,
                              sizeof(d_particles->dz), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_particles->px), &px,
                              sizeof(d_particles->px), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_particles->py), &py,
                              sizeof(d_particles->py), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_particles->pz), &pz,
                              sizeof(d_particles->pz), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_particles->tof), &tof,
                              sizeof(d_particles->tof), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(d_particles->prng_state_1), &prng_state_1,
                              sizeof(d_particles->prng_state_1), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_particles->prng_state_2), &prng_state_2,
                              sizeof(d_particles->prng_state_2), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_particles->prng_state_3), &prng_state_3,
                              sizeof(d_particles->prng_state_3), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_particles->prng_state_4), &prng_state_4,
                              sizeof(d_particles->prng_state_4), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_particles->prng_state_5), &prng_state_5,
                              sizeof(d_particles->prng_state_5), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(d_particles->geometry_id), &geometry_id,
                              sizeof(d_particles->geometry_id), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_particles->E_index), &E_index,
                              sizeof(d_particles->E_index), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_particles->scatter_order), &scatter_order,
                              sizeof(d_particles->scatter_order), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(d_particles->next_interaction_distance), &next_interaction_distance,
                              sizeof(d_particles->next_interaction_distance), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_particles->next_discrete_process), &next_discrete_process,
                              sizeof(d_particles->next_discrete_process), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(d_particles->status), &status,
                              sizeof(d_particles->status), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_particles->level), &level,
                              sizeof(d_particles->level), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_particles->pname), &pname,
                              sizeof(d_particles->pname), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(d_particles->size), &n,
                              sizeof(d_particles->size), cudaMemcpyHostToDevice ) );

}

// Init particle seeds with the main seed
void ParticleManager::m_cpu_init_stack_seed ( ui32 seed )
{

    srand ( seed );
    ui32 i=0;
    while ( i<m_particles_size )
    {
        // init random seed
        h_particles->prng_state_1[i] = rand();
        h_particles->prng_state_2[i] = rand();
        h_particles->prng_state_3[i] = rand();
        h_particles->prng_state_4[i] = rand();
        h_particles->prng_state_5[i] = 0;      // carry

        ++i;
    }
}

/*
void ParticleManager::copy_gpu2cpu( Particles &part )
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

void ParticleManager::print_stack( Particles part, ui32 n )
{
    std::vector< std::string > status;
    status.push_back("Alive");
    status.push_back("Dead");
    status.push_back("Freeze");

    ui32 i = 0; while ( i < n ) {
        printf("%i - E %f - p %f %f %f - d %f %f %f - tof %f - Status %s\n", i, part.data_h.E[i], part.data_h.px[i],
               part.data_h.py[i], part.data_h.pz[i], part.data_h.dx[i], part.data_h.dy[i], part.data_h.dz[i], part.data_h.tof[i],
               status[ part.data_h.endsimu[ i ] ].c_str() );
        ++i;
    }

}
*/















#endif
