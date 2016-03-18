// GGEMS Copyright (C) 2015

/*!
 * \file phasespace_source.cu
 * \brief phasespace source class
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 9 mars 2016
 *
 * phasespace source class
 *
 */

#ifndef PHASESPACE_SOURCE_CU
#define PHASESPACE_SOURCE_CU

#include "phasespace_source.cuh"

///////// GPU code ////////////////////////////////////////////////////

// Internal function that create a new particle to the buffer at the slot id
__host__ __device__ void PHSPSRC::phsp_source ( ParticlesData particles_data,
                                                IaeaType phasespace, PhSpTransform transform, ui32 id )



{
    // Get a random sources (i.e. virtual sources that use the same phasespace)
    ui32 source_id = 0;
    if ( transform.nb_sources > 1 )
    {
        source_id = binary_search( prng_uniform( particles_data, id ), transform.cdf, transform.nb_sources );
    }

#ifdef DEBUG
    assert( source_id < transform.nb_sources );
#endif

    // Get a random particle from the phasespace
    ui32 phsp_id = (ui32) ( f32(phasespace.tot_particles) * prng_uniform( particles_data, id ) );

#ifdef DEBUG
    assert( phsp_id < phasespace.tot_particles );
#endif

    // Then set the mandatory field to create a new particle
    particles_data.E[id] = phasespace.energy[ phsp_id ];     // Energy in MeV

    // Aplly trsnformation
    f32xyz pos = make_f32xyz( phasespace.pos_x[ phsp_id ], phasespace.pos_y[ phsp_id ], phasespace.pos_z[ phsp_id ] );
    f32xyz dir = make_f32xyz( phasespace.dir_x[ phsp_id ], phasespace.dir_y[ phsp_id ], phasespace.dir_z[ phsp_id ] );
    f32xyz t = make_f32xyz( transform.tx[ source_id ], transform.ty[ source_id ], transform.tz[ source_id ] );

    //printf("ID %i rot %f %f %f\n", id, transform.rx, transform.ry, transform.rz);

    pos = fxyz_add( pos, t );                                        // translate
    pos = fxyz_rotate_z_axis( pos, transform.rz[ source_id ] );      // Rotate: yaw, pitch, and roll convetion (RzRyRx)
    pos = fxyz_rotate_y_axis( pos, transform.ry[ source_id ] );      //         for Euler convention RzRyRz
    pos = fxyz_rotate_x_axis( pos, transform.rx[ source_id ] );      //         Here is a right-hand rule
    //                TODO add scaling - JB
    dir = fxyz_rotate_z_axis( dir, transform.rz[ source_id ] );
    dir = fxyz_rotate_y_axis( dir, transform.ry[ source_id ] );
    dir = fxyz_rotate_x_axis( dir, transform.rx[ source_id ] );
    dir = fxyz_unit( dir );

    particles_data.px[id] = pos.x;     // Position in mm
    particles_data.py[id] = pos.y;     //
    particles_data.pz[id] = pos.z;     //

    particles_data.dx[id] = dir.x;     // Direction (unit vector)
    particles_data.dy[id] = dir.y;     //
    particles_data.dz[id] = dir.z;     //

    particles_data.tof[id] = 0.0f;                             // Time of flight
    particles_data.endsimu[id] = PARTICLE_ALIVE;               // Status of the particle

    particles_data.level[id] = PRIMARY;                        // It is a primary particle
    particles_data.pname[id] = phasespace.ptype[ phsp_id ];    // a photon or an electron

    particles_data.geometry_id[id] = 0;                        // Some internal variables
    particles_data.next_discrete_process[id] = NO_PROCESS;     //
    particles_data.next_interaction_distance[id] = 0.0;        //

//    printf(" ID %i E %e pos %e %e %e ptype %i\n", id, particles_data.E[id],
//           particles_data.px[id], particles_data.py[id], particles_data.pz[id], particles_data.pname[id]);
}

__global__ void PHSPSRC::phsp_point_source ( ParticlesData particles_data,
                                             IaeaType phasespace, PhSpTransform transform )
{
    // Get thread id
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles_data.size ) return;

    // Get a new particle
    phsp_source( particles_data, phasespace, transform, id );
}

//////// Class //////////////////////////////////////////////////////////

// Constructor
PhaseSpaceSource::PhaseSpaceSource(): GGEMSSource()
{
    // Set the name of the source
    set_name( "phasespace_source" );

    // Init
    m_iaea = new IAEAIO();
    m_phasespace.tot_particles = 0;

    // Default transformation
    m_transform.nb_sources = 0;

}

// Destructor
PhaseSpaceSource::~PhaseSpaceSource() {
//    // ensure all memory is deallocated
//    cudaDeviceSynchronize();
//    cudaFree(m_spectrum->energies);
//    cudaFree(m_spectrum->cdf);
//    delete m_spectrum;

    // TODO delete m_iaea
}

//========== Setting ===============================================

// Setting position of the source
void PhaseSpaceSource::set_translation( f32 tx, f32 ty, f32 tz )
{
    if( m_transform.nb_sources == 0)
    {
        // Allocation of one transformation
        m_transform_allocation( 1 );
    }

    m_transform.tx[ 0 ] = tx;
    m_transform.ty[ 0 ] = ty;
    m_transform.tz[ 0 ] = tz;
}

// Setting rotation of the source
void PhaseSpaceSource::set_rotation( f32 aroundx, f32 aroundy, f32 aroundz )
{
    if( m_transform.nb_sources == 0)
    {
        // Allocation of one transformation
        m_transform_allocation( 1 );
    }

    m_transform.rx[ 0 ] = aroundx;  // right hand rule - JB
    m_transform.ry[ 0 ] = aroundy;
    m_transform.rz[ 0 ] = aroundz;
}

/*
// Setting scaling of the source
void PhaseSpaceSource::set_scaling( f32 sx, f32 sy, f32 sz )
{
    m_transform.sx = sx;
    m_transform.sy = sy;
    m_transform.sz = sz;
}
*/

//========= Private ============================================

// Check if everything is ok to initialize this source
bool PhaseSpaceSource::m_check_mandatory()
{
    if ( m_phasespace.tot_particles == 0 ) return false;
    else return true;
}

// Transform data allocation
void PhaseSpaceSource::m_transform_allocation( ui32 nb_sources )
{
    // Allocation
    HANDLE_ERROR( cudaMallocManaged( &(m_transform.tx), nb_sources*sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_transform.ty), nb_sources*sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_transform.tz), nb_sources*sizeof( f32 ) ) );

    HANDLE_ERROR( cudaMallocManaged( &(m_transform.rx), nb_sources*sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_transform.ry), nb_sources*sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_transform.rz), nb_sources*sizeof( f32 ) ) );

    HANDLE_ERROR( cudaMallocManaged( &(m_transform.sx), nb_sources*sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_transform.sy), nb_sources*sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_transform.sz), nb_sources*sizeof( f32 ) ) );

    HANDLE_ERROR( cudaMallocManaged( &(m_transform.cdf), nb_sources*sizeof( f32 ) ) );

    m_transform.nb_sources = nb_sources;
}

//========= Main function ============================================

// Load phasespace file
void PhaseSpaceSource::load_phasespace_file( std::string filename )
{
    std::string ext = filename.substr( filename.find_last_of( "." ) + 1 );
    if ( ext != "IAEAheader" )
    {
        GGcerr << "Phasespace source can only read data in IAEA format (.IAEAheader)!" << GGendl;
        exit_simulation();
    }

    m_iaea->read_header( filename );
    m_phasespace = m_iaea->read_data();
}

// Mandatory function, abstract from GGEMSSource. This function is called
// by GGEMS to initialize and load all necessary data into the graphic card
void PhaseSpaceSource::initialize ( GlobalSimulationParameters params )
{
    // Check if everything was set properly
    if ( !m_check_mandatory() )
    {
        GGcerr << "Phasespace source was not set properly!" << GGendl;
        exit_simulation();
    }

    // Store global parameters
    m_params = params;   

    // Check if the physics is set properly considering the phasespace file
    bool there_is_photon = m_params.data_h.physics_list[PHOTON_COMPTON] ||
                           m_params.data_h.physics_list[PHOTON_PHOTOELECTRIC] ||
                           m_params.data_h.physics_list[PHOTON_RAYLEIGH];

    bool there_is_electron = m_params.data_h.physics_list[ELECTRON_IONISATION] ||
                             m_params.data_h.physics_list[ELECTRON_BREMSSTRAHLUNG] ||
                             m_params.data_h.physics_list[ELECTRON_MSC];

    if ( !there_is_electron && m_phasespace.nb_electrons != 0 )
    {
        GGcerr << "Phasespace file contains " << m_phasespace.nb_electrons
               << " electrons and there are no electron physics effects enabled!"
               << GGendl;
        exit_simulation();
    }

    if ( !there_is_photon && m_phasespace.nb_photons != 0 )
    {
        GGcerr << "Phasespace file contains " << m_phasespace.nb_photons
               << " photons and there are no photon physics effects enabled!"
               << GGendl;
        exit_simulation();
    }

    if ( m_phasespace.nb_positrons != 0 )
    {
        GGcerr << "Phasespace file contains " << m_phasespace.nb_positrons
               << " positrons and there are no positron physics effects enabled!"
               << GGendl;
        exit_simulation();
    }

    // Check transformation
    if ( m_transform.nb_sources == 0 )
    {
        // Allocation of one transformation
        m_transform_allocation( 1 );

        m_transform.tx[ 0 ] = 0;
        m_transform.ty[ 0 ] = 0;
        m_transform.tz[ 0 ] = 0;

        m_transform.rx[ 0 ] = 0;
        m_transform.ry[ 0 ] = 0;
        m_transform.rz[ 0 ] = 0;

        m_transform.sx[ 0 ] = 1;
        m_transform.sy[ 0 ] = 1;
        m_transform.sz[ 0 ] = 1;

        m_transform.cdf[ 0 ] = 1;
        m_transform.nb_sources = 1;
    }

    // Some verbose if required
    if ( params.data_h.display_memory_usage )
    {
        ui32 mem = 29 * m_phasespace.tot_particles;
        GGcout_mem("PhaseSpace source", mem);
    }

}

// Mandatory function, abstract from GGEMSSource. This function is called
// by GGEMS to fill particle buffer of new fresh particles, which is the role
// of any source.
void PhaseSpaceSource::get_primaries_generator ( Particles particles )
{

    // If CPU running, do it on CPU
    if ( m_params.data_h.device_target == CPU_DEVICE )
    {
        ui32 id=0;

        // Loop over the particle buffer
        while( id < particles.size )
        {
            PHSPSRC::phsp_source( particles.data_h, m_phasespace, m_transform, id );
            ++id;
        }
    }
    // If GPU running, do it on GPU
    else if ( m_params.data_h.device_target == GPU_DEVICE )
    {
        // Defined threads and grid
        dim3 threads, grid;
        threads.x = m_params.data_h.gpu_block_size;
        grid.x = ( particles.size + m_params.data_h.gpu_block_size - 1 ) / m_params.data_h.gpu_block_size;

        PHSPSRC::phsp_point_source<<<grid, threads>>>( particles.data_d, m_phasespace, m_transform );
        cuda_error_check( "Error ", " Kernel_phasespace_source" );

    }

}

#endif

















