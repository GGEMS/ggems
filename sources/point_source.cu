// GGEMS Copyright (C) 2015

/*!
 * \file point_source.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 *
 *
 */

#ifndef POINT_SOURCE_CU
#define POINT_SOURCE_CU

#include "point_source.cuh"

///////// GPU code ////////////////////////////////////////////////////

// Internal function that create a new particle to the buffer at the slot id
__host__ __device__ void point_source ( ParticlesData particles_data,
                                        f32 px, f32 py, f32 pz, f32 energy, ui8 ptype, ui32 id)
{
    // First get an isotropic particle direction
    f32 phi = prng_uniform( particles_data, id );
    f32 theta = prng_uniform( particles_data, id );
    phi  *= gpu_twopi;
    theta = acosf ( 1.0f - 2.0f*theta );
    f32 dx = cosf( phi ) * sinf( theta );
    f32 dy = sinf( phi ) * sinf( theta );
    f32 dz = cosf( theta );

    // Then set the mandatory field to create a new particle
    particles_data.E[id] = energy;                             // Energy in MeV

    particles_data.px[id] = px;                                // Position in mm
    particles_data.py[id] = py;                                //
    particles_data.pz[id] = pz;                                //

    particles_data.dx[id] = dx;                                // Direction (unit vector)
    particles_data.dy[id] = dy;                                //
    particles_data.dz[id] = dz;                                //

    particles_data.tof[id] = 0.0f;                             // Time of flight
    particles_data.endsimu[id] = PARTICLE_ALIVE;               // Status of the particle

    particles_data.level[id] = PRIMARY;                        // It is a primary particle
    particles_data.pname[id] = ptype;                          // a photon or an electron

    particles_data.geometry_id[id] = 0;                        // Some internal variables
    particles_data.next_discrete_process[id] = NO_PROCESS;     //
    particles_data.next_interaction_distance[id] = 0.0;        //
}

// Kernel to create new particles. This kernel will only call the host/device function
// point source in order to get one new particle.
__global__ void kernel_point_source ( ParticlesData particles_data,
                                      f32 px, f32 py, f32 pz, f32 energy, ui8 ptype )
{
    // Get thread id
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles_data.size ) return;

    // Get a new particle
    point_source( particles_data, px, py, pz, energy, ptype, id );
}

//////// Class //////////////////////////////////////////////////////////

// Constructor
PointSource::PointSource()
    : GGEMSSource(),
      m_px( 0.0 ),
      m_py( 0.0 ),
      m_pz( 0.0 ),
      m_energy( 0.0 ),
      m_particle_type( PHOTON )
{
    // Set the name of the source
    set_name( "point_source" );
}

// Destructor
PointSource::~PointSource() {}

// Setting position of the source
void PointSource::set_position( f32 posx, f32 posy, f32 posz )
{
    m_px = posx;
    m_py = posy;
    m_pz = posz;
}

//========== Setting ===============================================

// Setting particle type (photon or electron)
void PointSource::set_particle_type( std::string pname )
{
    // Transform the name of the particle in small letter
    std::transform( pname.begin(), pname.end(), pname.begin(), ::tolower );

    if( pname == "photon" )
    {
        m_particle_type = PHOTON;
    }
    else if( pname == "electron" )
    {
        m_particle_type = ELECTRON;
    }
    else
    {
        GGcerr << "Particle '" << pname << "' not recognized!!!" << GGendl;
        exit_simulation();
    }
}

// Setting energy
void PointSource::set_energy( f32 energy )
{
    m_energy = energy;
}

//========= Main function ============================================

// Check if everything is ok to initialize this source
bool PointSource::m_check_mandatory()
{
    if ( m_energy == 0.0 ) return false;
    else return true;
}

// Mandatory function, abstract from GGEMSSource. This function is called
// by GGEMS to initialize and load all necessary data into the graphic card
void PointSource::initialize ( GlobalSimulationParameters params )
{
    // Check if everything was set properly
    if ( !m_check_mandatory() )
    {
        GGcerr << "Point source was not set properly!" << GGendl;
        exit_simulation();
    }

    // Store global parameters: params are provided by GGEMS and are used to
    // know different information about the simulation. For example if the targeted
    // device is a CPU or a GPU.
    m_params = params;

    // Handle GPU device if needed. Here nothing is load to the GPU (simple source). But
    // in case of the use of a spectrum data should be allocated and transfered here.
    if ( m_params.data_h.device_target == GPU_DEVICE )
    {
        // GPU mem allocation

        // GPU mem copy
    }

}

// Mandatory function, abstract from GGEMSSource. This function is called
// by GGEMS to fill particle buffer of new fresh particles, which is the role
// of any source.
void PointSource::get_primaries_generator ( Particles particles )
{

    // If CPU running, do it on CPU
    if ( m_params.data_h.device_target == CPU_DEVICE )
    {
        // Loop over the particle buffer
        ui32 id=0;
        while( id < particles.size )
        {
            // Call a point source that get a new particle at a time. In this case data from host (CPU)
            // is passed to the function (particles.data_h).
            point_source( particles.data_h, m_px, m_py, m_pz, m_energy, m_particle_type, id );
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

        // Call GPU kernel of a point source that get fill the complete particle buffer. In this case data
        // from device (GPU) is passed to the kernel (particles.data_d).
        kernel_point_source<<<grid, threads>>>( particles.data_d, m_px, m_py, m_pz, m_energy, m_particle_type );
        cuda_error_check( "Error ", " Kernel_point_source" );
    }

}

#endif

















