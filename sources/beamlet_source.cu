// GGEMS Copyright (C) 2015

/*!
 * \file beamlet_source.cu
 * \brief Beamlet source
 * \author Julien Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date Thursday November 19, 2015
*/

#ifndef BEAMLET_SOURCE_CU
#define BEAMLET_SOURCE_CU

#include "beamlet_source.cuh"

///////// GPU code ////////////////////////////////////////////////////

// Internal function that create a new particle to the buffer at the slot id
__host__ __device__ void beamlet_source ( ParticlesData particles, f32xyz pos, f32xyz foc_pos, f32xy size, f32xyz angle,
                                          f32 *spectrum_E, f32 *spectrum_CDF,
                                          ui32 nb_of_energy_bins, ui8 ptype, ui32 id)
{
    // 1. First chose a position within the beamlet
    f32xyz part_pos;
    part_pos.x = size.x*prng_uniform( particles, id ) - 0.5*size.x;
    part_pos.y = size.y*prng_uniform( particles, id ) - 0.5*size.y;
    part_pos.x = pos.x + part_pos.x;
    part_pos.y = pos.y + part_pos.y;

    // 3. Then compute direction
    f32xyz part_dir = fxyz_sub( foc_pos, part_pos );
    part_dir = fxyz_unit( part_dir );

    // 4. First rotation (Carousel)
    part_pos = fxyz_rotate_y_axis(part_pos, angle.y);  // angle
    part_dir = fxyz_rotate_y_axis(part_dir, angle.y);  // angle

    // 5. Then, beam rotation (z-axis = granty angle, x-axis = non-coplanar beam)
    part_pos = fxyz_rotate_z_axis(part_pos, angle.z);  // angle
    part_dir = fxyz_rotate_z_axis(part_dir, angle.z);  // angle

    part_pos = fxyz_rotate_x_axis(part_pos, angle.x);  // angle
    part_dir = fxyz_rotate_x_axis(part_dir, angle.x);  // angle

    part_dir = fxyz_unit( part_dir ); // to be sure

    // 6. Get energy
    if( nb_of_energy_bins == 1 ) // mono energy
    {
        particles.E[ id ] = spectrum_E[ 0 ];
    }
    else // poly
    {
        f32 rndm = prng_uniform( particles, id );
        ui32 pos = binary_search( rndm, spectrum_CDF, nb_of_energy_bins );
        if ( pos == ( nb_of_energy_bins - 1 ) )
        {
            particles.E[ id ] = spectrum_E[ pos ];
        }
        else
        {
            particles.E[ id ] = linear_interpolation ( spectrum_CDF[ pos ],     spectrum_E[ pos ],
                                                       spectrum_CDF[ pos + 1 ], spectrum_E[ pos + 1 ], rndm );
        }

    }

    // 7. Then set the mandatory field to create a new particle
    particles.px[id] = part_pos.x;                        // Position in mm
    particles.py[id] = part_pos.y;                        //
    particles.pz[id] = part_pos.z;                        //

    particles.dx[id] = part_dir.x;                        // Direction (unit vector)
    particles.dy[id] = part_dir.y;                        //
    particles.dz[id] = part_dir.z;                        //

    particles.tof[id] = 0.0f;                             // Time of flight
    particles.endsimu[id] = PARTICLE_ALIVE;               // Status of the particle

    particles.level[id] = PRIMARY;                        // It is a primary particle
    particles.pname[id] = ptype;                          // a photon or an electron

    particles.geometry_id[id] = 0;                        // Some internal variables
    particles.next_discrete_process[id] = NO_PROCESS;     //
    particles.next_interaction_distance[id] = 0.0;        //

}

// Kernel to create new particles. This kernel will only call the host/device function
// beamlet source in order to get one new particle.
__global__ void kernel_beamlet_source ( ParticlesData particles, f32xyz pos, f32xyz foc_pos, f32xy size, f32xyz angle,
                                        f32 *spectrum_E, f32 *spectrum_CDF,
                                        ui32 nb_of_energy_bins, ui8 particle_type )
{
    // Get thread id
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles.size ) return;

    // Get a new particle
    beamlet_source( particles, pos, foc_pos, size, angle, spectrum_E, spectrum_CDF, nb_of_energy_bins,
                    particle_type, id );
}

//////// Class //////////////////////////////////////////////////////////

// Constructor
BeamletSource::BeamletSource() : GGEMSSource()
{
    // Set the name of the source
    set_name( "point_source" );

    // Init vars
    m_pos = make_f32xyz( 0.0, 0.0, 0.0 );
    m_foc_pos = make_f32xyz( 0.0, 0.0, 0.0 );
    m_angle = make_f32xyz( 0.0, 0.0, 0.0 );
    m_beamlet_size = make_f32xy( 0.0, 0.0 );
    m_particle_type = PHOTON;
    m_spectrum_E = NULL;
    m_spectrum_CDF = NULL;
    m_nb_of_energy_bins = 0;
    m_energy = 0;
    m_spectrum_filename = "";
}

// Destructor
BeamletSource::~BeamletSource() {}

//========== Private ===============================================

void BeamletSource::m_load_spectrum()
{
    // Open the histogram file
    std::ifstream input( m_spectrum_filename.c_str(), std::ios::in );
    if( !input )
    {
        GGcerr << "Error to open the file'" << m_spectrum_filename << "'!" << GGendl;
        exit_simulation();
    }

    // Compute number of energy bins
    std::string line;
    while( std::getline( input, line ) ) ++m_nb_of_energy_bins;

    // Returning to beginning of the file to read it again
    input.clear();
    input.seekg( 0, std::ios::beg );

    // Allocating buffers to store data
    HANDLE_ERROR( cudaMallocManaged( &m_spectrum_E, m_nb_of_energy_bins * sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &m_spectrum_CDF, m_nb_of_energy_bins * sizeof( f32 ) ) );

    // Store data from file
    size_t idx = 0;
    f64 sum = 0.0;
    while( std::getline( input, line ) )
    {
        std::istringstream iss( line );
        iss >> m_spectrum_E[ idx ] >> m_spectrum_CDF[ idx ];
        sum += m_spectrum_CDF[ idx ];
        ++idx;
    }

    // Compute CDF and normalized in same time by security
    m_spectrum_CDF[ 0 ] /= sum;
    for( ui32 i = 1; i < m_nb_of_energy_bins; ++i )
    {
        m_spectrum_CDF[ i ] = m_spectrum_CDF[ i ] / sum
                              + m_spectrum_CDF[ i - 1 ];
    }

    // Watch dog
    m_spectrum_CDF[ m_nb_of_energy_bins - 1 ] = 1.0;

    // Close the file
    input.close();
}

//========== Setting ===============================================

// Setting position of the beamlet
void BeamletSource::set_position( f32 posx, f32 posy, f32 posz )
{
    m_pos = make_f32xyz( posx, posy, posz );
}

// Setting position of the focal beamlet
void BeamletSource::set_beamlet_focal_point( f32 posx, f32 posy, f32 posz )
{
    m_foc_pos = make_f32xyz( posx, posy, posz );
}

// Setting orientation of the beamlet
void BeamletSource::set_rotation( f32 agx, f32 agy, f32 agz )
{
    m_angle = make_f32xyz( agx, agy, agz );
}

// Setting energy
void BeamletSource::set_mono_energy( f32 energy )
{
    m_energy = energy;
}

// Setting spectrum
void BeamletSource::set_energy_spectrum( std::string filename )
{
    m_spectrum_filename = filename;
    // Watchdog (avoid to set the two option mono energy and spectrum)
    m_energy = 0;
}

// Setting particle type (photon or electron)
void BeamletSource::set_particle_type( std::string pname )
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

//========= Main function ============================================

// Mandatory function, abstract from GGEMSSource. This function is called
// by GGEMS to initialize and load all necessary data into the graphic card
void BeamletSource::initialize ( GlobalSimulationParameters params )
{
    // Check if everything was set properly
    if ( m_energy == 0 && m_spectrum_filename == "" )
    {
        GGcerr << "No energy or spectrum file specified!" << GGendl;
        exit_simulation();
    }
    if ( m_beamlet_size.x == 0 || m_beamlet_size.y == 0 )
    {
        GGcerr << "Size of the beamlet was not defined!" << GGendl;
        exit_simulation();
    }

    // If mono energy
    if ( m_energy != 0 )
    {
        HANDLE_ERROR( cudaMallocManaged( &m_spectrum_E, sizeof( f32 ) ) );
        HANDLE_ERROR( cudaMallocManaged( &m_spectrum_CDF, sizeof( f32 ) ) );
        m_spectrum_E[ 0 ] = m_energy;
        m_spectrum_CDF[ 0 ] = 1.0;
        m_nb_of_energy_bins = 1;
    }
    else // else load a spectrum
    {
        m_load_spectrum();
    }

    // Store global parameters: params are provided by GGEMS and are used to
    // know different information about the simulation. For example if the targeted
    // device is a CPU or a GPU.
    m_params = params;   

}

// Mandatory function, abstract from GGEMSSource. This function is called
// by GGEMS to fill particle buffer of new fresh particles, which is the role
// of any source.
void BeamletSource::get_primaries_generator ( Particles particles )
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
            beamlet_source( particles.data_h, m_pos, m_foc_pos, m_beamlet_size, m_angle,
                            m_spectrum_E, m_spectrum_CDF, m_nb_of_energy_bins,
                            m_particle_type, id );
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
        kernel_beamlet_source<<<grid, threads>>>( particles.data_h, m_pos, m_foc_pos, m_beamlet_size, m_angle,
                                                  m_spectrum_E, m_spectrum_CDF, m_nb_of_energy_bins,
                                                  m_particle_type );
        cuda_error_check( "Error ", " Kernel_beamlet_source" );
    }

}

#endif

















