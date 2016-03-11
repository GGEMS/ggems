// GGEMS Copyright (C) 2015

/*!
 * \file geom_source.cu
 * \brief Geom source class
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 9 mars 2016
 *
 * Geom source class
 *
 */

#ifndef GEOM_SOURCE_CU
#define GEOM_SOURCE_CU

#include "geom_source.cuh"

///////// GPU code ////////////////////////////////////////////////////

// Get energy from spectrum
__host__ __device__ f32 GEOMSRC::get_energy( ParticlesData particles_data, f32 *energy, f32 *cdf, ui32 nb_bins, ui32 id )
{
    if( nb_bins == 1 )
    {
        return energy[ 0 ];
    }
    else
    {
        // Get the position in spectrum
        f32 rndm = prng_uniform( particles_data, id );
        ui32 pos = binary_search( rndm, cdf, nb_bins );

        if ( pos == ( nb_bins - 1 ) )
        {
            return energy[ pos ];
        }
        else
        {
            return linear_interpolation ( cdf[ pos ],     energy[ pos ],
                                          cdf[ pos + 1 ], energy[ pos + 1 ], rndm );
        }
    }
}

// Internal function that create a new particle to the buffer at the slot id
__host__ __device__ void GEOMSRC::point_source ( ParticlesData particles_data,
                                                 f32xyz pos, f32 *energy, f32 *cdf, ui32 nb_bins,
                                                 ui8 ptype, ui32 id )



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
    particles_data.E[id] = get_energy( particles_data, energy, cdf, nb_bins, id);  // Energy in MeV

    particles_data.px[id] = pos.x;                                // Position in mm
    particles_data.py[id] = pos.y;                                //
    particles_data.pz[id] = pos.z;                                //

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
__global__ void GEOMSRC::kernel_point_source ( ParticlesData particles_data,
                                               f32xyz pos, f32 *energy, f32 *cdf, ui32 nb_bins,
                                               ui8 ptype )
{
    // Get thread id
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles_data.size ) return;

    // Get a new particle
    point_source( particles_data, pos, energy, cdf, nb_bins, ptype, id );
}

//////// Class //////////////////////////////////////////////////////////

// Constructor
GeomSource::GeomSource(): GGEMSSource()
{
    // Set the name of the source
    set_name( "geom_source" );

    // Init
    m_shape = "point";
    m_shape_mode = "volume";
    m_pos = make_f32xyz( 0.0f, 0.0f, 0.0f );
    m_rot = make_f32xyz( 0.0f, 0.0f, 0.0f );
    m_length = make_f32xyz( 0.0f, 0.0f, 0.0f );
    m_radius = 0.0f;
    m_particle_type = PHOTON;
    m_spectrum = NULL;
    m_source = NULL;
}

// Destructor
GeomSource::~GeomSource() {
    // ensure all memory is deallocated
    cudaDeviceSynchronize();
    cudaFree(m_spectrum->energies);
    cudaFree(m_spectrum->cdf);
    delete m_spectrum;
}

//========== Setting ===============================================

// Setting shape
void GeomSource::set_shape( std::string shape_name )
{
    set_shape( shape_name, "volume" );
}

// Setting shape
void GeomSource::set_shape( std::string shape_name, std::string shape_mode )
{
    // Transform the name in small letter
    std::transform( shape_name.begin(), shape_name.end(), shape_name.begin(), ::tolower );

    // Shape
    if ( shape_name == "point" )
    {
        m_shape = "point";
    }
    else if ( shape_name == "cylinder" )
    {
        m_shape = "cylinder";
    }
    else if ( shape_name == "sphere" )
    {
        m_shape = "sphere";
    }
    else if ( shape_name == "capsule" )
    {
        m_shape = "capsule";
    }
    else
    {
        GGcerr << "GeomSource: shape '" << shape_name << "'' not recognized!" << GGendl;
        exit_simulation();
    }

    // And mode
    std::transform( shape_mode.begin(), shape_mode.end(), shape_mode.begin(), ::tolower );

    if ( shape_mode == "surface" )
    {
        m_shape_mode = "surface";
    }
    else if ( shape_mode == "volume" )
    {
        m_shape_mode = "volume";
    }
    else
    {
        GGcerr << "GeomSource: shape mode '" << shape_mode << "'' not recognized!" << GGendl;
        exit_simulation();
    }
}

// Setting position of the source
void GeomSource::set_position( f32 posx, f32 posy, f32 posz )
{
    m_pos = make_f32xyz( posx, posy, posz );
}

// Setting rotation of the source
void GeomSource::set_rotation( f32 aroundx, f32 aroundy, f32 aroundz )
{
    m_rot = make_f32xyz( aroundx, aroundy, aroundz );
}

// Setting length of the source
void GeomSource::set_length( f32 alongx, f32 alongy, f32 alongz )
{
    m_length = make_f32xyz( alongx, alongy, alongz );
}

// Setting radius
void GeomSource::set_radius( f32 radius )
{
    m_radius = radius;
}

// Setting particle type (photon or electron)
void GeomSource::set_particle_type( std::string pname )
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
void GeomSource::set_mono_energy( f32 energy )
{
    m_spectrum = new Spectrum;

    m_spectrum->nb_of_energy_bins = 1;
    HANDLE_ERROR( cudaMallocManaged( &(m_spectrum->energies), m_spectrum->nb_of_energy_bins * sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_spectrum->cdf), m_spectrum->nb_of_energy_bins * sizeof( f32 ) ) );

    m_spectrum->energies[ 0 ] = energy;
    m_spectrum->cdf[ 0 ] = 1.0f;
}

// Setting spectrum
void GeomSource::set_energy_spectrum( std::string filename )
{
    // Open the histogram file
    std::ifstream input( filename.c_str(), std::ios::in );
    if( !input )
    {
        std::ostringstream oss( std::ostringstream::out );
#ifdef _WIN32
        char buffer_error[ 256 ];
        oss << "Error opening file '" << filename << "': "
            << strerror_s( buffer_error, 256, errno );
#else
        oss << "Error opening file '" << filename << "': "
            << strerror( errno );
#endif
        std::string error_msg = oss.str();
        throw std::runtime_error( error_msg );
    }

    // Compute number of energy bins
    std::string line;
    ui32 nb_of_energy_bins;
    while( std::getline( input, line ) ) ++nb_of_energy_bins;

    // Returning to beginning of the file to read it again
    input.clear();
    input.seekg( 0, std::ios::beg );

    // Allocating buffers to store data
    m_spectrum = new Spectrum;
    m_spectrum->nb_of_energy_bins = nb_of_energy_bins;
    HANDLE_ERROR( cudaMallocManaged( &(m_spectrum->energies), m_spectrum->nb_of_energy_bins * sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_spectrum->cdf), m_spectrum->nb_of_energy_bins * sizeof( f32 ) ) );

    // Store data from file
    size_t idx = 0;
    f64 sum = 0.0;
    while( std::getline( input, line ) )
    {
        std::istringstream iss( line );
        iss >> m_spectrum->energies[ idx ] >> m_spectrum->cdf[ idx ];
        sum += m_spectrum->cdf[ idx ];
        ++idx;
    }

    // Compute CDF and normalized in same time by security
    m_spectrum->cdf[ 0 ] /= sum;
    for( ui32 i = 1; i < nb_of_energy_bins; ++i )
    {
        m_spectrum->cdf[ i ] = m_spectrum->cdf[ i ] / sum + m_spectrum->cdf[ i - 1 ];
    }

    // Watch dog
    m_spectrum->cdf[ nb_of_energy_bins - 1 ] = 1.0;

    // Close the file
    input.close();
}


//========= Main function ============================================

// Check if everything is ok to initialize this source
bool GeomSource::m_check_mandatory()
{
    if ( m_spectrum == NULL ) return false;
    else return true;
}

// Mandatory function, abstract from GGEMSSource. This function is called
// by GGEMS to initialize and load all necessary data into the graphic card
void GeomSource::initialize ( GlobalSimulationParameters params )
{
    // Check if everything was set properly
    if ( !m_check_mandatory() )
    {
        GGcerr << "Geom source was not set properly!" << GGendl;
        exit_simulation();
    }

    // Store global parameters
    m_params = params;   

}

// Mandatory function, abstract from GGEMSSource. This function is called
// by GGEMS to fill particle buffer of new fresh particles, which is the role
// of any source.
void GeomSource::get_primaries_generator ( Particles particles )
{

    // If CPU running, do it on CPU
    if ( m_params.data_h.device_target == CPU_DEVICE )
    {
        ui32 id=0;

        if ( m_shape == "point" )
        {
            // Loop over the particle buffer
            while( id < particles.size )
            {
                GEOMSRC::point_source( particles.data_h, m_pos, m_spectrum->energies, m_spectrum->cdf,
                                       m_spectrum->nb_of_energy_bins, m_particle_type, id );
                ++id;
            }
        }



    }
    // If GPU running, do it on GPU
    else if ( m_params.data_h.device_target == GPU_DEVICE )
    {
        // Defined threads and grid
        dim3 threads, grid;
        threads.x = m_params.data_h.gpu_block_size;
        grid.x = ( particles.size + m_params.data_h.gpu_block_size - 1 ) / m_params.data_h.gpu_block_size;


        if ( m_shape == "point" )
        {
            GEOMSRC::kernel_point_source<<<grid, threads>>>( particles.data_d, m_pos, m_spectrum->energies, m_spectrum->cdf,
                                                             m_spectrum->nb_of_energy_bins, m_particle_type );
            cuda_error_check( "Error ", " Kernel_geom_source (point)" );
        }
    }

}

#endif

















