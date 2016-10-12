// GGEMS Copyright (C) 2015

/*!
 * \file cone_beam_CT_source.cu
 * \brief Cone beam source for CT
 * \author Didier Benoit <didier.benoit13@gmail.com>
 * \version 0.1
 * \date Friday January 8, 2015
*/

#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <iomanip>
#include <cerrno>
#include <cstring>

#include "ggems_source.cuh"
#include "fun.cuh"
#include "prng.cuh"
#include "cone_beam_CT_source.cuh"


__host__ __device__ void cone_beam_ct_source( ParticlesData &particles, ui8 ptype,
                                              f32 *spectrum_E, f32 *spectrum_CDF, ui32 nb_of_energy_bins, f32 aperture,
                                              f32xyz foc, const f32matrix44 transform, ui32 id )
{            
    // This part is in double precision due to aliasing issue - JB 06/2016

    // Get deflection (local)
    f64 phi = prng_uniform( particles, id );
    f64 theta = prng_uniform( particles, id );
    f64 val_aper = 1.0f - cos( f64( aperture ) );
    phi  *= f64( gpu_twopi );
    theta = acos( f64( 1.0 ) - val_aper * theta );   // Aliasing issue, ( 1 - small number )

    // From here, simple precision can be used - JB 06/2016

    f32xyz rd = { cosf( phi ) * sinf( theta ),
                  sinf( phi ) * sinf( theta ),
                  cosf( theta ) };    

    // Get direction of the cone beam. The beam is targeted to the isocenter, then
    // the direction is directly related to the position of the source.
    f32xyz gbl_pos = fxyz_local_to_global_position( transform, make_f32xyz( 0.0, 0.0, 0.0 ) );
    f32xyz dir = fxyz_unit( fxyz_sub( make_f32xyz( 0.0, 0.0, 0.0 ), gbl_pos ) );  // 0x0x0 is the isocenter position

    // Apply deflection (global)
    dir = rotateUz( rd, dir );
    dir = fxyz_unit( dir );

    // Postition with focal (local)
    gbl_pos.x = foc.x * ( prng_uniform( particles, id ) - 0.5f );
    gbl_pos.y = foc.y * ( prng_uniform( particles, id ) - 0.5f );
    gbl_pos.z = foc.z * ( prng_uniform( particles, id ) - 0.5f );

    // Apply transformation (local to global frame)
    gbl_pos = fxyz_local_to_global_position( transform, gbl_pos );

    // Get energy
    if( nb_of_energy_bins == 1 ) // mono energy
    {
        particles.E[ id ] = spectrum_E[ 0 ];
    }
    else // poly
    {
        f32 rndm = prng_uniform( particles, id );
        ui32 pos = binary_search_left( rndm, spectrum_CDF, nb_of_energy_bins );
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

    // Then set the mandatory field to create a new particle
    particles.px[id] = gbl_pos.x;                        // Position in mm
    particles.py[id] = gbl_pos.y;                        //
    particles.pz[id] = gbl_pos.z;                        //

    particles.dx[id] = dir.x;                        // Direction (unit vector)
    particles.dy[id] = dir.y;                        //
    particles.dz[id] = dir.z;                        //

    particles.tof[id] = 0.0f;                             // Time of flight
    particles.endsimu[id] = PARTICLE_ALIVE;               // Status of the particle

    particles.level[id] = PRIMARY;                        // It is a primary particle
    particles.pname[id] = ptype;                          // a photon or an electron

    particles.geometry_id[id] = 0;                        // Some internal variables
    particles.next_discrete_process[id] = NO_PROCESS;     //
    particles.next_interaction_distance[id] = 0.0;        //
    particles.scatter_order[ id ] = 0;                    //

}

__global__ void kernel_cone_beam_ct_source( ParticlesData particles, ui8 ptype,
                                            f32 *spectrum_E, f32 *spectrum_CDF, ui32 nb_of_energy_bins, f32 aperture,
                                            f32xyz foc, const f32matrix44 transform )
{
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;;
    if( id >= particles.size ) return;

    cone_beam_ct_source( particles, ptype, spectrum_E, spectrum_CDF, nb_of_energy_bins, aperture,
                         foc, transform, id );
}

ConeBeamCTSource::ConeBeamCTSource()
    : GGEMSSource(),     
      m_aperture( 360.0 ),
      m_particle_type( PHOTON ),     
      m_spectrum_E( nullptr ),
      m_spectrum_CDF( nullptr ),
      m_energy( 0.0 ),
      m_nb_of_energy_bins( 0 )
{
    // Set the name of the source
    set_name( "ConeBeamCTSource" );

    m_pos = make_f32xyz( 0.0, 0.0, 0.0 );
    m_angles = make_f32xyz( 0.0, 0.0, 0.0 );
    m_foc = make_f32xyz( 0.0, 0.0, 0.0 );
}

ConeBeamCTSource::~ConeBeamCTSource()
{

    if( m_spectrum_E )
    {
        cudaFree( m_spectrum_E );
    }

    if( m_spectrum_CDF )
    {
        cudaFree( m_spectrum_CDF );
    }
}

void ConeBeamCTSource::set_position( f32 px, f32 py, f32 pz )
{
    m_pos = make_f32xyz( px, py, pz );
}

void ConeBeamCTSource::set_focal_size(f32 xfoc, f32 yfoc, f32 zfoc)
{
    m_foc = make_f32xyz( xfoc, yfoc, zfoc );
}

void ConeBeamCTSource::set_beam_aperture( f32 aperture )
{
    m_aperture = aperture;
}

void ConeBeamCTSource::set_rotation( f32 rx, f32 ry, f32 rz )
{
    m_angles = make_f32xyz( rx, ry, rz );
}

// Setting the axis transformation matrix
void ConeBeamCTSource::set_local_axis( f32 m00, f32 m01, f32 m02,
                                       f32 m10, f32 m11, f32 m12,
                                       f32 m20, f32 m21, f32 m22 )
{
    m_local_axis.m00 = m00;
    m_local_axis.m01 = m01;
    m_local_axis.m02 = m02;
    m_local_axis.m10 = m10;
    m_local_axis.m11 = m11;
    m_local_axis.m12 = m12;
    m_local_axis.m20 = m20;
    m_local_axis.m21 = m21;
    m_local_axis.m22 = m22;
}


void ConeBeamCTSource::set_particle_type( std::string pname )
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
        std::ostringstream oss( std::ostringstream::out );
        oss << "Particle '" << pname << "' not recognized!!!";
        throw std::runtime_error( oss.str() );
    }
}

void ConeBeamCTSource::set_mono_energy( f32 energy )
{
    m_energy = energy;
}

void ConeBeamCTSource::set_energy_spectrum( std::string filename )
{
    m_spectrum_filename = filename;
    // Watchdog (avoid to set the two option mono energy and spectrum)
    m_energy = 0;
}

void ConeBeamCTSource::m_load_spectrum()
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

f32xyz ConeBeamCTSource::get_position()
{
    // Compute position
    return m_pos;
}

f32xyz ConeBeamCTSource::get_orbiting_angles()
{
    return m_angles;
}

f32 ConeBeamCTSource::get_aperture()
{
    return m_aperture;
}

f32matrix44 ConeBeamCTSource::get_transformation_matrix()
{
    return m_transform;
}

std::ostream& operator<<( std::ostream& os, ConeBeamCTSource const& cbct )
{
    os << std::setfill( '#' ) << std::setw( 80 ) << "" << GGendl;
    os << "--> Cone-Beam CT source infos:" << GGendl;
    os << "    --------------------------" << GGendl;
    os << GGendl;
    //os << "+ Source name:   " << cbct.get_name() << GGendl;  // Compilation error - JB
    os << "+ Particle type: "
       << ( cbct.m_particle_type == ELECTRON ? "electron" :
                                               ( cbct.m_particle_type == PHOTON ? "photon" : "positron" ) )
       << GGendl;
    os << "+ Aperture:      " << cbct.m_aperture << " [deg]" << GGendl;

    os << GGendl;
    os << "                 " << std::setfill( ' ' ) << std::setw( 12 )
       << "X" << std::setw( 20 )
       << "Y" << std::setw( 20 )
       << "Z" << GGendl;
    os << "+ Position:      " << std::fixed << std::setprecision( 3 )
       << std::setfill( ' ' )
       << std::setw( 15 ) << cbct.m_pos.x/mm << " [mm]"
       << std::setw( 15 ) << cbct.m_pos.y/mm << " [mm]"
       << std::setw( 15 ) << cbct.m_pos.z/mm << " [mm]" << GGendl;
    os << GGendl;

    os << "+ Energy:        ";

    if( cbct.m_nb_of_energy_bins == 1 )
    {
        os << cbct.m_spectrum_E[ 0 ]/keV << " [keV] (monoenergy)" << GGendl;
    }
    else if( cbct.m_nb_of_energy_bins == 0 )
    {
        os << GGendl;
    }
    else
    {
        os << "polychromatic spectrum" << GGendl;
        os << GGendl;
        os << "    " << "energy [keV]" << std::setfill( ' ' ) << std::setw( 13 )
           << "CDF" << GGendl;
        for( ui32 i = 0; i < cbct.m_nb_of_energy_bins; ++i )
        {
            os << std::fixed << std::setprecision( 3 ) << std::setfill( ' ' )
               << std::setw( 12 ) << cbct.m_spectrum_E[ i ]/keV
               << std::setw( 18 )
               << cbct.m_spectrum_CDF[ i ] << GGendl;
        }
    }

    os << std::setfill( '#' ) << std::setw( 80 ) << "";
    return os;
}

void ConeBeamCTSource::initialize( GlobalSimulationParameters params )
{
    // Check if everything was set properly
    if ( m_energy == 0 && m_spectrum_filename == "" )
    {
        GGcerr << "No energy or spectrum file specified!" << GGendl;
        exit_simulation();
    }

    // Store global parameters
    m_params = params;

    // Compute the transformation matrix of the source that map local frame to global frame
    TransformCalculator *trans = new TransformCalculator;
    trans->set_translation( m_pos.x, m_pos.y, m_pos.z );
    trans->set_rotation( m_angles );
    trans->set_axis_transformation( m_local_axis );
    m_transform = trans->get_transformation_matrix();
    delete trans;

    // Read and allocate data

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

}

void ConeBeamCTSource::get_primaries_generator( Particles particles )
{
    if( m_params.data_h.device_target == CPU_DEVICE )
    {
        ui32 id = 0;
        while( id < particles.size )
        {
            cone_beam_ct_source( particles.data_h, m_particle_type, m_spectrum_E, m_spectrum_CDF,
                                 m_nb_of_energy_bins, m_aperture, m_foc, m_transform, id );
            ++id;
        }
    }
    else if( m_params.data_h.device_target == GPU_DEVICE )
    {
        dim3 threads, grid;
        threads.x = m_params.data_h.gpu_block_size;
        grid.x = ( particles.size + m_params.data_h.gpu_block_size - 1 )
                / m_params.data_h.gpu_block_size;

        kernel_cone_beam_ct_source<<<grid, threads>>>( particles.data_d, m_particle_type, m_spectrum_E, m_spectrum_CDF,
                                                       m_nb_of_energy_bins, m_aperture, m_foc, m_transform );
        cuda_error_check( "Error ", " kernel_cone_beam_ct_source" );
        cudaDeviceSynchronize();
    }
}



