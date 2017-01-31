// GGEMS Copyright (C) 2017

/*!
 * \file ggems.cuh
 * \brief Main header of GGEMS lib
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.2
 * \date 13 novembre 2015
 *
 * Header of the main GGEMS lib
 *
 * v0.2: JB - Change all structs and remove CPU exec
 */

#ifndef GGEMS_CU
#define GGEMS_CU

#include <fcntl.h>

#ifdef _WIN32
#include <windows.h>
#include <wincrypt.h>
#else
#include <unistd.h>
#endif
#include <cerrno>
#include <stdexcept>
#include "ggems.cuh"

GGEMS::GGEMS()
{
    if(std::string(getenv("GGEMSHOME")) == "")
    {
        print_error("GGEMSHOME not found... Please source /path/to/bin/ggems.sh");
        exit_simulation();
    }

    // Allocate struct
    h_parameters = (GlobalSimulationParametersData*)malloc( sizeof(GlobalSimulationParametersData) );
    d_parameters = nullptr;

    // Init physics list and secondaries list
    h_parameters->physics_list = ( ui8* ) malloc ( NB_PROCESSES*sizeof ( ui8 ) );
    h_parameters->secondaries_list = ( ui8* ) malloc ( NB_PARTICLES*sizeof ( ui8 ) );

    ui32 i = 0;
    while ( i < NB_PROCESSES )
    {
        h_parameters->physics_list[i] = DISABLED;
        ++i;
    }
    i = 0;
    while ( i < NB_PARTICLES )
    {
        h_parameters->secondaries_list[i] = DISABLED;
        ++i;
    }

    // Parameters
    h_parameters->nb_of_particles = 0;
    h_parameters->size_of_particles_batch = 1000000;
    h_parameters->nb_of_batches = 0;
    h_parameters->time = 0;
    h_parameters->seed = 0;
    h_parameters->cs_table_nbins = 220;
    h_parameters->cs_table_min_E = 990*eV;
    h_parameters->cs_table_max_E = 250*MeV;
    h_parameters->photon_cut = 1 *um;
    h_parameters->electron_cut = 1 *um;
    h_parameters->nb_of_secondaries = 0;
    h_parameters->geom_tolerance = 100.0 *nm;

    // Init by default others parameters   
    h_parameters->gpu_id = 0;
    h_parameters->gpu_block_size = 192;

    // Others parameters
    h_parameters->display_run_time = ENABLED;
    h_parameters->display_memory_usage = DISABLED;
    h_parameters->display_energy_cuts = DISABLED;
    h_parameters->verbose = ENABLED;

    // To know if initialisation was performed
    m_flag_init = false;

#ifdef _WIN32
    h_parameters->display_in_color = DISABLED;
#else
    h_parameters->display_in_color = ENABLED;
#endif

    // Element of the simulation
    m_source = nullptr;
    //m_phantom = nullptr;
    m_phantoms.clear();
//    m_detector = nullptr;

}

GGEMS::~GGEMS()
{       
    // Reset device
    reset_gpu_device();
}

////// :: Setting ::

/// Params

// Set the GPU id
void GGEMS::set_GPU_ID ( ui32 valid )
{
    h_parameters->gpu_id = valid;
}

// Set the GPU block size
void GGEMS::set_GPU_block_size ( ui32 val )
{
    h_parameters->gpu_block_size = val;
}

// Add a process to the physics list
void GGEMS::set_process ( std::string process_name )
{
    // Transform the name of the process in small letter
    std::transform( process_name.begin(), process_name.end(),
      process_name.begin(), ::tolower );

    if ( process_name == "compton" )
    {
        h_parameters->physics_list[PHOTON_COMPTON] = ENABLED;

    }
    else if ( process_name == "photoelectric" )
    {
        h_parameters->physics_list[PHOTON_PHOTOELECTRIC] = ENABLED;

    }
    else if ( process_name == "rayleigh" )
    {
        h_parameters->physics_list[PHOTON_RAYLEIGH] = ENABLED;

    }
    else if ( process_name == "eionisation" )
    {
        h_parameters->physics_list[ELECTRON_IONISATION] = ENABLED;

    }
    else if ( process_name == "ebremsstrahlung" )
    {
        h_parameters->physics_list[ELECTRON_BREMSSTRAHLUNG] = ENABLED;

    }
    else if ( process_name == "emultiplescattering" )
    {
        h_parameters->physics_list[ELECTRON_MSC] = ENABLED;

    }
    else
    {
        print_warning ( "This process is unknown!!\n" );
        printf ( "     -> %s\n", process_name.c_str() );
        exit_simulation();
    }
}

// Add cut on particle tracking
void GGEMS::set_particle_cut ( std::string pname, f32 E )
{
    // Transform the name of the particle in small letter
    std::transform( pname.begin(), pname.end(), pname.begin(), ::tolower );

    if ( pname == "photon" ) h_parameters->photon_cut = E;
    else if ( pname == "electron" )
    {
        h_parameters->electron_cut = E;
    }
}

// Enable the simulation of a particular secondary particle
void GGEMS::set_secondary ( std::string pname )
{
    // Transform the name of the particle in small letter
    std::transform( pname.begin(), pname.end(), pname.begin(), ::tolower );

    if ( pname == "photon" )
    {
        //h_parameters->secondaries_list[PHOTON] = ENABLED;
        GGwarn << "Photon particle as secondary (ex Bremsstrhalung) is not available yet!" << GGendl;
        h_parameters->secondaries_list[PHOTON] = DISABLED;
    }
    else if ( pname == "electron" )
    {
        h_parameters->secondaries_list[ELECTRON] = ENABLED;
    }
    else
    {
        print_warning ( "Secondary particle type is unknow!!" );
        printf ( "     -> %s\n", pname.c_str() );
        exit_simulation();
    }
}

// Set the number of particles required for the simulation
void GGEMS::set_number_of_particles ( ui64 nb )
{
    h_parameters->nb_of_particles = nb;
}

// Set the geometry tolerance
void GGEMS::set_geometry_tolerance( f32 tolerance )
{
    tolerance = min ( 1.0 *mm, tolerance );
    tolerance = max ( 1.0 *nm, tolerance );

    h_parameters->geom_tolerance = tolerance;
}

// Set the size of particles batch
void GGEMS::set_size_of_particles_batch ( ui64 nb )
{
    h_parameters->size_of_particles_batch = nb;
}

// Set parameters to generate cross sections table
void GGEMS::set_CS_table_nbins ( ui32 valbin )
{
    h_parameters->cs_table_nbins = valbin;
}

void GGEMS::set_CS_table_E_min ( f32 valE )
{
    h_parameters->cs_table_min_E = valE;
}

void GGEMS::set_CS_table_E_max ( f32 valE )
{
    h_parameters->cs_table_max_E = valE;
}

void GGEMS::set_electron_cut ( f32 valE )
{
    h_parameters->electron_cut = valE;
}

void GGEMS::set_photon_cut ( f32 valE )
{
    h_parameters->photon_cut = valE;
}

// Set the seed number
void GGEMS::set_seed ( ui32 vseed )
{
  if( vseed == 0 ) // Compute a seed
  {
    #ifdef _WIN32
    HCRYPTPROV seedWin32;
    if( CryptAcquireContext(
      &seedWin32,
      NULL,
      NULL,
      PROV_RSA_FULL,
      CRYPT_VERIFYCONTEXT ) == FALSE )
    {
      std::ostringstream oss( std::ostringstream::out );
      char buffer_error[ 256 ];
      oss << "Error finding a seed: " <<
        strerror_s( buffer_error, 256, errno ) << std::endl;
      std::string error_msg = oss.str();
      throw std::runtime_error( error_msg );
    }
    vseed = static_cast<ui32>( seedWin32 );
    #else
    // Open a system random file
    int fd = ::open( "/dev/urandom", O_RDONLY | O_NONBLOCK );
    if( fd < 0 )
    {
      std::ostringstream oss( std::ostringstream::out );
      oss << "Error opening the file '/dev/urandom': " << strerror( errno )
        << std::endl;
      std::string error_msg = oss.str();
      throw std::runtime_error( error_msg );
    }

    // Buffer storing 4 characters
    char seedArray[ sizeof( ui32 ) ];
    ::read( fd, (void*)seedArray, sizeof( ui32 ) );
    ::close( fd );
    ui32 *seedUInt32 = reinterpret_cast<ui32*>( seedArray );
    vseed = *seedUInt32;
    #endif
  }

  h_parameters->seed = vseed;
}

/// Sources
void GGEMS::set_source ( GGEMSSource* aSource )
{
    m_source = aSource;
}

/// Phantoms
void GGEMS::set_phantom ( GGEMSPhantom* aPhantom )
{
    //m_phantom = aPhantom;
    m_phantoms.push_back( aPhantom );
}

///// Detector
void GGEMS::set_detector( GGEMSDetector* aDetector )
{
  m_detector = aDetector;
}

/// Utils

// Display run time
void GGEMS::set_display_run_time( bool flag )
{
    if ( flag )
    {
        h_parameters->display_run_time = ENABLED;
    }
    else
    {
        h_parameters->display_run_time = DISABLED;
    }
}

// Display memory usage
void GGEMS::set_display_memory_usage( bool flag )
{
    if ( flag )
    {
        h_parameters->display_memory_usage = ENABLED;
    }
    else
    {
        h_parameters->display_memory_usage = DISABLED;
    }
}

// Display energy cut
void GGEMS::set_display_energy_cuts( bool flag )
{
    if ( flag )
    {
        h_parameters->display_energy_cuts = ENABLED;
    }
    else
    {
        h_parameters->display_energy_cuts = DISABLED;
    }
}

// Display in color
void GGEMS::set_display_in_color( bool flag )
{
    if ( flag )
    {
        #ifdef _WIN32
            GGcerr << "Display in color is not supported by Windows terminal: option set to FALSE" << GGendl;
            h_parameters->display_in_color = DISABLED;
        #else
            h_parameters->display_in_color = ENABLED;
        #endif
    }
    else
    {
        h_parameters->display_in_color = DISABLED;
    }
}

// Main verbose
void GGEMS::set_verbose( bool flag )
{
    if ( flag )
    {
        h_parameters->verbose = ENABLED;
    }
    else
    {
        h_parameters->verbose = DISABLED;
    }
}

void GGEMS::set_secondaries_level ( ui32 level )
{
    h_parameters->nb_of_secondaries = level;
}
/*
void GGEMS::print_stack(ui32 n = 0)
{
    if ( m_particles_manager.particles.size > n ) n = m_particles_manager.particles.size;
    if ( n == 0 ) n = m_particles_manager.particles.size;

    if ( h_parameters->device_target == GPU_DEVICE )
    {
        m_particles_manager.copy_gpu2cpu( m_particles_manager.particles );
    }

    m_particles_manager.print_stack( m_particles_manager.particles, n );
}
*/
////// :: Private functions ::

// Check mandatory parameters
bool GGEMS::m_check_mandatory()
{
    bool flag_error = false;

    if ( m_source == NULL )
    {
        print_error ( "No source defined." );
        flag_error = true;
    }

    if ( h_parameters->nb_of_particles == 0 )
    {
        print_error ( "Nb_of_particles = 0." );
        flag_error = true;
    }

    if ( h_parameters->size_of_particles_batch == 0 )
    {
        print_error ( "Size_of_particles_batch = 0." );
        flag_error = true;
    }

    if ( h_parameters->seed == 0 )
    {
        print_error ( "Seed value set to 0." );
        flag_error = true;
    }

    if ( flag_error ) exit_simulation();
    
    return flag_error;
}

// Copy the global simulation parameters to the GPU
void GGEMS::m_copy_parameters_cpu2gpu()
{
    // First, struct allocation
    HANDLE_ERROR( cudaMalloc( (void**) &d_parameters, sizeof( GlobalSimulationParametersData ) ) );

    // Tmp device pointer
    ui8 *d_physics_list, *d_secondaries_list;
    HANDLE_ERROR( cudaMalloc( (void**) &d_physics_list, sizeof(ui8)*NB_PROCESSES ) );
    HANDLE_ERROR( cudaMalloc( (void**) &d_secondaries_list, sizeof(ui8)*NB_PARTICLES ) );

    // Copy data
    HANDLE_ERROR( cudaMemcpy( d_physics_list, h_parameters->physics_list,
                              sizeof(ui8)*NB_PROCESSES,
                              cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( d_secondaries_list, h_parameters->secondaries_list,
                                sizeof(ui8)*NB_PARTICLES, cudaMemcpyHostToDevice ) );

    // Bind data to the struct
    HANDLE_ERROR( cudaMemcpy( &(d_parameters->physics_list), &d_physics_list,
                              sizeof(d_parameters->physics_list), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_parameters->secondaries_list), &d_secondaries_list,
                              sizeof(d_parameters->secondaries_list), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(d_parameters->photon_cut), &(h_parameters->photon_cut),
                              sizeof(d_parameters->photon_cut), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_parameters->electron_cut), &(h_parameters->electron_cut),
                              sizeof(d_parameters->electron_cut), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(d_parameters->nb_of_secondaries), &(h_parameters->nb_of_secondaries),
                              sizeof(d_parameters->nb_of_secondaries), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(d_parameters->nb_of_particles), &(h_parameters->nb_of_particles),
                              sizeof(d_parameters->nb_of_particles), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_parameters->size_of_particles_batch), &(h_parameters->size_of_particles_batch),
                              sizeof(d_parameters->size_of_particles_batch), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_parameters->nb_of_batches), &(h_parameters->nb_of_batches),
                              sizeof(d_parameters->nb_of_batches), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(d_parameters->gpu_id), &(h_parameters->gpu_id),
                              sizeof(d_parameters->gpu_id), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_parameters->gpu_block_size), &(h_parameters->gpu_block_size),
                              sizeof(d_parameters->gpu_block_size), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_parameters->gpu_grid_size), &(h_parameters->gpu_grid_size),
                              sizeof(d_parameters->gpu_grid_size), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_parameters->time), &(h_parameters->time),
                              sizeof(d_parameters->time), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_parameters->seed), &(h_parameters->seed),
                              sizeof(d_parameters->seed), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(d_parameters->display_run_time), &(h_parameters->display_run_time),
                              sizeof(d_parameters->display_run_time), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_parameters->display_memory_usage), &(h_parameters->display_memory_usage),
                              sizeof(d_parameters->display_memory_usage), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_parameters->display_in_color), &(h_parameters->display_in_color),
                              sizeof(d_parameters->display_in_color), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_parameters->display_energy_cuts), &(h_parameters->display_energy_cuts),
                              sizeof(d_parameters->display_energy_cuts), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_parameters->verbose), &(h_parameters->verbose),
                              sizeof(d_parameters->verbose), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(d_parameters->geom_tolerance), &(h_parameters->geom_tolerance),
                              sizeof(d_parameters->geom_tolerance), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(d_parameters->cs_table_nbins), &(h_parameters->cs_table_nbins),
                              sizeof(d_parameters->cs_table_nbins), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_parameters->cs_table_min_E), &(h_parameters->cs_table_min_E),
                              sizeof(d_parameters->cs_table_min_E), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_parameters->cs_table_max_E), &(h_parameters->cs_table_max_E),
                              sizeof(d_parameters->cs_table_max_E), cudaMemcpyHostToDevice ) );
}


////// :: Main functions ::

// Init simualtion
void GGEMS::init_simulation()
{
    // Verbose
    if ( !h_parameters->verbose )
    {
        h_parameters->display_energy_cuts = DISABLED;
        h_parameters->display_in_color = DISABLED;
        h_parameters->display_memory_usage = DISABLED;
        h_parameters->display_run_time = DISABLED;
    }

    // Banner
    if ( h_parameters->verbose )
    {
        print_banner("V2.1", h_parameters );
    }

    // Check
    m_check_mandatory();

    // Run time
    f64 t_start = 0;
    if ( h_parameters->display_run_time )
    {
        t_start = get_time();
    }

    // CPU PRNG
    srand( h_parameters->seed );

    h_parameters->nb_of_batches = ui32 ( ( f32 ) h_parameters->nb_of_particles / ( f32 ) h_parameters->size_of_particles_batch );

    if( h_parameters->nb_of_particles % h_parameters->size_of_particles_batch )
    {
        h_parameters->nb_of_batches++;
    }

    // Print some information
    if ( h_parameters->verbose )
    {
        GGnewline();
        GGcout_timestamp();
        GGcout_version();
        GGcout_def();
        GGcout_params( h_parameters );
    }


    // Set the gpu id
    set_gpu_device( h_parameters->gpu_id );

    // Reset device
    reset_gpu_device();

    // Copy params to the GPU
    m_copy_parameters_cpu2gpu();

    /// Init Sources /////////////////////////////////
    m_source->initialize( h_parameters );

    /// Init Phantoms ////////////////////////////////
    if ( m_phantoms.size() != 0 )
    {
        ui16 i = 0; while ( i < m_phantoms.size() )
        {
            m_phantoms[ i++ ]->initialize( h_parameters, d_parameters );
        }
    }

    /// Init Detectors /////////////////////////
    // The detector is not mandatory
    if ( m_detector ) m_detector->initialize( h_parameters );

    /// Init Particles Stack /////////////////////////
    m_particles_manager.initialize( h_parameters );

    /// Verbose information //////////////////////////

    // Display memory usage
    if (h_parameters->display_memory_usage) {
        // Particle stack
        ui64 n = m_particles_manager.h_particles->size;        
        ui64 mem = n * ( 12 * sizeof( f32 ) + 5 * sizeof( ui32 )  + 4 * sizeof( ui8 ) );

        GGcout_mem("Particle stacks", mem);

        GGnewline();
    }

    // Run time
    if ( h_parameters->display_run_time ) {
        GGcout_time ( "Initialization", get_time()-t_start );
        GGnewline();
    }

    // Succesfull init
    m_flag_init = true;

}

/*
void progress_bar(float progress, int etape, int nbatch )
{

//         while (progress < 1.0) {
    int barWidth = 70;

    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) 
    {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " % (Batch : " << etape << "/"<< nbatch<< ")\r";
    std::cout.flush();            
}
*/

void GGEMS::start_simulation()
{

    if ( !m_flag_init )
    {
        GGcerr << "GGEMS simulation not initialized!!" << GGendl;
        exit_simulation();
    }

    // Run time
    f64 t_start = 0;
    if ( h_parameters->display_run_time )
    {
        t_start = get_time();
    }

    // Main loop
    ui32 ibatch=0;
    while ( ibatch < h_parameters->nb_of_batches )
    {

        if ( h_parameters->verbose )
        {
            GGcout << "----> Launching batch " << ibatch+1 << "/" << h_parameters->nb_of_batches << GGendl;
            GGcout << "      + Generating " << h_parameters->size_of_particles_batch << " particles from "
                   << m_source->get_name() << GGendl;
        }
        m_source->get_primaries_generator( m_particles_manager.d_particles );

        // Nav between source and phantom
        if ( m_phantoms.size() != 0 )
        {
            ui16 i = 0; while ( i < m_phantoms.size() )
            {
                // Nav between source to phantom
                if ( h_parameters->verbose )
                {
                    GGcout << "      + Navigation to the phantom " << m_phantoms[ i ]->get_name()
                           << " (" << i << ")" << GGendl;
                }
                m_phantoms[ i ]->track_to_in( m_particles_manager.d_particles );

                // Nav within the phantom
                if ( h_parameters->verbose )
                {
                    GGcout << "      + Navigation within the phantom " << m_phantoms[ i ]->get_name()
                           << " (" << i << ")" << GGendl;
                }
                m_phantoms[ i ]->track_to_out( m_particles_manager.d_particles );

                ++i;
            }
        }

        // Nav between phantom and detector
        if( m_detector )
        {
            if ( h_parameters->verbose )
            {
                GGcout << "      + Navigation to the detector " << m_detector->get_name() << GGendl;
            }
            m_detector->track_to_in( m_particles_manager.d_particles );

            if ( h_parameters->verbose )
            {
                GGcout << "      + Navigation within the detector " << m_detector->get_name() << GGendl;
            }
            m_detector->track_to_out( m_particles_manager.d_particles );

            if ( h_parameters->verbose )
            {
                GGcout << "      + Digitizer from " << m_detector->get_name() << GGendl;
            }
            m_detector->digitizer( m_particles_manager.d_particles );
        }

        if ( h_parameters->verbose )
        {
            GGcout << "----> Batch finished" << GGendl << GGendl;
        }

        ++ibatch;
    }
    std::cout << std::endl;

    // Run time
    if ( h_parameters->display_run_time ) {
        // Sync all kernel to get the GPU run time
        cudaDeviceSynchronize();

        GGcout_time ( "Simulation run time", get_time()-t_start );
        GGnewline();
    }
}








#endif
