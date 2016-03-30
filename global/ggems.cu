// GGEMS Copyright (C) 2015

/*!
 * \file ggems.cuh
 * \brief Main header of GGEMS lib
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 13 novembre 2015
 *
 * Header of the main GGEMS lib
 *
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

    // Init physics list and secondaries list
    m_parameters.data_h.physics_list = ( bool* ) malloc ( NB_PROCESSES*sizeof ( bool ) );
    m_parameters.data_h.secondaries_list = ( bool* ) malloc ( NB_PARTICLES*sizeof ( bool ) );

    ui32 i = 0;
    while ( i < NB_PROCESSES )
    {
        m_parameters.data_h.physics_list[i] = DISABLED;
        ++i;
    }
    i = 0;
    while ( i < NB_PARTICLES )
    {
        m_parameters.data_h.secondaries_list[i] = DISABLED;
        ++i;
    }

    // Parameters
    m_parameters.data_h.nb_of_particles = 0;
    m_parameters.data_h.size_of_particles_batch = 1000000;
    m_parameters.data_h.nb_of_batches = 0;
    m_parameters.data_h.time = 0;
    m_parameters.data_h.seed = 0;
    m_parameters.data_h.cs_table_nbins = 220;
    m_parameters.data_h.cs_table_min_E = 990*eV;
    m_parameters.data_h.cs_table_max_E = 250*MeV;
    m_parameters.data_h.photon_cut = 1 *um;
    m_parameters.data_h.electron_cut = 1 *um;
    m_parameters.data_h.nb_of_secondaries = 0;
    m_parameters.data_h.geom_tolerance = 100.0 *nm;

    // Init by default others parameters
    m_parameters.data_h.device_target = GPU_DEVICE;
    m_parameters.data_h.gpu_id = 0;
    m_parameters.data_h.gpu_block_size = 192;

    // Others parameters
    m_parameters.data_h.display_run_time = ENABLED;
    m_parameters.data_h.display_memory_usage = DISABLED;
    m_parameters.data_h.display_energy_cuts = DISABLED;

    // To know if initialisation was performed
    m_flag_init = false;

#ifdef _WIN32
    m_parameters.data_h.display_in_color = DISABLED;
#else
    m_parameters.data_h.display_in_color = ENABLED;
#endif

    // Element of the simulation
    m_source = nullptr;
    m_phantom = nullptr;
    m_detector = nullptr;

}

GGEMS::~GGEMS()
{
   
    if ( m_parameters.data_h.device_target == GPU_DEVICE )
    {
        // Reset device
        reset_gpu_device();
    }
   //delete m_parameters;
    //delete m_source;
}

////// :: Setting ::

/// Params

// Set the GGEMS license
void GGEMS::set_license(std::string license_path) {
    m_license.read_license ( license_path );
    m_license.check_license();
}

// Set the hardware used for the simulation CPU or GPU (CPU by default)
void GGEMS::set_hardware_target ( std::string value )
{

    // Transform the name of the process in small letter
    std::transform( value.begin(), value.end(),
      value.begin(), ::tolower );

    if ( value == "gpu" )
    {
        m_parameters.data_h.device_target = GPU_DEVICE;
    }
    else
    {
        m_parameters.data_h.device_target = CPU_DEVICE;
    }
}

// Set the GPU id
void GGEMS::set_GPU_ID ( ui32 valid )
{
    m_parameters.data_h.gpu_id = valid;
}

// Set the GPU block size
void GGEMS::set_GPU_block_size ( ui32 val )
{
    m_parameters.data_h.gpu_block_size = val;
}

// Add a process to the physics list
void GGEMS::set_process ( std::string process_name )
{
    // Transform the name of the process in small letter
    std::transform( process_name.begin(), process_name.end(),
      process_name.begin(), ::tolower );

    if ( process_name == "compton" )
    {
        m_parameters.data_h.physics_list[PHOTON_COMPTON] = ENABLED;

    }
    else if ( process_name == "photoelectric" )
    {
        m_parameters.data_h.physics_list[PHOTON_PHOTOELECTRIC] = ENABLED;

    }
    else if ( process_name == "rayleigh" )
    {
        m_parameters.data_h.physics_list[PHOTON_RAYLEIGH] = ENABLED;

    }
    else if ( process_name == "eionisation" )
    {
        m_parameters.data_h.physics_list[ELECTRON_IONISATION] = ENABLED;

    }
    else if ( process_name == "ebremsstrahlung" )
    {
        m_parameters.data_h.physics_list[ELECTRON_BREMSSTRAHLUNG] = ENABLED;

    }
    else if ( process_name == "emultiplescattering" )
    {
        m_parameters.data_h.physics_list[ELECTRON_MSC] = ENABLED;

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

    if ( pname == "photon" ) m_parameters.data_h.photon_cut = E;
    else if ( pname == "electron" )
    {
        m_parameters.data_h.electron_cut = E;
    }
}

// Enable the simulation of a particular secondary particle
void GGEMS::set_secondary ( std::string pname )
{
    // Transform the name of the particle in small letter
    std::transform( pname.begin(), pname.end(), pname.begin(), ::tolower );

    if ( pname == "photon" )
    {
        //m_parameters.data_h.secondaries_list[PHOTON] = ENABLED;
        GGwarn << "Photon particle as secondary (ex Bremsstrhalung) is not available yet!" << GGendl;
        m_parameters.data_h.secondaries_list[PHOTON] = DISABLED;
    }
    else if ( pname == "electron" )
    {
        m_parameters.data_h.secondaries_list[ELECTRON] = ENABLED;
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
    m_parameters.data_h.nb_of_particles = nb;
}

// Set the geometry tolerance
void GGEMS::set_geometry_tolerance( f32 tolerance )
{
    tolerance = min ( 1.0 *mm, tolerance );
    tolerance = max ( 1.0 *nm, tolerance );

    m_parameters.data_h.geom_tolerance = tolerance;
}

/*   TO BE REMOVED - JB
// Set the number of particles required for the simulation
void GGEMS::set_number_of_particles ( std::string str )
{
    int id = 0;
    int batch = 1;

    std::string unit;
    std::vector<std::string> tokens;
    std::stringstream stream;

    tokens=split_vector(str," ");
    if(tokens.size()!=2 )
    {
        GGcerr << "There must be only 2 values : The number of particles and the unit (thousand, million, billion)" << GGendl;
        exit_simulation();
    }

    stream << tokens[0];
    stream >> id;
    stream.clear(); //clear the sstream
    stream << tokens[1];
    stream >> unit;
    stream.clear(); //clear the sstream

    if(unit=="thousand")
    {
        id *= 1000;
    }
    else if(unit=="million")
    {       
        id *= 1000000;
    }
    else if(unit=="billion")
    {
        batch = id;
        id = 1000000000;
    }
    else
    {
        GGcerr << "Multiple " << unit.c_str() << " unknown (only thousand, million, billion accepted)" << GGendl;
        exit_simulation();
    }

    if(id<0)
    {
        printf("The number of particles must be a positive value \n");
        exit_simulation();
    }

    set_number_of_particles(id);
    set_size_of_particles_batch(batch);

}
*/

// Set the size of particles batch
void GGEMS::set_size_of_particles_batch ( ui64 nb )
{
    m_parameters.data_h.size_of_particles_batch = nb;
}

// Set parameters to generate cross sections table
void GGEMS::set_CS_table_nbins ( ui32 valbin )
{
    m_parameters.data_h.cs_table_nbins = valbin;
}

void GGEMS::set_CS_table_E_min ( f32 valE )
{
    m_parameters.data_h.cs_table_min_E = valE;
}

void GGEMS::set_CS_table_E_max ( f32 valE )
{
    m_parameters.data_h.cs_table_max_E = valE;
}

void GGEMS::set_electron_cut ( f32 valE )
{
    m_parameters.data_h.electron_cut = valE;
}

void GGEMS::set_photon_cut ( f32 valE )
{
    m_parameters.data_h.photon_cut = valE;
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

  m_parameters.data_h.seed = vseed;
}

/// Sources
void GGEMS::set_source ( GGEMSSource* aSource )
{
    m_source = aSource;
}

/// Phantoms
void GGEMS::set_phantom ( GGEMSPhantom* aPhantom )
{
    m_phantom = aPhantom;
}

/// Detector
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
        m_parameters.data_h.display_run_time = ENABLED;
    }
    else
    {
        m_parameters.data_h.display_run_time = DISABLED;
    }
}

// Display memory usage
void GGEMS::set_display_memory_usage( bool flag )
{
    if ( flag )
    {
        m_parameters.data_h.display_memory_usage = ENABLED;
    }
    else
    {
        m_parameters.data_h.display_memory_usage = DISABLED;
    }
}

// Display energy cut
void GGEMS::set_display_energy_cuts( bool flag )
{
    if ( flag )
    {
        m_parameters.data_h.display_energy_cuts = ENABLED;
    }
    else
    {
        m_parameters.data_h.display_energy_cuts = DISABLED;
    }
}

// Display in color
void GGEMS::set_display_in_color( bool flag )
{
    if ( flag )
    {
        #ifdef _WIN32
            GGcerr << "Display in color is not supported by Windows terminal: option set to FALSE" << GGendl;
            m_parameters.data_h.display_in_color = DISABLED;
        #else
            m_parameters.data_h.display_in_color = ENABLED;
        #endif
    }
    else
    {
        m_parameters.data_h.display_in_color = DISABLED;
    }
}

void GGEMS::set_secondaries_level ( ui32 level )
{
    m_parameters.data_h.nb_of_secondaries = level;
}

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

    /*if ( m_phantom == NULL )
    {
        print_error ( "No phantom defined." );
        flag_error = true;
    }*/

    if ( m_parameters.data_h.nb_of_particles == 0 )
    {
        print_error ( "Nb_of_particles = 0." );
        flag_error = true;
    }

    if ( m_parameters.data_h.size_of_particles_batch == 0 )
    {
        print_error ( "Size_of_particles_batch = 0." );
        flag_error = true;
    }

    if ( m_parameters.data_h.seed == 0 )
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

    // Mem allocation
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &m_parameters.data_d.physics_list, NB_PROCESSES*sizeof ( bool ) ) );
    HANDLE_ERROR ( cudaMalloc ( ( void** ) &m_parameters.data_d.secondaries_list, NB_PARTICLES*sizeof ( bool ) ) );

    // Copy data
    HANDLE_ERROR ( cudaMemcpy ( m_parameters.data_d.physics_list, m_parameters.data_h.physics_list,
                                sizeof ( ui8 ) *NB_PROCESSES, cudaMemcpyHostToDevice ) );
    HANDLE_ERROR ( cudaMemcpy ( m_parameters.data_d.secondaries_list, m_parameters.data_h.secondaries_list,
                                sizeof ( ui8 ) *NB_PARTICLES, cudaMemcpyHostToDevice ) );

    m_parameters.data_d.nb_of_particles = m_parameters.data_h.nb_of_particles;
    m_parameters.data_d.size_of_particles_batch = m_parameters.data_h.size_of_particles_batch;
    m_parameters.data_d.nb_of_batches = m_parameters.data_h.nb_of_batches;

    m_parameters.data_d.device_target = m_parameters.data_h.device_target;
    m_parameters.data_d.gpu_id = m_parameters.data_h.gpu_id;
    m_parameters.data_d.gpu_block_size = m_parameters.data_h.gpu_block_size;

    m_parameters.data_d.time = m_parameters.data_h.time;
    m_parameters.data_d.seed = m_parameters.data_h.seed;

    m_parameters.data_d.display_run_time = m_parameters.data_h.display_run_time;
    m_parameters.data_d.display_memory_usage = m_parameters.data_h.display_memory_usage;

    m_parameters.data_d.cs_table_nbins = m_parameters.data_h.cs_table_nbins;
    m_parameters.data_d.cs_table_min_E = m_parameters.data_h.cs_table_min_E;
    m_parameters.data_d.cs_table_max_E = m_parameters.data_h.cs_table_max_E;
    m_parameters.data_d.photon_cut = m_parameters.data_h.photon_cut;
    m_parameters.data_d.electron_cut = m_parameters.data_h.electron_cut;
    m_parameters.data_d.nb_of_secondaries = m_parameters.data_h.nb_of_secondaries;
    m_parameters.data_d.geom_tolerance = m_parameters.data_h.geom_tolerance;
}


////// :: Main functions ::

// Init simualtion
void GGEMS::init_simulation()
{
    // License and banner
    if (!m_license.info.clearence)
    {
        print_error("Your license has expired or is invalid!\n");
        exit_simulation();
    }
    print_banner(m_license.info.institution, m_license.info.expired_day, m_license.info.expired_month,
                 m_license.info.expired_year, "V1.3", m_parameters.data_h );

    // Check
    m_check_mandatory();

    // Run time
    f64 t_start = 0;
    if ( m_parameters.data_h.display_run_time )
    {
        t_start = get_time();
    }

    // Memory usage
    //     ui32 mem = 0;

    // CPU PRNG
    srand ( m_parameters.data_h.seed );

    // Get Nb of batch            
    m_parameters.data_h.size_of_particles_batch = fminf( m_parameters.data_h.nb_of_particles, m_parameters.data_h.size_of_particles_batch );

    m_parameters.data_h.nb_of_batches = ui32 ( ( f32 ) m_parameters.data_h.nb_of_particles / ( f32 ) m_parameters.data_h.size_of_particles_batch );

    if( m_parameters.data_h.nb_of_particles % m_parameters.data_h.size_of_particles_batch )
    {
        m_parameters.data_h.nb_of_batches++;
    }

    // Print some information
    GGnewline();
    GGcout_timestamp();
    GGcout_version();
    GGcout_def();

    // Print params
    GGcout_params( m_parameters.data_h );

    //// Need to clean this bunch of crap - JB

    /*
    if (m_parameters.data_h.nb_of_particles % m_parameters.data_h.size_of_particles_batch)
    {
        m_parameters.data_h.nb_of_batches = (m_parameters.data_h.nb_of_particles / m_parameters.data_h.size_of_particles_batch) + 1;
    }
    else
    {
        m_parameters.data_h.nb_of_batches = m_parameters.data_h.nb_of_particles / m_parameters.data_h.size_of_particles_batch;
    }
    m_parameters.data_h.size_of_particles_batch = m_parameters.data_h.nb_of_particles / m_parameters.data_h.nb_of_batches;
    m_parameters.data_h.nb_of_particles = m_parameters.data_h.size_of_particles_batch * m_parameters.data_h.nb_of_batches;
    
    
    m_parameters.data_h.gpu_grid_size = (m_parameters.data_h.size_of_particles_batch + m_parameters.data_h.gpu_block_size - 1) / m_parameters.data_h.gpu_block_size;
    
    m_parameters.data_h.size_of_particles_batch = m_parameters.data_h.gpu_block_size * m_parameters.data_h.gpu_grid_size;
    

    //     printf("Particle Stack size : %d \n",m_stack_size);
    m_parameters.data_h.nb_of_particles = m_parameters.data_h.size_of_particles_batch * m_parameters.data_h.nb_of_batches;
    //     m_parameters.data_h.nb_of_batches *= m_parameters.data_h.size_of_particles_batch;
    
    */
    
    
    // Init the GPU if need
    if ( m_parameters.data_h.device_target == GPU_DEVICE )
    {       
        // Set the gpu id
        set_gpu_device ( m_parameters.data_h.gpu_id );

        // Reset device
        reset_gpu_device();

        // Copy params to the GPU
        m_copy_parameters_cpu2gpu();
    }

    /// Init Sources /////////////////////////////////
    m_source->initialize ( m_parameters );

    /// Init Phantoms ////////////////////////////////
    if ( m_phantom ) m_phantom->initialize ( m_parameters );

    /// Init Particles Stack /////////////////////////
    // The detector is not mandatory
    if ( m_detector ) m_detector->initialize ( m_parameters );

    /// Init Particles Stack /////////////////////////
    m_particles_manager.initialize ( m_parameters );

    /// Verbose information //////////////////////////

    // Display memory usage
    if (m_parameters.data_h.display_memory_usage) {
        // Particle stack
        ui64 n = m_particles_manager.particles.size;
        ui64 l = m_parameters.data_h.nb_of_secondaries;

        ui64 mem = n * ( 12 * sizeof( f32 ) + 5 * sizeof( ui32 )  + 4 * sizeof( ui8 ) ) +
                   n*l * ( 8 * sizeof ( f32 ) + sizeof( ui8 ) );

        GGcout_mem("Particle stacks", mem);

        GGnewline();
    }

    // Run time
    if ( m_parameters.data_h.display_run_time ) {
        GGcout_time ( "Initialization", get_time()-t_start );
        GGnewline();
    }

    // Succesfull init
    m_flag_init = true;

}


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

void GGEMS::start_simulation()
{

    if ( !m_flag_init )
    {
        GGcerr << "GGEMS simulation not initialized!!" << GGendl;
        exit_simulation();
    }

    // Run time
    f64 t_start = 0;
    if ( m_parameters.data_h.display_run_time )
    {
        t_start = get_time();
    }

    //float progress = 0.0;
    // Main loop
    ui32 ibatch=0;
    //GGcout << "Total number of particles to generate: " << m_parameters.data_h.nb_of_particles << GGendl;
    while ( ibatch < m_parameters.data_h.nb_of_batches )
    {

        GGcout << "----> Launching batch " << ibatch+1 << "/" << m_parameters.data_h.nb_of_batches << " ..." << GGendl;
        // Get primaries
        GGcout << "      Number of particles to generate: " << m_parameters.data_h.size_of_particles_batch << GGendl;
        GGcout << "      Generating the particles ..." << GGendl;
        //         progress_bar(progress,"generate primaries");
        m_source->get_primaries_generator( m_particles_manager.particles );

        //m_particles_manager.copy_gpu2cpu( m_particles_manager.particles );
        //m_particles_manager.print_stack( m_particles_manager.particles );


        // Nav between source and phantom
        if ( m_phantom )
        {
            // Nav between source to phantom
            GGcout << "      Navigation between the source and the phantom ..." << GGendl;
            //         progress_bar(progress,"track to phantom");
            m_phantom->track_to_in( m_particles_manager.particles );

            // Nav within the phantom
            GGcout << "      Navigation within the phantom ..." << GGendl;
            //         progress_bar(progress,"batch");
            m_phantom->track_to_out( m_particles_manager.particles );
        }

        // Nav between phantom and detector
        if( m_detector )
        {
            GGcout << "      Navigation between the phantom and the detector ..." << GGendl;
            m_detector->track_to_in( m_particles_manager.particles );

            GGcout << "      Navigation within the detector ..." << GGendl;
            m_detector->track_to_out( m_particles_manager.particles );

            GGcout << "      Digitizer ..." << GGendl;
            m_detector->digitizer( m_particles_manager.particles );
        }

        GGcout << "----> Batch finished ..." << GGendl << GGendl;
        //progress_bar(progress, ibatch , m_parameters.data_h.nb_of_batches);

        //progress += 1./ (float)(m_parameters.data_h.nb_of_batches);

        //             progress += 0.16; // for demonstration only
        //         }

        ++ibatch;
    }
    //        progress_bar(progress, m_parameters.data_h.nb_of_batches , m_parameters.data_h.nb_of_batches);
    std::cout << std::endl;

    // Run time
    if ( m_parameters.data_h.display_run_time ) {
        // Sync all kernel to get the GPU run time
        if ( m_parameters.data_h.device_target == GPU_DEVICE ) cudaDeviceSynchronize();

        GGcout_time ( "Simulation run time", get_time()-t_start );
        GGnewline();
    }
}








#endif
