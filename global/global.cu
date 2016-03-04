// GGEMS Copyright (C) 2015

#ifndef GLOBAL_CU
#define GLOBAL_CU
#include "global.cuh"

// Some usefull functions

// Reset the GPU
void reset_gpu_device()
{
    cudaDeviceReset();
}

// comes from "cuda by example" book
void HandleError ( cudaError_t err,
                   const char *file,
                   int line )
{
    if ( err != cudaSuccess )
    {
        printf ( "%s in %s at line %d\n", cudaGetErrorString ( err ),
                 file, line );
        exit ( EXIT_FAILURE );
    }
}

// comes from "cuda programming" book
__host__ void cuda_error_check ( const char * prefix, const char * postfix )
{
    if ( cudaPeekAtLastError() != cudaSuccess )
    {
        printf ( "\n%s%s%s\n",prefix, cudaGetErrorString ( cudaGetLastError() ),postfix );
        cudaDeviceReset();
        exit ( EXIT_FAILURE );
    }
}

// Set a GPU device
void set_gpu_device ( int deviceChoice )
{

    f32 minversion = 3.0;

    i32 deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount( &deviceCount );

    if (error_id != cudaSuccess)
    {
        GGcerr << "cudaGetDeviceCount returned " << ( i32 ) error_id
               << " " << cudaGetErrorString(error_id) << GGendl;
        exit_simulation();
    }

    if ( deviceCount == 0 )
    {
        GGcerr << "There are no available device(s) that support CUDA" << GGendl;
        exit_simulation();
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties ( &prop, deviceChoice%deviceCount );

    if ( prop.major < minversion )
    {
        GGcerr << "Your device is not compatible with " << minversion << " version" << GGendl;
        exit_simulation();
    }

    cudaSetDevice ( deviceChoice%deviceCount );
    GGcout << "GPU found: " << prop.name << " (id: " << deviceChoice%deviceCount << ") " << GGendl;
    GGnewline();

    
/*    // Debug, pour calculer automatiquement le nbre de threads  et blocks en fonction du GPU
    int numBlocks;        // Occupancy in terms of active blocks
    int blockSize = 32;

    // These variables are used to convert occupancy to warps

    int activeWarps;
    int maxWarps;

    
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
        MyKernel,
        blockSize,
        0);

    activeWarps = numBlocks * blockSize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    
    std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;*/
    
}



// Print out for error
void print_error ( std::string msg )
{
    printf ( "\033[31;03m[ERROR] - %s\033[00m", msg.c_str() );
}

// Print out for warning
void print_warning ( std::string msg )
{
    printf ( "\033[33;03m[WARNING] - %s\033[00m", msg.c_str() );
}

// Print out run time
void GGcout_time ( std::string txt, f64 t )
{

    f64 res;
    ui32 time_h = ( ui32 ) ( t / 3600.0 );
    res = t - ( time_h*3600.0 );
    ui32 time_m = ( ui32 ) ( res / 60.0 );
    res -= ( time_m * 60.0 );
    ui32 time_s = ( ui32 ) ( res );
    res -= time_s;
    ui32 time_ms = ( ui32 ) ( res*1000.0 );

    printf ( "[GGEMS] %s: ", txt.c_str() );

    if ( time_h != 0 ) printf ( "%i h ", time_h );
    if ( time_m != 0 ) printf ( "%i m ", time_m );
    if ( time_s != 0 ) printf ( "%i s ", time_s );
    printf ( "%i ms\n", time_ms );

}

// Print date and time
void GGcout_timestamp ()
{
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);

    printf("[GGEMS] %d-%d-%d %02d:%02d:%02d\n", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
}

// Print some version information
void GGcout_version ()
{
    i32 Version = 0;
    //i32 DrvVersion = 0;

    cudaRuntimeGetVersion( &Version );
    //cudaRuntimeGetVersion( &DrvVersion );

    std::string VerTxt = (Version) ? std::to_string( Version/1000.0 ) : "Unknown";
    //std::string DrvTxt = (DrvVersion) ? std::to_string( DrvVersion ) : "Unknown";

    GGcout << "GCC: " << __GNUC__ << "." << __GNUC_MINOR__
           << " NVCC: " << VerTxt
           << GGendl;
}

// Print out memory usage
void GGcout_mem ( std::string txt, ui64 valmem )
{

    std::vector<std::string> pref;
    pref.push_back ( "B" );
    pref.push_back ( "kB" );
    pref.push_back ( "MB" );
    pref.push_back ( "GB" );

    ui32 iemem = ( ui32 ) ( log ( valmem ) / log ( 1000 ) );
    f32 mem = f32 ( f64 ( valmem ) / ( pow ( 1000, iemem ) ) );

    printf ( "[GGEMS] %s: %5.2f %s\n", txt.c_str(), mem, pref[iemem].c_str() );
}


std::string Green_str( std::string txt )
{
    return "\033[32;01m" + txt + "\033[00m";
}

std::string Check_str( std::string txt )
{
    return "[X] " + txt;
}

std::string Red_str( std::string txt )
{
    return "\033[31;03m" + txt + "\033[00m";
}

std::string NoCheck_str( std::string txt )
{
    return "[ ] " + txt;
}

std::string Energy_str( f32 E )
{
    E /= eV;

    std::vector<std::string> pref;
    pref.push_back ( "eV" );
    pref.push_back ( "keV" );
    pref.push_back ( "MeV" );
    pref.push_back ( "GeV" );

    ui32 exp = ( ui32 ) ( log ( E ) / log ( 1000 ) );
    f32 val = f32 ( E ) / ( pow ( 1000, exp ) );

    char tmp[ 100 ];
    sprintf( tmp, "%5.2f %s", val, pref[ exp ].c_str());

    return std::string( tmp );
}

std::string Range_str( f32 range )
{
    range /= nm;

    std::vector<std::string> pref;
    pref.push_back ( "nm" );
    pref.push_back ( "um" );
    pref.push_back ( "mm" );
    pref.push_back ( "m" );

    ui32 exp = ( ui32 ) ( log ( range ) / log ( 1000 ) );
    f32 val = f32 ( range ) / ( pow ( 1000, exp ) );

    char tmp[ 100 ];
    sprintf( tmp, "%5.2f %s", val, pref[ exp ].c_str());

    return std::string( tmp );
}

// Print params
void GGcout_params( GlobalSimulationParametersData params )
{

    if ( params.display_in_color )
    {
        printf("\n");
        printf("[GGEMS] Physics list:\n");
        printf("[GGEMS]    Gamma: %s   %s   %s\n", ( params.physics_list[ PHOTON_COMPTON ] ) ? Green_str("Compton").c_str() : Red_str("Compton").c_str(),
                             ( params.physics_list[ PHOTON_PHOTOELECTRIC ] ) ? Green_str("Photoelectric").c_str() : Red_str("Photoelectric").c_str(),
                                            ( params.physics_list[ PHOTON_RAYLEIGH ] ) ? Green_str("Rayleigh").c_str() : Red_str("Rayleigh").c_str() );

        printf("[GGEMS]    Electron: %s   %s   %s\n", ( params.physics_list[ ELECTRON_IONISATION ] ) ? Green_str("Ionisation").c_str() : Red_str("Ionisation").c_str(),
                                      ( params.physics_list[ ELECTRON_BREMSSTRAHLUNG ] ) ? Green_str("Bremsstrahlung").c_str() : Red_str("Bremsstrahlung").c_str(),
                                       ( params.physics_list[ ELECTRON_MSC ] ) ? Green_str("Multiple scattering").c_str() : Red_str("Multiple scattering").c_str() );

        printf("[GGEMS]    Tables: MinE %s   MaxE %s   Nb of energy bin %i\n", Energy_str( params.cs_table_min_E ).c_str(),
                                                                           Energy_str( params.cs_table_max_E ).c_str(),
                                                                           params.cs_table_nbins );
        printf("[GGEMS]    Range cuts: Gamma %s   Electron %s\n", Range_str( params.photon_cut ).c_str(),
                                                                 Range_str( params.electron_cut ).c_str() );

        printf("[GGEMS] Secondary particles:\n");
        printf("[GGEMS]    Particles: %s   %s\n", ( params.secondaries_list[ PHOTON ] ) ? Green_str("Gamma").c_str() : Red_str("Gamma").c_str(),
                                                ( params.secondaries_list[ ELECTRON ] ) ? Green_str("Electron").c_str() : Red_str("Electron").c_str() );

        printf("[GGEMS]    Levels: %i\n", params.nb_of_secondaries);

        printf("[GGEMS] Geometry tolerance:\n");
        printf("[GGEMS]    Range: %s\n", Range_str( params.geom_tolerance ).c_str() );
        printf("[GGEMS] Simulation:\n");
        printf("[GGEMS]    Device target: %s\n", ( params.device_target == CPU_DEVICE ) ? "CPU" : "GPU" );
        printf("[GGEMS]    Total Nb of particles: %i\n", params.nb_of_particles);
        printf("[GGEMS]    Size of batch: %i\n", params.size_of_particles_batch);
        printf("[GGEMS]    Nb of batches: %i\n", params.nb_of_batches);
        printf("[GGEMS]    Seed value %i\n", params.seed);

        printf("\n");
    }
    else
    {
        printf("\n");
        printf("[GGEMS] Physics list:\n");
        printf("[GGEMS]    Gamma: %s   %s   %s\n", ( params.physics_list[ PHOTON_COMPTON ] ) ? Check_str("Compton").c_str() : NoCheck_str("Compton").c_str(),
                             ( params.physics_list[ PHOTON_PHOTOELECTRIC ] ) ? Check_str("Photoelectric").c_str() : NoCheck_str("Photoelectric").c_str(),
                                            ( params.physics_list[ PHOTON_RAYLEIGH ] ) ? Check_str("Rayleigh").c_str() : NoCheck_str("Rayleigh").c_str() );

        printf("[GGEMS]    Electron: %s   %s   %s\n", ( params.physics_list[ ELECTRON_IONISATION ] ) ? Check_str("Ionisation").c_str() : NoCheck_str("Ionisation").c_str(),
                                      ( params.physics_list[ ELECTRON_BREMSSTRAHLUNG ] ) ? Check_str("Bremsstrahlung").c_str() : NoCheck_str("Bremsstrahlung").c_str(),
                                       ( params.physics_list[ ELECTRON_MSC ] ) ? Check_str("Multiple scattering").c_str() : NoCheck_str("Multiple scattering").c_str() );

        printf("[GGEMS]    Tables: MinE %s   MaxE %s   Nb of energy bin %i\n", Energy_str( params.cs_table_min_E ).c_str(),
                                                                           Energy_str( params.cs_table_max_E ).c_str(),
                                                                           params.cs_table_nbins );
        printf("[GGEMS]    Range cuts: Gamma %s   Electron %s\n", Range_str( params.photon_cut ).c_str(),
                                                                 Range_str( params.electron_cut ).c_str() );

        printf("[GGEMS] Secondary particles:\n");
        printf("[GGEMS]    Particles: %s   %s\n", ( params.secondaries_list[ PHOTON ] ) ? Check_str("Gamma").c_str() : NoCheck_str("Gamma").c_str(),
                                                ( params.secondaries_list[ ELECTRON ] ) ? Check_str("Electron").c_str() : NoCheck_str("Electron").c_str() );

        printf("[GGEMS]    Levels: %i\n", params.nb_of_secondaries);

        printf("[GGEMS] Geometry tolerance:\n");
        printf("[GGEMS]    Range: %s\n", Range_str( params.geom_tolerance ).c_str() );
        printf("[GGEMS] Simulation:\n");
        printf("[GGEMS]    Device target: %s\n", ( params.device_target == CPU_DEVICE ) ? "CPU" : "GPU" );
        printf("[GGEMS]    Total Nb of particles: %i\n", params.nb_of_particles);
        printf("[GGEMS]    Size of batch: %i\n", params.size_of_particles_batch);
        printf("[GGEMS]    Nb of batches: %i\n", params.nb_of_batches);
        printf("[GGEMS]    Seed value %i\n", params.seed);

        printf("\n");
    }



}

// Empty line
void GGnewline( )
{
    printf("\n");
}

//ui64 nb_of_particles;
//ui64 size_of_particles_batch;
//ui32 nb_of_batches;

//ui8 device_target;
//ui32 gpu_id;
//ui32 gpu_block_size;
//ui32 gpu_grid_size;
//f32 time;
//ui32 seed;




// Print GGEMS banner
void print_banner( std::string institution, std::string exp_day, std::string exp_month, std::string exp_year,
                   std::string version, GlobalSimulationParametersData params )
{
    if ( params.display_in_color )
    {
        printf("      \033[32;01m____\033[00m                  \n");
        printf(".--. \033[32;01m/\\__/\\\033[00m .--.            \n");
        printf("`\033[33;01mO\033[00m  \033[32;01m/ /  \\ \\\033[00m  .`     GGEMS %s  \n", version.c_str());
        printf("  `-\033[32;01m| |  | |\033[00m\033[33;01mO\033[00m`              \n");
        printf("   -\033[32;01m|\033[00m`\033[32;01m|\033[00m..\033[32;01m|\033[00m`\033[32;01m|\033[00m-        License:  \n");
        printf(" .` \033[32;01m\\\033[00m.\033[32;01m\\__/\033[00m.\033[32;01m/\033[00m `.        %s       \n", institution.c_str());
        printf("'.-` \033[32;01m\\/__\\/\033[00m `-.'       %s-%s-%s \n", exp_day.c_str(), exp_month.c_str(), exp_year.c_str());
        printf("\n");
    }
    else
    {
        printf("      ____                  \n");
        printf(".--. /\\__/\\ .--.            \n");
        printf("`O  / /  \\ \\  .`     GGEMS %s  \n", version.c_str());
        printf("  `-| |  | |O`              \n");
        printf("   -|`|..|`|-        License:  \n");
        printf(" .` \\.\\__/./ `.        %s       \n", institution.c_str());
        printf("'.-` \\/__\\/ `-.'       %s-%s-%s \n", exp_day.c_str(), exp_month.c_str(), exp_year.c_str());
        printf("\n");
    }
}

// Abort the current simulation
void exit_simulation()
{
    printf ( "\n[\033[31;03mSimulation aborded\033[00m]\n" );
    exit ( EXIT_FAILURE );
}

/*
// Create a color
Color make_color(f32 r, f32 g, f32 b) {
    Color c;
    c.r = r;
    c.g = g;
    c.b = b;
    return c;
}
*/

// Get time
f64 get_time()
{
    timeval tv;
    gettimeofday ( &tv, NULL );
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}







#endif
