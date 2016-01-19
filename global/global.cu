// GGEMS Copyright (C) 2015

#ifndef GLOBAL_CU
#define GLOBAL_CU
#include "global.cuh"

// Some usefull functions

// Reset the GPU
void reset_gpu_device()
{
    printf ( "[\033[32;01mok\033[00m] Reset device .. \n" );
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
void set_gpu_device ( int deviceChoice, f32 minversion )
{

    int deviceCount = 0;
    cudaGetDeviceCount ( &deviceCount );

    if ( deviceCount == 0 )
    {
        printf ( "[\033[31;03mWARNING\033[00m] There is no device supporting CUDA\n" );
        exit ( EXIT_FAILURE );
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties ( &prop, deviceChoice%deviceCount );

    if ( prop.major<minversion )
    {
        printf ( "[\033[31;03mWARNING\033[00m] Your device is not compatible with %1.1f version\n",minversion );
        exit ( EXIT_FAILURE );
    }

    cudaSetDevice ( deviceChoice%deviceCount );
    printf ( "[\033[32;01mok\033[00m] \033[32;01m%s\033[00m found\n", prop.name );

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
void print_time ( std::string txt, f64 t )
{

    f64 res;
    ui32 time_h = ( ui32 ) ( t / 3600.0 );
    res = t - ( time_h*3600.0 );
    ui32 time_m = ( ui32 ) ( res / 60.0 );
    res -= ( time_m * 60.0 );
    ui32 time_s = ( ui32 ) ( res );
    res -= time_s;
    ui32 time_ms = ( ui32 ) ( res*1000.0 );

    printf ( "[\033[32;01mRun time\033[00m] %s: ", txt.c_str() );

    if ( time_h != 0 ) printf ( "%i h ", time_h );
    if ( time_m != 0 ) printf ( "%i m ", time_m );
    if ( time_s != 0 ) printf ( "%i s ", time_s );
    printf ( "%i ms\n", time_ms );

}

// Print out memory usage
void print_memory ( std::string txt, ui32 t )
{

    std::vector<std::string> pref;
    pref.push_back ( "B" );
    pref.push_back ( "kB" );
    pref.push_back ( "MB" );
    pref.push_back ( "GB" );

    ui32 iemem = ( ui32 ) ( log ( t ) / log ( 1000 ) );
    f32 mem = f32 ( t ) / ( pow ( 1000, iemem ) );

    printf ( "[\033[34;01mMemory usage\033[00m] %s: %5.2f %s\n", txt.c_str(), mem, pref[iemem].c_str() );

}

// Print GGEMS banner
void print_banner(std::string institution, std::string exp_day, std::string exp_month, std::string exp_year, std::string version) {

    printf("      \033[32;01m____\033[00m                  \n");
    printf(".--. \033[32;01m/\\__/\\\033[00m .--.            \n");
    printf("`\033[33;01mO\033[00m  \033[32;01m/ /  \\ \\\033[00m  .`     GGEMS %s  \n", version.c_str());
    printf("  `-\033[32;01m| |  | |\033[00m\033[33;01mO\033[00m`              \n");
    printf("   -\033[32;01m|\033[00m`\033[32;01m|\033[00m..\033[32;01m|\033[00m`\033[32;01m|\033[00m-        License:  \n");
    printf(" .` \033[32;01m\\\033[00m.\033[32;01m\\__/\033[00m.\033[32;01m/\033[00m `.        %s       \n", institution.c_str());
    printf("'.-` \033[32;01m\\/__\\/\033[00m `-.'       %s-%s-%s \n", exp_day.c_str(), exp_month.c_str(), exp_year.c_str());
    printf("\n");

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
