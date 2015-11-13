// This file is part of GGEMS
//
// GGEMS is free software: you can redistribute it and/or modify
// GGEMS Copyright (C) 2015

#ifndef GLOBAL_CUH
#define GLOBAL_CUH

/////// INCLUDES ///////////////////////////////////////////////

#include "G4SystemOfUnits.hh"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <map>
#include <algorithm>
#include <cfloat>
#include <assert.h>
#include <math.h>
#include <sys/time.h>

/////// CONSTANTS //////////////////////////////////////////////

// Run on CPU
#define CPU_DEVICE 0
// Run on GPU
#define GPU_DEVICE 1

// Maximum number of processes
#define NB_PROCESSES 20
// Maximum number of photon processes
#define NB_PHOTON_PROCESSES 3
// Maximum number of photon processes
#define NB_ELECTRON_PROCESSES 3
// Maximum number of different particles
#define NB_PARTICLES 5

// Type of particle
#define PHOTON 0
#define ELECTRON 1

// Photon processes
#define PHOTON_COMPTON 0
#define PHOTON_PHOTOELECTRIC 1
#define PHOTON_RAYLEIGH 2
#define PHOTON_BOUNDARY_VOXEL 3

// Electron processes
//#define ELECTRON_IONISATION 4
//#define ELECTRON_MSC 5
//#define ELECTRON_BREMSSTRAHLUNG 6

// Particle state
//#define PRIMARY 0
//#define GEOMETRY_BOUNDARY 99
//#define PARTICLE_ALIVE 0
//#define PARTICLE_DEAD 1

// Misc
#define DISABLED 0
#define ENABLED 1

#define TRUE    1
#define FALSE   0

//#define EKINELIMIT 1*eV
//#define elec_radius          (2.8179409421853486E-15*m)      // Metre
//#define N_avogadro           (6.0221367E+23/mole)

//#define DEBUGOK "[\033[32;01mok\033[00m]"
//#define PRINTFUNCTION printf("%s %s\n",DEBUGOK,__FUNCTION__);
//#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

//#define EPSILON3 1.0e-03f
//#define EPSILON6 1.0e-06f

// Pi
//#define gpu_pi               3.141592653589793116
//#define gpu_twopi            2.0*gpu_pi

/////// TYPEDEF ////////////////////////////////////////////////

#ifndef SINGLE_PRECISION
    // Double precision
    typedef float f32;
    typedef float2 f32xy;
    typedef float3 f32xyz;
    typedef float4 f32xyzw;

    typedef double f64;
    typedef double2 f64xy;
    typedef double3 f64xyz;
    typedef double4 f64xyzw;

    typedef int i32;
    typedef int2 i32xy;
    typedef int3 i32xyz;
    typedef int4 i32xyzw;

    typedef short int i16;
    typedef char i8;

    typedef unsigned int ui32;
    typedef uint2 ui32xy;
    typedef uint3 ui32xyz;
    typedef uint4 ui32xyzw;

    typedef unsigned short int ui16;
    typedef ushort2 ui16xy;
    typedef ushort3 ui16xyz;
    typedef ushort4 ui16xyzw;
    
    typedef unsigned long int ui64;

    typedef unsigned char ui8;

    #define F32_MAX FLT_MAX;
    #define F64_MAX DBL_MAX;

#else
    // Single precision
    typedef float f32;
    typedef float2 f32xy;
    typedef float3 f32xyz;
    typedef float4 f32xyzw;

    typedef double f64;
    typedef double2 f64xy;
    typedef double3 f64xyz;
    typedef double4 f64xyzw;

    typedef int i32;
    typedef int2 i32xy;
    typedef int3 i32xyz;
    typedef int4 i32xyzw;

    typedef short int i16;
    typedef char i8;

    typedef unsigned int ui32;
    typedef uint2 ui32xy;
    typedef uint3 ui32xyz;
    typedef uint4 ui32xyzw;

    typedef unsigned short int ui16;
    typedef ushort2 ui16xy;
    typedef ushort3 ui16xyz;
    typedef ushort4 ui16xyzw;

    typedef unsigned char ui8;

    #define F32_MAX FLT_MAX;
    #define F64_MAX FLT_MAX;
#endif

////////////////////////////////////////////////////////////////

// GPU
void set_gpu_device(int deviceChoice, float minversion=3.0);
void reset_gpu_device();
void HandleError(cudaError_t err, const char *file, int line );
void cuda_error_check (const char * prefix, const char * postfix);

// Utils
void print_error(std::string msg);
void print_warning(std::string msg);
void print_time(std::string txt, f64 t);
void print_memory(std::string txt, ui32 t);
void exit_simulation();
f64 get_time();

// Global simulation parameters
struct GlobalSimulationParameters {
    ui8 *physics_list;
    ui8 *secondaries_list;   

    ui64 nb_of_particles;
    ui64 size_of_particles_batch;
    ui32 nb_of_batches;

    ui8 device_target;
    ui32 gpu_id;
    ui32 gpu_block_size;

    f32 time;
    ui32 seed;

    bool display_run_time;
    bool display_memory_usage;

    // To build cross sections table
    ui32 cs_table_nbins;
    f32 cs_table_min_E;
    f32 cs_table_max_E;
};

// Struct that handle colors
//struct Color {
//    f32 r, g, b;
//};
//Color make_color(f32 r, f32 g, f32 b);

//// Struct that handle nD variable     TODO the other types
static __inline__ __host__ __device__ f32xyz make_f32xyz(f32 vx, f32 vy, f32 vz) {
    f32xyz t; t.x = vx; t.y = vy; t.z = vz; return t;
}
static __inline__ __host__ __device__ f64xyz make_f64xyz(f64 vx, f64 vy, f64 vz) {
    f64xyz t; t.x = vx; t.y = vy; t.z = vz; return t;
}

#endif
