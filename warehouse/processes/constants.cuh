// GGEMS Copyright (C) 2015

#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "G4SystemOfUnits.hh"

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
#define ELECTRON_IONISATION 4
#define ELECTRON_MSC 5
#define ELECTRON_BREMSSTRAHLUNG 6

// Particle state
#define PRIMARY 0
#define GEOMETRY_BOUNDARY 99
#define PARTICLE_ALIVE 0
#define PARTICLE_DEAD 1

// Misc
#define DISABLED 0
#define ENABLED 1

#define TRUE    1
#define FALSE   0

#define EKINELIMIT 1*eV
#define elec_radius          (2.8179409421853486E-15*m)      // Metre
#define N_avogadro           (6.0221367E+23/mole)

#define DEBUGOK "[\033[32;01mok\033[00m]"
#define PRINTFUNCTION printf("%s %s\n",DEBUGOK,__FUNCTION__);

#define EPSILON3 1.0e-03f
#define EPSILON6 1.0e-06f

// Pi
#define gpu_pi               3.141592653589793116
#define gpu_twopi            2.0*gpu_pi

#endif
