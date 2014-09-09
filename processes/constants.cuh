// This file is part of GGEMS
//
// FIREwork is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// FIREwork is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with FIREwork.  If not, see <http://www.gnu.org/licenses/>.
//
// GGEMS Copyright (C) 2013-2014 Julien Bert

#ifndef CONSTANTS_H
#define CONSTANTS_H

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

// Pi
#define gpu_pi               3.141592653589793116
#define gpu_twopi            2.0*gpu_pi

/*

// Units
#define GeV                  (1.E3*MeV)           // MeV
#define MeV                  1.                           // MeV
#define keV                  (1.E-3*MeV)      // MeV
#define eV                   (1.E-6*MeV)      // MeV

#define gramme               1.                             // Gramme

#define mole                 1.                             // Mole

#define sec                  1.                             // Seconde

#define m                    1.E3                       // Millimetre
#define cm                   (1.E-2*m)          // Millimetre
#define mm                   (1.E-3*m)          // Millimetre
#define um                   (1.E-6*m)          // Millimetre

#define m2                   (1.*m*m)             // Metre2
#define cm2                  (1.E-4*m2)         // Metre2
#define mm2                  (1.E-6*m2)         // Metre2
#define barn                 (1.E-28*m2)      // Metre2

#define m3                   (1.*m*m*m)// Metre3
#define cm3                  (1.E-6*m3)         // Metre3
#define mm3                  (1.E-9*m3)         // Metre3

#define deg                  (180./pi)                                               // rad
#define N_avogadro           (6.0221367E+23/mole)                        // Mole
#define electron_mass_c2     (0.51099906*MeV)                            // MeV
#define elec_radius          (2.8179409421853486E-15*m)      // Metre
#define eplus                1.60217646E-19                             // Coulomb
#define pi                   3.141592653589793116
#define c_light              (2.99792458E+8*m/sec)                       // m/s
#define Bohr_radius          (52.91772621718944E-12*m)               // Metre
#define hbarc                (197.32705406375647E-15*MeV*m)  // MeV*m
#define twopi_mc2_rcl2       2.54954929921041011529E-23          //2.*pi*electron_mass_c2*elec_radius*elec_radius
#define fine_struct          0.00729735301337329

#define Magmom_proton        2.79285E-6              // MgN
#define amu_c2               (931.494*MeV)        // MeV
#define m_proton             (938.272013*MeV)   // MeV
#define m_neutron            (939.572*MeV)          // MeV
#define m_deuton             (1876.0957309*MeV) // MeV
#define m_alpha              (3728.24*MeV)          // MeV
*/

#endif
