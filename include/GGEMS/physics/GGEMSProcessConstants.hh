#ifndef GUARD_GGEMS_PHYSICS_GGEMSPROCESSCONSTANTS_HH
#define GUARD_GGEMS_PHYSICS_GGEMSPROCESSCONSTANTS_HH

/*!
  \file GGEMSEMProcessConstants.hh

  \brief Storing some constant variables for process

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday April 13, 2020
*/

#include "GGEMS/tools/GGEMSSystemOfUnits.hh"

/*#ifdef OPENCL_COMPILER

// Processes
#define NUMBER_PROCESSES 3
#define NUMBER_PHOTON_PROCESSES 3
#define COMPTON_SCATTERING 0
#define PHOTOELECTRIC_EFFECT 1
#define RAYLEIGH_SCATTERING 2
#define NO_PROCESS 99

// Cross sections
#define MAX_CROSS_SECTION_TABLE_NUMBER_BINS 2048

#else*/

/*!
  \namespace GGEMSProcess
  \brief Namespace storing constants about processes
*/
#ifndef OPENCL_COMPILER
namespace GGEMSProcess
{
#endif
  // Processes
  __constant GGuchar NUMBER_PROCESSES = 3; /*!< Maximum number of processes */

  #ifndef OPENCL_COMPILER
  __constant GGuchar NUMBER_PHOTON_PROCESSES = 3; /*!< Maximum number of photon processes */
  #else
  #define NUMBER_PHOTON_PROCESSES 3
  #endif

  // Photon
  __constant GGuchar COMPTON_SCATTERING = 0; /*!< Compton process */
  __constant GGuchar PHOTOELECTRIC_EFFECT = 1; /*!< Photoelectric process */
  __constant GGuchar RAYLEIGH_SCATTERING = 2; /*!< Rayleigh process */

  __constant GGuchar NO_PROCESS = 99; /*!< No process */
  __constant GGuchar TRANSPORTATION = 99; /*!< Transportation process */

  //__constant GGuchar NUMBER_ELECTRON_PROCESSES = 3; /*!< Maximum number of electron processes */
  //__constant GGuchar NUMBER_PARTICLES = 5; /*!< Maximum number of different particles for secondaries */
  //__constant GGuchar PHOTON_BONDARY_VOXEL = 77; /*!< Photon on the boundaries */
  //__constant GGuchar ELECTRON_IONISATION = 4; /*!< Electron ionisation process */
  //__constant GGuchar ELECTRON_MSC = 5; /*!< Electron multiple scattering process */
  //__constant GGuchar ELECTRON_BREMSSTRAHLUNG = 6; /*!< Bremsstralung electron process */

  // Cross sections
  #ifndef OPENCL_COMPILER
  __constant GGfloat KINETIC_ENERGY_MIN = 1.0f*GGEMSUnits::eV; /*!< Min kinetic energy */
  __constant GGfloat CROSS_SECTION_TABLE_ENERGY_MIN = 990.0f*GGEMSUnits::eV; /*!< Min energy in the cross section table */
  __constant GGfloat CROSS_SECTION_TABLE_ENERGY_MAX = 250.0f*GGEMSUnits::MeV; /*!< Max energy in the cross section table */
  __constant GGushort MAX_CROSS_SECTION_TABLE_NUMBER_BINS = 2048; /*!< Max number of bins in the cross section table */
  #else
  __constant GGfloat KINETIC_ENERGY_MIN = 1.e-6f; /*!< Min kinetic energy */
  __constant GGfloat CROSS_SECTION_TABLE_ENERGY_MIN = 990.0f*1.e-6f; /*!< Min energy in the cross section table */
  __constant GGfloat CROSS_SECTION_TABLE_ENERGY_MAX = 250.0f*1.0f; /*!< Max energy in the cross section table */
  #define MAX_CROSS_SECTION_TABLE_NUMBER_BINS 2048
  #endif

  __constant GGushort CROSS_SECTION_TABLE_NUMBER_BINS = 220; /*!< Number of bins in the cross section table */
#ifndef OPENCL_COMPILER
}
#endif

/*!
  \namespace GGEMSProcessCut
  \brief Namespace storing constants about process cuts
*/
#ifndef OPENCL_COMPILER
namespace GGEMSProcessCut
{
#endif
  #ifndef OPENCL_COMPILER
  __constant GGfloat PHOTON_DISTANCE_CUT = 1.0f*GGEMSUnits::um; /*!< Photon cut */
  __constant GGfloat ELECTRON_DISTANCE_CUT = 1.0f*GGEMSUnits::um; /*!< Electron cut */
  __constant GGfloat POSITRON_DISTANCE_CUT = 1.0f*GGEMSUnits::um; /*!< Positron cut */
  #else
  __constant GGfloat PHOTON_DISTANCE_CUT = 1.e-3f; /*!< Photon cut */
  __constant GGfloat ELECTRON_DISTANCE_CUT = 1.e-3f; /*!< Electron cut */
  __constant GGfloat POSITRON_DISTANCE_CUT = 1.e-3f; /*!< Positron cut */
  #endif
#ifndef OPENCL_COMPILER
}
#endif

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSEMPROCESSCONSTANTS_HH