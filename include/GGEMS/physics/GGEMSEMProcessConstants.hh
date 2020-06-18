#ifndef GUARD_GGEMS_PHYSICS_GGEMSEMPROCESSCONSTANTS_HH
#define GUARD_GGEMS_PHYSICS_GGEMSEMPROCESSCONSTANTS_HH

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

/*!
  \namespace GGEMSProcess
  \brief Namespace storing constants about processes, using preprocessor for OpenCL
*/
#ifdef OPENCL_COMPILER

#define NUMBER_PROCESSES 3
#define NUMBER_PHOTON_PROCESSES 3
#define COMPTON_SCATTERING 0
#define PHOTOELECTRIC_EFFECT 1
#define RAYLEIGH_SCATTERING 2
#define NO_PROCESS 99

#else

namespace GGEMSProcess
{
  __constant GGuchar NUMBER_PROCESSES = 3; /*!< Maximum number of processes */
  __constant GGuchar NUMBER_PHOTON_PROCESSES = 3; /*!< Maximum number of photon processes */
  //__constant GGuchar NUMBER_ELECTRON_PROCESSES = 3; /*!< Maximum number of electron processes */
  //__constant GGuchar NUMBER_PARTICLES = 5; /*!< Maximum number of different particles for secondaries */

  __constant GGuchar COMPTON_SCATTERING = 0; /*!< Compton process */
  __constant GGuchar PHOTOELECTRIC_EFFECT = 1; /*!< Photoelectric process */
  __constant GGuchar RAYLEIGH_SCATTERING = 2; /*!< Rayleigh process */
  //__constant GGuchar PHOTON_BONDARY_VOXEL = 77; /*!< Photon on the boundaries */

  //__constant GGuchar ELECTRON_IONISATION = 4; /*!< Electron ionisation process */
  //__constant GGuchar ELECTRON_MSC = 5; /*!< Electron multiple scattering process */
  //__constant GGuchar ELECTRON_BREMSSTRAHLUNG = 6; /*!< Bremsstralung electron process */

  __constant GGuchar NO_PROCESS = 99; /*!< No process */
}
#endif

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSEMPROCESSCONSTANTS_HH