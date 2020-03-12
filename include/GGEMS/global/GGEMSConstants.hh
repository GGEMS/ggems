#ifndef GUARD_GGEMS_GLOBAL_GGEMSCONSTANTS_HH
#define GUARD_GGEMS_GLOBAL_GGEMSCONSTANTS_HH

/*!
  \file GGEMSConstants.hh

  \brief Different namespaces storing constants useful for GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, Brest, FRANCE
  \version 1.0
  \date Wednesday October 2, 2019
*/

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"

/*!
  \namespace GGEMSProcessName
  \brief Namespace storing constants about processes
*/
#ifndef OPENCL_COMPILER
namespace GGEMSProcessName
{
#endif
  __constant GGuchar NUMBER_PROCESSES = 7; /*!< Maximum number of processes */
  __constant GGuchar NUMBER_PHOTON_PROCESSES = 3; /*!< Maximum number of photon processes */
  __constant GGuchar NUMBER_ELECTRON_PROCESSES = 3; /*!< Maximum number of electron processes */
  __constant GGuchar NUMBER_PARTICLES = 3; /*!< Maximum number of different particles */

  __constant GGuchar PHOTON_COMPTON = 0; /*!< Compton process */
  __constant GGuchar PHOTON_PHOTOELECTRIC = 1; /*!< Photoelectric process */
  __constant GGuchar PHOTON_RAYLEIGH = 2; /*!< Rayleigh process */
  __constant GGuchar PHOTON_BONDARY_VOXEL = 3; /*!< Photon on the boundaries */

  __constant GGuchar ELECTRON_IONISATION = 4; /*!< Electron ionisation process */
  __constant GGuchar ELECTRON_MSC = 5; /*!< Electron multiple scattering process */
  __constant GGuchar ELECTRON_BREMSSTRAHLUNG = 6; /*!< Bremsstralung electron process */

  __constant GGuchar NO_PROCESS = 99; /*!< No process */
#ifndef OPENCL_COMPILER
}
#endif

/*!
  \namespace GGEMSParticleName
  \brief Namespace storing particles handling by GGEMS
*/
#ifndef OPENCL_COMPILER
namespace GGEMSParticleName
{
#endif
  __constant GGuchar PHOTON = 0; /*!< Photon particle */
  __constant GGuchar ELECTRON = 1; /*!< Electron particle */
  __constant GGuchar POSITRON = 2; /*!< Positron particle */
#ifndef OPENCL_COMPILER
}
#endif

/*!
  \namespace GGEMSParticleState
  \brief Namespace storing the state of the particle
*/
#ifndef OPENCL_COMPILER
namespace GGEMSParticleState
{
#endif
  __constant GGuchar PRIMARY = 0; /*!< Primary particle */
  __constant GGuchar GEOMETRY_BOUNDARY = 99; /*!< Particle on the boundary */
  __constant GGuchar ALIVE = 0; /*!< Particle alive */
  __constant GGuchar DEAD = 1; /*!< Particle dead */
  __constant GGuchar FREEZE = 2; /*!< Particle freeze */
#ifndef OPENCL_COMPILER
}
#endif

/*!
  \namespace GGEMSTolerance
  \brief Namespace storing the tolerance for the float computations
*/
#ifndef OPENCL_COMPILER
namespace GGEMSTolerance
{
#endif
  __constant GGfloat EPSILON2 = 1.0e-02f; /*!< Epsilon of 0.01 */
  __constant GGfloat EPSILON3 = 1.0e-03f; /*!< Epsilon of 0.001 */
  __constant GGfloat EPSILON6 = 1.0e-06f; /*!< Epsilon of 0.000001 */
  __constant GGfloat GEOMETRY = 100.0f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::nm; /*!< Tolerance for the geometry navigation */
  #else
  (1.e-9f *1000.f*1.0f);
  #endif
#ifndef OPENCL_COMPILER
}
#endif

/*!
  \namespace GGEMSState
  \brief Namespace storing the state of the particle
*/
#ifndef OPENCL_COMPILER
namespace GGEMSState
{
#endif
  __constant GGuchar SOLID = 0; /*!< Solid state */
  __constant GGuchar GAS = 1; /*!< Gas state */
#ifndef OPENCL_COMPILER
}
#endif

/*!
  \namespace GGEMSLimit
  \brief Namespace storing the energy threshold
*/
#ifndef OPENCL_COMPILER
namespace GGEMSLimit
{
#endif
  __constant GGfloat KINETIC_ENERGY_MIN = 1.0f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::eV; /*!< Min kinetic energy */
  #else
  1.e-6f*1.f;
  #endif

  __constant GGushort CROSS_SECTION_TABLE_NUMBER_BINS = 220; /*!< Number of bins in the cross section table */
  __constant GGfloat CROSS_SECTION_TABLE_ENERGY_MIN = 990.0f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::eV; /*!< Min energy in the cross section table */
  #else
  1.e-6f*1.f;
  #endif

  __constant GGfloat CROSS_SECTION_TABLE_ENERGY_MAX = 250.0f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::MeV; /*!< Max energy in the cross section table */
  #else
  1.f;
  #endif

  __constant GGfloat PHOTON_CUT = 1.0f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::um; /*!< Photon cut */
  #else
  1.e-6f *1000.f*1.0f;
  #endif

  __constant GGfloat ELECTRON_CUT = 1.0f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::um; /*!< Electron cut */
  #else
  1.e-6f *1000.f*1.0f;
  #endif

  __constant GGfloat POSITRON_CUT = 1.0f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::um; /*!< Positron cut */
  #else
  1.e-6f*1000.f*1.0f;
  #endif
#ifndef OPENCL_COMPILER
}
#endif

#endif // End of GUARD_GGEMS_GLOBAL_GGEMSCONSTANTS_HH
