#ifndef GUARD_GGEMS_GLOBAL_GGCONSTANTS_HH
#define GUARD_GGEMS_GLOBAL_GGCONSTANTS_HH

/*!
  \file GGConstants.hh

  \brief Different namespaces storing constants useful for GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, Brest, FRANCE
  \version 1.0
  \date Wednesday October 2, 2019
*/

#include "GGEMS/global/GGExport.hh"
#include "GGEMS/tools/GGSystemOfUnits.hh"

/*!
  \namespace GGProcessName
  \brief Namespace storing constants about processes
*/
#ifndef OPENCL_COMPILER
namespace GGProcessName
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
  \namespace GGParticleName
  \brief Namespace storing particles handling by GGEMS
*/
#ifndef OPENCL_COMPILER
namespace GGParticleName
{
#endif
  __constant GGuchar PHOTON = 0; /*!< Photon particle */
  __constant GGuchar ELECTRON = 1; /*!< Electron particle */
  __constant GGuchar POSITRON = 2; /*!< Positron particle */
#ifndef OPENCL_COMPILER
}
#endif

/*!
  \namespace ParticleState
  \brief Namespace storing the state of the particle
*/
#ifndef OPENCL_COMPILER
namespace GGParticleState
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
  \namespace Tolerance
  \brief Namespace storing the tolerance for the float computations
*/
#ifndef OPENCL_COMPILER
namespace GGTolerance
{
#endif
  __constant GGdouble EPSILON2 = 1.0e-02; /*!< Epsilon of 0.01 */
  __constant GGdouble EPSILON3 = 1.0e-03; /*!< Epsilon of 0.001 */
  __constant GGdouble EPSILON6 = 1.0e-06; /*!< Epsilon of 0.000001 */
  __constant GGdouble GEOMETRY = 100.0*
  #ifndef OPENCL_COMPILER
  GGUnits::nm; /*!< Tolerance for the geometry navigation */
  #else
  (1.e-9 *1000.*1.0);
  #endif
#ifndef OPENCL_COMPILER
}
#endif

/*!
  \namespace GGState
  \brief Namespace storing the state of the particle
*/
#ifndef OPENCL_COMPILER
namespace GGState
{
#endif
  __constant GGuchar SOLID = 0; /*!< Solid state */
  __constant GGuchar GAS = 1; /*!< Gas state */
#ifndef OPENCL_COMPILER
}
#endif

/*!
  \namespace GGLimit
  \brief Namespace storing the energy threshold
*/
#ifndef OPENCL_COMPILER
namespace Limit
{
#endif
  __constant GGdouble KINETIC_ENERGY_MIN = 1.0*
  #ifndef OPENCL_COMPILER
  GGUnits::eV; /*!< Min kinetic energy */
  #else
  1.e-6*1.;
  #endif

  __constant GGuint CROSS_SECTION_TABLE_NUMBER_BINS = 220; /*!< Number of bins in the cross section table */
  __constant GGdouble CROSS_SECTION_TABLE_ENERGY_MIN = 990.0*
  #ifndef OPENCL_COMPILER
  GGUnits::eV; /*!< Min energy in the cross section table */
  #else
  1.e-6*1.;
  #endif

  __constant GGdouble CROSS_SECTION_TABLE_ENERGY_MAX = 250.0*
  #ifndef OPENCL_COMPILER
  GGUnits::MeV; /*!< Max energy in the cross section table */
  #else
  1.;
  #endif

  __constant GGdouble PHOTON_CUT = 1.0*
  #ifndef OPENCL_COMPILER
  GGUnits::um; /*!< Photon cut */
  #else
  1.e-6 *1000.*1.0;
  #endif

  __constant GGdouble ELECTRON_CUT = 1.0*
  #ifndef OPENCL_COMPILER
  GGUnits::um; /*!< Electron cut */
  #else
  1.e-6 *1000.*1.0;
  #endif

  __constant GGdouble POSITRON_CUT = 1.0*
  #ifndef OPENCL_COMPILER
  GGUnits::um; /*!< Positron cut */
  #else
  1.e-6 *1000.*1.0;
  #endif
#ifndef OPENCL_COMPILER
}
#endif

#endif // End of GUARD_GGEMS_GLOBAL_GGEMSCONSTANTS_HH
