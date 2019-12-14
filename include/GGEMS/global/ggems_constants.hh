#ifndef GUARD_GGEMS_GLOBAL_GGEMSCONSTANTS_HH
#define GUARD_GGEMS_GLOBAL_GGEMSCONSTANTS_HH

/*!
  \file ggems_constants.hh

  \brief Different namespaces storing constants useful for GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, Brest, FRANCE
  \version 1.0
  \date Wednesday October 2, 2019
*/

#include "GGEMS/global/ggems_export.hh"
#include "GGEMS/tools/system_of_units.hh"

/*!
  \namespace ProcessName
  \brief Namespace storing constants about processes
*/
#ifdef __cplusplus
namespace ProcessName
{
#endif
  __constant int NUMBER_PROCESSES = 7; /*!< Maximum number of processes */
  __constant int NUMBER_PHOTON_PROCESSES = 3; /*!< Maximum number of photon processes */
  __constant int NUMBER_ELECTRON_PROCESSES = 3; /*!< Maximum number of electron processes */
  __constant int NUMBER_PARTICLES = 3; /*!< Maximum number of different particles */

  __constant int PHOTON_COMPTON = 0; /*!< Compton process */
  __constant int PHOTON_PHOTOELECTRIC = 1; /*!< Photoelectric process */
  __constant int PHOTON_RAYLEIGH = 2; /*!< Rayleigh process */
  __constant int PHOTON_BONDARY_VOXEL = 3; /*!< Photon on the boundaries */

  __constant int ELECTRON_IONISATION = 4; /*!< Electron ionisation process */
  __constant int ELECTRON_MSC = 5; /*!< Electron multiple scattering process */
  __constant int ELECTRON_BREMSSTRAHLUNG = 6; /*!< Bremsstralung electron process */

  __constant int NO_PROCESS = 99; /*!< No process */
#ifdef __cplusplus
}
#endif

/*!
  \namespace ParticleName
  \brief Namespace storing particles handling by GGEMS
*/
#ifdef __cplusplus
namespace ParticleName
{
#endif
  __constant int PHOTON = 0; /*!< Photon particle */
  __constant int ELECTRON = 1; /*!< Electron particle */
  __constant int POSITRON = 2; /*!< Positron particle */
#ifdef __cplusplus
}
#endif

/*!
  \namespace ParticleState
  \brief Namespace storing the state of the particle
*/
#ifdef __cplusplus
namespace ParticleState
{
#endif
  __constant int PRIMARY = 0; /*!< Primary particle */
  __constant int GEOMETRY_BOUNDARY = 99; /*!< Particle on the boundary */
  __constant int ALIVE = 0; /*!< Particle alive */
  __constant int DEAD = 1; /*!< Particle dead */
  __constant int FREEZE = 2; /*!< Particle freeze */
#ifdef __cplusplus
}
#endif

/*!
  \namespace Tolerance
  \brief Namespace storing the tolerance for the float computations
*/
#ifdef __cplusplus
namespace Tolerance
{
#endif
  __constant double EPSILON2 = 1.0e-02; /*!< Epsilon of 0.01 */
  __constant double EPSILON3 = 1.0e-03; /*!< Epsilon of 0.001 */
  __constant double EPSILON6 = 1.0e-06; /*!< Epsilon of 0.000001 */
  __constant double GEOMETRY = 100.0*
  #ifdef __cplusplus
  Units::nm; /*!< Tolerance for the geometry navigation */
  #else
  (1.e-9 *1000.*1.0);
  #endif
#ifdef __cplusplus
}
#endif

/*!
  \namespace State
  \brief Namespace storing the state of the particle
*/
#ifdef __cplusplus
namespace State
{
#endif
  __constant int SOLID = 0; /*!< Solid state */
  __constant int GAS = 1; /*!< Gas state */
#ifdef __cplusplus
}
#endif

/*!
  \namespace Limit
  \brief Namespace storing the energy threshold
*/
#ifdef __cplusplus
namespace Limit
{
#endif
  __constant double KINETIC_ENERGY_MIN = 1.0*
  #ifdef __cplusplus
  Units::eV; /*!< Min kinetic energy */
  #else
  1.e-6*1.;
  #endif

  __constant int CROSS_SECTION_TABLE_NUMBER_BINS = 220; /*!< Number of bins in the cross section table */
  __constant double CROSS_SECTION_TABLE_ENERGY_MIN = 990.0*
  #ifdef __cplusplus
  Units::eV; /*!< Min energy in the cross section table */
  #else
  1.e-6*1.;
  #endif

  __constant double CROSS_SECTION_TABLE_ENERGY_MAX = 250.0*
  #ifdef __cplusplus
  Units::MeV; /*!< Max energy in the cross section table */
  #else
  1.;
  #endif

  __constant double PHOTON_CUT = 1.0*
  #ifdef __cplusplus
  Units::um; /*!< Photon cut */
  #else
  1.e-6 *1000.*1.0;
  #endif

  __constant double ELECTRON_CUT = 1.0*
  #ifdef __cplusplus
  Units::um; /*!< Electron cut */
  #else
  1.e-6 *1000.*1.0;
  #endif

  __constant double POSITRON_CUT = 1.0*
  #ifdef __cplusplus
  Units::um; /*!< Positron cut */
  #else
  1.e-6 *1000.*1.0;
  #endif
#ifdef __cplusplus
}
#endif

#endif // End of GUARD_GGEMS_GLOBAL_GGEMSCONSTANTS_HH
