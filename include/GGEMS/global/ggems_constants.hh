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
namespace ProcessName
{
  inline static constexpr int NUMBER_PROCESSES = 7; /*!< Maximum number of processes */
  inline static constexpr int NUMBER_PHOTON_PROCESSES = 3; /*!< Maximum number of photon processes */
  inline static constexpr int NUMBER_ELECTRON_PROCESSES = 3; /*!< Maximum number of electron processes */
  inline static constexpr int NUMBER_PARTICLES = 3; /*!< Maximum number of different particles */

  inline static constexpr int PHOTON_COMPTON = 0; /*!< Compton process */
  inline static constexpr int PHOTON_PHOTOELECTRIC = 1; /*!< Photoelectric process */
  inline static constexpr int PHOTON_RAYLEIGH = 2; /*!< Rayleigh process */
  inline static constexpr int PHOTON_BONDARY_VOXEL = 3; /*!< Photon on the boundaries */

  inline static constexpr int ELECTRON_IONISATION = 4; /*!< Electron ionisation process */
  inline static constexpr int ELECTRON_MSC = 5; /*!< Electron multiple scattering process */
  inline static constexpr int ELECTRON_BREMSSTRAHLUNG = 6; /*!< Bremsstralung electron process */

  inline static constexpr int NO_PROCESS = 99; /*!< No process */
}

/*!
  \namespace ParticleName
  \brief Namespace storing particles handling by GGEMS
*/
namespace ParticleName
{
  inline static constexpr int PHOTON = 0; /*!< Photon particle */
  inline static constexpr int ELECTRON = 1; /*!< Electron particle */
  inline static constexpr int POSITRON = 2; /*!< Positron particle */
}

/*!
  \namespace ParticleState
  \brief Namespace storing the state of the particle
*/
namespace ParticleState
{
  inline static constexpr int PRIMARY = 0; /*!< Primary particle */
  inline static constexpr int GEOMETRY_BOUNDARY = 99; /*!< Particle on the boundary */
  inline static constexpr int ALIVE = 0; /*!< Particle alive */
  inline static constexpr int DEAD = 1; /*!< Particle dead */
  inline static constexpr int FREEZE = 2; /*!< Particle freeze */
}

/*!
  \namespace Tolerance
  \brief Namespace storing the tolerance for the float computations
*/
namespace Tolerance
{
  inline static constexpr double EPSILON2 = 1.0e-02; /*!< Epsilon of 0.01 */
  inline static constexpr double EPSILON3 = 1.0e-03; /*!< Epsilon of 0.001 */
  inline static constexpr double EPSILON6 = 1.0e-06; /*!< Epsilon of 0.000001 */
  inline static constexpr double GEOMETRY = 100.0*Units::nm; /*!< Tolerance for the geometry navigation */
}

/*!
  \namespace State
  \brief Namespace storing the state of the particle
*/
namespace State
{
  inline static constexpr int SOLID = 0; /*!< Solid state */
  inline static constexpr int GAS = 1; /*!< Gas state */
}

/*!
  \namespace Limit
  \brief Namespace storing the energy threshold
*/
namespace Limit
{
  inline static constexpr double KINETIC_ENERGY_MIN = 1.0*Units::eV; /*!< Min kinetic energy */
  inline static constexpr int CROSS_SECTION_TABLE_NUMBER_BINS = 220; /*!< Number of bins in the cross section table */
  inline static constexpr double CROSS_SECTION_TABLE_ENERGY_MIN =
    990.0*Units::eV; /*!< Min energy in the cross section table */
  inline static constexpr double CROSS_SECTION_TABLE_ENERGY_MAX =
    250.0*Units::MeV; /*!< Max energy in the cross section table */
  inline static constexpr double PHOTON_CUT = 1.0*Units::um; /*!< Photon cut */
  inline static constexpr double ELECTRON_CUT = 1.0*Units::um; /*!< Electron cut */
  inline static constexpr double POSITRON_CUT = 1.0*Units::um; /*!< Positron cut */
}

#endif // End of GUARD_GGEMS_GLOBAL_GGEMSCONSTANTS_HH
