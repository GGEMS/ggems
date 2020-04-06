#ifndef GUARD_GGEMS_PHYSICS_GGEMSPARTICLECROSSSECTIONSSTACK_HH
#define GUARD_GGEMS_PHYSICS_GGEMSPARTICLECROSSSECTIONSSTACK_HH

/*!
  \file GGEMSParticleCrossSectionsStack.hh

  \brief Structure storing the particle (photon, electron, positron) cross sections for OpenCL device

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Friday April 3, 2020
*/

#include "GGEMS/global/GGEMSConfiguration.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \struct GGEMSParticleCrossSections_t
  \brief Structure storing the photon cross sections for OpenCL device
*/
#ifdef OPENCL_COMPILER
typedef struct __attribute__((aligned (1))) GGEMSParticleCrossSections_t
#else
#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
typedef struct PACKED GGEMSParticleCrossSections_t
#endif
{
  // Variables for all particles
  GGushort number_of_bins_; /*!< Number of bins in the cross section tables */
  GGuchar number_of_materials_; /*!< Number of materials */
  
  GGfloat min_energy_; /*!< Min energy in the cross section table */
  GGfloat max_energy_; /*!< Max energy in the cross section table */
  GGfloat energy_bins_[1024]; /*!< Energy in bin (1024 max of bin, 220 by default) */

  /////////////////
  // All cross sections are stored in a one big array for each type of particles
  /////////////////

  // Photon
  // 3: N processes with 0 Compton, 1 Photoelectric and 2 Rayleigh [0...2]
  // 256: Max number of materials [0...255]
  // 1024: Max number of bins [0...1023]
  GGuchar number_of_activated_photon_processes_; /*!< Number of activated photon processes */
  #ifdef OPENCL_COMPILER
  GGuchar index_photon_cs[NUMBER_PHOTON_PROCESSES]; /*!< Index of activated photon process, ex: if only Rayleigh activate index_photon_cs[0] = 2 */
  GGfloat photon_cross_sections_[NUMBER_PHOTON_PROCESSES][256*1024]; /*!< Photon cross sections */
  #else
  GGuchar index_photon_cs[GGEMSProcess::NUMBER_PHOTON_PROCESSES]; /*!< Index of activated photon process, ex: if only Rayleigh activate index_photon_cs[0] = 2 */
  GGfloat photon_cross_sections_[GGEMSProcess::NUMBER_PHOTON_PROCESSES][256*1024]; /*!< Photon cross sections */
  #endif

  // Electron

  // Positron
} GGEMSParticleCrossSections; /*!< Using C convention name of struct to C++ (_t deletion) */
#ifndef OPENCL_COMPILER
#ifdef _MSC_VER
#pragma pack(pop)
#endif
#endif

#endif // GUARD_GGEMS_PHYSICS_GGEMSPARTICLECROSSSECTIONSSTACK_HH
