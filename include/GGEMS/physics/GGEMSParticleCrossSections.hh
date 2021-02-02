#ifndef GUARD_GGEMS_PHYSICS_GGEMSPARTICLECROSSSECTIONS_HH
#define GUARD_GGEMS_PHYSICS_GGEMSPARTICLECROSSSECTIONS_HH

// ************************************************************************
// * This file is part of GGEMS.                                          *
// *                                                                      *
// * GGEMS is free software: you can redistribute it and/or modify        *
// * it under the terms of the GNU General Public License as published by *
// * the Free Software Foundation, either version 3 of the License, or    *
// * (at your option) any later version.                                  *
// *                                                                      *
// * GGEMS is distributed in the hope that it will be useful,             *
// * but WITHOUT ANY WARRANTY; without even the implied warranty of       *
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        *
// * GNU General Public License for more details.                         *
// *                                                                      *
// * You should have received a copy of the GNU General Public License    *
// * along with GGEMS.  If not, see <https://www.gnu.org/licenses/>.      *
// *                                                                      *
// ************************************************************************

/*!
  \file GGEMSParticleCrossSections.hh

  \brief Structure storing the particle (photon, electron, positron) cross sections for OpenCL device

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Friday April 3, 2020
*/

#include "GGEMS/physics/GGEMSProcessConstants.hh"

/*!
  \struct GGEMSParticleCrossSections_t
  \brief Structure storing the photon cross sections for OpenCL device
*/
typedef struct GGEMSParticleCrossSections_t
{
  // Variables for all particles
  GGsize number_of_bins_; /*!< Number of bins in the cross section tables */
  GGsize number_of_materials_; /*!< Number of materials */
  GGfloat min_energy_; /*!< Min energy in the cross section table */
  GGfloat max_energy_; /*!< Max energy in the cross section table */
  GGfloat energy_bins_[MAX_CROSS_SECTION_TABLE_NUMBER_BINS]; /*!< Energy in bin (220 by default) */

  // Photon
  // 256: Max number of materials [0...255]
  // MAX_CROSS_SECTION_TABLE_NUMBER_BINS: Max number of bins [0...2047]
  GGfloat photon_cross_sections_[NUMBER_PHOTON_PROCESSES][256*MAX_CROSS_SECTION_TABLE_NUMBER_BINS]; /*!< Photon cross sections per material in mm-1 */
  GGfloat photon_cross_sections_per_atom_[NUMBER_PHOTON_PROCESSES][101*MAX_CROSS_SECTION_TABLE_NUMBER_BINS]; /*!< Photon cross sections per atom in mm-1, 100 chemical elements + 1 first empty element */
  GGsize number_of_activated_photon_processes_; /*!< Number of activated photon processes, 3 processes -> 0: Compton, 1: Photoelectric, 2: Rayleigh */
  GGchar photon_cs_id_[NUMBER_PHOTON_PROCESSES]; /*!< Index of activated photon process, ex: if only Rayleigh activate index_photon_cs[0] = 2 */

  GGchar material_names_[256][64]; /*!< Name of the materials */
} GGEMSParticleCrossSections; /*!< Using C convention name of struct to C++ (_t deletion) */

#endif // GUARD_GGEMS_PHYSICS_GGEMSPARTICLECROSSSECTIONS_HH
