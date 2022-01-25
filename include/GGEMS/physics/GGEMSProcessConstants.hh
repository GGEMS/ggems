#ifndef GUARD_GGEMS_PHYSICS_GGEMSPROCESSCONSTANTS_HH
#define GUARD_GGEMS_PHYSICS_GGEMSPROCESSCONSTANTS_HH

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
  \file GGEMSProcessConstants.hh

  \brief Storing some __constant variables for process

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday April 13, 2020
*/

#include "GGEMS/tools/GGEMSSystemOfUnits.hh"

__constant GGchar NUMBER_PROCESSES = 3; /*!< Maximum number of processes */

__constant GGchar NO_PROCESS = 99; /*!< No process */
__constant GGchar TRANSPORTATION = 99; /*!< Transportation process */

// PHOTON PROCESSES
#define NUMBER_PHOTON_PROCESSES 3 /*!< Number of photon processes */
__constant GGchar COMPTON_SCATTERING = 0; /*!< Compton process */
__constant GGchar PHOTOELECTRIC_EFFECT = 1; /*!< Photoelectric process */
__constant GGchar RAYLEIGH_SCATTERING = 2; /*!< Rayleigh process */

//__constant GGuchar NUMBER_ELECTRON_PROCESSES = 3; /*!< Maximum number of electron processes */
//__constant GGuchar NUMBER_PARTICLES = 5; /*!< Maximum number of different particles for secondaries */
//__constant GGuchar PHOTON_BONDARY_VOXEL = 77; /*!< Photon on the boundaries */
//__constant GGuchar ELECTRON_IONISATION = 4; /*!< Electron ionisation process */
//__constant GGuchar ELECTRON_MSC = 5; /*!< Electron multiple scattering process */
//__constant GGuchar ELECTRON_BREMSSTRAHLUNG = 6; /*!< Bremsstralung electron process */

// CROSS SECTIONS
__constant GGfloat KINETIC_ENERGY_MIN = 1.e-6f; /*!< Min kinetic energy, 1eV */
__constant GGfloat CROSS_SECTION_TABLE_ENERGY_MIN = 990.0f*1.e-6f; /*!< Min energy in the cross section table, 990 eV */
__constant GGfloat CROSS_SECTION_TABLE_ENERGY_MAX = 250.0f*1.0f; /*!< Max energy in the cross section table, 250 MeV */
#define MAX_CROSS_SECTION_TABLE_NUMBER_BINS 2048 /*!< Number of maximum bins in cross section table */
__constant GGshort CROSS_SECTION_TABLE_NUMBER_BINS = 220; /*!< Number of bins in the cross section table */

// ATTENUATIONS
__constant GGfloat ATTENUATION_ENERGY_MIN = 0.001f; /*!< Min energy for attenuation is 0.001 keV */
__constant GGfloat ATTENUATION_ENERGY_MAX = 1.0f; /*!< Max energy for attenuation is 1 MeV */
#define ATTENUATION_TABLE_NUMBER_BINS 220 /*!< Number of bins in attenuation table */

// CUTS
__constant GGfloat PHOTON_DISTANCE_CUT = 1.e-3f; /*!< Photon cut, 1 um */
__constant GGfloat ELECTRON_DISTANCE_CUT = 1.e-3f; /*!< Electron cut, 1 um */
__constant GGfloat POSITRON_DISTANCE_CUT = 1.e-3f; /*!< Positron cut, 1 um */

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSEMPROCESSCONSTANTS_HH
