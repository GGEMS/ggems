#ifndef GUARD_GGEMS_PHYSICS_GGEMSMUDATA_HH
#define GUARD_GGEMS_PHYSICS_GGEMSMUDATA_HH

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
  \file GGEMSMuData.hh

  \brief Structure storing the attenuation and energy-absorption coefficient values for OpenCL device

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author Mateo VILLA <ingmatvillaa@gmail.com>

  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \date Monday June 10, 2020
*/

#include "GGEMS/tools/GGEMSSystemOfUnits.hh"
#include "GGEMS/physics/GGEMSProcessConstants.hh"

/*!
  \struct GGEMSMuMuEnData_t
  \brief Mu and Mu_en table used by TLE
*/
typedef struct GGEMSMuMuEnData_t
{
  GGfloat energy_bins_[MAX_CROSS_SECTION_TABLE_NUMBER_BINS]; /*! Number of energy bins */
  GGfloat mu_[256*MAX_CROSS_SECTION_TABLE_NUMBER_BINS]; /*!< attenuation coefficient values for each material (n*k) */
  GGfloat mu_en_[256*MAX_CROSS_SECTION_TABLE_NUMBER_BINS]; /*!< energy-absorption coefficient for each material (n*k) */

  GGint number_of_materials_; /*!< Number of materials : k */
  GGint number_of_bins_; /*!< Number of bins : n */

  GGfloat energy_min_; /*!< Minimum of energy */
  GGfloat energy_max_; /*!< Maximum of energy */
} GGEMSMuMuEnData;

#endif
