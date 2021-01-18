#ifndef GUARD_GGEMS_NAVIGATORS_GGEMSDOSEPARAMS_HH
#define GUARD_GGEMS_NAVIGATORS_GGEMSDOSEPARAMS_HH

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
  \file GGEMSDoseParams.hh

  \brief Structure storing dosimetry infos

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday January 18, 2021
*/

#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \struct GGEMSDoseParams_t
  \brief Structure storing dosimetry infos
*/
#pragma pack(push, 1)
typedef struct GGEMSDoseParams_t
{
  GGint3 number_of_dosels_; /*!< Number of dosels per dimension */
  GGfloat3 size_of_dosels_; /*!< Size of dosels per dimension */
  GGfloat3 inv_size_of_dosels_; /*!< Inverse of dosel sizes */
  GGint total_number_of_dosels_; /*!< Total number of dosels */
  GGint slice_number_of_dosels_; /*!< Number of dosels per slice */

  // // Offset
  // f32xyz offset;
  // // Volume Of Interest
  // f32 xmin, xmax;
  // f32 ymin, ymax;
  // f32 zmin, zmax;
} GGEMSDoseParams; /*!< Using C convention name of struct to C++ (_t deletion) */
#pragma pack(pop)

#endif // End of GUARD_GGEMS_NAVIGATORS_GGEMSDOSEPARAMS_HH
