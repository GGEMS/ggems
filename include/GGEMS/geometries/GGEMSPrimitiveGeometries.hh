#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSPRIMITIVEGEOMETRIES_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSPRIMITIVEGEOMETRIES_HH

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
  \file GGEMSPrimitiveGeometries.hh

  \brief Structure storing some primitive geometries

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday October 22, 2020
*/

#include "GGEMS/maths/GGEMSMatrixTypes.hh"

/*!
  \struct GGEMSOBB_t
  \brief Structure storing OBB (Oriented Bounding Box) geometry
*/
#pragma pack(push, 1)
typedef struct GGEMSOBB_t
{
  GGfloat border_min_xyz_[3]; /*!< Min. of border in X, Y and Z */
  GGfloat border_max_xyz_[3]; /*!< Max. of border in X, Y and Z */
  GGfloat44 matrix_transformation_; /*!< Matrix of transformation including angle of rotation */
} GGEMSOBB; /*!< Using C convention name of struct to C++ (_t deletion) */
#pragma pack(pop)

#endif // GUARD_GGEMS_GEOMETRIES_GGEMSPRIMITIVEGEOMETRIES_HH
