#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSVOXELIZEDSOLIDSTACK_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSVOXELIZEDSOLIDSTACK_HH

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
  \file GGEMSVoxelizedSolidData.hh

  \brief Structure storing the stack of data for voxelized and analytical solid

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday March 2, 2020
*/

#include "GGEMS/geometries/GGEMSPrimitiveGeometries.hh"

/*!
  \struct GGEMSVoxelizedSolidData_t
  \brief Structure storing the stack of data for voxelized solid
*/
#pragma pack(push, 1)
typedef struct GGEMSVoxelizedSolidData_t
{
  GGint solid_id_; /*!< Navigator index */
  GGint number_of_voxels_; /*!< Total number of voxels */
  GGint3 number_of_voxels_xyz_; /*!< Number of voxel in X, Y and Z */
  GGfloat3 voxel_sizes_xyz_; /*!< Size of voxels in X, Y and Z */
  GGEMSOBB obb_geometry_; /*!< OBB storing border of voxelized solid and matrix of transformation */
} GGEMSVoxelizedSolidData; /*!< Using C convention name of struct to C++ (_t deletion) */
#pragma pack(pop)

#endif // GUARD_GGEMS_GEOMETRIES_GGEMSVOXELIZEDSOLIDSTACK_HH
