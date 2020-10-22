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
  \file GGEMSVoxelizedSolidStack.hh

  \brief Structure storing the stack of data for voxelized and analytical solid

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday March 2, 2020
*/

#include "GGEMS/global/GGEMSConfiguration.hh"
#include "GGEMS/tools/GGEMSTypes.hh"
#include "GGEMS/geometries/GGEMSPrimitiveGeometriesStack.hh"

/*!
  \struct GGEMSVoxelizedSolidData_t
  \brief Structure storing the stack of data for voxelized solid
*/
#ifdef __OPENCL_C_VERSION__
typedef struct __attribute__((aligned (1))) GGEMSVoxelizedSolidData_t
#else
typedef struct PACKED GGEMSVoxelizedSolidData_t
#endif
{
  GGushort3 number_of_voxels_xyz_; /*!< Number of voxel in X, Y and Z [0, 65535] */
  GGuint number_of_voxels_; /*!< Total number of voxels */
  GGfloat3 voxel_sizes_xyz_; /*!< Size of voxels in X, Y and Z */
  GGfloat3 position_xyz_; /*!< Position of phantom in X, Y and Z */
  GGEMSOBB obb_geometry_; /*!< OBB storing border of voxelized solid and matrix of transformation */
  GGint solid_id_; /*!< Navigator index */
} GGEMSVoxelizedSolidData; /*!< Using C convention name of struct to C++ (_t deletion) */

#endif // GUARD_GGEMS_GEOMETRIES_GGEMSVOXELIZEDSOLIDSTACK_HH
