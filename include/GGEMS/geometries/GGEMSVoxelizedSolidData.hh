#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSVOXELIZEDSOLIDDATA_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSVOXELIZEDSOLIDDATA_HH

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

#ifndef __OPENCL_C_VERSION__
#include <cstddef>
#endif

/*!
  \struct GGEMSVoxelizedSolidData_t
  \brief Structure storing the stack of data for voxelized solid
*/
#ifdef __OPENCL_C_VERSION__
typedef struct GGEMSVoxelizedSolidData_t
#else
// Some MSVC OpenCL headers do not align cl_float3/cl_int3 to 16 bytes.
typedef struct alignas(16) GGEMSVoxelizedSolidData_t
#endif
{
  GGEMSOBB obb_geometry_; /*!< OBB storing border of voxelized solid and matrix of transformation */
  GGfloat3 voxel_sizes_xyz_; /*!< Size of voxels in X, Y and Z */
  GGint3 number_of_voxels_xyz_; /*!< Number of voxel in X, Y and Z */
  GGint solid_id_; /*!< Navigator index */
  GGint number_of_voxels_; /*!< Total number of voxels */
  GGfloat geometry_tolerance_; /*!< Navigation tolerance in mm; initialized for each voxelized solid */
} GGEMSVoxelizedSolidData; /*!< Using C convention name of struct to C++ (_t deletion) */

#ifndef __OPENCL_C_VERSION__
static_assert(sizeof(GGEMSVoxelizedSolidData) == 144, "Voxelized solid OpenCL ABI size mismatch");
static_assert(offsetof(GGEMSVoxelizedSolidData, geometry_tolerance_) == 136, "Voxelized solid OpenCL ABI offset mismatch");
#endif

#endif // GUARD_GGEMS_GEOMETRIES_GGEMSVOXELIZEDSOLIDDATA_HH
