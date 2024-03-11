#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSMESHEDSOLIDDATA_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSMESHEDSOLIDDATA_HH

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
  \file GGEMSMeshedSolidData.hh

  \brief Structure storing the stack of data for meshed solid

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday June 28, 2020
*/

#include "GGEMS/geometries/GGEMSPrimitiveGeometries.hh"

/*!
  \struct GGEMSMeshedSolidData_t
  \brief Structure storing the stack of data for meshed solid
*/
typedef struct GGEMSMeshedSolidData_t
{
  //GGEMSOBB obb_geometry_; /*!< OBB storing border of meshed solid and matrix of transformation */
  //GGfloat3 voxel_sizes_xyz_; /*!< Size of voxels in X, Y and Z */
  //GGint3 number_of_voxels_xyz_; /*!< Number of voxel in X, Y and Z */
  GGint solid_id_; /*!< Navigator index */
  //GGint number_of_voxels_; /*!< Total number of voxels */
} GGEMSMeshedSolidData; /*!< Using C convention name of struct to C++ (_t deletion) */

#endif // GUARD_GGEMS_GEOMETRIES_GGEMSMESHEDSOLIDDATA_HH
