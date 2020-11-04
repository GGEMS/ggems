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
  \file DrawGGEMSSphere.cl

  \brief OpenCL kernel drawing a sphere in voxelized image

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday November 4, 2020
*/

#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \fn __kernel void draw_ggems_sphere(GGuint const voxel_id_limit, GGfloat3 const element_sizes, GGuint3 const phantom_dimensions, GGfloat3 const positions, GGfloat const label_value, GGfloat const radius,  global GGchar* voxelized_phantom)
  \param voxel_id_limit - voxel id limit
  \param element_sizes - size of voxels
  \param phantom_dimensions - dimension of phantom
  \param positions - position of volume
  \param label_value - label of volume
  \param radius - radius of tube
  \param voxelized_phantom - buffer storing voxelized phantom
  \brief Draw sphere solid in voxelized image
*/
kernel void draw_ggems_sphere(
  GGuint const voxel_id_limit,
  GGfloat3 const element_sizes,
  GGuint3 const phantom_dimensions,
  GGfloat3 const positions,
  GGfloat const label_value,
  GGfloat const radius,
  #ifdef MET_CHAR
  global GGchar* voxelized_phantom
  #elif MET_UCHAR
  global GGuchar* voxelized_phantom
  #elif MET_SHORT
  global GGshort* voxelized_phantom
  #elif MET_USHORT
  global GGushort* voxelized_phantom
  #elif MET_INT
  global GGint* voxelized_phantom
  #elif MET_UINT
  global GGuint* voxelized_phantom
  #elif MET_FLOAT
  global GGfloat* voxelized_phantom
  #else
  #warning "Type Unknown, please specified a type by compiling!!!"
  #endif
)
{
  // Getting index of thread
  GGint kGlobalVoxelID = get_global_id(0);

  // Return if index > to voxel limit
  if (kGlobalVoxelID >= voxel_id_limit) return;

  // Get dimension of voxelized phantom
  GGuint n_x = phantom_dimensions.x;
  GGuint n_y = phantom_dimensions.y;
  GGuint n_z = phantom_dimensions.z;

  // Get size of voxels
  GGfloat size_x = element_sizes.x;
  GGfloat size_y = element_sizes.y;
  GGfloat size_z = element_sizes.z;

  // Get the isocenter position of solid
  GGfloat isocenter_x = positions.x;
  GGfloat isocenter_y = positions.y;
  GGfloat isocenter_z = positions.z;

  // Radius square and half of height
  GGfloat radius2 = radius * radius;

  // Get index i, j and k of current voxel
  GGuint j = (kGlobalVoxelID % (n_x * n_y)) / n_x;
  GGuint i = (kGlobalVoxelID % (n_x * n_y)) - j * n_x;
  GGuint k = kGlobalVoxelID / (n_x * n_y);

  // Get the coordinates of the current voxel
  GGfloat x = (size_x / 2.0f) * (1.0f - (GGfloat)n_x + 2.0f * i);
  GGfloat y = (size_y / 2.0f) * (1.0f - (GGfloat)n_y + 2.0f * j);
  GGfloat z = (size_z / 2.0f) * (1.0f - (GGfloat)n_z + 2.0f * k);

  // Apply solid isocenter
  x -= isocenter_x;
  y -= isocenter_y;
  z -= isocenter_z;

  // Check if voxel is outside/inside analytical volume
  if (x*x + y*y + z*z <= radius2) {
    #ifdef MET_CHAR
    voxelized_phantom[kGlobalVoxelID] = (GGchar)label_value;
    #elif MET_UCHAR
    voxelized_phantom[kGlobalVoxelID] = (GGuchar)label_value;
    #elif MET_SHORT
    voxelized_phantom[kGlobalVoxelID] = (GGshort)label_value;
    #elif MET_USHORT
    voxelized_phantom[kGlobalVoxelID] = (GGushort)label_value;
    #elif MET_INT
    voxelized_phantom[kGlobalVoxelID] = (GGint)label_value;
    #elif MET_UINT
    voxelized_phantom[kGlobalVoxelID] = (GGuint)label_value;
    #elif MET_FLOAT
    voxelized_phantom[kGlobalVoxelID] = (GGfloat)label_value;
    #endif
  }
}
