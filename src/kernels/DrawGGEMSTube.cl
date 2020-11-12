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
  \file DrawGGEMSTube.cl

  \brief OpenCL kernel drawing a tube in voxelized image

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday January 13, 2020
*/

#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \fn __kernel void draw_ggems_tube(GGint const voxel_id_limit, GGfloat3 const element_sizes, GGint3 const phantom_dimensions, GGfloat3 const positions, GGfloat const label_value, GGfloat const height, GGfloat const radius_x, GGfloat const radius_y, __global GGchar* voxelized_phantom)
  \param voxel_id_limit - voxel id limit
  \param element_sizes - size of voxels
  \param phantom_dimensions - dimension of phantom
  \param positions - position of volume
  \param label_value - label of volume
  \param height - height of tube
  \param radius_x - radius of tube in X axis
  \param radius_y - radius of tube in Y axis
  \param voxelized_phantom - buffer storing voxelized phantom
  \brief Draw tube solid in voxelized image
*/
kernel void draw_ggems_tube(
  GGint const voxel_id_limit,
  GGfloat3 const element_sizes,
  GGint3 const phantom_dimensions,
  GGfloat3 const positions,
  GGfloat const label_value,
  GGfloat const height,
  GGfloat const radius_x,
  GGfloat const radius_y,
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
  GGint global_id = get_global_id(0);

  // Return if index > to voxel limit
  if (global_id >= voxel_id_limit) return;

  // Get dimension of voxelized phantom
  GGint n_x = phantom_dimensions.x;
  GGint n_y = phantom_dimensions.y;
  GGint n_z = phantom_dimensions.z;

  // Get size of voxels
  GGfloat size_x = element_sizes.x;
  GGfloat size_y = element_sizes.y;
  GGfloat size_z = element_sizes.z;

  // Get the isocenter position of solid
  GGfloat isocenter_x = positions.x;
  GGfloat isocenter_y = positions.y;
  GGfloat isocenter_z = positions.z;

  // Radius square and half of height
  GGfloat radius_x2 = radius_x*radius_x;
  GGfloat radius_y2 = radius_y*radius_y;
  GGfloat half_height = height/2.0f;

  // Get index i, j and k of current voxel
  GGint j = (global_id % (n_x*n_y)) / n_x;
  GGint i = (global_id % (n_x*n_y)) - j*n_x;
  GGint k = global_id / (n_x*n_y);

  // Get the coordinates of the current voxel
  GGfloat x = (size_x/2.0f) * (1.0f - (GGfloat)n_x + 2.0f*i);
  GGfloat y = (size_y/2.0f) * (1.0f - (GGfloat)n_y + 2.0f*j);
  GGfloat z = (size_z/2.0f) * (1.0f - (GGfloat)n_z + 2.0f*k);

  // Apply solid isocenter
  x -= isocenter_x;
  y -= isocenter_y;
  z -= isocenter_z;

  // Check if voxel is outside/inside analytical volume
  if (z <= half_height && z >= -half_height) {
    if (x*x/radius_x2 + y*y/radius_y2 <= 1) {
      #ifdef MET_CHAR
      voxelized_phantom[global_id] = (GGchar)label_value;
      #elif MET_UCHAR
      voxelized_phantom[global_id] = (GGuchar)label_value;
      #elif MET_SHORT
      voxelized_phantom[global_id] = (GGshort)label_value;
      #elif MET_USHORT
      voxelized_phantom[global_id] = (GGushort)label_value;
      #elif MET_INT
      voxelized_phantom[global_id] = (GGint)label_value;
      #elif MET_UINT
      voxelized_phantom[global_id] = (GGuint)label_value;
      #elif MET_FLOAT
      voxelized_phantom[global_id] = (GGfloat)label_value;
      #endif
    }
  }
}
