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
  \file DrawGGEMSBox.cl

  \brief OpenCL kernel drawing a box in voxelized image

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday August 31, 2020
*/

#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \fn kernel void draw_ggems_box(GGint const voxel_id_limit, GGfloat3 const element_sizes, GGuint3 const phantom_dimensions, GGfloat3 const positions, GGfloat const label_value, GGfloat const height, GGfloat const width, GGfloat const depth, global GGchar* voxelized_phantom)
  \param voxel_id_limit - voxel id limit
  \param element_sizes - size of voxels
  \param phantom_dimensions - dimension of phantom
  \param positions - position of volume
  \param label_value - label of volume
  \param height - height of box
  \param width - width of box
  \param depth - depth of box
  \param voxelized_phantom - buffer storing voxelized phantom
  \brief Draw box solid in voxelized image
 */
kernel void draw_ggems_box(GGint const voxel_id_limit, GGfloat3 const element_sizes, GGuint3 const phantom_dimensions, GGfloat3 const positions, GGfloat const label_value, GGfloat const height, GGfloat const width, GGfloat const depth,
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

  // Radius square and half of height
  GGfloat half_height = height/2.0f;
  GGfloat half_width = width/2.0f;
  GGfloat half_depth = depth/2.0f;

  // Get index i, j and k of current voxel
  GGint3 indices;
  indices.y = (global_id % (phantom_dimensions.x*phantom_dimensions.y)) / phantom_dimensions.x;
  indices.x = (global_id % (phantom_dimensions.x*phantom_dimensions.y)) - indices.y*phantom_dimensions.x;
  indices.z = global_id / (phantom_dimensions.x*phantom_dimensions.y);

  // Get the coordinates of the current voxel
  GGfloat3 voxel_pos = (element_sizes/2.0f) * (1.0f - convert_float3(phantom_dimensions) + 2.0f*convert_float3(indices));
  voxel_pos -= positions;

  // Check if voxel is outside/inside analytical volume
  if (voxel_pos.z <= half_depth && voxel_pos.z >= -half_depth) {
    if (voxel_pos.y <= half_height && voxel_pos.y >= -half_height) {
      if (voxel_pos.x <= half_width && voxel_pos.x >= -half_width) {
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
}
