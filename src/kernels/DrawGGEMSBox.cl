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
  \fn __kernel void draw_ggems_box(GGfloat3 const element_sizes, GGuint3 const phantom_dimensions, GGfloat3 const positions, GGfloat const label_value, GGfloat const height, GGfloat const radius,  __global GGchar* voxelized_phantom)
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
kernel void draw_ggems_box(
  GGuint const thread_id_limit,
  GGfloat3 const element_sizes,
  GGuint3 const phantom_dimensions,
  GGfloat3 const positions,
  GGfloat const label_value,
  GGfloat const height,
  GGfloat const width,
  GGfloat const depth,
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
  GGint const kGlobalVoxelID = get_global_id(0);

  // Return if index > to thread limit
  if (kGlobalVoxelID >= thread_id_limit) return;

  // Get dimension of voxelized phantom
  GGuint const kX = phantom_dimensions.x;
  GGuint const kY = phantom_dimensions.y;
  GGuint const kZ = phantom_dimensions.z;

  // Get size of voxels
  GGfloat const kSizeX = element_sizes.x;
  GGfloat const kSizeY = element_sizes.y;
  GGfloat const kSizeZ = element_sizes.z;

  // Get the isocenter position of solid
  GGfloat const kPosIsoX = positions.x;
  GGfloat const kPosIsoY = positions.y;
  GGfloat const kPosIsoZ = positions.z;

  // Radius square and half of height
  GGfloat const kHalfHeight = height / 2.0f;
  GGfloat const kHalfWidth = width / 2.0f;
  GGfloat const kHalfDepth = depth / 2.0f;

  // Get index i, j and k of current voxel
  GGuint const j = (kGlobalVoxelID % (kX * kY)) / kX;
  GGuint const i = (kGlobalVoxelID % (kX * kY)) - j * kX;
  GGuint const k = kGlobalVoxelID / (kX * kY);

  // Get the coordinates of the current voxel
  GGfloat x = (kSizeX / 2.0f) * (1.0f - (GGfloat)kX + 2.0f * i);
  GGfloat y = (kSizeY / 2.0f) * (1.0f - (GGfloat)kY + 2.0f * j);
  GGfloat z = (kSizeZ / 2.0f) * (1.0f - (GGfloat)kZ + 2.0f * k);

  // Apply solid isocenter
  x -= kPosIsoX;
  y -= kPosIsoY;
  z -= kPosIsoZ;

  // Check if voxel is outside/inside analytical volume
  if (z <= kHalfDepth && z >= -kHalfDepth) {
    if (y <= kHalfHeight && y >= -kHalfHeight) {
      if (x <= kHalfWidth && x >= -kHalfWidth) {
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
  }
}
