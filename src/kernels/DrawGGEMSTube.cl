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
  \fn __kernel void draw_ggems_tube(GGfloat3 const element_sizes, GGuint3 const phantom_dimensions, GGfloat3 const positions, GGfloat const label_value, GGfloat const height, GGfloat const radius,  __global GGchar* voxelized_phantom)
  \param element_sizes - size of voxels
  \param phantom_dimensions - dimension of phantom
  \param positions - position of volume
  \param label_value - label of volume
  \param height - height of tube
  \param radius - radius of tube
  \param voxelized_phantom - buffer storing voxelized phantom
  \brief Draw tube solid in voxelized image
*/
__kernel void draw_ggems_tube(
  GGfloat3 const element_sizes,
  GGuint3 const phantom_dimensions,
  GGfloat3 const positions,
  GGfloat const label_value,
  GGfloat const height,
  GGfloat const radius,
  #ifdef MET_CHAR
  __global GGchar* voxelized_phantom
  #elif MET_UCHAR
  __global GGuchar* voxelized_phantom
  #elif MET_SHORT
  __global GGshort* voxelized_phantom
  #elif MET_USHORT
  __global GGushort* voxelized_phantom
  #elif MET_INT
  __global GGint* voxelized_phantom
  #elif MET_UINT
  __global GGuint* voxelized_phantom
  #elif MET_FLOAT
  __global GGfloat* voxelized_phantom
  #else
  #warning "Type Unknown, please specified a type by compiling!!!"
  #endif
)
{
  // Getting index of thread
  GGint const kVoxelID = get_global_id(0);

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
  GGfloat const kR2 = radius * radius;
  GGfloat const kHalfHeight = height / 2.0;

  // Get index i, j and k of current voxel
  GGuint const j = (kVoxelID % (kX * kY)) / kX;
  GGuint const i = (kVoxelID % (kX * kY)) - j * kX;
  GGuint const k = kVoxelID / (kX * kY);

  // Get the coordinates of the current voxel
  GGfloat x = (kSizeX / 2.0) * (1.0 - (GGfloat)kX + 2.0 * i);
  GGfloat y = (kSizeY / 2.0) * (1.0 - (GGfloat)kY + 2.0 * j);
  GGfloat z = (kSizeZ / 2.0) * (1.0 - (GGfloat)kZ + 2.0 * k);

  // Apply solid isocenter
  x -= kPosIsoX;
  y -= kPosIsoY;
  z -= kPosIsoZ;

  // Check if voxel is outside/inside analytical volume
  if (z <= kHalfHeight && z >= -kHalfHeight) {
    if (x * x + y * y <= kR2) {
      #ifdef MET_CHAR
      voxelized_phantom[kVoxelID] = (GGchar)label_value;
      #elif MET_UCHAR
      voxelized_phantom[kVoxelID] = (GGuchar)label_value;
      #elif MET_SHORT
      voxelized_phantom[kVoxelID] = (GGshort)label_value;
      #elif MET_USHORT
      voxelized_phantom[kVoxelID] = (GGushort)label_value;
      #elif MET_INT
      voxelized_phantom[kVoxelID] = (GGint)label_value;
      #elif MET_UINT
      voxelized_phantom[kVoxelID] = (GGuint)label_value;
      #elif MET_FLOAT
      voxelized_phantom[kVoxelID] = (GGfloat)label_value;
      #endif
    }
  }
}
