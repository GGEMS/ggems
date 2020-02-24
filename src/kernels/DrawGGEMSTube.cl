#include "GGEMS/tools/GGEMSTypes.hh"

__kernel void draw_ggems_tube(
  GGdouble3 const element_sizes,
  GGuint3 const phantom_dimensions,
  GGdouble3 const positions,
  GGfloat const label_value,
  GGdouble const height,
  GGdouble const radius,
  __global GGfloat* voxelized_phantom)
{
  // Getting index of thread
  GGint const kGlobalIndex = get_global_id(0);

  // Get dimension of voxelized phantom
  GGuint const kX = phantom_dimensions.x;
  GGuint const kY = phantom_dimensions.y;
  GGuint const kZ = phantom_dimensions.z;

  // Get size of voxels
  GGdouble const kSizeX = element_sizes.x;
  GGdouble const kSizeY = element_sizes.y;
  GGdouble const kSizeZ = element_sizes.z;

  // Get the isocenter position of solid
  GGdouble const kPosIsoX = positions.x;
  GGdouble const kPosIsoY = positions.y;
  GGdouble const kPosIsoZ = positions.z;

  // Radius square and half of height
  GGdouble const kR2 = radius * radius;
  GGdouble const kHalfHeight = height / 2.0;

  // Get index i, j and k of current voxel
  GGuint const j = (kGlobalIndex % (kX * kY)) / kX;
  GGuint const i = (kGlobalIndex % (kX * kY)) - j * kX;
  GGuint const k = kGlobalIndex / (kX * kY);

  // Get the coordinates of the current voxel
  GGdouble x = (kSizeX / 2.0) * (1.0 - (GGdouble)kX + 2.0 * i);
  GGdouble y = (kSizeY / 2.0) * (1.0 - (GGdouble)kY + 2.0 * j);
  GGdouble z = (kSizeZ / 2.0) * (1.0 - (GGdouble)kZ + 2.0 * k);

  // Apply solid isocenter
  x -= kPosIsoX;
  y -= kPosIsoY;
  z -= kPosIsoZ;

  // Check if voxel is outside/inside analytical volume
  if (z <= kHalfHeight && z >= -kHalfHeight) {
    if (x * x + y * y <= kR2) {
      voxelized_phantom[kGlobalIndex] = label_value;
    }
  }
}
