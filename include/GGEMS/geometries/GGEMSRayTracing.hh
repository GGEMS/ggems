#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSRAYTRACING_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSRAYTRACING_HH

/*!
  \file GGEMSRayTracing.hh

  \brief Functions for ray tracing computation used as auxiliary functions in OpenCL

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday May 25, 2020
*/

#ifdef OPENCL_COMPILER

#include "GGEMS/physics/GGEMSParticleConstants.hh"
#include "GGEMS/geometries/GGEMSGeometryConstants.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn GGfloat3 TransportGetSafetyInsideVoxelizedNavigator(GGfloat3 const* position, __global GGEMSVoxelizedSolidData* voxelized_solid_data)
  \param position - pointer on primary particle
  \param voxelized_solid_data - voxelized data infos
  \return new postion of moved particle
  \brief Moving particle slightly inside a voxelized navigator
*/
// Get a safety position inside an AABB geometry
GGfloat3 TransportGetSafetyInsideVoxelizedNavigator(GGfloat3 const* position, __global GGEMSVoxelizedSolidData* voxelized_solid_data)
{
  // Getting positions
  GGfloat3 new_position = {position->x, position->y, position->z};

  // Borders of voxelized solid
  GGfloat const x_min = voxelized_solid_data->border_min_xyz_.x;
  GGfloat const x_max = voxelized_solid_data->border_max_xyz_.x;
  GGfloat const y_min = voxelized_solid_data->border_min_xyz_.y;
  GGfloat const y_max = voxelized_solid_data->border_max_xyz_.y;
  GGfloat const z_min = voxelized_solid_data->border_min_xyz_.z;
  GGfloat const z_max = voxelized_solid_data->border_max_xyz_.z;

  // Tolerance
  GGfloat const kTolerance = voxelized_solid_data->tolerance_;

  // on x
  GGfloat SafXmin = fabs(new_position.x - x_min);
  GGfloat SafXmax = fabs(new_position.x - x_max);

  new_position.x = (SafXmin < kTolerance) ? x_min + kTolerance : new_position.x;
  new_position.x = (SafXmax < kTolerance) ? x_max - kTolerance : new_position.x;

  // on y
  GGfloat SafYmin = fabs(new_position.y - y_min);
  GGfloat SafYmax = fabs(new_position.y - y_max);

  new_position.y = (SafYmin < kTolerance) ? y_min + kTolerance : new_position.y;
  new_position.y = (SafYmax < kTolerance) ? y_max - kTolerance : new_position.y;

  // on z
  GGfloat SafZmin = fabs(new_position.z - z_min);
  GGfloat SafZmax = fabs(new_position.z - z_max);

  new_position.z = (SafZmin < kTolerance) ? z_min + kTolerance : new_position.z;
  new_position.z = (SafZmax < kTolerance) ? z_max - kTolerance : new_position.z;

  return new_position;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGuchar IsParticleInVoxelizedNavigator(GGfloat3 const* position, __global GGEMSVoxelizedSolidData* voxelized_solid_data)
  \param position - pointer on primary particle
  \param voxelized_solid_data - voxelized data infos
  \return false if particle outside voxelized navigator, and true if particle inside voxelized navigator
  \brief Check if particle is inside or outside voxelized navigator
*/
inline GGuchar IsParticleInVoxelizedNavigator(GGfloat3 const* position, __global GGEMSVoxelizedSolidData* voxelized_solid_data)
{
  // Getting tolerance
  GGdouble const kTolerance = voxelized_solid_data->tolerance_;
  if (position->x < (voxelized_solid_data->border_min_xyz_.x + kTolerance) || position->x > (voxelized_solid_data->border_max_xyz_.x - kTolerance)) return OPENCL_FALSE;
  if (position->y < (voxelized_solid_data->border_min_xyz_.y + kTolerance) || position->y > (voxelized_solid_data->border_max_xyz_.y - kTolerance)) return OPENCL_FALSE;
  if (position->z < (voxelized_solid_data->border_min_xyz_.z + kTolerance) || position->z > (voxelized_solid_data->border_max_xyz_.z - kTolerance)) return OPENCL_FALSE;

  return OPENCL_TRUE;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGfloat ComputeDistanceToVoxelizedNavigator(GGfloat3 const* position, GGfloat3 const* direction, __global GGEMSVoxelizedSolidData* voxelized_solid_data)
  \param position - pointer on position of primary particle
  \param direction - pointer on direction of primary particle
  \param voxelized_solid_data - voxelized data infos
  \return distance to navigator
  \brief Compute the distance between particle and voxelized navigator using Smits algorithm
*/
inline GGfloat ComputeDistanceToVoxelizedNavigator(GGfloat3 const* position, GGfloat3 const* direction, __global GGEMSVoxelizedSolidData* voxelized_solid_data)
{
  // Variables for algorithm
  GGfloat idx = 0.0f;
  GGfloat idy = 0.0f;
  GGfloat idz = 0.0f;
  GGfloat tmp = 0.0f;
  GGfloat tmin = FLT_MIN;
  GGfloat tmax = FLT_MAX;
  GGfloat tymin = 0.0f;
  GGfloat tymax = 0.0f;
  GGfloat tzmin = 0.0f;
  GGfloat tzmax = 0.0f;

  // Getting positions
  GGfloat const pos_x = position->x;
  GGfloat const pos_y = position->y;
  GGfloat const pos_z = position->z;

  // Getting directions
  GGfloat const dir_x = direction->x;
  GGfloat const dir_y = direction->y;
  GGfloat const dir_z = direction->z;

  // Borders of voxelized solid
  GGfloat const x_min = voxelized_solid_data->border_min_xyz_.x;
  GGfloat const x_max = voxelized_solid_data->border_max_xyz_.x;
  GGfloat const y_min = voxelized_solid_data->border_min_xyz_.y;
  GGfloat const y_max = voxelized_solid_data->border_max_xyz_.y;
  GGfloat const z_min = voxelized_solid_data->border_min_xyz_.z;
  GGfloat const z_max = voxelized_solid_data->border_max_xyz_.z;

  // On X axis
  if (fabs(dir_x) < EPSILON6) {
    if (pos_x < x_min || pos_x > x_max) return OUT_OF_WORLD;
  }
  else {
    idx = 1.0f / dir_x;
    tmin = (x_min - pos_x) * idx;
    tmax = (x_max - pos_x) * idx;
    if (tmin > tmax) {
      tmp = tmin;
      tmin = tmax;
      tmax = tmp;
    }
    if (tmin > tmax) return OUT_OF_WORLD;
  }

  // On Y axis
  if (fabs(dir_y) < EPSILON6) {
    if (pos_y < y_min || pos_y > y_max) return OUT_OF_WORLD;
  }
  else {
    idy = 1.0f / dir_y;
    tymin = (y_min - pos_y) * idy;
    tymax = (y_max - pos_y) * idy;
    if (tymin > tymax) {
      tmp = tymin;
      tymin = tymax;
      tymax = tmp;
    }
    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;
    if (tmin > tmax) return OUT_OF_WORLD;
  }

  // On Z axis
  if (fabs(dir_z) < EPSILON6) {
    if (pos_z < z_min || pos_z > z_max) return OUT_OF_WORLD;
  }
  else {
    idz = 1.0f / dir_z;
    tzmin = (z_min - pos_z) * idz;
    tzmax = (z_max - pos_z) * idz;
    if (tzmin > tzmax) {
      tmp = tzmin;
      tzmin = tzmax;
      tzmax = tmp;
    }
    if (tzmin > tmin) tmin = tzmin;
    if (tzmax < tmax) tmax = tzmax;
    if (tmin > tmax) return OUT_OF_WORLD;
  }

  // Return the smaller positive value diff to zero
  if (tmin < 0.0f && (tmax < 0.0f || tmax == 0.0f)) return OUT_OF_WORLD;

  // Checking if particle cross navigator sufficiently
  if ((tmax-tmin) < (2.0*voxelized_solid_data->tolerance_)) return OUT_OF_WORLD;

  if (tmin <= 0.0f) return tmax;
  else return tmin;
}

#endif

#endif // End of GUARD_GGEMS_GEOMETRIES_GGEMSRAYTRACING_HH
