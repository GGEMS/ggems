#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSRAYTRACING_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSRAYTRACING_HH

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
  \fn inline void TransportGetSafetyInsideAABB(GGfloat3 const* position, GGfloat const xmin, GGfloat const xmax, GGfloat const ymin, GGfloat const ymax, GGfloat const zmin, GGfloat const zmax, GGfloat const tolerance)
  \param position - pointer on primary particle position
  \param xmin - min. border in x axis
  \param xmax - max. border in x axis
  \param ymin - min. border in y axis
  \param ymax - max. border in y axis
  \param zmin - min. border in z axis
  \param zmax - max. border in z axis
  \param tolerance - tolerance for geometry
  \brief Get a safety position inside an AABB geometry
*/
inline void TransportGetSafetyInsideAABB(GGfloat3 const* position, GGfloat const xmin, GGfloat const xmax, GGfloat const ymin, GGfloat const ymax, GGfloat const zmin, GGfloat const zmax, GGfloat const tolerance)
{
  // on x
  GGfloat SafXmin = fabs(position->x - xmin);
  GGfloat SafXmax = fabs(position->x - xmax);

  position->x = (SafXmin < tolerance) ? xmin + tolerance : position->x;
  position->x = (SafXmax < tolerance) ? xmax - tolerance : position->x;

  // on y
  GGfloat SafYmin = fabs(position->y - ymin);
  GGfloat SafYmax = fabs(position->y - ymax);

  position->y = (SafYmin < tolerance) ? ymin + tolerance : position->y;
  position->y = (SafYmax < tolerance) ? ymax - tolerance : position->y;

  // on z
  GGfloat SafZmin = fabs(position->z - zmin);
  GGfloat SafZmax = fabs(position->z - zmax);

  position->z = (SafZmin < tolerance) ? zmin + tolerance : position->z;
  position->z = (SafZmax < tolerance) ? zmax - tolerance : position->z;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline void TransportGetSafetyInsideVoxelizedNavigator(GGfloat3 const* position, __global GGEMSVoxelizedSolidData* voxelized_solid_data)
  \param position - pointer on primary particle position
  \param voxelized_solid_data - voxelized data infos
  \brief Moving particle slightly inside a voxelized navigator
*/
inline void TransportGetSafetyInsideVoxelizedNavigator(GGfloat3 const* position, __global GGEMSVoxelizedSolidData* voxelized_solid_data)
{
  // Borders of voxelized solid
  GGfloat x_min = voxelized_solid_data->border_min_xyz_.x;
  GGfloat x_max = voxelized_solid_data->border_max_xyz_.x;
  GGfloat y_min = voxelized_solid_data->border_min_xyz_.y;
  GGfloat y_max = voxelized_solid_data->border_max_xyz_.y;
  GGfloat z_min = voxelized_solid_data->border_min_xyz_.z;
  GGfloat z_max = voxelized_solid_data->border_max_xyz_.z;

  // Tolerance
  GGfloat tolerance = voxelized_solid_data->tolerance_;

  TransportGetSafetyInsideAABB(position, x_min, x_max, y_min, y_max, z_min, z_max, tolerance);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline void TransportGetSafetyOutsideAABB(GGfloat3 const* position, GGfloat const xmin, GGfloat const xmax, GGfloat const ymin, GGfloat const ymax, GGfloat const zmin, GGfloat const zmax, GGfloat const tolerance)
  \param position - pointer on primary particle position
  \param xmin - min. border in x axis
  \param xmax - max. border in x axis
  \param ymin - min. border in y axis
  \param ymax - max. border in y axis
  \param zmin - min. border in z axis
  \param zmax - max. border in z axis
  \param tolerance - tolerance for geometry
  \return new position of moved particle
  \brief Get a safety position outside an AABB geometry
*/
inline void TransportGetSafetyOutsideAABB(GGfloat3 const* position, GGfloat const xmin, GGfloat const xmax, GGfloat const ymin, GGfloat const ymax, GGfloat const zmin, GGfloat const zmax, GGfloat const tolerance)
{
  // on x
  GGfloat SafXmin = fabs(position->x - xmin);
  GGfloat SafXmax = fabs(position->x - xmax);

  position->x = (SafXmin < tolerance) ? xmin - tolerance : position->x;
  position->x = (SafXmax < tolerance) ? xmax + tolerance : position->x;

  // on y
  GGfloat SafYmin = fabs(position->y - ymin);
  GGfloat SafYmax = fabs(position->y - ymax);

  position->y = (SafYmin < tolerance) ? ymin - tolerance : position->y;
  position->y = (SafYmax < tolerance) ? ymax + tolerance : position->y;

  // on z
  GGfloat SafZmin = fabs(position->z - zmin);
  GGfloat SafZmax = fabs(position->z - zmax);

  position->z = (SafZmin < tolerance) ? zmin - tolerance : position->z;
  position->z = (SafZmax < tolerance) ? zmax + tolerance : position->z;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline void TransportGetSafetyOutsideVoxelizedNavigator(GGfloat3 const* position, __global GGEMSVoxelizedSolidData* voxelized_solid_data)
  \param position - pointer on primary particle position
  \param voxelized_solid_data - voxelized data infos
  \brief Moving particle slightly outside a voxelized navigator
*/
inline void TransportGetSafetyOutsideVoxelizedNavigator(GGfloat3 const* position, __global GGEMSVoxelizedSolidData* voxelized_solid_data)
{
  // Borders of voxelized solid
  GGfloat x_min = voxelized_solid_data->border_min_xyz_.x;
  GGfloat x_max = voxelized_solid_data->border_max_xyz_.x;
  GGfloat y_min = voxelized_solid_data->border_min_xyz_.y;
  GGfloat y_max = voxelized_solid_data->border_max_xyz_.y;
  GGfloat z_min = voxelized_solid_data->border_min_xyz_.z;
  GGfloat z_max = voxelized_solid_data->border_max_xyz_.z;

  // Tolerance
  GGfloat tolerance = voxelized_solid_data->tolerance_;

  TransportGetSafetyOutsideAABB(position, x_min, x_max, y_min, y_max, z_min, z_max, tolerance);
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
  \fn inline GGfloat ComputeDistanceToAABB(GGfloat3 const* position, GGfloat3 const* direction, GGfloat const x_min, GGfloat const x_max, GGfloat const y_min, GGfloat const y_max, GGfloat const z_min, GGfloat const z_max, GGfloat const tolerance)
  \param position - pointer on primary particle position
  \param direction - pointer on primary particle direction
  \param x_min - min. border in x axis
  \param x_max - max. border in x axis
  \param y_min - min. border in y axis
  \param y_max - max. border in y axis
  \param z_min - min. border in z axis
  \param z_max - max. border in z axis
  \param tolerance - tolerance for geometry
  \return distance to AABB boundary
  \brief Get the distance to AABB boundary
*/
inline GGfloat ComputeDistanceToAABB(GGfloat3 const* position, GGfloat3 const* direction, GGfloat const x_min, GGfloat const x_max, GGfloat const y_min, GGfloat const y_max, GGfloat const z_min, GGfloat const z_max, GGfloat const tolerance)
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
  if ((tmax-tmin) < (2.0*tolerance)) return OUT_OF_WORLD;

  if (tmin <= 0.0f) return tmax;
  else return tmin;
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
  // Borders of voxelized solid
  GGfloat x_min = voxelized_solid_data->border_min_xyz_.x;
  GGfloat x_max = voxelized_solid_data->border_max_xyz_.x;
  GGfloat y_min = voxelized_solid_data->border_min_xyz_.y;
  GGfloat y_max = voxelized_solid_data->border_max_xyz_.y;
  GGfloat z_min = voxelized_solid_data->border_min_xyz_.z;
  GGfloat z_max = voxelized_solid_data->border_max_xyz_.z;

  // Tolerance
  GGfloat tolerance = voxelized_solid_data->tolerance_;

  return ComputeDistanceToAABB(position, direction, x_min, x_max, y_min, y_max, z_min, z_max, tolerance);
}

#endif

#endif // End of GUARD_GGEMS_GEOMETRIES_GGEMSRAYTRACING_HH
