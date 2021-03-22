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

#ifdef __OPENCL_C_VERSION__

#include "GGEMS/physics/GGEMSParticleConstants.hh"

#include "GGEMS/geometries/GGEMSGeometryConstants.hh"
#include "GGEMS/geometries/GGEMSVoxelizedSolidData.hh"

#include "GGEMS/maths/GGEMSMatrixOperations.hh"
#include "GGEMS/maths/GGEMSReferentialTransformation.hh"

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
  GGfloat safmin = fabs(position->x - xmin);
  GGfloat safmax = fabs(position->x - xmax);

  position->x = (safmin < tolerance) ? xmin + tolerance : position->x;
  position->x = (safmax < tolerance) ? xmax - tolerance : position->x;

  // on y
  safmin = fabs(position->y - ymin);
  safmax = fabs(position->y - ymax);

  position->y = (safmin < tolerance) ? ymin + tolerance : position->y;
  position->y = (safmax < tolerance) ? ymax - tolerance : position->y;

  // on z
  safmin = fabs(position->z - zmin);
  safmax = fabs(position->z - zmax);

  position->z = (safmin < tolerance) ? zmin + tolerance : position->z;
  position->z = (safmax < tolerance) ? zmax - tolerance : position->z;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline void TransportGetSafetyInsideOBB(GGfloat3 const* position, global GGEMSOBB* obb_data)
  \param position - pointer on primary particle position
  \param obb_data - OBB data infos
  \brief Moving particle slightly inside a OBB solid
*/
inline void TransportGetSafetyInsideOBB(GGfloat3 const* position, global GGEMSOBB* obb_data)
{
  // Get the position in local position
  GGfloat3 local_position = GlobalToLocalPosition(&obb_data->matrix_transformation_, position);

  TransportGetSafetyInsideAABB(
    &local_position,
    obb_data->border_min_xyz_.x, obb_data->border_max_xyz_.x,
    obb_data->border_min_xyz_.y, obb_data->border_max_xyz_.y,
    obb_data->border_min_xyz_.z, obb_data->border_max_xyz_.z,
    GEOMETRY_TOLERANCE
  );
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
  GGfloat safmin = fabs(position->x - xmin);
  GGfloat safmax = fabs(position->x - xmax);

  position->x = (safmin < tolerance) ? xmin - tolerance : position->x;
  position->x = (safmax < tolerance) ? xmax + tolerance : position->x;

  // on y
  safmin = fabs(position->y - ymin);
  safmax = fabs(position->y - ymax);

  position->y = (safmin < tolerance) ? ymin - tolerance : position->y;
  position->y = (safmax < tolerance) ? ymax + tolerance : position->y;

  // on z
  safmin = fabs(position->z - zmin);
  safmax = fabs(position->z - zmax);

  position->z = (safmin < tolerance) ? zmin - tolerance : position->z;
  position->z = (safmax < tolerance) ? zmax + tolerance : position->z;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGuchar IsParticleInAABB(GGfloat3 const* position, GGfloat const x_min, GGfloat const x_max, GGfloat const y_min, GGfloat const y_max, GGfloat const z_min, GGfloat const z_max, GGfloat const tolerance)
  \param position - pointer on primary particle
  \param x_min - min. border in x axis
  \param x_max - max. border in x axis
  \param y_min - min. border in y axis
  \param y_max - max. border in y axis
  \param z_min - min. border in z axis
  \param z_max - max. border in z axis
  \param tolerance - tolerance for geometry
  \return false if particle outside AABB object, and true if particle inside AABB object
  \brief Check if particle is inside or outside AABB object
*/
inline GGchar IsParticleInAABB(GGfloat3 const* position, GGfloat const x_min, GGfloat const x_max, GGfloat const y_min, GGfloat const y_max, GGfloat const z_min, GGfloat const z_max, GGfloat const tolerance)
{
  if (position->s0 < (x_min + GEOMETRY_TOLERANCE) || position->s0 > (x_max - GEOMETRY_TOLERANCE)) return FALSE;
  if (position->s1 < (y_min + GEOMETRY_TOLERANCE) || position->s1 > (y_max - GEOMETRY_TOLERANCE)) return FALSE;
  if (position->s2 < (z_min + GEOMETRY_TOLERANCE) || position->s2 > (z_max - GEOMETRY_TOLERANCE)) return FALSE;

  return TRUE;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGuchar IsParticleInOBB(GGfloat3 const* position, global GGEMSOBB* obb_data)
  \param position - pointer on primary particle
  \param obb_data - OBB data infos
  \return false if particle outside OBB object, and true if particle inside OBB object
  \brief Check if particle is inside or outside OBB object
*/
inline GGchar IsParticleInOBB(GGfloat3 const* position, global GGEMSOBB* obb_data)
{
  // Get the position in local position
  GGfloat3 local_position = GlobalToLocalPosition(&obb_data->matrix_transformation_, position);

  return IsParticleInAABB(
    &local_position,
    obb_data->border_min_xyz_.x, obb_data->border_max_xyz_.x,
    obb_data->border_min_xyz_.y, obb_data->border_max_xyz_.y,
    obb_data->border_min_xyz_.z, obb_data->border_max_xyz_.z,
    GEOMETRY_TOLERANCE
  );
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

  // On X axis
  if (fabs(direction->x) < EPSILON6) {
    if (position->x < x_min || position->x > x_max) return OUT_OF_WORLD;
  }
  else {
    idx = 1.0f / direction->x;
    tmin = (x_min - position->x) * idx;
    tmax = (x_max - position->x) * idx;
    if (tmin > tmax) {
      tmp = tmin;
      tmin = tmax;
      tmax = tmp;
    }
    if (tmin > tmax) return OUT_OF_WORLD;
  }

  // On Y axis
  if (fabs(direction->y) < EPSILON6) {
    if (position->y < y_min || position->y > y_max) return OUT_OF_WORLD;
  }
  else {
    idy = 1.0f / direction->y;
    tymin = (y_min - position->y) * idy;
    tymax = (y_max - position->y) * idy;

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
  if (fabs(direction->z) < EPSILON6) {
    if (position->z < z_min || position->z > z_max) return OUT_OF_WORLD;
  }
  else {
    idz = 1.0f / direction->z;
    tzmin = (z_min - position->z) * idz;
    tzmax = (z_max - position->z) * idz;
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

  if (tmin <= EPSILON6) return tmax;
  else return tmin;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGfloat ComputeDistanceToOBB(GGfloat3 const* position, GGfloat3 const* direction, global GGEMSOBB const* obb_data)
  \param position - pointer on position of primary particle
  \param direction - pointer on direction of primary particle
  \param obb_data - OBB data infos
  \return distance to OBB solid
  \brief Compute the distance between particle and OBB using Smits algorithm
*/
inline GGfloat ComputeDistanceToOBB(GGfloat3 const* position, GGfloat3 const* direction, global GGEMSOBB const* obb_data)
{
  // Get the position in local position
  GGfloat3 local_position = GlobalToLocalPosition(&obb_data->matrix_transformation_, position);
  GGfloat3 local_direction = GlobalToLocalDirection(&obb_data->matrix_transformation_, direction);

  return ComputeDistanceToAABB(
    &local_position,
    &local_direction,
    obb_data->border_min_xyz_.x, obb_data->border_max_xyz_.x,
    obb_data->border_min_xyz_.y, obb_data->border_max_xyz_.y,
    obb_data->border_min_xyz_.z, obb_data->border_max_xyz_.z,
    GEOMETRY_TOLERANCE
  );
}

#endif

#endif // End of GUARD_GGEMS_GEOMETRIES_GGEMSRAYTRACING_HH
