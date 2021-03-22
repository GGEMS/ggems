#ifndef GUARD_GGEMS_MATHS_GGEMSREFERENTIALTRANSFORMATION_HH
#define GUARD_GGEMS_MATHS_GGEMSREFERENTIALTRANSFORMATION_HH

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
  \file GGEMSReferentialTransformation.hh

  \brief Definitions of functions changing referential computation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday November 5, 2020
*/

#include "GGEMS/maths/GGEMSMatrixOperations.hh"

#ifdef __OPENCL_C_VERSION__

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGfloat3 RotateUnitZ(GGfloat3* vector, GGfloat3 const* new_uz)
  \param vector - vector to change
  \param new_uz - new direction
  \return a vector of 3x1 float
  \brief rotateUz, function from CLHEP
*/
inline GGfloat3 RotateUnitZ(GGfloat3* vector, GGfloat3 const* new_uz)
{
  GGfloat u1 = new_uz->x;
  GGfloat u2 = new_uz->y;
  GGfloat u3 = new_uz->z;

  GGfloat up = u1*u1 + u2*u2;
  if (up > 0) {
    up = sqrt(up);
    GGfloat px = vector->x,  py = vector->y, pz = vector->z;
    vector->x = (u1*u3*px - u2*py) /up + u1*pz;
    vector->y = (u2*u3*px + u1*py) /up + u2*pz;
    vector->z =    -up*px +             u3*pz;
  }
  else if (u3 < 0.) {
    vector->x = -vector->x;    // phi=0  theta=gpu_pi
    vector->z = -vector->z;
  }

  GGfloat3 tmp = {vector->x, vector->y, vector->z};
  return tmp;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
 \fn inline GGfloat3 GlobalToLocalPosition(global GGfloat44 const* matrix, GGfloat3 const* point)
 \param matrix - A matrix (4x4)
 \param point - Point in 3D (x, y, z)
 \return The point expresses in the local frame
 \brief Transform a 3D point from local to global frame
*/
inline GGfloat3 GlobalToLocalPosition(global GGfloat44 const* matrix, GGfloat3 const* point)
{
  // Current point minus translation
  GGfloat3 new_point = {
    point->x - matrix->m0_[3],
    point->y - matrix->m1_[3],
    point->z - matrix->m2_[3]
  };

  return GGfloat33TransposeMultGGfloat3(matrix, &new_point);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
 \fn inline GGfloat3 LocalToGlobalPosition(global GGfloat44* matrix, GGfloat3 const* point)
 \param matrix - A matrix (4x4)
 \param point - Point in 3D (x, y, z)
 \return The point expresses in the global frame
 \brief Transform a 3D point from local to global frame
*/
inline GGfloat3 LocalToGlobalPosition(global GGfloat44 const* matrix, GGfloat3 const* point)
{
  return GGfloat44MultGGfloat3(matrix, point);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
 \fn inline GGfloat3 GlobalToLocalDirection(global GGfloat44 const* matrix, GGfloat3 const* point)
 \param matrix - A matrix (4x4)
 \param point - Point in 3D (x, y, z)
 \return The direction expresses in the global frame
 \brief Transform a 3D direction from global to local frame
*/
inline GGfloat3 GlobalToLocalDirection(global GGfloat44 const* matrix, GGfloat3 const* point)
{
  return normalize(GGfloat33TransposeMultGGfloat3(matrix, point));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
 \fn inline GGfloat3 LocalToGlobalDirection(global GGfloat44 const* matrix, GGfloat3 const* point)
 \param matrix - A matrix (4x4)
 \param point - Point in 3D (x, y, z)
 \return The direction expresses in the local frame
 \brief Transform a 3D direction from local to global frame
*/
inline GGfloat3 LocalToGlobalDirection(global GGfloat44 const* matrix, GGfloat3 const* point)
{
  return normalize(GGfloat33MultGGfloat3(matrix, point));
}

#endif

#endif // End of GUARD_GGEMS_MATHS_GGEMSREFERENTIALTRANSFORMATION_HH
