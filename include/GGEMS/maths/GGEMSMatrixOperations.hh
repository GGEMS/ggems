#ifndef GUARD_GGEMS_MATHS_GGEMSMATRIXOPERATIONS_HH
#define GUARD_GGEMS_MATHS_GGEMSMATRIXOPERATIONS_HH

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
  \file GGEMSMatrixOperations.hh

  \brief Definitions of functions using matrix

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday December 16, 2019
*/

#include "GGEMS/maths/GGEMSMatrixTypes.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGfloat3 RotateUnitZ(GGfloat3 vector, GGfloat3 const new_uz)
  \param vector - vector to change
  \param new_uz - new direction
  \return a vector of 3x1 float
  \brief rotateUz, function from CLHEP
*/
inline GGfloat3 RotateUnitZ(GGfloat3 vector, GGfloat3 const new_uz)
{
  #ifdef OPENCL_COMPILER
  GGfloat const u1 = new_uz.x;
  GGfloat const u2 = new_uz.y;
  GGfloat const u3 = new_uz.z;
  #else
  GGfloat const u1 = new_uz.s[0];
  GGfloat const u2 = new_uz.s[1];
  GGfloat const u3 = new_uz.s[2];
  #endif

  GGfloat up = u1*u1 + u2*u2;
  if (up > 0) {
    #ifdef OPENCL_COMPILER
    up = sqrt(up);
    GGfloat px = vector.x,  py = vector.y, pz = vector.z;
    vector.x = (u1*u3*px - u2*py) /up + u1*pz;
    vector.y = (u2*u3*px + u1*py) /up + u2*pz;
    vector.z =    -up*px +             u3*pz;
    #else
    up = sqrtf(up);
    GGfloat px = vector.s[0], py = vector.s[1], pz = vector.s[2];
    vector.s[0] = (u1*u3*px - u2*py)/up + u1*pz;
    vector.s[1] = (u2*u3*px + u1*py)/up + u2*pz;
    vector.s[2] = -up*px + u3*pz;
    #endif
  }
  else if (u3 < 0.) {
    #ifdef OPENCL_COMPILER
    vector.x = -vector.x;    // phi=0  theta=gpu_pi
    vector.z = -vector.z;
    #else
    vector.s[0] = -vector.s[0];    // phi=0  theta=gpu_pi
    vector.s[2] = -vector.s[2];
    #endif
  }

  #ifdef OPENCL_COMPILER
  return MakeFloat3(vector.x, vector.y, vector.z);
  #else
  return MakeFloat3(vector.s[0], vector.s[1], vector.s[2]);
  #endif
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGfloat3 GGfloat3UnitVector(GGfloat3 const u)
  \param u - 3D vector
  \return an unitary vector
  \brief compute an unitary vector
*/
inline GGfloat3 GGfloat3UnitVector(GGfloat3 const u)
{
  #ifdef OPENCL_COMPILER
  GGfloat norm = 1.0f / sqrt(u.x*u.x + u.y*u.y + u.z*u.z);
  return MakeFloat3(u.x*norm, u.y*norm, u.z*norm);
  #else
  GGfloat norm = 1.0f / sqrtf(u.s[0]*u.s[0] + u.s[1]*u.s[1] + u.s[2]*u.s[2]);
  return MakeFloat3(u.s[0]*norm, u.s[1]*norm, u.s[2]*norm);
  #endif
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGfloat3 GGfloat3Sub(GGfloat3 const u, GGfloat3 const v)
  \param u - 3D vector
  \param v - 3D vector
  \return the 3D vector u - v
  \brief Substract the vector v to vector u
*/
inline GGfloat3 GGfloat3Sub(GGfloat3 const u, GGfloat3 const v)
{
  GGfloat3 vector = {
    #ifdef OPENCL_COMPILER
    u.x-v.x, u.y-v.y, u.z-v.z
    #else
    {u.s[0]-v.s[0], u.s[1]-v.s[1], u.s[2]-v.s[2]}
    #endif
  };
  return vector;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGfloat3 GGfloat3Add(GGfloat3 const u, GGfloat3 const v)
  \param u - 3D vector
  \param v - 3D vector
  \return the 3D vector u + v
  \brief Add the vector v to vector u
*/
inline GGfloat3 GGfloat3Add(GGfloat3 const u, GGfloat3 const v)
{
  GGfloat3 vector = {
    #ifdef OPENCL_COMPILER
    u.x+v.x, u.y+v.y, u.z+v.z
    #else
    {u.s[0]+v.s[0], u.s[1]+v.s[1], u.s[2]+v.s[2]}
    #endif
  };
  return vector;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGfloat3 GGfloat3Scale(GGfloat3 const u, GGfloat const s)
  \param u - 3D vector
  \param s - a scalar
  \return a 3D vector u * s
  \brief Scale vector u with scalar s
*/
inline GGfloat3 GGfloat3Scale(GGfloat3 const u, GGfloat const s)
{
  GGfloat3 vector = {
    #ifdef OPENCL_COMPILER
    u.x*s, u.y*s, u.z*s
    #else
    {u.s[0]*s, u.s[1]*s, u.s[2]*s}
    #endif
  };
  return vector;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifdef OPENCL_COMPILER
/*!
  \fn inline GGfloat3 GGfloat44MultGGfloat3(__global GGfloat44 const* matrix, GGfloat3 const point)
  \param matrix - A matrix (4x4)
  \param point - Point in 3D (x, y, z)
  \return a vector 3x1
  \brief Compute the multiplication of matrix 4x4 and a point 3x1
*/
inline GGfloat3 GGfloat44MultGGfloat3(__global GGfloat44 const* matrix, GGfloat3 const point)
#else
/*!
  \fn inline GGfloat3 GGfloat44MultGGfloat3(GGfloat44 const* matrix, GGfloat3 const point)
  \param matrix - A matrix (4x4)
  \param point - Point in 3D (x, y, z)
  \return a vector 3x1
  \brief Compute the multiplication of matrix 4x4 and a point 3x1
*/
inline GGfloat3 GGfloat44MultGGfloat3(GGfloat44 const* matrix, GGfloat3 const point)
#endif
{
  #ifdef OPENCL_COMPILER
  GGfloat3 vector = {
    matrix->m00_*point.x + matrix->m01_*point.y + matrix->m02_*point.z + matrix->m03_*1.0f,
    matrix->m10_*point.x + matrix->m11_*point.y + matrix->m12_*point.z + matrix->m13_*1.0f,
    matrix->m20_*point.x + matrix->m21_*point.y + matrix->m22_*point.z + matrix->m23_*1.0f
  };
  #else
  GGfloat3 vector = {
    {matrix->m00_*point.s[0] + matrix->m01_*point.s[1] + matrix->m02_*point.s[2] + matrix->m03_*1.0f,
    matrix->m10_*point.s[0] + matrix->m11_*point.s[1] + matrix->m12_*point.s[2] + matrix->m13_*1.0f,
    matrix->m20_*point.s[0] + matrix->m21_*point.s[1] + matrix->m22_*point.s[2] + matrix->m23_*1.0f}
  };
  #endif

  return vector;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifdef OPENCL_COMPILER
/*!
 \fn inline GGfloat3 LocalToGlobalPosition(__global GGfloat44 const* matrix, GGfloat3 const point)
 \param matrix - A matrix (4x4)
 \param point - Point in 3D (x, y, z)
 \return The point expresses in the global frame
 \brief Transform a 3D point from local to global frame
*/
inline GGfloat3 LocalToGlobalPosition(__global GGfloat44 const* matrix, GGfloat3 const point)
#else
/*!
 \fn inline GGfloat3 LocalToGlobalPosition(GGfloat44 const* matrix, GGfloat3 const point)
 \param matrix - A matrix (4x4)
 \param point - Point in 3D (x, y, z)
 \return The point expresses in the global frame
 \brief Transform a 3D point from local to global frame
*/
inline GGfloat3 LocalToGlobalPosition(GGfloat44 const* matrix, GGfloat3 const point)
#endif
{
  return GGfloat44MultGGfloat3(matrix, point);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGfloat44 GGfloat44MultGGfloat44(GGfloat44 const A, GGfloat44 const B)
  \param A - first matrix
  \param B - second matrix
  \brief Perform the matrix (4x4) multiplication
  \return New matrix AxB
*/
inline GGfloat44 GGfloat44MultGGfloat44(GGfloat44 const A, GGfloat44 const B)
{
  GGfloat44 tmp = MakeFloat44Zeros();

  // Row 1
  tmp.m00_ = A.m00_*B.m00_ + A.m01_*B.m10_ + A.m02_*B.m20_ + A.m03_*B.m30_;
  tmp.m01_ = A.m00_*B.m01_ + A.m01_*B.m11_ + A.m02_*B.m21_ + A.m03_*B.m31_;
  tmp.m02_ = A.m00_*B.m02_ + A.m01_*B.m12_ + A.m02_*B.m22_ + A.m03_*B.m32_;
  tmp.m03_ = A.m00_*B.m03_ + A.m01_*B.m13_ + A.m02_*B.m23_ + A.m03_*B.m33_;

  // Row 2
  tmp.m10_ = A.m10_*B.m00_ + A.m11_*B.m10_ + A.m12_*B.m20_ + A.m13_*B.m30_;
  tmp.m11_ = A.m10_*B.m01_ + A.m11_*B.m11_ + A.m12_*B.m21_ + A.m13_*B.m31_;
  tmp.m12_ = A.m10_*B.m02_ + A.m11_*B.m12_ + A.m12_*B.m22_ + A.m13_*B.m32_;
  tmp.m13_ = A.m10_*B.m03_ + A.m11_*B.m13_ + A.m12_*B.m23_ + A.m13_*B.m33_;

  // Row 3
  tmp.m20_ = A.m20_*B.m00_ + A.m21_*B.m10_ + A.m22_*B.m20_ + A.m23_*B.m30_;
  tmp.m21_ = A.m20_*B.m01_ + A.m21_*B.m11_ + A.m22_*B.m21_ + A.m23_*B.m31_;
  tmp.m22_ = A.m20_*B.m02_ + A.m21_*B.m12_ + A.m22_*B.m22_ + A.m23_*B.m32_;
  tmp.m23_ = A.m20_*B.m03_ + A.m21_*B.m13_ + A.m22_*B.m23_ + A.m23_*B.m33_;

  // Row 4
  tmp.m30_ = A.m30_*B.m00_ + A.m31_*B.m10_ + A.m32_*B.m20_ + A.m33_*B.m30_;
  tmp.m31_ = A.m30_*B.m01_ + A.m31_*B.m11_ + A.m32_*B.m21_ + A.m33_*B.m31_;
  tmp.m32_ = A.m30_*B.m02_ + A.m31_*B.m12_ + A.m32_*B.m22_ + A.m33_*B.m32_;
  tmp.m33_ = A.m30_*B.m03_ + A.m31_*B.m13_ + A.m32_*B.m23_ + A.m33_*B.m33_;

  return tmp;
}

#endif // End of GUARD_GGEMS_MATHS_MATRIX_FUNCTIONS_HH
