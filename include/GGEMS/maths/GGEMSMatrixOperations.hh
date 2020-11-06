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

#ifdef __OPENCL_C_VERSION__

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
// inline GGfloat3 GGfloat3Sub(GGfloat3 const u, GGfloat3 const v)
// {
//   GGfloat3 vector = {
//     #ifdef __OPENCL_C_VERSION__
//     u.x-v.x, u.y-v.y, u.z-v.z
//     #else
//     {u.s[0]-v.s[0], u.s[1]-v.s[1], u.s[2]-v.s[2]}
//     #endif
//   };
//   return vector;
// }

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
// inline GGfloat3 GGfloat3Add(GGfloat3 const u, GGfloat3 const v)
// {
//   GGfloat3 vector = {
//     #ifdef __OPENCL_C_VERSION__
//     u.x+v.x, u.y+v.y, u.z+v.z
//     #else
//     {u.s[0]+v.s[0], u.s[1]+v.s[1], u.s[2]+v.s[2]}
//     #endif
//   };
//   return vector;
// }

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
// inline GGfloat3 GGfloat3Scale(GGfloat3 const u, GGfloat const s)
// {
//   GGfloat3 vector = {
//     #ifdef __OPENCL_C_VERSION__
//     u.x*s, u.y*s, u.z*s
//     #else
//     {u.s[0]*s, u.s[1]*s, u.s[2]*s}
//     #endif
//   };
//   return vector;
// }

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGfloat3 GGfloat44MultGGfloat3(global GGfloat44 const* matrix, GGfloat3 const point)
  \param matrix - A matrix (4x4)
  \param point - Point in 3D (x, y, z)
  \return a vector 3x1
  \brief Compute the multiplication of matrix 4x4 and a point 3x1
*/
inline GGfloat3 GGfloat44MultGGfloat3(global GGfloat44 const* matrix, GGfloat3 const point)
{
  GGfloat4 point4D = {point.x, point.y, point.z, 1.0f};
  GGfloat4 row0 = {matrix->m0_[0], matrix->m0_[1], matrix->m0_[2], matrix->m0_[3]};
  GGfloat4 row1 = {matrix->m1_[0], matrix->m1_[1], matrix->m1_[2], matrix->m1_[3]};
  GGfloat4 row2 = {matrix->m2_[0], matrix->m2_[1], matrix->m2_[2], matrix->m2_[3]};

  GGfloat3 vector = {
    dot(row0, point4D),
    dot(row1, point4D),
    dot(row2, point4D)
  };

  return vector;
}

#endif

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
  GGfloat44 tmp;

  // Row 1
  tmp.m0_[0] = A.m0_[0]*B.m0_[0] + A.m0_[1]*B.m1_[0] + A.m0_[2]*B.m2_[0] + A.m0_[3]*B.m3_[0];
  tmp.m0_[1] = A.m0_[0]*B.m0_[1] + A.m0_[1]*B.m1_[1] + A.m0_[2]*B.m2_[1] + A.m0_[3]*B.m3_[1];
  tmp.m0_[2] = A.m0_[0]*B.m0_[2] + A.m0_[1]*B.m1_[2] + A.m0_[2]*B.m2_[2] + A.m0_[3]*B.m3_[2];
  tmp.m0_[3] = A.m0_[0]*B.m0_[3] + A.m0_[1]*B.m1_[3] + A.m0_[2]*B.m2_[3] + A.m0_[3]*B.m3_[3];

  // Row 2
  tmp.m1_[0] = A.m1_[0]*B.m0_[0] + A.m1_[1]*B.m1_[0] + A.m1_[2]*B.m2_[0] + A.m1_[3]*B.m3_[0];
  tmp.m1_[1] = A.m1_[0]*B.m0_[1] + A.m1_[1]*B.m1_[1] + A.m1_[2]*B.m2_[1] + A.m1_[3]*B.m3_[1];
  tmp.m1_[2] = A.m1_[0]*B.m0_[2] + A.m1_[1]*B.m1_[2] + A.m1_[2]*B.m2_[2] + A.m1_[3]*B.m3_[2];
  tmp.m1_[3] = A.m1_[0]*B.m0_[3] + A.m1_[1]*B.m1_[3] + A.m1_[2]*B.m2_[3] + A.m1_[3]*B.m3_[3];

  // Row 3
  tmp.m2_[0] = A.m2_[0]*B.m0_[0] + A.m2_[1]*B.m1_[0] + A.m2_[2]*B.m2_[0] + A.m2_[3]*B.m3_[0];
  tmp.m2_[1] = A.m2_[0]*B.m0_[1] + A.m2_[1]*B.m1_[1] + A.m2_[2]*B.m2_[1] + A.m2_[3]*B.m3_[1];
  tmp.m2_[2] = A.m2_[0]*B.m0_[2] + A.m2_[1]*B.m1_[2] + A.m2_[2]*B.m2_[2] + A.m2_[3]*B.m3_[2];
  tmp.m2_[3] = A.m2_[0]*B.m0_[3] + A.m2_[1]*B.m1_[3] + A.m2_[2]*B.m2_[3] + A.m2_[3]*B.m3_[3];

  // Row 4
  tmp.m3_[0] = A.m3_[0]*B.m0_[0] + A.m3_[1]*B.m1_[0] + A.m3_[2]*B.m2_[0] + A.m3_[3]*B.m3_[0];
  tmp.m3_[1] = A.m3_[0]*B.m0_[1] + A.m3_[1]*B.m1_[1] + A.m3_[2]*B.m2_[1] + A.m3_[3]*B.m3_[1];
  tmp.m3_[2] = A.m3_[0]*B.m0_[2] + A.m3_[1]*B.m1_[2] + A.m3_[2]*B.m2_[2] + A.m3_[3]*B.m3_[2];
  tmp.m3_[3] = A.m3_[0]*B.m0_[3] + A.m3_[1]*B.m1_[3] + A.m3_[2]*B.m2_[3] + A.m3_[3]*B.m3_[3];

  return tmp;
}

#endif // End of GUARD_GGEMS_MATHS_MATRIX_FUNCTIONS_HH
