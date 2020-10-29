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
  \fn inline GGfloat33 MakeFloat33(GGGfloat3 const m0, GGfloat3 const m1, GGfloat3 const m2)
  \param m0 - Row 0 in the matrix 3x3 for local axis
  \param m1 - Row 1 in the matrix 3x3 for local axis
  \param m2 - Row 2 in the matrix 3x3 for local axis
  \return a 3x3 float matrix
  \brief Make a GGfloat33 with custom values
*/
inline GGfloat33 MakeFloat33(GGfloat3 const m0, GGfloat3 const m1, GGfloat3 const m2)
{
  GGfloat33 tmp;
  tmp.m0_[0] = m0.s0; tmp.m0_[1] = m0.s1; tmp.m0_[2] = m0.s2;
  tmp.m1_[0] = m1.s0; tmp.m1_[1] = m1.s1; tmp.m1_[2] = m1.s2;
  tmp.m2_[0] = m2.s0; tmp.m2_[1] = m2.s1; tmp.m2_[2] = m2.s2;
  return tmp;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGfloat33 MakeFloat33Zeros(void)
  \return a 3x3 float matrix of 0
  \brief Make a GGfloat33 with zeros for value
*/
inline GGfloat33 MakeFloat33Zeros(void)
{
  GGfloat33 tmp;
  tmp.m0_[0] = 0.0f; tmp.m0_[1] = 0.0f; tmp.m0_[2] = 0.0f;
  tmp.m1_[0] = 0.0f; tmp.m1_[1] = 0.0f; tmp.m1_[2] = 0.0f;
  tmp.m2_[0] = 0.0f; tmp.m2_[1] = 0.0f; tmp.m2_[2] = 0.0f;
  return tmp;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGfloat44 MakeFloat44(GGfloat4 const m0, GGfloat4 const m1, GGfloat4 const m2, GGfloat4 const m3)
  \param m0 - Row 0 in the matrix 4x4 for local axis
  \param m1 - Row 1 in the matrix 4x4 for local axis
  \param m2 - Row 2 in the matrix 4x4 for local axis
  \param m3 - Row 3 in the matrix 4x4 for local axis
  \return a 4x4 float matrix
  \brief Make a GGfloat44 with custom values
*/
inline GGfloat44 MakeFloat44(GGfloat4 const m0, GGfloat4 const m1, GGfloat4 const m2, GGfloat4 const m3)
{
  GGfloat44 tmp;
  tmp.m0_[0] = m0.s0; tmp.m0_[1] = m0.s1; tmp.m0_[2] = m0.s2; tmp.m0_[3] = m0.s3;
  tmp.m1_[0] = m1.s0; tmp.m1_[1] = m1.s1; tmp.m1_[2] = m1.s2; tmp.m1_[3] = m1.s3;
  tmp.m2_[0] = m2.s0; tmp.m2_[1] = m2.s1; tmp.m2_[2] = m2.s2; tmp.m2_[3] = m2.s3;
  tmp.m3_[0] = m3.s0; tmp.m3_[1] = m3.s1; tmp.m3_[2] = m3.s2; tmp.m3_[3] = m3.s3;
  return tmp;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline GGfloat44 MakeFloat44Zeros(void)
  \return a 4x4 float matrix
  \brief Make a GGfloat44 with zeros for value
*/
inline GGfloat44 MakeFloat44Zeros(void)
{
  GGfloat44 tmp;
  tmp.m0_[0] = 0.0f; tmp.m0_[1] = 0.0f; tmp.m0_[2] = 0.0f; tmp.m0_[3] = 0.0f;
  tmp.m1_[0] = 0.0f; tmp.m1_[1] = 0.0f; tmp.m1_[2] = 0.0f; tmp.m1_[3] = 0.0f;
  tmp.m2_[0] = 0.0f; tmp.m2_[1] = 0.0f; tmp.m2_[2] = 0.0f; tmp.m2_[3] = 0.0f;
  tmp.m3_[0] = 0.0f; tmp.m3_[1] = 0.0f; tmp.m3_[2] = 0.0f; tmp.m3_[3] = 0.0f;
  return tmp;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifdef __OPENCL_C_VERSION__

/*!
  \fn inline GGfloat3 RotateUnitZ(GGfloat3 vector, GGfloat3 const new_uz)
  \param vector - vector to change
  \param new_uz - new direction
  \return a vector of 3x1 float
  \brief rotateUz, function from CLHEP
*/
inline GGfloat3 RotateUnitZ(GGfloat3 vector, GGfloat3 const new_uz)
{
  GGfloat const u1 = new_uz.x;
  GGfloat const u2 = new_uz.y;
  GGfloat const u3 = new_uz.z;

  GGfloat up = u1*u1 + u2*u2;
  if (up > 0) {
    up = sqrt(up);
    GGfloat px = vector.x,  py = vector.y, pz = vector.z;
    vector.x = (u1*u3*px - u2*py) /up + u1*pz;
    vector.y = (u2*u3*px + u1*py) /up + u2*pz;
    vector.z =    -up*px +             u3*pz;
  }
  else if (u3 < 0.) {
    vector.x = -vector.x;    // phi=0  theta=gpu_pi
    vector.z = -vector.z;
  }

  return MakeFloat3(vector.x, vector.y, vector.z);
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
  \fn inline GGfloat3 GGfloat44MultGGfloat3(__global GGfloat44 const* matrix, GGfloat3 const point)
  \param matrix - A matrix (4x4)
  \param point - Point in 3D (x, y, z)
  \return a vector 3x1
  \brief Compute the multiplication of matrix 4x4 and a point 3x1
*/
inline GGfloat3 GGfloat44MultGGfloat3(__global GGfloat44 const* matrix, GGfloat3 const point)
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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

inline GGfloat3 GlobalToLocalPosition(__global GGfloat44 const* matrix, GGfloat3 const point)
{
  // Extract translation
  GGfloat3 translation;
  //translation.x = matrix->m0_.x;
  //= {matrix->m0_.w, matrix->m1_.w, matrix->m2_.w};

  return translation;
}

/*__host__ __device__ f32xyz fxyz_global_to_local_position( const f32matrix44 &G, f32xyz u)
{
    // first, extract the translation
    f32xyz T = { G.m03, G.m13, G.m23 };
    // Then the sub matrix (R and P)
    f32matrix33 g = { G.m00, G.m01, G.m02,
                      G.m10, G.m11, G.m12,
                      G.m20, G.m21, G.m22 };
    // Inverse transform
    f32matrix33 ginv = fmatrix_trans( g );
    u = fxyz_sub( u, T );
    u = fmatrix_mul_fxyz( ginv, u );

    return u;
}*/

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
 \fn inline GGfloat3 LocalToGlobalPosition(__global GGfloat44 const* matrix, GGfloat3 const point)
 \param matrix - A matrix (4x4)
 \param point - Point in 3D (x, y, z)
 \return The point expresses in the global frame
 \brief Transform a 3D point from local to global frame
*/
inline GGfloat3 LocalToGlobalPosition(__global GGfloat44 const* matrix, GGfloat3 const point)
{
  return GGfloat44MultGGfloat3(matrix, point);
}
#endif

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifndef __OPENCL_C_VERSION__
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
#endif

#endif // End of GUARD_GGEMS_MATHS_MATRIX_FUNCTIONS_HH
