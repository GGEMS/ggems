#ifndef GUARD_GGEMS_MATHS_MATRIX_FUNCTIONS_HH
#define GUARD_GGEMS_MATHS_MATRIX_FUNCTIONS_HH

/*!
  \file matrix_functions.hh

  \brief Definitions of functions using matrix

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday December 16, 2019
*/

#include "GGEMS/maths/matrix_types.hh"

/*!
  \fn inline f323cl_t RotateUz(f323cl_t vector, f323cl_t const new_uz)
  \param vector - vector to change
  \param new_uz - new direction
  \brief rotateUz, function from CLHEP
*/
inline f323cl_t RotateUz(f323cl_t vector, f323cl_t const new_uz)
{
  #ifdef OPENCL_COMPILER
  f32cl_t const u1 = new_uz.x;
  f32cl_t const u2 = new_uz.y;
  f32cl_t const u3 = new_uz.z;
  #else
  f32cl_t const u1 = new_uz.s[0];
  f32cl_t const u2 = new_uz.s[1];
  f32cl_t const u3 = new_uz.s[2];
  #endif

  f32cl_t up = u1*u1 + u2*u2;

  if (up > 0) {
    up = sqrt(up);
    #ifdef OPENCL_COMPILER
    f32cl_t px = vector.x,  py = vector.y, pz = vector.z;
    vector.x = (u1*u3*px - u2*py) /up + u1*pz;
    vector.y = (u2*u3*px + u1*py) /up + u2*pz;
    vector.z =    -up*px +             u3*pz;
    #else
    f32cl_t px = vector.s[0], py = vector.s[1], pz = vector.s[2];
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
    vector.s[0] = -vector.s[1];    // phi=0  theta=gpu_pi
    vector.s[2] = -vector.s[2];
    #endif
  }

  #ifdef OPENCL_COMPILER
  return MakeFloat3x1(vector.x, vector.y, vector.z);
  #else
  return MakeFloat3x1(vector.s[0], vector.s[1], vector.s[2]);
  #endif
}

/*!
  \fn inline f323cl_t f323x1_unit(f323cl_t const u)
  \param u - 3D vector
  \return an unitary vector
  \brief compute an unitary vector
*/
inline f323cl_t f323x1_unit(f323cl_t const u)
{
  #ifdef OPENCL_COMPILER
  f32cl_t norm = 1.0f / sqrt(u.x*u.x + u.y*u.y + u.z*u.z);
  return MakeFloat3x1(u.x*norm, u.y*norm, u.z*norm);
  #else
  f32cl_t norm = 1.0f / sqrt(u.s[0]*u.s[0] + u.s[1]*u.s[1] + u.s[2]*u.s[2]);
  return MakeFloat3x1(u.s[0]*norm, u.s[1]*norm, u.s[2]*norm);
  #endif
}

/*!
  \fn inline f323cl_t f323x1_sub(f323cl_t const u, f323cl_t const v)
  \param u - 3D vector
  \param v - 3D vector
  \return the 3D vector u - v
  \brief Substract the vector v to vector u
*/
inline f323cl_t f323x1_sub(f323cl_t const u, f323cl_t const v)
{
  f323cl_t vector = {
    #ifdef OPENCL_COMPILER
    {u.x-v.x, u.y-v.y, u.z-v.z}
    #else
    {u.s[0]-v.s[0], u.s[1]-v.s[1], u.s[2]-v.s[2]}
    #endif
  };
  return vector;
}

/*!
  \fn inline f323cl_t MatrixMult4x4_3x1(float4x4 const matrix, f323cl_t const point)
  \param matrix - A matrix (4x4)
  \param point - Point in 3D (x, y, z)
  \return a vector 3x1
  \brief Compute the multiplication of matrix 4x4 and a point 3x1
*/
#ifdef OPENCL_COMPILER
inline f323cl_t MatrixMult4x4_3x1(__global float4x4 const* matrix,
  f323cl_t const point)
#else
inline f323cl_t MatrixMult4x4_3x1(float4x4 const* matrix, f323cl_t const point)
#endif
{
  #ifdef OPENCL_COMPILER
  f323cl_t vector = {
    {matrix->m00_*point.x + matrix->m01_*point.y + matrix->m02_*point.z
      + matrix->m03_*1.0f,
    matrix->m10_*point.x + matrix->m11_*point.y + matrix->m12_*point.z
      + matrix->m13_*1.0f,
    matrix->m20_*point.x + matrix->m21_*point.y + matrix->m22_*point.z
      + matrix->m23_*1.0f}
  };
  #else
  f323cl_t vector = {
    {matrix->m00_*point.s[0] + matrix->m01_*point.s[1] + matrix->m02_*point.s[2]
      + matrix->m03_*1.0f,
    matrix->m10_*point.s[0] + matrix->m11_*point.s[1] + matrix->m12_*point.s[2]
      + matrix->m13_*1.0f,
    matrix->m20_*point.s[0] + matrix->m21_*point.s[1] + matrix->m22_*point.s[2]
      + matrix->m23_*1.0f}
  };
  #endif

  return vector;
}

/*!
 \fn inline f323cl_t local_to_global_position(float4x4 const* matrix, f323cl_t const point)
 \param matrix - A matrix (4x4)
 \param point - Point in 3D (x, y, z)
 \return The point expresses in the global frame
 \brief Transform a 3D point from local to global frame
*/
#ifdef OPENCL_COMPILER
inline f323cl_t LocalToGlobalPosition(__global float4x4 const* matrix,
  f323cl_t const point)
#else
inline f323cl_t LocalToGlobalPosition(float4x4 const* matrix,
  f323cl_t const point)
#endif
{
  return MatrixMult4x4_3x1(matrix, point);
}

/*!
  \fn inline float4x4 MatrixMult4x4_4x4(float4x4 const A, float4x4 const B)
  \param A - first matrix
  \param B - second matrix
  \brief Perform the matrix (4x4) multiplication
  \return New matrix AxB
*/
inline float4x4 MatrixMult4x4_4x4(float4x4 const A, float4x4 const B)
{
  float4x4 tmp = MakeFloat4x4Zeros();

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
