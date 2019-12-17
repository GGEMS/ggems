#ifndef GUARD_GGEMS_MATHS_MATRIX_TYPES_HH
#define GUARD_GGEMS_MATHS_MATRIX_TYPES_HH

/*!
  \file matrix_types.hh

  \brief Class managing the matrix types

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday November 13, 2019
*/

#include "GGEMS/global/ggems_configuration.hh"
#include "GGEMS/opencl/types.hh"

/*!
  \struct float3x3_t
  \brief Structure storing float 3 x 3 matrix
*/
#ifdef OPENCL_COMPILER
typedef struct __attribute__((aligned (1))) float3x3_t
#else
#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
typedef struct PACKED float3x3_t
#endif
{
  f32cl_t m00_, m01_, m02_;
  f32cl_t m10_, m11_, m12_;
  f32cl_t m20_, m21_, m22_;
} float3x3;
#ifndef OPENCL_COMPILER
#ifdef _MSC_VER
#pragma pack(pop)
#endif
#endif

/*!
  \struct float4x4_t
  \brief Structure storing float 4 x 4 matrix
*/
#ifdef OPENCL_COMPILER
typedef struct __attribute__((aligned (1))) float4x4_t
#else
#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
typedef struct PACKED float4x4_t
#endif
{
  f32cl_t m00_, m01_, m02_, m03_;
  f32cl_t m10_, m11_, m12_, m13_;
  f32cl_t m20_, m21_, m22_, m23_;
  f32cl_t m30_, m31_, m32_, m33_;
} float4x4;
#ifndef OPENCL_COMPILER
#ifdef _MSC_VER
#pragma pack(pop)
#endif
#endif

/*!
  \fn inline f323cl_t MakeFloatXYZ(f32cl_t const x, f32cl_t const y, f32cl_t const z)
  \param x - x parameter
  \param y - y parameter
  \param z - z parameter
  \brief Make a float X, Y and Z with custom values
*/
inline f323cl_t MakeFloatXYZ(f32cl_t const x, f32cl_t const y, f32cl_t const z)
{
  f323cl_t tmp;
  #ifdef OPENCL_COMPILER
  tmp.x = x;
  tmp.y = y;
  tmp.z = z;
  #else
  tmp.s[0] = x;
  tmp.s[1] = y;
  tmp.s[2] = z;
  #endif
  return tmp;
}

/*!
  \fn inline f323cl_t MakeFloatXYZZeros()
  \brief Make a float X, Y and Z with zeros for value
*/
inline f323cl_t MakeFloatXYZZeros()
{
  f323cl_t tmp;
  #ifdef OPENCL_COMPILER
  tmp.x = 0.0f;
  tmp.y = 0.0f;
  tmp.z = 0.0f;
  #else
  tmp.s[0] = 0.0f;
  tmp.s[1] = 0.0f;
  tmp.s[2] = 0.0f;
  #endif
  return tmp;
}

/*!
  \fn inline float3x3 MakeFloat3x3(f32cl_t const m00, f32cl_t const m01, f32cl_t const m02, f32cl_t const m10, f32cl_t const m11, f32cl_t const m12, f32cl_t const m20, f32cl_t const m21, f32cl_t const m22)
  \param m00 - Element 0,0 in the matrix 3x3 for local axis
  \param m01 - Element 0,1 in the matrix 3x3 for local axis
  \param m02 - Element 0,2 in the matrix 3x3 for local axis
  \param m10 - Element 1,0 in the matrix 3x3 for local axis
  \param m11 - Element 1,1 in the matrix 3x3 for local axis
  \param m12 - Element 1,2 in the matrix 3x3 for local axis
  \param m20 - Element 2,0 in the matrix 3x3 for local axis
  \param m21 - Element 2,1 in the matrix 3x3 for local axis
  \param m22 - Element 2,2 in the matrix 3x3 for local axis
  \brief Make a float3x3 with custom values
*/
inline float3x3 MakeFloat3x3(
  f32cl_t const m00, f32cl_t const m01, f32cl_t const m02,
  f32cl_t const m10, f32cl_t const m11, f32cl_t const m12,
  f32cl_t const m20, f32cl_t const m21, f32cl_t const m22)
{
  float3x3 tmp;
  // Row 1
  tmp.m00_ = m00; tmp.m01_ = m01; tmp.m02_ = m02;
  // Row 2
  tmp.m10_ = m10; tmp.m11_ = m11; tmp.m12_ = m12;
  // Row 3
  tmp.m20_ = m20; tmp.m21_ = m21; tmp.m22_ = m22;
  return tmp;
}

/*!
  \fn inline float3x3 MakeFloat3x3Zeros()
  \brief Make a float3x3 with zeros for value
*/
inline float3x3 MakeFloat3x3Zeros()
{
  float3x3 tmp;
  // Row 1
  tmp.m00_ = 0.0f; tmp.m01_ = 0.0f; tmp.m02_ = 0.0f;
  // Row 2
  tmp.m10_ = 0.0f; tmp.m11_ = 0.0f; tmp.m12_ = 0.0f;
  // Row 3
  tmp.m20_ = 0.0f; tmp.m21_ = 0.0f; tmp.m22_ = 0.0f;
  return tmp;
}

/*!
  \fn inline float4x4 MakeFloat4x4(float const m00, float const m01, float const m02, float const m03, float const m10, float const m11, float const m12, float const m13, float const m20, float const m21, float const m22, float const m23, float const m30, float const m31, float const m32, float const m33)
  \param m00 - Element 0,0 in the matrix 4x4 for local axis
  \param m01 - Element 0,1 in the matrix 4x4 for local axis
  \param m02 - Element 0,2 in the matrix 4x4 for local axis
  \param m03 - Element 0,3 in the matrix 4x4 for local axis
  \param m10 - Element 1,0 in the matrix 4x4 for local axis
  \param m11 - Element 1,1 in the matrix 4x4 for local axis
  \param m12 - Element 1,2 in the matrix 4x4 for local axis
  \param m13 - Element 1,3 in the matrix 4x4 for local axis
  \param m20 - Element 2,0 in the matrix 4x4 for local axis
  \param m21 - Element 2,1 in the matrix 4x4 for local axis
  \param m22 - Element 2,2 in the matrix 4x4 for local axis
  \param m23 - Element 2,3 in the matrix 4x4 for local axis
  \param m30 - Element 3,0 in the matrix 4x4 for local axis
  \param m31 - Element 3,1 in the matrix 4x4 for local axis
  \param m32 - Element 3,2 in the matrix 4x4 for local axis
  \param m33 - Element 3,3 in the matrix 4x4 for local axis
  \brief Make a float4x4 with custom values
*/
inline float4x4 MakeFloat4x4(
  f32cl_t const m00, f32cl_t const m01, f32cl_t const m02, f32cl_t const m03,
  f32cl_t const m10, f32cl_t const m11, f32cl_t const m12, f32cl_t const m13,
  f32cl_t const m20, f32cl_t const m21, f32cl_t const m22, f32cl_t const m23,
  f32cl_t const m30, f32cl_t const m31, f32cl_t const m32, f32cl_t const m33)
{
  float4x4 tmp;
  // Row 1
  tmp.m00_ = m00; tmp.m01_ = m01; tmp.m02_ = m02; tmp.m03_ = m03;
  // Row 2
  tmp.m10_ = m10; tmp.m11_ = m11; tmp.m12_ = m12; tmp.m13_ = m13;
  // Row 3
  tmp.m20_ = m20; tmp.m21_ = m21; tmp.m22_ = m22; tmp.m23_ = m23;
  // Row 4
  tmp.m30_ = m30; tmp.m31_ = m31; tmp.m32_ = m32; tmp.m33_ = m33;
  return tmp;
}

/*!
  \fn inline float4x4 MakeFloat3x3Zeros()
  \brief Make a float4x4 with zeros for value
*/
inline float4x4 MakeFloat4x4Zeros()
{
  float4x4 tmp;
  // Row 1
  tmp.m00_ = 0.0f; tmp.m01_ = 0.0f; tmp.m02_ = 0.0f; tmp.m03_ = 0.0f;
  // Row 2
  tmp.m10_ = 0.0f; tmp.m11_ = 0.0f; tmp.m12_ = 0.0f; tmp.m13_ = 0.0f;
  // Row 3
  tmp.m20_ = 0.0f; tmp.m21_ = 0.0f; tmp.m22_ = 0.0f; tmp.m23_ = 0.0f;
  // Row 4
  tmp.m30_ = 0.0f; tmp.m31_ = 0.0f; tmp.m32_ = 0.0f; tmp.m33_ = 0.0f;
  return tmp;
}

#endif // End of GUARD_GGEMS_MATHS_MATRIX_TYPES_HH
