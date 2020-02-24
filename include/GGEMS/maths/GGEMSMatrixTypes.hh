#ifndef GUARD_GGEMS_MATHS_GGEMSMATRIXTYPES_HH
#define GUARD_GGEMS_MATHS_GGEMSMATRIXTYPES_HH

/*!
  \file GGEMSMatrixTypes.hh

  \brief Class managing the matrix types

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday November 13, 2019
*/

#include "GGEMS/global/GGEMSConfiguration.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \struct GGfloat33_t
  \brief Structure storing float 3 x 3 matrix
*/
#ifdef OPENCL_COMPILER
typedef struct __attribute__((aligned (1))) GGfloat33_t
#else
#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
typedef struct PACKED GGfloat33_t
#endif
{
  GGfloat m00_, m01_, m02_;
  GGfloat m10_, m11_, m12_;
  GGfloat m20_, m21_, m22_;
} GGfloat33;
#ifndef OPENCL_COMPILER
#ifdef _MSC_VER
#pragma pack(pop)
#endif
#endif

/*!
  \struct GGfloat44_t
  \brief Structure storing float 4 x 4 matrix
*/
#ifdef OPENCL_COMPILER
typedef struct __attribute__((aligned (1))) GGfloat44_t
#else
#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
typedef struct PACKED GGfloat44_t
#endif
{
  GGfloat m00_, m01_, m02_, m03_;
  GGfloat m10_, m11_, m12_, m13_;
  GGfloat m20_, m21_, m22_, m23_;
  GGfloat m30_, m31_, m32_, m33_;
} GGfloat44;
#ifndef OPENCL_COMPILER
#ifdef _MSC_VER
#pragma pack(pop)
#endif
#endif

/*!
  \fn inline GGfloat33 MakeFloat33(GGfloat const m00, GGfloat const m01, GGfloat const m02, GGfloat const m10, GGfloat const m11, GGfloat const m12, GGfloat const m20, GGfloat const m21, GGfloat const m22)
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
inline GGfloat33 MakeFloat33(GGfloat const m00, GGfloat const m01, GGfloat const m02, GGfloat const m10, GGfloat const m11, GGfloat const m12, GGfloat const m20, GGfloat const m21, GGfloat const m22)
{
  GGfloat33 tmp;
  tmp.m00_ = m00; tmp.m01_ = m01; tmp.m02_ = m02;
  tmp.m10_ = m10; tmp.m11_ = m11; tmp.m12_ = m12;
  tmp.m20_ = m20; tmp.m21_ = m21; tmp.m22_ = m22;
  return tmp;
}

/*!
  \fn inline float3x3 MakeFloat33Zeros()
  \brief Make a float3x3 with zeros for value
*/
inline GGfloat33 MakeFloat33Zeros()
{
  GGfloat33 tmp;
  tmp.m00_ = 0.0f; tmp.m01_ = 0.0f; tmp.m02_ = 0.0f;
  tmp.m10_ = 0.0f; tmp.m11_ = 0.0f; tmp.m12_ = 0.0f;
  tmp.m20_ = 0.0f; tmp.m21_ = 0.0f; tmp.m22_ = 0.0f;
  return tmp;
}

/*!
  \fn inline float4x4 MakeFloat44(float const m00, float const m01, float const m02, float const m03, float const m10, float const m11, float const m12, float const m13, float const m20, float const m21, float const m22, float const m23, float const m30, float const m31, float const m32, float const m33)
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
  \brief Make a GGfloat44 with custom values
*/
inline GGfloat44 MakeFloat44(GGfloat const m00, GGfloat const m01, GGfloat const m02, GGfloat const m03, GGfloat const m10, GGfloat const m11, GGfloat const m12, GGfloat const m13, GGfloat const m20, GGfloat const m21, GGfloat const m22, GGfloat const m23, GGfloat const m30, GGfloat const m31, GGfloat const m32, GGfloat const m33)
{
  GGfloat44 tmp;
  tmp.m00_ = m00; tmp.m01_ = m01; tmp.m02_ = m02; tmp.m03_ = m03;
  tmp.m10_ = m10; tmp.m11_ = m11; tmp.m12_ = m12; tmp.m13_ = m13;
  tmp.m20_ = m20; tmp.m21_ = m21; tmp.m22_ = m22; tmp.m23_ = m23;
  tmp.m30_ = m30; tmp.m31_ = m31; tmp.m32_ = m32; tmp.m33_ = m33;
  return tmp;
}

/*!
  \fn inline GGfloat44 MakeFloat44Zeros()
  \brief Make a GGfloat44 with zeros for value
*/
inline GGfloat44 MakeFloat44Zeros()
{
  GGfloat44 tmp;
  tmp.m00_ = 0.0f; tmp.m01_ = 0.0f; tmp.m02_ = 0.0f; tmp.m03_ = 0.0f;
  tmp.m10_ = 0.0f; tmp.m11_ = 0.0f; tmp.m12_ = 0.0f; tmp.m13_ = 0.0f;
  tmp.m20_ = 0.0f; tmp.m21_ = 0.0f; tmp.m22_ = 0.0f; tmp.m23_ = 0.0f;
  tmp.m30_ = 0.0f; tmp.m31_ = 0.0f; tmp.m32_ = 0.0f; tmp.m33_ = 0.0f;
  return tmp;
}

#endif // End of GUARD_GGEMS_MATHS_GGEMSMATRIXTYPES_HH
