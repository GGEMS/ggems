#ifndef GUARD_GGEMS_TOOLS_MATRIX_HH
#define GUARD_GGEMS_TOOLS_MATRIX_HH

/*!
  \file matrix.hh

  \brief Class managing the matrix computation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday November 13, 2019
*/

#include "GGEMS/global/ggems_configuration.hh"
#include "GGEMS/global/opencl_manager.hh"

/*!
  \namespace Matrix
  \brief namespace storing miscellaneous functions for Matrix
*/
namespace Matrix
{
  /*!
    \struct float3x3_t
    \brief Structure storing float 3 x 3 matrix
  */
  #ifdef _MSC_VER
  #pragma pack(push, 1)
  #endif
  typedef struct PACKED float3x3_t
  {
    cl_float m00_, m01_, m02_;
    cl_float m10_, m11_, m12_;
    cl_float m20_, m21_, m22_;
  } float3x3;
  #ifdef _MSC_VER
  #pragma pack(pop)
  #endif

  /*!
    \struct float4x4_t
    \brief Structure storing float 4 x 4 matrix
  */
  #ifdef _MSC_VER
  #pragma pack(push, 1)
  #endif
  typedef struct PACKED float4x4_t
  {
    cl_float m00_, m01_, m02_, m03_;
    cl_float m10_, m11_, m12_, m13_;
    cl_float m20_, m21_, m22_, m23_;
    cl_float m30_, m31_, m32_, m33_;
  } float4x4;
  #ifdef _MSC_VER
  #pragma pack(pop)
  #endif

  /*!
    \fn inline cl_float3 MakeFloatXYZ(float const& x, float const& y, float const& z)
    \param x - x parameter
    \param y - y parameter
    \param z - z parameter
    \brief Make a float X, Y and Z with custom values
  */
  inline cl_float3 MakeFloatXYZ(float const& x, float const& y, float const& z)
  {
    cl_float3 tmp;
    tmp.s[0] = x;
    tmp.s[1] = y;
    tmp.s[2] = z;
    return tmp;
  }

  /*!
    \fn inline cl_float3 MakeFloatXYZZeros()
    \brief Make a float X, Y and Z with zeros for value
  */
  inline cl_float3 MakeFloatXYZZeros()
  {
    cl_float3 tmp;
    tmp.s[0] = 0.0f;
    tmp.s[1] = 0.0f;
    tmp.s[2] = 0.0f;
    return tmp;
  }

  /*!
    \fn inline float3x3 MakeFloat3x3(float const& m00, float const& m01, float const& m02, float const& m10, float const& m11, float const& m12, float const& m20, float const& m21, float const& m22)
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
    float const& m00, float const& m01, float const& m02,
    float const& m10, float const& m11, float const& m12,
    float const& m20, float const& m21, float const& m22)
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
}

#endif // End of GUARD_GGEMS_TOOLS_MATRIX_HH
