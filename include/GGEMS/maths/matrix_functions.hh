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

inline float4x4 MatrixMult4x4(float4x4 const A, float4x4 const B)
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
