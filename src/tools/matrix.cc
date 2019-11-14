/*!
  \file matrix.hh

  \brief Class managing the matrix computation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday November 13, 2019
*/

#include <cmath>

#include "GGEMS/tools/matrix.hh"
#include "GGEMS/tools/print.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

TransformCalculator::TransformCalculator()
{
  GGEMScout("TransformCalculator", "TransformCalculator", 1)
    << "Allocation of TransformCalculator..." << GGEMSendl;

  // Initializing translation matrix
  translation_ = Matrix::MakeFloat4x4(
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f);

  // Initializing rotation matrix
  rotation_ = Matrix::MakeFloat4x4(
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f);

  // Initializing orthographic projection matrix
  orthographic_projection_ = Matrix::MakeFloat4x4(
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

TransformCalculator::~TransformCalculator()
{
  GGEMScout("TransformCalculator", "~TransformCalculator", 1)
    << "Deallocation of TransformCalculator..." << GGEMSendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void TransformCalculator::SetTranslation(float const& tx, float const& ty,
  float const& tz)
{
  translation_ = Matrix::MakeFloat4x4(
    1.0f, 0.0f, 0.0f, tx,
    0.0f, 1.0f, 0.0f, ty,
    0.0f, 0.0f, 1.0f, tz,
    0.0f, 0.0f, 0.0f, 1.0f);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void TransformCalculator::SetTranslation(cl_float3 const& txyz)
{
  SetTranslation(txyz.s[0], txyz.s[1], txyz.s[2]);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void TransformCalculator::SetRotation(float const& rx, float const& ry,
  float const& rz)
{
  // Definition of cosinus and sinus
  double cosinus = 0.0, sinus = 0.0;

  // X axis
  cosinus = cos(rx);
  sinus = sin(rx);

  Matrix::float4x4 const kRotationX = Matrix::MakeFloat4x4(
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, static_cast<float>(cosinus), -static_cast<float>(sinus), 0.0f,
    0.0f, static_cast<float>(sinus), static_cast<float>(cosinus), 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
  );

  // Y axis
  cosinus = cos(ry);
  sinus = sin(ry);

  Matrix::float4x4 const kRotationY = Matrix::MakeFloat4x4(
    static_cast<float>(cosinus), 0.0f, static_cast<float>(sinus), 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    -static_cast<float>(sinus), 0.0f, static_cast<float>(cosinus), 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
  );

  // Z axis
  cosinus = cos(rz);
  sinus = sin(rz);

  Matrix::float4x4 const kRotationZ = Matrix::MakeFloat4x4(
    static_cast<float>(cosinus), -static_cast<float>(sinus), 0.0f, 0.0f,
    static_cast<float>(sinus), static_cast<float>(cosinus), 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f);

  // Get the total rotation matrix
  rotation_ = Matrix::MatrixMult4x4(kRotationY, kRotationX);
  rotation_ = Matrix::MatrixMult4x4(kRotationZ, rotation_);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void TransformCalculator::SetRotation(cl_float3 const& rxyz)
{
  SetRotation(rxyz.s[0], rxyz.s[1], rxyz.s[2]);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void TransformCalculator::SetAxisTransformation(
  float const& m00, float const& m01, float const& m02,
  float const& m10, float const& m11, float const& m12,
  float const& m20, float const& m21, float const& m22)
{
  Matrix::float3x3 const kTmp = Matrix::MakeFloat3x3(
    m00, m01, m02,
    m10, m11, m12,
    m20, m21, m22);

  SetAxisTransformation(kTmp);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void TransformCalculator::SetAxisTransformation(Matrix::float3x3 const& axis)
{
  orthographic_projection_ = Matrix::MakeFloat4x4(
    axis.m00_, axis.m01_, axis.m02_, 0.0f,
    axis.m10_, axis.m11_, axis.m12_, 0.0f,
    axis.m20_, axis.m21_, axis.m22_, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f);
}
