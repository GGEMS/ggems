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
: need_updated_(false),
  position_(Matrix::MakeFloatXYZ(0.0f, 0.0f, 0.0f)),
  rotation_(Matrix::MakeFloatXYZ(0.0f, 0.0f, 0.0f))
{
  GGEMScout("TransformCalculator", "TransformCalculator", 1)
    << "Allocation of TransformCalculator..." << GGEMSendl;

  // Initialize the local axis
  local_axis_ = Matrix::MakeFloat3x3(
    1.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 1.0f);

  // Initializing translation matrix
  matrix_translation_ = Matrix::MakeFloat4x4(
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f);

  // Initializing rotation matrix
  matrix_rotation_ = Matrix::MakeFloat4x4(
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f);

  // Initializing orthographic projection matrix
  matrix_orthographic_projection_ = Matrix::MakeFloat4x4(
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f);

  // Initializing the transformation matrix
  matrix_transformation_ = Matrix::MakeFloat4x4(
    0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f);
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
  // Fill the position buffer first
  position_ = Matrix::MakeFloatXYZ(tx, ty, tz);

  // Filling the translation matrix
  matrix_translation_ = Matrix::MakeFloat4x4(
    1.0f, 0.0f, 0.0f, position_.s[0],
    0.0f, 1.0f, 0.0f, position_.s[1],
    0.0f, 0.0f, 1.0f, position_.s[2],
    0.0f, 0.0f, 0.0f, 1.0f);

  // Need to be updated if the Transformation matrix is called
  need_updated_ = true;
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
  // Filling the rotation buffer
  rotation_ = Matrix::MakeFloatXYZ(rx, ry, rz);

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
  matrix_rotation_ = Matrix::MatrixMult4x4(kRotationY, kRotationX);
  matrix_rotation_ = Matrix::MatrixMult4x4(kRotationZ, matrix_rotation_);

  // Need to be updated if the Transformation matrix is called
  need_updated_ = true;
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
  // Filling the local axis buffer first
  local_axis_ = Matrix::MakeFloat3x3(
    axis.m00_, axis.m01_, axis.m02_,
    axis.m10_, axis.m11_, axis.m12_,
    axis.m20_, axis.m21_, axis.m22_);

  matrix_orthographic_projection_ = Matrix::MakeFloat4x4(
    axis.m00_, axis.m01_, axis.m02_, 0.0f,
    axis.m10_, axis.m11_, axis.m12_, 0.0f,
    axis.m20_, axis.m21_, axis.m22_, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f);

  // Need to be updated if the Transformation matrix is called
  need_updated_ = true;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void TransformCalculator::UpdateTransformationMatrix(void)
{
  matrix_transformation_ = Matrix::MatrixMult4x4( matrix_rotation_,
      Matrix::MatrixMult4x4(matrix_translation_,
      matrix_orthographic_projection_));

  // Update is done
  need_updated_ = false;
}