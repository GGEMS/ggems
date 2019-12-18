/*!
  \file geometry_transformation.cc

  \brief Class managing the geometry transformation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday November 13, 2019
*/

#include <cmath>
#include <limits>

#include "GGEMS/maths/geometry_transformation.hh"
#include "GGEMS/maths/matrix_functions.hh"
#include "GGEMS/tools/print.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GeometryTransformation::GeometryTransformation()
: opencl_manager_(OpenCLManager::GetInstance()),
  is_need_updated_(false)
{
  GGEMScout("GeometryTransformation", "GeometryTransformation", 3)
    << "Allocation of GeometryTransformation..." << GGEMSendl;

  // Initialize the position with min. float
  position_ = MakeFloat3x1(
    std::numeric_limits<float>::min(),
    std::numeric_limits<float>::min(),
    std::numeric_limits<float>::min()
  );

  // Initialize the rotation with min. float
  rotation_ = MakeFloat3x1(
    std::numeric_limits<float>::min(),
    std::numeric_limits<float>::min(),
    std::numeric_limits<float>::min()
  );

  // Initialize the local axis
  local_axis_ = MakeFloat3x3(
    1.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 1.0f);

  // Initializing translation matrix
  matrix_translation_ = MakeFloat4x4(
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f);

  // Initializing rotation matrix
  matrix_rotation_ = MakeFloat4x4(
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f);

  // Initializing orthographic projection matrix
  matrix_orthographic_projection_ = MakeFloat4x4(
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f);

  // Allocation of matrix transformation on OpenCL device
  p_matrix_transformation_ = opencl_manager_.Allocate(nullptr,
    sizeof(float4x4), CL_MEM_READ_WRITE);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GeometryTransformation::~GeometryTransformation()
{
  // Freeing memory
  if (p_matrix_transformation_) {
    opencl_manager_.Deallocate(p_matrix_transformation_, sizeof(float4x4));
    p_matrix_transformation_ = nullptr;
  }

  GGEMScout("GeometryTransformation", "~GeometryTransformation", 3)
    << "Deallocation of GeometryTransformation..." << GGEMSendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GeometryTransformation::SetTranslation(float const& tx, float const& ty,
  float const& tz)
{
  // Fill the position buffer first
  position_ = MakeFloat3x1(tx, ty, tz);

  // Filling the translation matrix
  matrix_translation_ = MakeFloat4x4(
    1.0f, 0.0f, 0.0f, position_.s[0],
    0.0f, 1.0f, 0.0f, position_.s[1],
    0.0f, 0.0f, 1.0f, position_.s[2],
    0.0f, 0.0f, 0.0f, 1.0f);

  // Need to be updated if the Transformation matrix is called
  is_need_updated_ = true;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GeometryTransformation::SetTranslation(f323cl_t const& txyz)
{
  SetTranslation(txyz.s[0], txyz.s[1], txyz.s[2]);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GeometryTransformation::SetRotation(float const& rx, float const& ry,
  float const& rz)
{
  // Filling the rotation buffer
  rotation_ = MakeFloat3x1(rx, ry, rz);

  // Definition of cosinus and sinus
  double cosinus = 0.0, sinus = 0.0;

  // X axis
  cosinus = cos(rx);
  sinus = sin(rx);

  float4x4 const kRotationX = MakeFloat4x4(
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, static_cast<float>(cosinus), -static_cast<float>(sinus), 0.0f,
    0.0f, static_cast<float>(sinus), static_cast<float>(cosinus), 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
  );

  // Y axis
  cosinus = cos(ry);
  sinus = sin(ry);

  float4x4 const kRotationY = MakeFloat4x4(
    static_cast<float>(cosinus), 0.0f, static_cast<float>(sinus), 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    -static_cast<float>(sinus), 0.0f, static_cast<float>(cosinus), 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
  );

  // Z axis
  cosinus = cos(rz);
  sinus = sin(rz);

  float4x4 const kRotationZ = MakeFloat4x4(
    static_cast<float>(cosinus), -static_cast<float>(sinus), 0.0f, 0.0f,
    static_cast<float>(sinus), static_cast<float>(cosinus), 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f);

  // Get the total rotation matrix
  matrix_rotation_ = MatrixMult4x4_4x4(kRotationY, kRotationX);
  matrix_rotation_ = MatrixMult4x4_4x4(kRotationZ, matrix_rotation_);

  // Need to be updated if the Transformation matrix is called
  is_need_updated_ = true;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GeometryTransformation::SetRotation(cl_float3 const& rxyz)
{
  SetRotation(rxyz.s[0], rxyz.s[1], rxyz.s[2]);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GeometryTransformation::SetAxisTransformation(
  float const& m00, float const& m01, float const& m02,
  float const& m10, float const& m11, float const& m12,
  float const& m20, float const& m21, float const& m22)
{
  float3x3 const kTmp = MakeFloat3x3(
    m00, m01, m02,
    m10, m11, m12,
    m20, m21, m22);

  SetAxisTransformation(kTmp);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GeometryTransformation::SetAxisTransformation(float3x3 const& axis)
{
  // Filling the local axis buffer first
  local_axis_ = MakeFloat3x3(
    axis.m00_, axis.m01_, axis.m02_,
    axis.m10_, axis.m11_, axis.m12_,
    axis.m20_, axis.m21_, axis.m22_);

  matrix_orthographic_projection_ = MakeFloat4x4(
    axis.m00_, axis.m01_, axis.m02_, 0.0f,
    axis.m10_, axis.m11_, axis.m12_, 0.0f,
    axis.m20_, axis.m21_, axis.m22_, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f);

  // Need to be updated if the Transformation matrix is called
  is_need_updated_ = true;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GeometryTransformation::UpdateTransformationMatrix(void)
{
  GGEMScout("GeometryTransformation", "UpdateTransformationMatrix", 3)
    << "Updating the transformation matrix..." << GGEMSendl;

  /*matrix_transformation_ = MatrixMult4x4(matrix_rotation_,
      MatrixMult4x4(matrix_translation_,
      matrix_orthographic_projection_));*/

  // Update the transformation matrix on OpenCL device
  // Get the pointer on device
  float4x4* p_matrix = opencl_manager_.GetDeviceBuffer<float4x4>(
    p_matrix_transformation_, sizeof(float4x4));

  // Compute a temporary matrix then copy it on OpenCL device
  float4x4 matrix_tmp = MatrixMult4x4_4x4(matrix_rotation_,
    MatrixMult4x4_4x4(matrix_translation_, matrix_orthographic_projection_));

  // Copy step
  p_matrix->m00_ = matrix_tmp.m00_;
  p_matrix->m01_ = matrix_tmp.m01_;
  p_matrix->m02_ = matrix_tmp.m02_;
  p_matrix->m03_ = matrix_tmp.m03_;

  p_matrix->m10_ = matrix_tmp.m10_;
  p_matrix->m11_ = matrix_tmp.m11_;
  p_matrix->m12_ = matrix_tmp.m12_;
  p_matrix->m13_ = matrix_tmp.m13_;

  p_matrix->m20_ = matrix_tmp.m20_;
  p_matrix->m21_ = matrix_tmp.m21_;
  p_matrix->m22_ = matrix_tmp.m22_;
  p_matrix->m23_ = matrix_tmp.m23_;

  p_matrix->m30_ = matrix_tmp.m30_;
  p_matrix->m31_ = matrix_tmp.m31_;
  p_matrix->m32_ = matrix_tmp.m32_;
  p_matrix->m33_ = matrix_tmp.m33_;

  // Release the pointer, mandatory step!!!
  opencl_manager_.ReleaseDeviceBuffer(p_matrix_transformation_, p_matrix);

  // Update is done
  is_need_updated_ = false;
}
