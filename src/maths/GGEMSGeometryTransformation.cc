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
  \file GGEMSGeometryTransformation.cc

  \brief Class managing the geometry transformation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday November 13, 2019
*/

#include <cmath>
#include <limits>

#include "GGEMS/maths/GGEMSGeometryTransformation.hh"
#include "GGEMS/maths/GGEMSMatrixOperations.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSGeometryTransformation::GGEMSGeometryTransformation()
: opencl_manager_(GGEMSOpenCLManager::GetInstance()),
  is_need_updated_(false)
{
  GGcout("GGEMSGeometryTransformation", "GGEMSGeometryTransformation", 3) << "Allocation of GGEMSGeometryTransformation..." << GGendl;

  // Initialize the position with min. float
  position_ = MakeFloat3(std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min());

  // Initialize the rotation with min. float
  rotation_ = MakeFloat3(std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min());

  // Initialize the local axis
  local_axis_ = MakeFloat33(
    1.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 1.0f
  );

  // Initializing translation matrix
  matrix_translation_ = MakeFloat44(
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
  );

  // Initializing rotation matrix
  matrix_rotation_ = MakeFloat44(
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
  );

  // Initializing orthographic projection matrix
  matrix_orthographic_projection_ = MakeFloat44(
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
  );

  // Allocation of matrix transformation on OpenCL device
  matrix_transformation_cl_ = opencl_manager_.Allocate(nullptr, sizeof(GGfloat44), CL_MEM_READ_WRITE);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSGeometryTransformation::~GGEMSGeometryTransformation()
{
  GGcout("GGEMSGeometryTransformation", "~GGEMSGeometryTransformation", 3) << "Deallocation of GGEMSGeometryTransformation..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSGeometryTransformation::SetTranslation(GGfloat const& tx, GGfloat const& ty, GGfloat const& tz)
{
  // Fill the position buffer first
  position_ = MakeFloat3(tx, ty, tz);

  // Filling the translation matrix
  matrix_translation_ = MakeFloat44(
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

void GGEMSGeometryTransformation::SetTranslation(GGfloat3 const& txyz)
{
  SetTranslation(txyz.s[0], txyz.s[1], txyz.s[2]);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSGeometryTransformation::SetRotation(GGfloat const& rx, GGfloat const& ry, GGfloat const& rz)
{
  // Filling the rotation buffer
  rotation_ = MakeFloat3(rx, ry, rz);

  // Definition of cosinus and sinus
  GGdouble cosinus = 0.0, sinus = 0.0;

  // X axis
  cosinus = cos(rx);
  sinus = sin(rx);

  GGfloat44 const kRotationX = MakeFloat44(
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, static_cast<GGfloat>(cosinus), -static_cast<GGfloat>(sinus), 0.0f,
    0.0f, static_cast<GGfloat>(sinus), static_cast<GGfloat>(cosinus), 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
  );

  // Y axis
  cosinus = cos(ry);
  sinus = sin(ry);

  GGfloat44 const kRotationY = MakeFloat44(
    static_cast<GGfloat>(cosinus), 0.0f, static_cast<GGfloat>(sinus), 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    -static_cast<GGfloat>(sinus), 0.0f, static_cast<GGfloat>(cosinus), 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
  );

  // Z axis
  cosinus = cos(rz);
  sinus = sin(rz);

  GGfloat44 const kRotationZ = MakeFloat44(
    static_cast<GGfloat>(cosinus), -static_cast<GGfloat>(sinus), 0.0f, 0.0f,
    static_cast<GGfloat>(sinus), static_cast<GGfloat>(cosinus), 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f);

  // Get the total rotation matrix
  matrix_rotation_ = GGfloat44MultGGfloat44(kRotationY, kRotationX);
  matrix_rotation_ = GGfloat44MultGGfloat44(kRotationZ, matrix_rotation_);

  // Need to be updated if the Transformation matrix is called
  is_need_updated_ = true;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSGeometryTransformation::SetRotation(GGfloat3 const& rxyz)
{
  SetRotation(rxyz.s[0], rxyz.s[1], rxyz.s[2]);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSGeometryTransformation::SetAxisTransformation(GGfloat const& m00, GGfloat const& m01, GGfloat const& m02, GGfloat const& m10, GGfloat const& m11, GGfloat const& m12, GGfloat const& m20, GGfloat const& m21, GGfloat const& m22)
{
  GGfloat33 const kTmp = MakeFloat33(
    m00, m01, m02,
    m10, m11, m12,
    m20, m21, m22
  );

  SetAxisTransformation(kTmp);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSGeometryTransformation::SetAxisTransformation(GGfloat33 const& axis)
{
  // Filling the local axis buffer first
  local_axis_ = MakeFloat33(
    axis.m00_, axis.m01_, axis.m02_,
    axis.m10_, axis.m11_, axis.m12_,
    axis.m20_, axis.m21_, axis.m22_
  );

  matrix_orthographic_projection_ = MakeFloat44(
    axis.m00_, axis.m01_, axis.m02_, 0.0f,
    axis.m10_, axis.m11_, axis.m12_, 0.0f,
    axis.m20_, axis.m21_, axis.m22_, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
  );

  // Need to be updated if the Transformation matrix is called
  is_need_updated_ = true;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSGeometryTransformation::UpdateTransformationMatrix(void)
{
  GGcout("GGEMSGeometryTransformation", "UpdateTransformationMatrix", 3) << "Updating the transformation matrix..." << GGendl;

  // Update the transformation matrix on OpenCL device
  // Get the pointer on device
  GGfloat44* matrix_device = opencl_manager_.GetDeviceBuffer<GGfloat44>(matrix_transformation_cl_.get(), sizeof(GGfloat44));

  // Compute a temporary matrix then copy it on OpenCL device
  GGfloat44 matrix_tmp = GGfloat44MultGGfloat44(matrix_rotation_, GGfloat44MultGGfloat44(matrix_translation_, matrix_orthographic_projection_));

  // Copy step
  matrix_device->m00_ = matrix_tmp.m00_;
  matrix_device->m01_ = matrix_tmp.m01_;
  matrix_device->m02_ = matrix_tmp.m02_;
  matrix_device->m03_ = matrix_tmp.m03_;

  matrix_device->m10_ = matrix_tmp.m10_;
  matrix_device->m11_ = matrix_tmp.m11_;
  matrix_device->m12_ = matrix_tmp.m12_;
  matrix_device->m13_ = matrix_tmp.m13_;

  matrix_device->m20_ = matrix_tmp.m20_;
  matrix_device->m21_ = matrix_tmp.m21_;
  matrix_device->m22_ = matrix_tmp.m22_;
  matrix_device->m23_ = matrix_tmp.m23_;

  matrix_device->m30_ = matrix_tmp.m30_;
  matrix_device->m31_ = matrix_tmp.m31_;
  matrix_device->m32_ = matrix_tmp.m32_;
  matrix_device->m33_ = matrix_tmp.m33_;

  // Release the pointer, mandatory step!!!
  opencl_manager_.ReleaseDeviceBuffer(matrix_transformation_cl_.get(), matrix_device);

  // Update is done
  is_need_updated_ = false;
}
