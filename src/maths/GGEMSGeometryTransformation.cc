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
    {1.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f},
    {0.0f, 0.0f, 1.0f}
  );

  // Initializing translation matrix
  matrix_translation_ = MakeFloat44(
    {1.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 1.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, 1.0f}
  );

  // Initializing rotation matrix
  matrix_rotation_ = MakeFloat44(
    {1.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 1.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, 1.0f}
  );

  // Initializing orthographic projection matrix
  matrix_orthographic_projection_ = MakeFloat44(
    {1.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 1.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, 1.0f}
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
    {1.0f, 0.0f, 0.0f, position_.s[0]},
    {0.0f, 1.0f, 0.0f, position_.s[1]},
    {0.0f, 0.0f, 1.0f, position_.s[2]},
    {0.0f, 0.0f, 0.0f, 1.0f}
  );

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
  GGfloat cosinus = 0.0, sinus = 0.0;

  // X axis
  cosinus = cos(rx);
  sinus = sin(rx);

  GGfloat44 const kRotationX = MakeFloat44(
    {1.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, static_cast<GGfloat>(cosinus), -static_cast<GGfloat>(sinus), 0.0f},
    {0.0f, static_cast<GGfloat>(sinus), static_cast<GGfloat>(cosinus), 0.0f},
    {0.0f, 0.0f, 0.0f, 1.0f}
  );

  // Y axis
  cosinus = cos(ry);
  sinus = sin(ry);

  GGfloat44 const kRotationY = MakeFloat44(
    {static_cast<GGfloat>(cosinus), 0.0f, static_cast<GGfloat>(sinus), 0.0f},
    {0.0f, 1.0f, 0.0f, 0.0f},
    {-static_cast<GGfloat>(sinus), 0.0f, static_cast<GGfloat>(cosinus), 0.0f},
    {0.0f, 0.0f, 0.0f, 1.0f}
  );

  // Z axis
  cosinus = cos(rz);
  sinus = sin(rz);

  GGfloat44 const kRotationZ = MakeFloat44(
    {static_cast<GGfloat>(cosinus), -static_cast<GGfloat>(sinus), 0.0f, 0.0f},
    {static_cast<GGfloat>(sinus), static_cast<GGfloat>(cosinus), 0.0f, 0.0f},
    {0.0f, 0.0f, 1.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, 1.0f}
  );

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

void GGEMSGeometryTransformation::SetAxisTransformation(GGfloat3 const& m0, GGfloat3 const& m1, GGfloat3 const& m2)
{
  GGfloat33 const kTmp = MakeFloat33(m0, m1, m2);
  SetAxisTransformation(kTmp);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSGeometryTransformation::SetAxisTransformation(GGfloat33 const& axis)
{
  // Filling the local axis buffer first
  local_axis_ = MakeFloat33(axis.m0_, axis.m1_, axis.m2_);

  matrix_orthographic_projection_ = MakeFloat44(
    {axis.m0_.s[0], axis.m0_.s[1], axis.m0_.s[2], 0.0f},
    {axis.m1_.s[0], axis.m1_.s[1], axis.m1_.s[2], 0.0f},
    {axis.m2_.s[0], axis.m2_.s[1], axis.m2_.s[2], 0.0f},
    {0.0f, 0.0f, 0.0f, 1.0f}
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
  matrix_device->m0_ = matrix_tmp.m0_;
  matrix_device->m1_ = matrix_tmp.m1_;
  matrix_device->m2_ = matrix_tmp.m2_;
  matrix_device->m3_ = matrix_tmp.m3_;

  // Release the pointer, mandatory step!!!
  opencl_manager_.ReleaseDeviceBuffer(matrix_transformation_cl_.get(), matrix_device);

  // Update is done
  is_need_updated_ = false;
}
