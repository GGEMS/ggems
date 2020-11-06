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
: is_need_updated_(false)
{
  GGcout("GGEMSGeometryTransformation", "GGEMSGeometryTransformation", 3) << "Allocation of GGEMSGeometryTransformation..." << GGendl;

  // Initialize the position with min. float
  position_ = {std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min()};

  // Initialize the rotation with min. float
  rotation_ = {std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min()};

  // Initialize the local axis
  local_axis_ =
    {
      {1.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 1.0f}
    };

  // Initializing translation matrix
  matrix_translation_ =
    {
      {1.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 0.0f, 1.0f}
    };

  // Initializing rotation matrix
  matrix_rotation_ =
    {
      {1.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 0.0f, 1.0f}
    };

  // Initializing orthographic projection matrix
  matrix_orthographic_projection_ =
    {
      {1.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 0.0f, 1.0f}
    };

  // Get OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Allocation of matrix transformation on OpenCL device
  matrix_transformation_cl_ = opencl_manager.Allocate(nullptr, sizeof(GGfloat44), CL_MEM_READ_WRITE);
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
  position_ = {tx, ty, tz};

  // Filling the translation matrix
  matrix_translation_ =
    {
      {1.0f, 0.0f, 0.0f, position_.s0},
      {0.0f, 1.0f, 0.0f, position_.s1},
      {0.0f, 0.0f, 1.0f, position_.s2},
      {0.0f, 0.0f, 0.0f, 1.0f}
    };

  // Need to be updated if the Transformation matrix is called
  is_need_updated_ = true;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSGeometryTransformation::SetTranslation(GGfloat3 const& txyz)
{
  SetTranslation(txyz.s0, txyz.s1, txyz.s2);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSGeometryTransformation::SetRotation(GGfloat const& rx, GGfloat const& ry, GGfloat const& rz)
{
  // Filling the rotation buffer
  rotation_ = {rx, ry, rz};

  // Definition of cosinus and sinus
  GGfloat cosinus = 0.0, sinus = 0.0;

  // X axis
  cosinus = cos(rx);
  sinus = sin(rx);

  GGfloat44 rotation_x =
    {
      {1.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, cosinus, -sinus, 0.0f},
      {0.0f, sinus, cosinus, 0.0f},
      {0.0f, 0.0f, 0.0f, 1.0f}
    };

  // Y axis
  cosinus = cos(ry);
  sinus = sin(ry);

  GGfloat44 rotation_y =
    {
      {cosinus, 0.0f, sinus, 0.0f},
      {0.0f, 1.0f, 0.0f, 0.0f},
      {-sinus, 0.0f, cosinus, 0.0f},
      {0.0f, 0.0f, 0.0f, 1.0f}
    };

  // Z axis
  cosinus = cos(rz);
  sinus = sin(rz);

  GGfloat44 rotation_z =
   {
      {cosinus, -sinus, 0.0f, 0.0f},
      {sinus, cosinus, 0.0f, 0.0f},
      {0.0f, 0.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 0.0f, 1.0f}
   };

  // Get the total rotation matrix
  matrix_rotation_ = GGfloat44MultGGfloat44(rotation_y, rotation_x);
  matrix_rotation_ = GGfloat44MultGGfloat44(rotation_z, matrix_rotation_);

  // Need to be updated if the Transformation matrix is called
  is_need_updated_ = true;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSGeometryTransformation::SetRotation(GGfloat3 const& rxyz)
{
  SetRotation(rxyz.s0, rxyz.s1, rxyz.s2);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSGeometryTransformation::SetAxisTransformation(GGfloat3 const& m0, GGfloat3 const& m1, GGfloat3 const& m2)
{
  SetAxisTransformation(
    {
      {m0.s0, m0.s1, m0.s2},
      {m1.s0, m1.s1, m1.s1},
      {m2.s0, m2.s1, m2.s2}
    }
  );
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSGeometryTransformation::SetAxisTransformation(GGfloat33 const& axis)
{
  // Filling the local axis buffer first
  local_axis_ =
    {
      {axis.m0_[0], axis.m0_[1], axis.m0_[2]},
      {axis.m1_[0], axis.m1_[1], axis.m1_[2]},
      {axis.m2_[0], axis.m2_[1], axis.m2_[2]}
    };

  matrix_orthographic_projection_ =
    {
      {axis.m0_[0], axis.m0_[1], axis.m0_[2], 0.0f},
      {axis.m1_[0], axis.m1_[1], axis.m1_[2], 0.0f},
      {axis.m2_[0], axis.m2_[1], axis.m2_[2], 0.0f},
      {0.0f, 0.0f, 0.0f, 1.0f}
    };

  // Need to be updated if the Transformation matrix is called
  is_need_updated_ = true;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSGeometryTransformation::UpdateTransformationMatrix(void)
{
  GGcout("GGEMSGeometryTransformation", "UpdateTransformationMatrix", 3) << "Updating the transformation matrix..." << GGendl;

  // Get OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Update the transformation matrix on OpenCL device
  // Get the pointer on device
  GGfloat44* matrix_device = opencl_manager.GetDeviceBuffer<GGfloat44>(matrix_transformation_cl_.get(), sizeof(GGfloat44));

  // Compute a temporary matrix then copy it on OpenCL device
  GGfloat44 matrix_tmp = GGfloat44MultGGfloat44(matrix_rotation_, GGfloat44MultGGfloat44(matrix_translation_, matrix_orthographic_projection_));

  // Copy step
  matrix_device->m0_[0] = matrix_tmp.m0_[0]; matrix_device->m0_[1] = matrix_tmp.m0_[1]; matrix_device->m0_[2] = matrix_tmp.m0_[2]; matrix_device->m0_[3] = matrix_tmp.m0_[3];
  matrix_device->m1_[0] = matrix_tmp.m1_[0]; matrix_device->m1_[1] = matrix_tmp.m1_[1]; matrix_device->m1_[2] = matrix_tmp.m1_[2]; matrix_device->m1_[3] = matrix_tmp.m1_[3];
  matrix_device->m2_[0] = matrix_tmp.m2_[0]; matrix_device->m2_[1] = matrix_tmp.m2_[1]; matrix_device->m2_[2] = matrix_tmp.m2_[2]; matrix_device->m2_[3] = matrix_tmp.m2_[3];
  matrix_device->m3_[0] = matrix_tmp.m3_[0]; matrix_device->m3_[1] = matrix_tmp.m3_[1]; matrix_device->m3_[2] = matrix_tmp.m3_[2]; matrix_device->m3_[3] = matrix_tmp.m3_[3];

  // Release the pointer, mandatory step!!!
  opencl_manager.ReleaseDeviceBuffer(matrix_transformation_cl_.get(), matrix_device);

  // Update is done
  is_need_updated_ = false;
}
