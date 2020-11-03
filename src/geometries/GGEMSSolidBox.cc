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
  \file GGEMSSolidBox.hh

  \brief GGEMS class for solid box

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday October 27, 2020
*/

#include "GGEMS/geometries/GGEMSSolidBox.hh"
#include "GGEMS/maths/GGEMSGeometryTransformation.hh"
#include "GGEMS/geometries/GGEMSSolidBoxStack.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSolidBox::GGEMSSolidBox(GGfloat const& length_x, GGfloat const& length_y, GGfloat const& length_z)
: GGEMSSolid()
{
  GGcout("GGEMSSolidBox", "GGEMSSolidBox", 3) << "Allocation of GGEMSSolidBox..." << GGendl;

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Allocation of memory on OpenCL device for header data
  solid_data_cl_ = opencl_manager.Allocate(nullptr, sizeof(GGEMSSolidBoxData), CL_MEM_READ_WRITE);

  // Fill the lengths in OpenCL device
  GGEMSSolidBoxData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSSolidBoxData>(solid_data_cl_.get(), sizeof(GGEMSSolidBoxData));

  solid_data_device->length_xyz_.x = length_x;
  solid_data_device->length_xyz_.y = length_y;
  solid_data_device->length_xyz_.z = length_z;

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(solid_data_cl_.get(), solid_data_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSolidBox::~GGEMSSolidBox(void)
{
  GGcout("GGEMSSolidBox", "~GGEMSSolidBox", 3) << "Deallocation of GGEMSSolidBox..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolidBox::InitializeKernel(void)
{
  GGcout("GGEMSSolidBox", "InitializeKernel", 3) << "Initializing kernel for solid box..." << GGendl;

  // // Getting OpenCL manager
  // GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // // Getting the path to kernel
  // std::string const kOpenCLKernelPath = OPENCL_KERNEL_PATH;
  // std::string const kFilename1 = kOpenCLKernelPath + "/DistanceSolidBox.cl";
  // std::string const kFilename2 = kOpenCLKernelPath + "/ProjectToSolidBox.cl";
  // std::string const kFilename3 = kOpenCLKernelPath + "/TrackThroughSolidBox.cl";

  // Compiling the kernels
  //kernel_distance_cl_ = opencl_manager.CompileKernel(kFilename1, "distance_voxelized_solid");
  //kernel_project_to_cl_ = opencl_manager.CompileKernel(kFilename2, "project_to_voxelized_solid");
  //kernel_track_through_cl_ = opencl_manager.CompileKernel(kFilename3, "track_through_voxelized_solid", nullptr, const_cast<char*>(tracking_kernel_option_.c_str()));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolidBox::Initialize(std::weak_ptr<GGEMSMaterials> materials)
{
  GGcout("GGEMSSolidBox", "Initialize", 3) << "Initializing voxelized solid..." << GGendl;

  // Initializing kernels
  InitializeKernel();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolidBox::PrintInfos(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device
  GGEMSSolidBoxData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSSolidBoxData>(solid_data_cl_.get(), sizeof(GGEMSSolidBoxData));

  GGcout("GGEMSSolidBox", "PrintInfos", 0) << GGendl;
  GGcout("GGEMSSolidBox", "PrintInfos", 0) << "GGEMSSolidBox Infos:" << GGendl;
  GGcout("GGEMSSolidBox", "PrintInfos", 0) << "--------------------------" << GGendl;
  GGcout("GGEMSSolidBox", "PrintInfos", 0) << "* Lengths: (" << solid_data_device->length_xyz_.s[0] << "x" << solid_data_device->length_xyz_.s[1] << "x" << solid_data_device->length_xyz_.s[2] << ") mm3" << GGendl;
  GGcout("GGEMSSolidBox", "PrintInfos", 0) << "* Position: (" << solid_data_device->position_xyz_.s[0] << "x" << solid_data_device->position_xyz_.s[1] << "x" << solid_data_device->position_xyz_.s[2] << ") mm3" << GGendl;
  GGcout("GGEMSSolidBox", "PrintInfos", 0) << "* Oriented bounding box (OBB):" << GGendl;
  // GGcout("GGEMSSolidBox", "PrintInfos", 0) << "    - X: " << solid_data_device->obb_geometry_.border_min_xyz_.s[0] << " <-> " << solid_data_device->obb_geometry_.border_max_xyz_.s[0] << GGendl;
  // GGcout("GGEMSSolidBox", "PrintInfos", 0) << "    - Y: " << solid_data_device->obb_geometry_.border_min_xyz_.s[1] << " <-> " << solid_data_device->obb_geometry_.border_max_xyz_.s[1] << GGendl;
  // GGcout("GGEMSSolidBox", "PrintInfos", 0) << "    - Z: " << solid_data_device->obb_geometry_.border_min_xyz_.s[2] << " <-> " << solid_data_device->obb_geometry_.border_max_xyz_.s[2] << GGendl;
  // GGcout("GGEMSSolidBox", "PrintInfos", 0) << "    - Transformation matrix:" << GGendl;
  // GGcout("GGEMSSolidBox", "PrintInfos", 0) << "    [" << GGendl;
  // GGcout("GGEMSSolidBox", "PrintInfos", 0) << "        " << solid_data_device->obb_geometry_.matrix_transformation_.m0_.s[0] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m0_.s[1] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m0_.s[2] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m0_.s[3] << GGendl;
  // GGcout("GGEMSSolidBox", "PrintInfos", 0) << "        " << solid_data_device->obb_geometry_.matrix_transformation_.m1_.s[0] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m1_.s[1] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m1_.s[2] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m1_.s[3] << GGendl;
  // GGcout("GGEMSSolidBox", "PrintInfos", 0) << "        " << solid_data_device->obb_geometry_.matrix_transformation_.m2_.s[0] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m2_.s[1] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m2_.s[2] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m2_.s[3] << GGendl;
  // GGcout("GGEMSSolidBox", "PrintInfos", 0) << "        " << solid_data_device->obb_geometry_.matrix_transformation_.m3_.s[0] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m3_.s[1] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m3_.s[2] << " " << solid_data_device->obb_geometry_.matrix_transformation_.m3_.s[3] << GGendl;
  // GGcout("GGEMSSolidBox", "PrintInfos", 0) << "    ]" << GGendl;
  GGcout("GGEMSSolidBox", "PrintInfos", 0) << "* Solid index: " << static_cast<GGint>(solid_data_device->solid_id_) << GGendl;
  GGcout("GGEMSSolidBox", "PrintInfos", 0) << GGendl;

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(solid_data_cl_.get(), solid_data_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolidBox::SetPosition(GGfloat3 const& position_xyz)
{
  GGcout("GGEMSSolidBox", "SetPosition", 3) << "Setting position of solid box..." << GGendl;

  // Set position in geometric transformation
  geometry_transformation_->SetTranslation(position_xyz);

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device
  GGEMSSolidBoxData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSSolidBoxData>(solid_data_cl_.get(), sizeof(GGEMSSolidBoxData));
  GGfloat44* transformation_matrix_device = opencl_manager.GetDeviceBuffer<GGfloat44>(geometry_transformation_->GetTransformationMatrix(), sizeof(GGfloat44));

  for (GGuint i = 0; i < 3; ++i ) {
    // Offset
    solid_data_device->position_xyz_.s[i] = position_xyz.s[i];

    // Bounding box
    //solid_data_device->obb_geometry_.border_min_xyz_.s[i] = -solid_data_device->position_xyz_.s[i];
    //solid_data_device->obb_geometry_.border_max_xyz_.s[i] = solid_data_device->obb_geometry_.border_min_xyz_.s[i] + solid_data_device->number_of_voxels_xyz_.s[i] * solid_data_device->voxel_sizes_xyz_.s[i];
  }

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(solid_data_cl_.get(), solid_data_device);
  opencl_manager.ReleaseDeviceBuffer(geometry_transformation_->GetTransformationMatrix(), transformation_matrix_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolidBox::UpdateTransformationMatrix(void)
{
  geometry_transformation_->UpdateTransformationMatrix();

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Copy information to OBB
  GGEMSSolidBoxData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSSolidBoxData>(solid_data_cl_.get(), sizeof(GGEMSSolidBoxData));
  GGfloat44* transformation_matrix_device = opencl_manager.GetDeviceBuffer<GGfloat44>(geometry_transformation_->GetTransformationMatrix(), sizeof(GGfloat44));

  // solid_data_device->obb_geometry_.matrix_transformation_.m0_ = transformation_matrix_device->m0_;
  // solid_data_device->obb_geometry_.matrix_transformation_.m1_ = transformation_matrix_device->m1_;
  // solid_data_device->obb_geometry_.matrix_transformation_.m2_ = transformation_matrix_device->m2_;
  // solid_data_device->obb_geometry_.matrix_transformation_.m3_ = transformation_matrix_device->m3_;

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(solid_data_cl_.get(), solid_data_device);
  opencl_manager.ReleaseDeviceBuffer(geometry_transformation_->GetTransformationMatrix(), transformation_matrix_device);
}
