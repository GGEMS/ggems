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
  \file GGEMSSolid.cc

  \brief GGEMS class for solid. This class store geometry about phantom or detector

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 2, 2020
*/

#include "GGEMS/geometries/GGEMSSolid.hh"
#include "GGEMS/sources/GGEMSSourceManager.hh"
#include "GGEMS/physics/GGEMSCrossSections.hh"
#include "GGEMS/randoms/GGEMSPseudoRandomGenerator.hh"
#include "GGEMS/maths/GGEMSGeometryTransformation.hh"
#include "GGEMS/geometries/GGEMSSolidBoxData.hh"
#include "GGEMS/geometries/GGEMSVoxelizedSolidData.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSolid::GGEMSSolid(void)
: kernel_option_("")
{
  GGcout("GGEMSSolid", "GGEMSSolid", 3) << "GGEMSSolid creating..." << GGendl;

  // Allocation of geometry transformation
  geometry_transformation_ = new GGEMSGeometryTransformation();
  data_reg_type_ = "";

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  number_activated_devices_ = opencl_manager.GetNumberOfActivatedDevice();

  solid_data_ = new cl::Buffer*[number_activated_devices_];
  label_data_ = new cl::Buffer*[number_activated_devices_];
  for (GGsize i = 0; i < number_activated_devices_; ++i) label_data_[i] = nullptr;

  // Storing a kernel for each device
  kernel_particle_solid_distance_ = new cl::Kernel*[number_activated_devices_];
  kernel_project_to_solid_ = new cl::Kernel*[number_activated_devices_];
  kernel_track_through_solid_ = new cl::Kernel*[number_activated_devices_];

  GGcout("GGEMSSolid", "GGEMSSolid", 3) << "GGEMSSolid created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSolid::~GGEMSSolid(void)
{
  GGcout("GGEMSSolid", "~GGEMSSolid", 3) << "GGEMSSolid erasing..." << GGendl;

  if (kernel_particle_solid_distance_) {
    delete[] kernel_particle_solid_distance_;
    kernel_particle_solid_distance_ = nullptr;
  }

  if (kernel_project_to_solid_) {
    delete[] kernel_project_to_solid_;
    kernel_project_to_solid_ = nullptr;
  }

  if (kernel_track_through_solid_) {
    delete[] kernel_track_through_solid_;
    kernel_track_through_solid_ = nullptr;
  }

  if (geometry_transformation_) {
    delete geometry_transformation_;
    geometry_transformation_ = nullptr;
  }

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  if (label_data_) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      GGEMSVoxelizedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data_[i], sizeof(GGEMSVoxelizedSolidData), i);
      GGsize number_of_voxels = static_cast<GGsize>(solid_data_device->number_of_voxels_);
      opencl_manager.ReleaseDeviceBuffer(solid_data_[i], solid_data_device, i);
      opencl_manager.Deallocate(label_data_[i], number_of_voxels*sizeof(GGuchar), i);
    }
    delete[] label_data_;
    label_data_ = nullptr;
  }

  if (solid_data_) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      opencl_manager.Deallocate(solid_data_[i], sizeof(GGEMSSolidBoxData), i);
    }
    delete[] solid_data_;
    solid_data_ = nullptr;
  }

  // if (histogram_.histogram_) {
  //   for (GGsize i = 0; i < number_activated_devices_; ++i) {
  //     opencl_manager.Deallocate(histogram_.histogram_[i], histogram_.number_of_elements_*sizeof(GGint), i);
  //   }
  //   delete[] histogram_.histogram_;
  //   histogram_.histogram_ = nullptr;
  // }

  GGcout("GGEMSSolid", "~GGEMSSolid", 3) << "GGEMSSolid erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::EnableTracking(void)
{
  kernel_option_ += " -DGGEMS_TRACKING";
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::SetRotation(GGfloat3 const& rotation_xyz)
{
  geometry_transformation_->SetRotation(rotation_xyz);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::SetPosition(GGfloat3 const& position_xyz)
{
  geometry_transformation_->SetTranslation(position_xyz);
}
