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
  \file GGEMSSphere.cc

  \brief Class GGEMSSphere inheriting from GGEMSVolume handling Sphere solid

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday November 4, 2020
*/

#include "GGEMS/geometries/GGEMSSphere.hh"
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"
#include "GGEMS/tools/GGEMSProfilerManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSphere::GGEMSSphere(GGfloat const& radius, std::string const& unit)
: GGEMSVolume()
{
  GGcout("GGEMSSphere", "GGEMSSphere", 3) << "Allocation of GGEMSSphere..." << GGendl;

  radius_ = DistanceUnit(radius, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSphere::~GGEMSSphere(void)
{
  GGcout("GGEMSSphere", "~GGEMSSphere", 3) << "Deallocation of GGEMSSphere..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSphere::Initialize(void)
{
  GGcout("GGEMSSphere", "Initialize", 3) << "Initializing GGEMSSphere solid volume..." << GGendl;

  // Getting the path to kernel
  std::string const kOpenCLKernelPath = OPENCL_KERNEL_PATH;
  std::string const kFilename = kOpenCLKernelPath + "/DrawGGEMSSphere.cl";

  // Get the volume creator manager
  GGEMSVolumeCreatorManager& volume_creator_manager = GGEMSVolumeCreatorManager::GetInstance();

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get the data type and compiling kernel
  std::string const kDataType = "-D" + volume_creator_manager.GetDataType();
  kernel_draw_volume_ = opencl_manager.CompileKernel(kFilename, "draw_ggems_sphere", nullptr, const_cast<char*>(kDataType.c_str()));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSphere::Draw(void)
{
  GGcout("GGEMSSphere", "Draw", 3) << "Drawing Sphere..." << GGendl;

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get the volume creator manager
  GGEMSVolumeCreatorManager& volume_creator_manager = GGEMSVolumeCreatorManager::GetInstance();

  // Get command queue and event
  cl::CommandQueue* queue = opencl_manager.GetCommandQueue();
  cl::Event* event = opencl_manager.GetEvent();

  // Get parameters from phantom creator
  GGfloat3 voxel_sizes = volume_creator_manager.GetElementsSizes();

  GGint3 phantom_dimensions;
  phantom_dimensions.x = static_cast<GGint>(volume_creator_manager.GetVolumeDimensions().x_);
  phantom_dimensions.y = static_cast<GGint>(volume_creator_manager.GetVolumeDimensions().y_);
  phantom_dimensions.z = static_cast<GGint>(volume_creator_manager.GetVolumeDimensions().z_);

  GGsize number_of_elements = volume_creator_manager.GetNumberElements();
  cl::Buffer* voxelized_phantom = volume_creator_manager.GetVoxelizedVolume();

  // Getting work group size, and work-item number
  GGsize work_group_size = opencl_manager.GetWorkGroupSize();
  GGsize number_of_work_items = opencl_manager.GetBestWorkItem(number_of_elements);

  // Parameters for work-item in kernel
  cl::NDRange global_wi(number_of_work_items);
  cl::NDRange local_wi(work_group_size);

  // Set parameters for kernel
  std::shared_ptr<cl::Kernel> kernel = kernel_draw_volume_.lock();
  kernel->setArg(0, number_of_elements);
  kernel->setArg(1, voxel_sizes);
  kernel->setArg(2, phantom_dimensions);
  kernel->setArg(3, positions_);
  kernel->setArg(4, label_value_);
  kernel->setArg(5, radius_);
  kernel->setArg(6, *voxelized_phantom);

  // Launching kernel
  cl_int kernel_status = queue->enqueueNDRangeKernel(*kernel, 0, global_wi, local_wi, nullptr, event);
  opencl_manager.CheckOpenCLError(kernel_status, "GGEMSSphere", "Draw");
  queue->finish();

  // GGEMS Profiling
  GGEMSProfilerManager& profiler_manager = GGEMSProfilerManager::GetInstance();
  profiler_manager.HandleEvent(*event, "GGEMSSphere::Draw");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSphere* create_sphere(GGfloat const radius, char const* unit)
{
  return new(std::nothrow) GGEMSSphere(radius, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void delete_sphere(GGEMSSphere* sphere)
{
  if (sphere) {
    delete sphere;
    sphere = nullptr;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_position_sphere(GGEMSSphere* sphere, GGfloat const pos_x, GGfloat const pos_y, GGfloat const pos_z, char const* unit)
{
  sphere->SetPosition(pos_x, pos_y, pos_z, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_material_sphere(GGEMSSphere* sphere, char const* material)
{
  sphere->SetMaterial(material);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_label_value_sphere(GGEMSSphere* sphere, GGfloat const label_value)
{
  sphere->SetLabelValue(label_value);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void initialize_sphere(GGEMSSphere* sphere)
{
  sphere->Initialize();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void draw_sphere(GGEMSSphere* sphere)
{
  sphere->Draw();
}
