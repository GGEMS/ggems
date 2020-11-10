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
  \file GGEMSTube.cc

  \brief Class GGEMSTube inheriting from GGEMSVolumeSolid handling Tube solid

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday January 13, 2020
*/

#include "GGEMS/geometries/GGEMSTube.hh"

#include "GGEMS/tools/GGEMSTools.hh"
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSTube::GGEMSTube(GGfloat const& radius_x, GGfloat const& radius_y, GGfloat const& height, std::string const& unit)
: GGEMSVolume()
{
  GGcout("GGEMSTube", "GGEMSTube", 3) << "Allocation of GGEMSTube..." << GGendl;

  height_ = DistanceUnit(height, unit);
  radius_x_ = DistanceUnit(radius_x, unit);
  radius_y_ = DistanceUnit(radius_y, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSTube::~GGEMSTube(void)
{
  GGcout("GGEMSTube", "~GGEMSTube", 3) << "Deallocation of GGEMSTube..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSTube::Initialize(void)
{
  GGcout("GGEMSTube", "Initialize", 3) << "Initializing GGEMSTube solid volume..." << GGendl;

  // Getting the path to kernel
  std::string const kOpenCLKernelPath = OPENCL_KERNEL_PATH;
  std::string const kFilename = kOpenCLKernelPath + "/DrawGGEMSTube.cl";

  // Get the volume creator manager
  GGEMSVolumeCreatorManager& volume_creator_manager = GGEMSVolumeCreatorManager::GetInstance();

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get the data type and compiling kernel
  std::string const kDataType = "-D" + volume_creator_manager.GetDataType();
  kernel_draw_volume_cl_ = opencl_manager.CompileKernel(kFilename, "draw_ggems_tube", nullptr, const_cast<char*>(kDataType.c_str()));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSTube::Draw(void)
{
  GGcout("GGEMSTube", "Draw", 3) << "Drawing Tube..." << GGendl;

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get the volume creator manager
  GGEMSVolumeCreatorManager& volume_creator_manager = GGEMSVolumeCreatorManager::GetInstance();

  // Get command queue and event
  cl::CommandQueue* p_queue_cl = opencl_manager.GetCommandQueue();
  cl::Event* p_event_cl = opencl_manager.GetEvent();

  // Get parameters from phantom creator
  GGfloat3 voxel_sizes = volume_creator_manager.GetElementsSizes();
  GGint3 phantom_dimensions = volume_creator_manager.GetVolumeDimensions();
  GGint number_of_elements = volume_creator_manager.GetNumberElements();
  cl::Buffer* voxelized_phantom = volume_creator_manager.GetVoxelizedVolume();

  // Set parameters for kernel
  std::shared_ptr<cl::Kernel> kernel_cl = kernel_draw_volume_cl_.lock();
  kernel_cl->setArg(0, number_of_elements);
  kernel_cl->setArg(1, voxel_sizes);
  kernel_cl->setArg(2, phantom_dimensions);
  kernel_cl->setArg(3, positions_);
  kernel_cl->setArg(4, label_value_);
  kernel_cl->setArg(5, height_);
  kernel_cl->setArg(6, radius_x_);
  kernel_cl->setArg(7, radius_y_);
  kernel_cl->setArg(8, *voxelized_phantom);

  // Get number of max work group size
  std::size_t max_work_group_size = opencl_manager.GetMaxWorkGroupSize();

  // Compute work item number
  std::size_t number_of_work_items = number_of_elements + (max_work_group_size - number_of_elements%max_work_group_size);

  cl::NDRange global_wi(number_of_work_items);
  cl::NDRange offset_wi(0);
  cl::NDRange local_wi(max_work_group_size);

  // Launching kernel
  cl_int kernel_status = p_queue_cl->enqueueNDRangeKernel(*kernel_cl, offset_wi, global_wi, local_wi, nullptr, p_event_cl);
  opencl_manager.CheckOpenCLError(kernel_status, "GGEMSTube", "Draw");
  p_queue_cl->finish(); // Wait until the kernel status is finish

  // Displaying time in kernel
  opencl_manager.DisplayElapsedTimeInKernel("draw_ggems_tube");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSTube* create_tube(GGfloat const radius_x, GGfloat const radius_y, GGfloat const height, char const* unit)
{
  return new(std::nothrow) GGEMSTube(radius_x, radius_y, height, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void delete_tube(GGEMSTube* tube)
{
  if (tube) {
    delete tube;
    tube = nullptr;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_position_tube(GGEMSTube* tube, GGfloat const pos_x, GGfloat const pos_y, GGfloat const pos_z, char const* unit)
{
  tube->SetPosition(pos_x, pos_y, pos_z, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_material_tube(GGEMSTube* tube, char const* material)
{
  tube->SetMaterial(material);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_label_value_tube(GGEMSTube* tube, GGfloat const label_value)
{
  tube->SetLabelValue(label_value);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void initialize_tube(GGEMSTube* tube)
{
  tube->Initialize();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void draw_tube(GGEMSTube* tube)
{
  tube->Draw();
}
