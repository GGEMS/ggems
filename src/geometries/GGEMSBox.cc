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
  \file GGEMSBox.cc

  \brief Class GGEMSBox inheriting from GGEMSVolume handling Box solid

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday August 31, 2020
*/

#include "GGEMS/geometries/GGEMSBox.hh"
#include "GGEMS/tools/GGEMSProfilerManager.hh"
#include "GGEMS/tools/GGEMSTools.hh"
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSBox::GGEMSBox(GGfloat const& width, GGfloat const& height, GGfloat const& depth, std::string const& unit)
: GGEMSVolume()
{
  GGcout("GGEMSBox", "GGEMSBox", 3) << "GGEMSBox creating..." << GGendl;

  width_ = DistanceUnit(width, unit);
  height_ = DistanceUnit(height, unit);
  depth_ = DistanceUnit(depth, unit);

  GGcout("GGEMSBox", "GGEMSBox", 3) << "GGEMSBox created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSBox::~GGEMSBox(void)
{
  GGcout("GGEMSBox", "~GGEMSBox", 3) << "GGEMSBox erasing..." << GGendl;

  GGcout("GGEMSBox", "~GGEMSBox", 3) << "GGEMSBox erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSBox::Initialize(void)
{
  GGcout("GGEMSBox", "Initialize", 3) << "Initializing GGEMSBox solid volume..." << GGendl;

  // Getting the path to kernel
  std::string const kOpenCLKernelPath = OPENCL_KERNEL_PATH;
  std::string const kFilename = kOpenCLKernelPath + "/DrawGGEMSBox.cl";

  // Get the volume creator manager
  GGEMSVolumeCreatorManager& volume_creator_manager = GGEMSVolumeCreatorManager::GetInstance();

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get the data type and compiling kernel
  std::string const kDataType = "-D" + volume_creator_manager.GetDataType();
  opencl_manager.CompileKernel(kFilename, "draw_ggems_box", kernel_draw_volume_, nullptr, const_cast<char*>(kDataType.c_str()));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSBox::Draw(void)
{
  GGcout("GGEMSBox", "Draw", 3) << "Drawing Box..." << GGendl;

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get the volume creator manager
  GGEMSVolumeCreatorManager& volume_creator_manager = GGEMSVolumeCreatorManager::GetInstance();

  // Get command queue and event
  cl::CommandQueue* queue = opencl_manager.GetCommandQueue(0);
  cl::Event* event = opencl_manager.GetEvent(0);

  // Get Device name and storing methode name + device
  GGsize device_index = opencl_manager.GetIndexOfActivatedDevice(0);
  std::string device_name = opencl_manager.GetDeviceName(device_index);
  std::ostringstream oss(std::ostringstream::out);
  oss << "GGEMSBox::Draw on " << device_name << ", index " << device_index;

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
  kernel_draw_volume_[0]->setArg(0, number_of_elements);
  kernel_draw_volume_[0]->setArg(1, voxel_sizes);
  kernel_draw_volume_[0]->setArg(2, phantom_dimensions);
  kernel_draw_volume_[0]->setArg(3, positions_);
  kernel_draw_volume_[0]->setArg(4, label_value_);
  kernel_draw_volume_[0]->setArg(5, height_);
  kernel_draw_volume_[0]->setArg(6, width_);
  kernel_draw_volume_[0]->setArg(7, depth_);
  kernel_draw_volume_[0]->setArg(8, *voxelized_phantom);

  // Launching kernel
  GGint kernel_status = queue->enqueueNDRangeKernel(*kernel_draw_volume_[0], 0, global_wi, local_wi, nullptr, event);
  opencl_manager.CheckOpenCLError(kernel_status, "GGEMSBox", "Draw");

  // GGEMS Profiling
  GGEMSProfilerManager& profiler_manager = GGEMSProfilerManager::GetInstance();
  profiler_manager.HandleEvent(*event, oss.str());

  queue->finish();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSBox* create_box(GGfloat const width, GGfloat const height, GGfloat const depth, char const* unit)
{
  return new(std::nothrow) GGEMSBox(width, height, depth, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void delete_box(GGEMSBox* box)
{
  if (box) {
    delete box;
    box = nullptr;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_position_box(GGEMSBox* box, GGfloat const pos_x, GGfloat const pos_y, GGfloat const pos_z, char const* unit)
{
  box->SetPosition(pos_x, pos_y, pos_z, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_material_box(GGEMSBox* box, char const* material)
{
  box->SetMaterial(material);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_label_value_box(GGEMSBox* box, GGfloat const label_value)
{
  box->SetLabelValue(label_value);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void initialize_box(GGEMSBox* box)
{
  box->Initialize();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void draw_box(GGEMSBox* box)
{
  box->Draw();
}
