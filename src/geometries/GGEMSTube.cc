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
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"
#include "GGEMS/tools/GGEMSProfilerManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSTube::GGEMSTube(GGfloat const& radius_x, GGfloat const& radius_y, GGfloat const& height, std::string const& unit)
: GGEMSVolume()
{
  GGcout("GGEMSTube", "GGEMSTube", 3) << "GGEMSTube creating..." << GGendl;

  height_ = DistanceUnit(height, unit);
  radius_x_ = DistanceUnit(radius_x, unit);
  radius_y_ = DistanceUnit(radius_y, unit);

  GGcout("GGEMSTube", "GGEMSTube", 3) << "GGEMSTube created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSTube::~GGEMSTube(void)
{
  GGcout("GGEMSTube", "~GGEMSTube", 3) << "GGEMSTube erasing..." << GGendl;

  GGcout("GGEMSTube", "~GGEMSTube", 3) << "GGEMSTube erased!!!" << GGendl;
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
  opencl_manager.CompileKernel(kFilename, "draw_ggems_tube", kernel_draw_volume_, nullptr, const_cast<char*>(kDataType.c_str()));
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
  cl::CommandQueue* queue = opencl_manager.GetCommandQueue(0);

  // Get Device name and storing methode name + device
  GGsize device_index = opencl_manager.GetIndexOfActivatedDevice(0);
  std::string device_name = opencl_manager.GetDeviceName(device_index);
  std::ostringstream oss(std::ostringstream::out);
  oss << "GGEMSTube::Draw on " << device_name << ", index " << device_index;

  // Get parameters from phantom creator
  GGfloat3 voxel_sizes = volume_creator_manager.GetElementsSizes();

  GGint3 phantom_dimensions;
  phantom_dimensions.s[0] = static_cast<GGint>(volume_creator_manager.GetVolumeDimensions().x_);
  phantom_dimensions.s[1] = static_cast<GGint>(volume_creator_manager.GetVolumeDimensions().y_);
  phantom_dimensions.s[2] = static_cast<GGint>(volume_creator_manager.GetVolumeDimensions().z_);

  GGsize number_of_elements = volume_creator_manager.GetNumberElements();
  cl::Buffer* voxelized_phantom = volume_creator_manager.GetVoxelizedVolume();

  // Set parameters for kernel
  GGuint index_arg = 0;
  kernel_draw_volume_[0]->setArg(index_arg++, number_of_elements);
  kernel_draw_volume_[0]->setArg(index_arg++, voxel_sizes);
  kernel_draw_volume_[0]->setArg(index_arg++, phantom_dimensions);
  kernel_draw_volume_[0]->setArg(index_arg++, positions_);
  kernel_draw_volume_[0]->setArg(index_arg++, label_value_);
  kernel_draw_volume_[0]->setArg(index_arg++, height_);
  kernel_draw_volume_[0]->setArg(index_arg++, radius_x_);
  kernel_draw_volume_[0]->setArg(index_arg++, radius_y_);
  kernel_draw_volume_[0]->setArg(index_arg++, *voxelized_phantom);

  // Compute parameter to optimize work-item for OpenCL
  GGsize work_group_size = opencl_manager.GetKernelWorkGroupSize(kernel_draw_volume_[0]);
  GGsize max_work_item = opencl_manager.GetDeviceMaxWorkItemSize(device_index, 0) * work_group_size;

  // Number total of work items
  GGsize number_of_work_item = (((number_of_elements - 1) / work_group_size) + 1) * work_group_size; // Multiple of work group size

  // Organize work item in batch
  bool is_last_batch = false;
  GGsize number_of_work_item_in_batch = max_work_item;
  GGsize number_batchs_work_item = (number_of_work_item / (max_work_item + 1)) + 1;

  for (GGsize i = 0; i < number_batchs_work_item; ++i) {
    if (i == number_batchs_work_item - 1) is_last_batch = true;

    if (is_last_batch) {
      number_of_work_item_in_batch = (number_of_work_item % max_work_item) == 0 ?
        max_work_item :
        number_of_work_item % max_work_item;
    }

    cl::NDRange global_wi(number_of_work_item_in_batch);
    cl::NDRange offset_wi(i * max_work_item);
    cl::NDRange local_wi(number_of_work_item_in_batch / work_group_size);

    cl::Event event;
    GGint kernel_status = queue->enqueueNDRangeKernel(*kernel_draw_volume_[0], offset_wi, global_wi, local_wi, nullptr, &event);
    opencl_manager.CheckOpenCLError(kernel_status, "GGEMSTube", "Draw");

    // GGEMS Profiling
    GGEMSProfilerManager::GetInstance().HandleEvent(event, oss.str());

    queue->finish();
  }
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
