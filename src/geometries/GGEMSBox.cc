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
#include "GGEMS/tools/GGEMSTools.hh"
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSBox::GGEMSBox(GGfloat const& width, GGfloat const& height, GGfloat const& depth, std::string const& unit)
: GGEMSVolume()
{
  GGcout("GGEMSBox", "GGEMSBox", 3) << "Allocation of GGEMSBox..." << GGendl;

  width_ = DistanceUnit(width, unit);
  height_ = DistanceUnit(height, unit);
  depth_ = DistanceUnit(depth, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSBox::~GGEMSBox(void)
{
  GGcout("GGEMSBox", "~GGEMSBox", 3) << "Deallocation of GGEMSBox..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSBox::CheckParameters(void) const
{
  GGcout("GGEMSBox", "CheckParameters", 3) << "Checking mandatory parameters..." << GGendl;

  // Checking height
  if (height_ == 0.0f) {
    GGEMSMisc::ThrowException("GGEMSBox", "CheckParameters", "The box height has to be > 0!!!");
  }

  // Checking width
  if (width_ == 0.0f) {
    GGEMSMisc::ThrowException("GGEMSBox", "CheckParameters", "The box width has to be > 0!!!");
  }

  // Checking depth
  if (depth_ == 0.0f) {
    GGEMSMisc::ThrowException("GGEMSBox", "CheckParameters", "The box depth has to be > 0!!!");
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSBox::Initialize(void)
{
  GGcout("GGEMSBox", "Initialize", 3) << "Initializing GGEMSBox solid volume..." << GGendl;

  // Check mandatory parameters
  CheckParameters();

  // Getting the path to kernel
  std::string const kOpenCLKernelPath = OPENCL_KERNEL_PATH;
  std::string const kFilename = kOpenCLKernelPath + "/DrawGGEMSBox.cl";

  // Get the volume creator manager
  GGEMSVolumeCreatorManager& volume_creator_manager = GGEMSVolumeCreatorManager::GetInstance();

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get the data type and compiling kernel
  std::string const kDataType = "-D" + volume_creator_manager.GetDataType();
  kernel_draw_volume_cl_ = opencl_manager.CompileKernel(kFilename, "draw_ggems_box", nullptr, const_cast<char*>(kDataType.c_str()));
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
  cl::CommandQueue* p_queue_cl = opencl_manager.GetCommandQueue();
  cl::Event* p_event_cl = opencl_manager.GetEvent();

  // Get parameters from phantom creator
  GGfloat3 const kVoxelSizes = volume_creator_manager.GetElementsSizes();
  GGuint3 const kPhantomDimensions = volume_creator_manager.GetVolumeDimensions();
  GGuint const kNumberThreads = volume_creator_manager.GetNumberElements();
  cl::Buffer* voxelized_phantom = volume_creator_manager.GetVoxelizedVolume();

  // Set parameters for kernel
  std::shared_ptr<cl::Kernel> kernel_cl = kernel_draw_volume_cl_.lock();
  kernel_cl->setArg(0, kNumberThreads);
  kernel_cl->setArg(1, kVoxelSizes);
  kernel_cl->setArg(2, kPhantomDimensions);
  kernel_cl->setArg(3, positions_);
  kernel_cl->setArg(4, label_value_);
  kernel_cl->setArg(5, height_);
  kernel_cl->setArg(6, width_);
  kernel_cl->setArg(7, depth_);
  kernel_cl->setArg(8, *voxelized_phantom);

  // Get number of max work group size
  std::size_t const kMaxWorkGroupSize = opencl_manager.GetMaxWorkGroupSize();

  // Compute work item number
  std::size_t const kWorkItem = kNumberThreads + (kMaxWorkGroupSize - kNumberThreads%kMaxWorkGroupSize);

  cl::NDRange global(kWorkItem);
  cl::NDRange offset(0);
  cl::NDRange local(kMaxWorkGroupSize);

  // Launching kernel
  cl_int kernel_status = p_queue_cl->enqueueNDRangeKernel(*kernel_cl, offset, global, local, nullptr, p_event_cl);
  opencl_manager.CheckOpenCLError(kernel_status, "GGEMSBox", "Draw Box");
  p_queue_cl->finish(); // Wait until the kernel status is finish

  // Displaying time in kernel
  opencl_manager.DisplayElapsedTimeInKernel("draw_ggems_box");
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
