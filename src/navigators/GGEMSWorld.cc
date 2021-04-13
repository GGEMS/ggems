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
  \file GGEMSWorld.cc

  \brief GGEMS class handling global world (space between navigators) in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 11, 2021
*/

#include "GGEMS/navigators/GGEMSNavigatorManager.hh"
#include "GGEMS/navigators/GGEMSWorld.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/sources/GGEMSSourceManager.hh"
#include "GGEMS/global/GGEMSManager.hh"
#include "GGEMS/io/GGEMSMHDImage.hh"
#include "GGEMS/tools/GGEMSProfilerManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSWorld::GGEMSWorld()
: world_output_basename_("world")
{
  GGcout("GGEMSWorld", "GGEMSWorld", 3) << "Allocation of GGEMSWorld..." << GGendl;

  GGEMSNavigatorManager::GetInstance().StoreWorld(this);

  dimensions_.x_= 0;
  dimensions_.y_ = 0;
  dimensions_.z_ = 0;

  sizes_.x = -1.0f;
  sizes_.y = -1.0f;
  sizes_.z = -1.0f;

  is_photon_tracking_ = false;
  is_energy_tracking_ = false;
  is_energy_squared_tracking_ = false;
  is_momentum_ = false;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSWorld::~GGEMSWorld(void)
{
  GGcout("GGEMSWorld", "~GGEMSWorld", 3) << "Deallocation of GGEMSWorld..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSWorld::SetOutputWorldBasename(std::string const& output_basename)
{
  world_output_basename_ = output_basename;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSWorld::CheckParameters(void) const
{
  GGcout("GGEMSWorld", "CheckParameters", 3) << "Checking the mandatory parameters..." << GGendl;

  // Checking world dimensions
  if (dimensions_.x_ == 0 || dimensions_.y_ == 0 || dimensions_.z_ == 0) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Dimensions of world have to be set";
    GGEMSMisc::ThrowException("GGEMSWorld", "CheckParameters", oss.str());
  }

  // Checking elements size in world
  if (sizes_.x < 0.0 || sizes_.y < 0.0 || sizes_.z < 0.0) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Size of elements in world";
    GGEMSMisc::ThrowException("GGEMSWorld", "CheckParameters", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSWorld::SetDimension(GGsize const& dimension_x, GGsize const& dimension_y, GGsize const& dimension_z)
{
  dimensions_.x_ = dimension_x;
  dimensions_.y_ = dimension_y;
  dimensions_.z_ = dimension_z;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSWorld::SetElementSize(GGfloat const& size_x, GGfloat const& size_y, GGfloat const& size_z, std::string const& unit)
{
  sizes_.x = DistanceUnit(size_x, unit);
  sizes_.y = DistanceUnit(size_y, unit);
  sizes_.z = DistanceUnit(size_z, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSWorld::SetPhotonTracking(bool const& is_activated)
{
  is_photon_tracking_ = is_activated;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSWorld::SetEnergyTracking(bool const& is_activated)
{
  is_energy_tracking_ = is_activated;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSWorld::SetEnergySquaredTracking(bool const& is_activated)
{
  is_energy_squared_tracking_ = is_activated;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSWorld::SetMomentum(bool const& is_activated)
{
  is_momentum_ = is_activated;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSWorld::InitializeKernel(void)
{
  GGcout("GGEMSWorld", "InitializeKernel", 3) << "Initializing kernel for world tracking..." << GGendl;

  // Getting OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Getting the path to kernel
  std::string openCL_kernel_path = OPENCL_KERNEL_PATH;
  std::string world_tracking_filename = openCL_kernel_path + "/WorldTracking.cl";

  std::string kernel_option("");
  if (GGEMSManager::GetInstance().IsTrackingVerbose()) {
    kernel_option = "-DGGEMS_TRACKING";
  }

  // Compiling the kernels
  kernel_world_tracking_ = opencl_manager.CompileKernel(world_tracking_filename, "world_tracking", nullptr, const_cast<char*>(kernel_option.c_str()));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSWorld::Initialize(void)
{
  GGcout("GGEMSWorld", "Initialize", 3) << "Initializing a GGEMS world..." << GGendl;

  // Checking the parameters of world
  CheckParameters();

  // Getting OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Initializing OpenCL buffers
  GGsize total_number_voxel_world = dimensions_.x_ * dimensions_.y_ * dimensions_.z_;

  world_recording_.photon_tracking_ = is_photon_tracking_ ? opencl_manager.Allocate(nullptr, total_number_voxel_world * sizeof(GGint), CL_MEM_READ_WRITE, "GGEMSWorld") : nullptr;
  if (is_photon_tracking_) opencl_manager.CleanBuffer(world_recording_.photon_tracking_, total_number_voxel_world * sizeof(GGint));

  world_recording_.energy_tracking_ = is_energy_tracking_ ? opencl_manager.Allocate(nullptr, total_number_voxel_world*sizeof(GGDosiType), CL_MEM_READ_WRITE, "GGEMSWorld") : nullptr;
  if (is_energy_tracking_) opencl_manager.CleanBuffer(world_recording_.energy_tracking_, total_number_voxel_world*sizeof(GGDosiType));

  world_recording_.energy_squared_tracking_ = is_energy_squared_tracking_ ? opencl_manager.Allocate(nullptr, total_number_voxel_world*sizeof(GGDosiType), CL_MEM_READ_WRITE, "GGEMSWorld") : nullptr;
  if (is_energy_squared_tracking_) opencl_manager.CleanBuffer(world_recording_.energy_squared_tracking_, total_number_voxel_world*sizeof(GGDosiType));

  world_recording_.momentum_x_ = is_momentum_ ? opencl_manager.Allocate(nullptr, total_number_voxel_world*sizeof(GGDosiType), CL_MEM_READ_WRITE, "GGEMSWorld") : nullptr;
  if (is_momentum_) opencl_manager.CleanBuffer(world_recording_.momentum_x_, total_number_voxel_world*sizeof(GGDosiType));

  world_recording_.momentum_y_ = is_momentum_ ? opencl_manager.Allocate(nullptr, total_number_voxel_world*sizeof(GGDosiType), CL_MEM_READ_WRITE, "GGEMSWorld") : nullptr;
  if (is_momentum_) opencl_manager.CleanBuffer(world_recording_.momentum_y_, total_number_voxel_world*sizeof(GGDosiType));

  world_recording_.momentum_z_ = is_momentum_ ? opencl_manager.Allocate(nullptr, total_number_voxel_world*sizeof(GGDosiType), CL_MEM_READ_WRITE, "GGEMSWorld") : nullptr;
  if (is_momentum_) opencl_manager.CleanBuffer(world_recording_.momentum_z_, total_number_voxel_world*sizeof(GGDosiType));

  // Initialize OpenCL kernel tracking particles in world
  InitializeKernel();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSWorld::Tracking(void)
{
  // Getting the OpenCL manager and infos for work-item launching
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  cl::CommandQueue* queue = opencl_manager.GetCommandQueue();
  cl::Event* event = opencl_manager.GetEvent();

  // Pointer to primary particles, and number to particles in buffer
  GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();
  cl::Buffer* primary_particles = source_manager.GetParticles()->GetPrimaryParticles();
  GGsize number_of_particles = source_manager.GetParticles()->GetNumberOfParticles();

  // Getting work group size, and work-item number
  GGsize work_group_size = opencl_manager.GetWorkGroupSize();
  GGsize number_of_work_items = opencl_manager.GetBestWorkItem(number_of_particles);

  // Parameters for work-item in kernel
  cl::NDRange global_wi(number_of_work_items);
  cl::NDRange local_wi(work_group_size);

  // Getting kernel, and setting parameters
  std::shared_ptr<cl::Kernel> kernel = kernel_world_tracking_.lock();
  kernel->setArg(0, number_of_particles);
  kernel->setArg(1, *primary_particles);
  kernel->setArg(2, *world_recording_.photon_tracking_.get());
  kernel->setArg(3, *world_recording_.energy_tracking_.get());
  kernel->setArg(4, *world_recording_.energy_squared_tracking_.get());
  kernel->setArg(5, *world_recording_.momentum_x_.get());
  kernel->setArg(6, *world_recording_.momentum_y_.get());
  kernel->setArg(7, *world_recording_.momentum_z_.get());
  kernel->setArg(8, dimensions_.x_);
  kernel->setArg(9, dimensions_.y_);
  kernel->setArg(10, dimensions_.z_);
  kernel->setArg(11, sizes_.x);
  kernel->setArg(12, sizes_.y);
  kernel->setArg(13, sizes_.z);

  // Launching kernel
  GGint kernel_status = queue->enqueueNDRangeKernel(*kernel, 0, global_wi, local_wi, nullptr, event);
  opencl_manager.CheckOpenCLError(kernel_status, "GGEMSWorld", "Tracking");

  // GGEMS Profiling
  GGEMSProfilerManager& profiler_manager = GGEMSProfilerManager::GetInstance();
  profiler_manager.HandleEvent(*event, "GGEMSWorld::Tracking");
  queue->finish();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSWorld::SaveResults(void) const
{
  if (is_photon_tracking_) SavePhotonTracking();
  if (is_energy_tracking_) SaveEnergyTracking();
  if (is_energy_squared_tracking_) SaveEnergySquaredTracking();
  if (is_momentum_) SaveMomentum();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSWorld::SavePhotonTracking(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  GGsize total_number_of_voxels = dimensions_.x_ * dimensions_.y_ * dimensions_.z_;
  GGint* photon_tracking = new GGint[total_number_of_voxels];
  std::memset(photon_tracking, 0, total_number_of_voxels*sizeof(GGint));

  GGEMSMHDImage mhdImage;
  mhdImage.SetOutputFileName(world_output_basename_ + "_world_photon_tracking.mhd");
  mhdImage.SetDataType("MET_INT");
  mhdImage.SetDimensions(dimensions_);
  mhdImage.SetElementSizes(sizes_);

  GGint* photon_tracking_device = opencl_manager.GetDeviceBuffer<GGint>(world_recording_.photon_tracking_.get(), total_number_of_voxels*sizeof(GGint));

  for (GGsize i = 0; i < total_number_of_voxels; ++i) photon_tracking[i] = photon_tracking_device[i];

  // Writing data
  mhdImage.Write<GGint>(photon_tracking, total_number_of_voxels);
  opencl_manager.ReleaseDeviceBuffer(world_recording_.photon_tracking_.get(), photon_tracking_device);
  delete[] photon_tracking;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSWorld::SaveEnergyTracking(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  GGsize total_number_of_voxels = dimensions_.x_ * dimensions_.y_ * dimensions_.z_;
  GGDosiType* edep_tracking = new GGDosiType[total_number_of_voxels];
  std::memset(edep_tracking, 0, total_number_of_voxels*sizeof(GGDosiType));

  GGEMSMHDImage mhdImage;
  mhdImage.SetOutputFileName(world_output_basename_ + "_world_edep.mhd");
  if (sizeof(GGDosiType) == 4) mhdImage.SetDataType("MET_FLOAT");
  else if (sizeof(GGDosiType) == 8) mhdImage.SetDataType("MET_DOUBLE");
  mhdImage.SetDimensions(dimensions_);
  mhdImage.SetElementSizes(sizes_);

  GGDosiType* edep_device = opencl_manager.GetDeviceBuffer<GGDosiType>(world_recording_.energy_tracking_.get(), total_number_of_voxels*sizeof(GGDosiType));

  for (GGsize i = 0; i < total_number_of_voxels; ++i) edep_tracking[i] = edep_device[i];

  // Writing data
  mhdImage.Write<GGDosiType>(edep_tracking, total_number_of_voxels);
  opencl_manager.ReleaseDeviceBuffer(world_recording_.energy_tracking_.get(), edep_device);
  delete[] edep_tracking;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSWorld::SaveEnergySquaredTracking(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  GGsize total_number_of_voxels = dimensions_.x_ * dimensions_.y_ * dimensions_.z_;
  GGDosiType* edep_squared_tracking = new GGDosiType[total_number_of_voxels];
  std::memset(edep_squared_tracking, 0, total_number_of_voxels*sizeof(GGDosiType));

  GGEMSMHDImage mhdImage;
  mhdImage.SetOutputFileName(world_output_basename_ + "_world_edep_squared.mhd");
  if (sizeof(GGDosiType) == 4) mhdImage.SetDataType("MET_FLOAT");
  else if (sizeof(GGDosiType) == 8) mhdImage.SetDataType("MET_DOUBLE");
  mhdImage.SetDimensions(dimensions_);
  mhdImage.SetElementSizes(sizes_);

  GGDosiType* edep_squared_device = opencl_manager.GetDeviceBuffer<GGDosiType>(world_recording_.energy_squared_tracking_.get(), total_number_of_voxels*sizeof(GGDosiType));

  for (GGsize i = 0; i < total_number_of_voxels; ++i) edep_squared_tracking[i] = edep_squared_device[i];

  // Writing data
  mhdImage.Write<GGDosiType>(edep_squared_tracking, total_number_of_voxels);
  opencl_manager.ReleaseDeviceBuffer(world_recording_.energy_squared_tracking_.get(), edep_squared_device);
  delete[] edep_squared_tracking;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSWorld::SaveMomentum(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  GGsize total_number_of_voxels = dimensions_.x_ * dimensions_.y_ * dimensions_.z_;

  GGDosiType* momentum_x = new GGDosiType[total_number_of_voxels];
  std::memset(momentum_x, 0, total_number_of_voxels*sizeof(GGDosiType));

  GGDosiType* momentum_y = new GGDosiType[total_number_of_voxels];
  std::memset(momentum_y, 0, total_number_of_voxels*sizeof(GGDosiType));

  GGDosiType* momentum_z = new GGDosiType[total_number_of_voxels];
  std::memset(momentum_z, 0, total_number_of_voxels*sizeof(GGDosiType));

  GGEMSMHDImage mhdImage_momentum_x;
  mhdImage_momentum_x.SetOutputFileName(world_output_basename_ + "_world_momentum_x.mhd");
  if (sizeof(GGDosiType) == 4) mhdImage_momentum_x.SetDataType("MET_FLOAT");
  else if (sizeof(GGDosiType) == 8) mhdImage_momentum_x.SetDataType("MET_DOUBLE");
  mhdImage_momentum_x.SetDimensions(dimensions_);
  mhdImage_momentum_x.SetElementSizes(sizes_);

  GGEMSMHDImage mhdImage_momentum_y;
  mhdImage_momentum_y.SetOutputFileName(world_output_basename_ + "_world_momentum_y.mhd");
  if (sizeof(GGDosiType) == 4) mhdImage_momentum_y.SetDataType("MET_FLOAT");
  else if (sizeof(GGDosiType) == 8) mhdImage_momentum_y.SetDataType("MET_DOUBLE");
  mhdImage_momentum_y.SetDimensions(dimensions_);
  mhdImage_momentum_y.SetElementSizes(sizes_);

  GGEMSMHDImage mhdImage_momentum_z;
  mhdImage_momentum_z.SetOutputFileName(world_output_basename_ + "_world_momentum_z.mhd");
  if (sizeof(GGDosiType) == 4) mhdImage_momentum_z.SetDataType("MET_FLOAT");
  else if (sizeof(GGDosiType) == 8) mhdImage_momentum_z.SetDataType("MET_DOUBLE");
  mhdImage_momentum_z.SetDimensions(dimensions_);
  mhdImage_momentum_z.SetElementSizes(sizes_);

  GGDosiType* momentum_x_device = opencl_manager.GetDeviceBuffer<GGDosiType>(world_recording_.momentum_x_.get(), total_number_of_voxels*sizeof(GGDosiType));

  for (GGsize i = 0; i < total_number_of_voxels; ++i) momentum_x[i] = momentum_x_device[i];

  // Writing data
  mhdImage_momentum_x.Write<GGDosiType>(momentum_x, total_number_of_voxels);
  opencl_manager.ReleaseDeviceBuffer(world_recording_.momentum_x_.get(), momentum_x_device);

  GGDosiType* momentum_y_device = opencl_manager.GetDeviceBuffer<GGDosiType>(world_recording_.momentum_y_.get(), total_number_of_voxels*sizeof(GGDosiType));

  for (GGsize i = 0; i < total_number_of_voxels; ++i) momentum_y[i] = momentum_y_device[i];

  // Writing data
  mhdImage_momentum_y.Write<GGDosiType>(momentum_y, total_number_of_voxels);
  opencl_manager.ReleaseDeviceBuffer(world_recording_.momentum_y_.get(), momentum_y_device);

  GGDosiType* momentum_z_device = opencl_manager.GetDeviceBuffer<GGDosiType>(world_recording_.momentum_z_.get(), total_number_of_voxels*sizeof(GGDosiType));

  for (GGsize i = 0; i < total_number_of_voxels; ++i) momentum_z[i] = momentum_z_device[i];

  // Writing data
  mhdImage_momentum_z.Write<GGDosiType>(momentum_z, total_number_of_voxels);
  opencl_manager.ReleaseDeviceBuffer(world_recording_.momentum_z_.get(), momentum_z_device);

  delete[] momentum_x;
  delete[] momentum_y;
  delete[] momentum_z;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSWorld* create_ggems_world(void)
{
  return new(std::nothrow) GGEMSWorld();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_dimension_ggems_world(GGEMSWorld* world, GGsize const dimension_x, GGsize const dimension_y, GGsize const dimension_z)
{
  world->SetDimension(dimension_x, dimension_y, dimension_z);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_size_ggems_world(GGEMSWorld* world, GGfloat const size_x, GGfloat const size_y, GGfloat const size_z, char const* unit)
{
  world->SetElementSize(size_x, size_y, size_z, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void photon_tracking_ggems_world(GGEMSWorld* world, bool const is_activated)
{
  world->SetPhotonTracking(is_activated);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_output_ggems_world(GGEMSWorld* world, char const* world_output_basename)
{
  world->SetOutputWorldBasename(world_output_basename);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void energy_tracking_ggems_world(GGEMSWorld* world, bool const is_activated)
{
  world->SetEnergyTracking(is_activated);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void energy_squared_tracking_ggems_world(GGEMSWorld* world, bool const is_activated)
{
  world->SetEnergySquaredTracking(is_activated);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void momentum_ggems_world(GGEMSWorld* world, bool const is_activated)
{
  world->SetMomentum(is_activated);
}
