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
  \file GGEMSParticles.cc

  \brief Class managing the particles in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thrusday October 3, 2019
*/

#include "GGEMS/physics/GGEMSPrimaryParticles.hh"
#include "GGEMS/sources/GGEMSSourceManager.hh"
#include "GGEMS/tools/GGEMSRAMManager.hh"
#include "GGEMS/tools/GGEMSProfilerManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSParticles::GGEMSParticles(void)
: number_of_particles_(nullptr),
  primary_particles_(nullptr),
  kernel_alive_(nullptr)
{
  GGcout("GGEMSParticles", "GGEMSParticles", 3) << "GGEMSParticles creating..." << GGendl;

  GGcout("GGEMSParticles", "GGEMSParticles", 3) << "GGEMSParticles created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSParticles::~GGEMSParticles(void)
{
  GGcout("GGEMSParticles", "~GGEMSParticles", 3) << "GGEMSParticles erasing..." << GGendl;

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  if (number_of_particles_) {
    delete[] number_of_particles_;
    number_of_particles_ = nullptr;
  }

  if (primary_particles_) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      opencl_manager.Deallocate(primary_particles_[i], sizeof(GGEMSPrimaryParticles), i);
      opencl_manager.Deallocate(status_[i], sizeof(GGint), i);
    }
    delete[] primary_particles_;
    primary_particles_ = nullptr;
    delete[] status_;
    status_ = nullptr;
  }

  if (kernel_alive_) {
    delete[] kernel_alive_;
    kernel_alive_ = nullptr;
  }

  GGcout("GGEMSParticles", "~GGEMSParticles", 3) << "GGEMSParticles erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSParticles::SetNumberOfParticles(GGsize const& thread_index, GGsize const& number_of_particles)
{
  number_of_particles_[thread_index] = number_of_particles;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSParticles::InitializeKernel(void)
{
  GGcout("GGEMSParticles", "InitializeKernel", 3) << "Initializing kernel..." << GGendl;

  // Getting the path to kernel
  std::string openCL_kernel_path = OPENCL_KERNEL_PATH;
  std::string filename = openCL_kernel_path + "/IsAlive.cl";


  // Storing a kernel for each device
  kernel_alive_ = new cl::Kernel*[number_activated_devices_];

  // Compiling the kernel
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Compiling kernel on each device
  opencl_manager.CompileKernel(filename, "is_alive", kernel_alive_, nullptr, nullptr);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSParticles::Initialize(void)
{
  GGcout("GGEMSParticles", "Initialize", 1) << "Initialization of GGEMSParticles..." << GGendl;

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get number of activated device
  number_activated_devices_ = opencl_manager.GetNumberOfActivatedDevice();

  number_of_particles_ = new GGsize[number_activated_devices_];

  // Allocation of the PrimaryParticle structure
  AllocatePrimaryParticles();

  // Initializing kernel
  InitializeKernel();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

bool GGEMSParticles::IsAlive(GGsize const& thread_index) const
{
  // Get command queue and event
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  cl::CommandQueue* queue = opencl_manager.GetCommandQueue(thread_index);

  // Get Device name and storing methode name + device
  GGsize device_index = opencl_manager.GetIndexOfActivatedDevice(thread_index);
  std::string device_name = opencl_manager.GetDeviceName(device_index);
  std::ostringstream oss(std::ostringstream::out);
  oss << "GGEMSParticles::IsAlive on " << device_name << ", index " << device_index;

  // Get the OpenCL buffers
  cl::Buffer* particles = primary_particles_[thread_index];
  cl::Buffer* status = status_[thread_index];

  // Getting work group size, and work-item number
  GGsize work_group_size = opencl_manager.GetWorkGroupSize();
  GGsize number_of_work_items = opencl_manager.GetBestWorkItem(number_of_particles_[thread_index]);

  // Parameters for work-item in kernel
  cl::NDRange global_wi(number_of_work_items);
  cl::NDRange local_wi(work_group_size);

  // Set parameters for kernel
  kernel_alive_[thread_index]->setArg(0, number_of_particles_[thread_index]);
  kernel_alive_[thread_index]->setArg(1, *particles);
  kernel_alive_[thread_index]->setArg(2, *status);

  // Launching kernel
  cl::Event event;
  GGint kernel_status = queue->enqueueNDRangeKernel(*kernel_alive_[thread_index], 0, global_wi, local_wi, nullptr, &event);
  opencl_manager.CheckOpenCLError(kernel_status, "GGEMSParticles", "IsAlive");

  // GGEMS Profiling
  GGEMSProfilerManager::GetInstance().HandleEvent(event, oss.str());
  queue->finish();

  // Get status from OpenCL device
  GGint* status_device = opencl_manager.GetDeviceBuffer<GGint>(status_[thread_index], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGint), thread_index);

  GGint status_from_device = status_device[0];

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(status_[thread_index], status_device, thread_index);

  // Cleaning buffer
  opencl_manager.CleanBuffer(status_[thread_index], sizeof(GGint), thread_index);

  if (status_from_device == static_cast<GGint>(number_of_particles_[thread_index])) return false;
  else return true;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSParticles::AllocatePrimaryParticles(void)
{
  GGcout("GGEMSParticles", "AllocatePrimaryParticles", 1) << "Allocation of primary particles..." << GGendl;

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  primary_particles_ = new cl::Buffer*[number_activated_devices_];
  status_ = new cl::Buffer*[number_activated_devices_];

  // Loop over activated device and allocate particle buffer on each device
  for (GGsize i = 0; i < number_activated_devices_; ++i) {
    primary_particles_[i] = opencl_manager.Allocate(nullptr, sizeof(GGEMSPrimaryParticles), i, CL_MEM_READ_WRITE, "GGEMSParticles");
    status_[i] = opencl_manager.Allocate(nullptr, sizeof(GGint), i, CL_MEM_READ_WRITE, "GGEMSParticles");
    opencl_manager.CleanBuffer(status_[i], sizeof(GGint), i);
  }
}
