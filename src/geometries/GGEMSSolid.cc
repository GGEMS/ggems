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
#include "GGEMS/global/GGEMSManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSolid::GGEMSSolid(void)
: solid_data_cl_(nullptr),
  label_data_cl_(nullptr),
  tracking_kernel_option_("")
{
  GGcout("GGEMSSolid", "GGEMSSolid", 3) << "Allocation of GGEMSSolid..." << GGendl;

  // Allocation of geometry transformation
  geometry_transformation_.reset(new GGEMSGeometryTransformation());
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSolid::~GGEMSSolid(void)
{
  GGcout("GGEMSSolid", "~GGEMSSolid", 3) << "Deallocation of GGEMSSolid..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::EnableTracking(void)
{
  tracking_kernel_option_ = "-DGGEMS_TRACKING";
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

void GGEMSSolid::SetLocalAxis(GGfloat33 const& local_axis_xyz)
{
  geometry_transformation_->SetAxisTransformation(local_axis_xyz);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::ParticleSolidDistance(void)
{
  // Getting the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  cl::CommandQueue* queue_cl = opencl_manager.GetCommandQueue();
  cl::Event* event_cl = opencl_manager.GetEvent();

  // Getting the buffer of primary particles from source
  GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();
  GGEMSParticles* particles = source_manager.GetParticles();
  cl::Buffer* primary_particles_cl = particles->GetPrimaryParticles();

  // Getting the number of particles
  GGulong const kNumberOfParticles = particles->GetNumberOfParticles();

  // Set parameters for kernel
  std::shared_ptr<cl::Kernel> kernel_cl = kernel_distance_cl_.lock();
  kernel_cl->setArg(0, *primary_particles_cl);
  kernel_cl->setArg(1, *solid_data_cl_);

  // Define the number of work-item to launch
  cl::NDRange global(kNumberOfParticles);
  cl::NDRange offset(0);

  // Launching kernel
  GGint kernel_status = queue_cl->enqueueNDRangeKernel(*kernel_cl, offset, global, cl::NullRange, nullptr, event_cl);
  opencl_manager.CheckOpenCLError(kernel_status, "GGEMSSolid", "ParticleSolidDistance");
  queue_cl->finish(); // Wait until the kernel status is finish

  // Checking if kernel verbosity is activated
  if (GGEMSManager::GetInstance().IsKernelVerbose()) {
    opencl_manager.DisplayElapsedTimeInKernel("distance");
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::ProjectTo(void)
{
  // Getting the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  cl::CommandQueue* queue_cl = opencl_manager.GetCommandQueue();
  cl::Event* event_cl = opencl_manager.GetEvent();

  // Getting the buffer of primary particles from source
  GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();
  GGEMSParticles* particles = source_manager.GetParticles();
  cl::Buffer* primary_particles_cl = particles->GetPrimaryParticles();

  // Getting the number of particles
  GGulong const kNumberOfParticles = particles->GetNumberOfParticles();

  // Set parameters for kernel
  std::shared_ptr<cl::Kernel> kernel_cl = kernel_project_to_cl_.lock();
  kernel_cl->setArg(0, *primary_particles_cl);
  kernel_cl->setArg(1, *solid_data_cl_);

  // Define the number of work-item to launch
  cl::NDRange global(kNumberOfParticles);
  cl::NDRange offset(0);

  // Launching kernel
  GGint kernel_status = queue_cl->enqueueNDRangeKernel(*kernel_cl, offset, global, cl::NullRange, nullptr, event_cl);
  opencl_manager.CheckOpenCLError(kernel_status, "GGEMSSolid", "ProjectTo");
  queue_cl->finish(); // Wait until the kernel status is finish
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::TrackThrough(std::weak_ptr<GGEMSCrossSections> cross_sections, std::weak_ptr<GGEMSMaterials> materials)
{
  // Getting the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  cl::CommandQueue* queue_cl = opencl_manager.GetCommandQueue();
  cl::Event* event_cl = opencl_manager.GetEvent();

  // Getting the buffer of primary particles and random from source
  GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();
  GGEMSParticles* particles = source_manager.GetParticles();
  cl::Buffer* primary_particles_cl = particles->GetPrimaryParticles();
  cl::Buffer* randoms_cl = source_manager.GetPseudoRandomGenerator()->GetPseudoRandomNumbers();

  // Getting OpenCL buffer for cross section
  cl::Buffer* cross_sections_cl = cross_sections.lock()->GetCrossSections();

  // Getting OpenCL buffer for materials
  cl::Buffer* materials_cl = materials.lock()->GetMaterialTables().lock().get();

  // Getting the number of particles
  GGulong const kNumberOfParticles = particles->GetNumberOfParticles();

  // Set parameters for kernel
  std::shared_ptr<cl::Kernel> kernel_cl = kernel_track_through_cl_.lock();
  kernel_cl->setArg(0, *primary_particles_cl);
  kernel_cl->setArg(1, *randoms_cl);
  kernel_cl->setArg(2, *solid_data_cl_);
  kernel_cl->setArg(3, *label_data_cl_);
  kernel_cl->setArg(4, *cross_sections_cl);
  kernel_cl->setArg(5, *materials_cl);

  // Define the number of work-item to launch
  cl::NDRange global(kNumberOfParticles);
  cl::NDRange offset(0);

  // Launching kernel
  GGint kernel_status = queue_cl->enqueueNDRangeKernel(*kernel_cl, offset, global, cl::NullRange, nullptr, event_cl);
  opencl_manager.CheckOpenCLError(kernel_status, "GGEMSSolid", "TrackThrough");
  queue_cl->finish(); // Wait until the kernel status is finish
}
