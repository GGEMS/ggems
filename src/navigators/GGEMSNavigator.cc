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
  \file GGEMSNavigator.cc

  \brief GGEMS mother class for navigation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday February 11, 2020
*/

#include "GGEMS/navigators/GGEMSNavigatorManager.hh"
#include "GGEMS/geometries/GGEMSVoxelizedSolid.hh"
#include "GGEMS/physics/GGEMSCrossSections.hh"
#include "GGEMS/sources/GGEMSSourceManager.hh"
#include "GGEMS/randoms/GGEMSPseudoRandomGenerator.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSNavigator::GGEMSNavigator(std::string const& navigator_name)
: navigator_name_(navigator_name),
  position_xyz_({0.0f, 0.0f, 0.0f}),
  rotation_xyz_({0.0f, 0.0f, 0.0f}),
  navigator_id_(-1),
  is_update_pos_(false),
  is_update_rot_(false),
  output_basename_(""),
  kernel_particle_solid_distance_timer_(GGEMSChrono::Zero()),
  kernel_project_to_solid_timer_(GGEMSChrono::Zero()),
  kernel_track_through_solid_timer_(GGEMSChrono::Zero())
{
  GGcout("GGEMSNavigator", "GGEMSNavigator", 3) << "Allocation of GGEMSNavigator..." << GGendl;

  // Store the phantom navigator in phantom navigator manager
  GGEMSNavigatorManager::GetInstance().Store(this);

  // Allocation of materials
  materials_.reset(new GGEMSMaterials());

  // Allocation of cross sections including physics
  cross_sections_.reset(new GGEMSCrossSections());
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSNavigator::~GGEMSNavigator(void)
{
  GGcout("GGEMSNavigator", "~GGEMSNavigator", 3) << "Deallocation of GGEMSNavigator..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::SetPosition(GGfloat const& position_x, GGfloat const& position_y, GGfloat const& position_z, std::string const& unit)
{
  is_update_pos_ = true;
  position_xyz_.s0 = DistanceUnit(position_x, unit);
  position_xyz_.s1 = DistanceUnit(position_y, unit);
  position_xyz_.s2 = DistanceUnit(position_z, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::SetRotation(GGfloat const& rx, GGfloat const& ry, GGfloat const& rz, std::string const& unit)
{
  is_update_rot_ = true;
  rotation_xyz_.s0 = AngleUnit(rx, unit);
  rotation_xyz_.s1 = AngleUnit(ry, unit);
  rotation_xyz_.s2 = AngleUnit(rz, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::SetThreshold(GGfloat const& threshold, std::string const& unit)
{
  threshold_ = EnergyUnit(threshold, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::SetNavigatorID(std::size_t const& navigator_id)
{
  navigator_id_ = navigator_id;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::CheckParameters(void) const
{
  GGcout("GGEMSNavigator", "CheckParameters", 3) << "Checking the mandatory parameters..." << GGendl;

  // Checking id of the navigator
  if (navigator_id_ == -1) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Id of the navigator is not set!!!";
    GGEMSMisc::ThrowException("GGEMSNavigator", "CheckParameters", oss.str());
  }

  // Checking output name
  if (output_basename_.empty()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Output basename not set!!!";
    GGEMSMisc::ThrowException("GGEMSNavigator", "CheckParameters", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::Initialize(void)
{
  GGcout("GGEMSNavigator", "Initialize", 3) << "Initializing a GGEMS navigator..." << GGendl;

  // Checking the parameters of phantom
  CheckParameters();

  // Loading the materials and building tables to OpenCL device and converting cuts
  materials_->Initialize();

  // Initialization of electromagnetic process and building cross section tables for each particles and materials
  cross_sections_->Initialize(materials_.get());
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::StoreOutput(std::string basename)
{
  output_basename_= basename;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::ParticleSolidDistance(void)
{
  // Getting the OpenCL manager and infos for work-item launching
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  cl::CommandQueue* queue_cl = opencl_manager.GetCommandQueue();
  cl::Event* event_cl = opencl_manager.GetEvent();

  // Pointer to primary particles, and number to particles in buffer
  GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();
  cl::Buffer* primary_particles_cl = source_manager.GetParticles()->GetPrimaryParticles();
  GGlong number_of_particles = source_manager.GetParticles()->GetNumberOfParticles();

  // Getting work group size, and work-item number
  std::size_t work_group_size = opencl_manager.GetWorkGroupSize();
  std::size_t number_of_work_items = number_of_particles + (work_group_size - number_of_particles%work_group_size);

  // Parameters for work-item in kernel
  cl::NDRange global_wi(number_of_work_items);
  cl::NDRange local_wi(work_group_size);

  // Loop over all the solids
  for (auto&& s : solids_) {
    // Getting solid data infos
    cl::Buffer* solid_data_cl = s->GetSolidData();

    // Getting kernel, and setting parameters
    std::shared_ptr<cl::Kernel> kernel_cl = s->GetKernelParticleSolidDistance().lock();
    kernel_cl->setArg(0, number_of_particles);
    kernel_cl->setArg(1, *primary_particles_cl);
    kernel_cl->setArg(2, *solid_data_cl);

    // Launching kernel
    GGint kernel_status = queue_cl->enqueueNDRangeKernel(*kernel_cl, 0, global_wi, local_wi, nullptr, event_cl);
    opencl_manager.CheckOpenCLError(kernel_status, "GGEMSNavigator", "ParticleSolidDistance");
    queue_cl->finish(); // Wait until the kernel status is finish

    // Incrementing elapsed time in kernel
    kernel_particle_solid_distance_timer_ += opencl_manager.GetElapsedTimeInKernel();
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::ProjectToSolid(void)
{
  // Getting the OpenCL manager and infos for work-item launching
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  cl::CommandQueue* queue_cl = opencl_manager.GetCommandQueue();
  cl::Event* event_cl = opencl_manager.GetEvent();

  // Pointer to primary particles, and number to particles in buffer
  GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();
  cl::Buffer* primary_particles_cl = source_manager.GetParticles()->GetPrimaryParticles();
  GGlong number_of_particles = source_manager.GetParticles()->GetNumberOfParticles();

  // Getting work group size, and work-item number
  std::size_t work_group_size = opencl_manager.GetWorkGroupSize();
  std::size_t number_of_work_items = number_of_particles + (work_group_size - number_of_particles%work_group_size);

  // Parameters for work-item in kernel
  cl::NDRange global_wi(number_of_work_items);
  cl::NDRange local_wi(work_group_size);

  // Loop over all the solids
  for (auto&& s : solids_) {
    // Getting solid data infos
    cl::Buffer* solid_data_cl = s->GetSolidData();

    // Getting kernel, and setting parameters
    std::shared_ptr<cl::Kernel> kernel_cl = s->GetKernelProjectToSolid().lock();
    kernel_cl->setArg(0, number_of_particles);
    kernel_cl->setArg(1, *primary_particles_cl);
    kernel_cl->setArg(2, *solid_data_cl);

    // Launching kernel
    GGint kernel_status = queue_cl->enqueueNDRangeKernel(*kernel_cl, 0, global_wi, local_wi, nullptr, event_cl);
    opencl_manager.CheckOpenCLError(kernel_status, "GGEMSNavigator", "ProjectToSolid");
    queue_cl->finish(); // Wait until the kernel status is finish

    // Incrementing elapsed time in kernel
    kernel_project_to_solid_timer_ += opencl_manager.GetElapsedTimeInKernel();
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::TrackThroughSolid(void)
{
  // Getting the OpenCL manager and infos for work-item launching
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  cl::CommandQueue* queue_cl = opencl_manager.GetCommandQueue();
  cl::Event* event_cl = opencl_manager.GetEvent();

  // Pointer to primary particles, and number to particles in buffer
  GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();
  cl::Buffer* primary_particles_cl = source_manager.GetParticles()->GetPrimaryParticles();
  GGlong number_of_particles = source_manager.GetParticles()->GetNumberOfParticles();

  // Getting OpenCL pointer to random number
  cl::Buffer* randoms_cl = source_manager.GetPseudoRandomGenerator()->GetPseudoRandomNumbers();

  // Getting OpenCL buffer for cross section
  cl::Buffer* cross_sections_cl = cross_sections_->GetCrossSections();

  // Getting OpenCL buffer for materials
  cl::Buffer* materials_cl = materials_->GetMaterialTables().lock().get();

  // Getting work group size, and work-item number
  std::size_t work_group_size = opencl_manager.GetWorkGroupSize();
  std::size_t number_of_work_items = number_of_particles + (work_group_size - number_of_particles%work_group_size);

  // Parameters for work-item in kernel
  cl::NDRange global_wi(number_of_work_items);
  cl::NDRange local_wi(work_group_size);

  // Loop over all the solids
  for (auto&& s : solids_) {
    // Getting solid  and label (for GGEMSVoxelizedSolid) data infos
    cl::Buffer* solid_data_cl = s->GetSolidData();
    cl::Buffer* label_data_cl = s->GetLabelData();

    // Get type of registered data and OpenCL buffer to data
    std::string data_reg_type = s->GetRegisteredDataType();
    cl::Buffer* histogram_cl = nullptr;
    if (data_reg_type == "HISTOGRAM") {
      histogram_cl = s->GetHistogram()->histogram_cl_.get();
    }

    // Getting kernel, and setting parameters
    std::shared_ptr<cl::Kernel> kernel_cl = s->GetKernelTrackThroughSolid().lock();
    kernel_cl->setArg(0, number_of_particles);
    kernel_cl->setArg(1, *primary_particles_cl);
    kernel_cl->setArg(2, *randoms_cl);
    kernel_cl->setArg(3, *solid_data_cl);
    kernel_cl->setArg(4, *label_data_cl); // Useful only for GGEMSVoxelizedSolid
    kernel_cl->setArg(5, *cross_sections_cl);
    kernel_cl->setArg(6, *materials_cl);
    kernel_cl->setArg(7, threshold_);
    if (data_reg_type == "HISTOGRAM") {
      kernel_cl->setArg(8, *histogram_cl);
    }

    // Launching kernel
    GGint kernel_status = queue_cl->enqueueNDRangeKernel(*kernel_cl, 0, global_wi, local_wi, nullptr, event_cl);
    opencl_manager.CheckOpenCLError(kernel_status, "GGEMSNavigator", "TrackThroughSolid");
    queue_cl->finish(); // Wait until the kernel status is finish

    // Incrementing elapsed time in kernel
    kernel_track_through_solid_timer_ += opencl_manager.GetElapsedTimeInKernel();
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::PrintInfos(void) const
{
  GGcout("GGEMSNavigator", "PrintInfos", 0) << GGendl;
  GGcout("GGEMSNavigator", "PrintInfos", 0) << "GGEMSNavigator Infos:" << GGendl;
  GGcout("GGEMSNavigator", "PrintInfos", 0) << "---------------------" << GGendl;
  GGcout("GGEMSNavigator", "PrintInfos", 0) << "* Navigator name: " << navigator_name_ << GGendl;
  for (auto&&i : solids_) i->PrintInfos();
  materials_->PrintInfos();
  GGcout("GGEMSNavigator", "PrintInfos", 0) << "* Output: " << output_basename_ << GGendl;
  GGcout("GGEMSNavigator", "PrintInfos", 0) << GGendl;
}
