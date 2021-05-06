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

#include "GGEMS/geometries/GGEMSVoxelizedSolid.hh"
#include "GGEMS/physics/GGEMSCrossSections.hh"
#include "GGEMS/sources/GGEMSSourceManager.hh"
#include "GGEMS/randoms/GGEMSPseudoRandomGenerator.hh"
#include "GGEMS/navigators/GGEMSDosimetryCalculator.hh"
#include "GGEMS/tools/GGEMSProfilerManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSNavigator::GGEMSNavigator(std::string const& navigator_name)
: navigator_name_(navigator_name),
  navigator_id_(NAVIGATOR_NOT_INITIALIZED),
  is_update_pos_(false),
  is_update_rot_(false),
  is_tracking_(false),
  output_basename_(""),
  solids_(nullptr),
  number_of_solids_(0),
  dose_calculator_(nullptr),
  is_dosimetry_mode_(false)
{
  GGcout("GGEMSNavigator", "GGEMSNavigator", 3) << "GGEMSNavigator creating..." << GGendl;

  position_xyz_.x = 0.0f;
  position_xyz_.y = 0.0f;
  position_xyz_.z = 0.0f;

  rotation_xyz_.x = 0.0f;
  rotation_xyz_.y = 0.0f;
  rotation_xyz_.z = 0.0f;

  // Store the phantom navigator in phantom navigator manager
  GGEMSNavigatorManager::GetInstance().Store(this);

  // Allocation of materials
  materials_ = new GGEMSMaterials();

  // Allocation of cross sections including physics
  cross_sections_ = new GGEMSCrossSections();

  // Get the number of activated device
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  number_activated_devices_ = opencl_manager.GetNumberOfActivatedDevice();

  GGcout("GGEMSNavigator", "GGEMSNavigator", 3) << "GGEMSNavigator created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSNavigator::~GGEMSNavigator(void)
{
  GGcout("GGEMSNavigator", "~GGEMSNavigator", 3) << "GGEMSNavigator erasing..." << GGendl;

  if (solids_) {
    for (GGsize i = 0; i < number_of_solids_; ++i) {
      delete solids_[i];
      solids_[i] = nullptr;
    }
    delete[] solids_;
    solids_ = nullptr;
  }

  if (materials_) {
    delete materials_;
    materials_ = nullptr;
  }

  if (cross_sections_) {
    delete cross_sections_;
    cross_sections_ = nullptr;
  }

  GGcout("GGEMSNavigator", "~GGEMSNavigator", 3) << "GGEMSNavigator erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::SetDosimetryCalculator(GGEMSDosimetryCalculator* dosimetry_calculator)
{
  dose_calculator_ = dosimetry_calculator;
  is_dosimetry_mode_ = true;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::SetPosition(GGfloat const& position_x, GGfloat const& position_y, GGfloat const& position_z, std::string const& unit)
{
  is_update_pos_ = true;
  position_xyz_.s[0] = DistanceUnit(position_x, unit);
  position_xyz_.s[1] = DistanceUnit(position_y, unit);
  position_xyz_.s[2] = DistanceUnit(position_z, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::SetRotation(GGfloat const& rx, GGfloat const& ry, GGfloat const& rz, std::string const& unit)
{
  is_update_rot_ = true;
  rotation_xyz_.x = AngleUnit(rx, unit);
  rotation_xyz_.y = AngleUnit(ry, unit);
  rotation_xyz_.z = AngleUnit(rz, unit);
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

void GGEMSNavigator::SetNavigatorID(GGsize const& navigator_id)
{
  navigator_id_ = navigator_id;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::EnableTracking(void)
{
  is_tracking_ = true;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::CheckParameters(void) const
{
  GGcout("GGEMSNavigator", "CheckParameters", 3) << "Checking the mandatory parameters..." << GGendl;

  // Checking id of the navigator
  if (navigator_id_ == NAVIGATOR_NOT_INITIALIZED) {
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
  cross_sections_->Initialize(materials_);
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

void GGEMSNavigator::ParticleSolidDistance(GGsize const& thread_index)
{
  // Getting the OpenCL manager and infos for work-item launching
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  cl::CommandQueue* queue = opencl_manager.GetCommandQueue(thread_index);
  cl::Event* event = opencl_manager.GetEvent(thread_index);

  // Get Device name and storing methode name + device
  GGsize device_index = opencl_manager.GetIndexOfActivatedDevice(thread_index);
  std::string device_name = opencl_manager.GetDeviceName(device_index);
  std::ostringstream oss(std::ostringstream::out);
  oss << "GGEMSNavigator::ParticleSolidDistance on " << device_name << ", index " << device_index;

  // Pointer to primary particles, and number to particles in buffer
  GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();
  cl::Buffer* primary_particles = source_manager.GetParticles()->GetPrimaryParticles(thread_index);
  GGsize number_of_particles = source_manager.GetParticles()->GetNumberOfParticles(thread_index);

  // Getting work group size, and work-item number
  GGsize work_group_size = opencl_manager.GetWorkGroupSize();
  GGsize number_of_work_items = opencl_manager.GetBestWorkItem(number_of_particles);

  // Parameters for work-item in kernel
  cl::NDRange global_wi(number_of_work_items);
  cl::NDRange local_wi(work_group_size);

  // Loop over all the solids
  for (GGsize s = 0; s < number_of_solids_; ++s) {
    // Getting solid data infos
    cl::Buffer* solid_data = solids_[s]->GetSolidData(thread_index);

    // Getting kernel, and setting parameters
    cl::Kernel* kernel = solids_[s]->GetKernelParticleSolidDistance(thread_index);
    kernel->setArg(0, number_of_particles);
    kernel->setArg(1, *primary_particles);
    kernel->setArg(2, *solid_data);

    // Launching kernel
    GGint kernel_status = queue->enqueueNDRangeKernel(*kernel, 0, global_wi, local_wi, nullptr, event);
    opencl_manager.CheckOpenCLError(kernel_status, "GGEMSNavigator", "ParticleSolidDistance");
    queue->finish();

    // GGEMS Profiling
    GGEMSProfilerManager& profiler_manager = GGEMSProfilerManager::GetInstance();
    profiler_manager.HandleEvent(*event, oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::ProjectToSolid(GGsize const& thread_index)
{
  // Getting the OpenCL manager and infos for work-item launching
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  cl::CommandQueue* queue = opencl_manager.GetCommandQueue(thread_index);
  cl::Event* event = opencl_manager.GetEvent(thread_index);

  // Get Device name and storing methode name + device
  GGsize device_index = opencl_manager.GetIndexOfActivatedDevice(thread_index);
  std::string device_name = opencl_manager.GetDeviceName(device_index);
  std::ostringstream oss(std::ostringstream::out);
  oss << "GGEMSNavigator::ProjectToSolid on " << device_name << ", index " << device_index;

  // Pointer to primary particles, and number to particles in buffer
  GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();
  cl::Buffer* primary_particles = source_manager.GetParticles()->GetPrimaryParticles(thread_index);
  GGsize number_of_particles = source_manager.GetParticles()->GetNumberOfParticles(thread_index);

  // Getting work group size, and work-item number
  GGsize work_group_size = opencl_manager.GetWorkGroupSize();
  GGsize number_of_work_items = opencl_manager.GetBestWorkItem(number_of_particles);

  // Parameters for work-item in kernel
  cl::NDRange global_wi(number_of_work_items);
  cl::NDRange local_wi(work_group_size);

  // Loop over all the solids
  for (GGsize s = 0; s < number_of_solids_; ++s) {
    // Getting solid data infos
    cl::Buffer* solid_data = solids_[s]->GetSolidData(thread_index);

    // Getting kernel, and setting parameters
    cl::Kernel* kernel = solids_[s]->GetKernelProjectToSolid(thread_index);
    kernel->setArg(0, number_of_particles);
    kernel->setArg(1, *primary_particles);
    kernel->setArg(2, *solid_data);

    // Launching kernel
    GGint kernel_status = queue->enqueueNDRangeKernel(*kernel, 0, global_wi, local_wi, nullptr, event);
    opencl_manager.CheckOpenCLError(kernel_status, "GGEMSNavigator", "ProjectToSolid");
    queue->finish();

    // GGEMS Profiling
    GGEMSProfilerManager& profiler_manager = GGEMSProfilerManager::GetInstance();
    profiler_manager.HandleEvent(*event, oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::TrackThroughSolid(GGsize const& thread_index)
{
  // Getting the OpenCL manager and infos for work-item launching
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  cl::CommandQueue* queue = opencl_manager.GetCommandQueue(thread_index);
  cl::Event* event = opencl_manager.GetEvent(thread_index);

  // Get Device name and storing methode name + device
  GGsize device_index = opencl_manager.GetIndexOfActivatedDevice(thread_index);
  std::string device_name = opencl_manager.GetDeviceName(device_index);
  std::ostringstream oss(std::ostringstream::out);
  oss << "GGEMSNavigator::TrackThroughSolid on " << device_name << ", index " << device_index;

  // Pointer to primary particles, and number to particles in buffer
  GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();
  cl::Buffer* primary_particles = source_manager.GetParticles()->GetPrimaryParticles(thread_index);
  GGsize number_of_particles = source_manager.GetParticles()->GetNumberOfParticles(thread_index);

  // Getting OpenCL pointer to random number
  cl::Buffer* randoms = source_manager.GetPseudoRandomGenerator()->GetPseudoRandomNumbers(thread_index);

  // Getting OpenCL buffer for cross section
  cl::Buffer* cross_sections = cross_sections_->GetCrossSections(thread_index);

  // Getting OpenCL buffer for materials
  cl::Buffer* materials = materials_->GetMaterialTables(thread_index);

  // Getting work group size, and work-item number
  GGsize work_group_size = opencl_manager.GetWorkGroupSize();
  GGsize number_of_work_items = opencl_manager.GetBestWorkItem(number_of_particles);

  // Parameters for work-item in kernel
  cl::NDRange global_wi(number_of_work_items);
  cl::NDRange local_wi(work_group_size);

  // Loop over all the solids
  for (GGsize s = 0; s < number_of_solids_; ++s) {
    // Getting solid  and label (for GGEMSVoxelizedSolid) data infos
    cl::Buffer* solid_data = solids_[s]->GetSolidData(thread_index);
    cl::Buffer* label_data = solids_[s]->GetLabelData(thread_index);

    // Get type of registered data and OpenCL buffer to data
    std::string data_reg_type = solids_[s]->GetRegisteredDataType();

    // Get buffers depending on mode of simulation
    // Histogram mode (for system, CT ...)
    cl::Buffer* histogram = nullptr;
    cl::Buffer* scatter_histogram = nullptr;
    // Dosimetry mode (for voxelized phantom ...)
    cl::Buffer* photon_tracking_dosimetry = nullptr;
    cl::Buffer* hit_tracking_dosimetry = nullptr;
    cl::Buffer* edep_tracking_dosimetry = nullptr;
    cl::Buffer* edep_squared_tracking_dosimetry = nullptr;
    cl::Buffer* dosimetry_params = nullptr;
    if (data_reg_type == "HISTOGRAM") {
      histogram = solids_[s]->GetHistogram(thread_index);
      scatter_histogram = solids_[s]->GetScatterHistogram(thread_index);
    }
    else if (data_reg_type == "DOSIMETRY") {
      dosimetry_params = dose_calculator_->GetDoseParams(thread_index);
      photon_tracking_dosimetry = dose_calculator_->GetPhotonTrackingBuffer(thread_index);
      hit_tracking_dosimetry = dose_calculator_->GetHitTrackingBuffer(thread_index);
      edep_tracking_dosimetry = dose_calculator_->GetEdepBuffer(thread_index);
      edep_squared_tracking_dosimetry = dose_calculator_->GetEdepSquaredBuffer(thread_index);
    }

    // Getting kernel, and setting parameters
    cl::Kernel* kernel = solids_[s]->GetKernelTrackThroughSolid(thread_index);
    kernel->setArg(0, number_of_particles);
    kernel->setArg(1, *primary_particles);
    kernel->setArg(2, *randoms);
    kernel->setArg(3, *solid_data);
    kernel->setArg(4, *label_data); // Useful only for GGEMSVoxelizedSolid
    kernel->setArg(5, *cross_sections);
    kernel->setArg(6, *materials);
    kernel->setArg(7, threshold_);
    if (data_reg_type == "HISTOGRAM") {
      kernel->setArg(8, *histogram);
      kernel->setArg(9, *scatter_histogram);
    }
    else if (data_reg_type == "DOSIMETRY") {
      kernel->setArg(8, *dosimetry_params);
      kernel->setArg(9, *edep_tracking_dosimetry);
      kernel->setArg(10, *edep_squared_tracking_dosimetry);
      kernel->setArg(11, *hit_tracking_dosimetry);
      kernel->setArg(12, *photon_tracking_dosimetry);
    }

    // Launching kernel
    GGint kernel_status = queue->enqueueNDRangeKernel(*kernel, 0, global_wi, local_wi, nullptr, event);
    opencl_manager.CheckOpenCLError(kernel_status, "GGEMSNavigator", "TrackThroughSolid");

    // GGEMS Profiling
    GGEMSProfilerManager& profiler_manager = GGEMSProfilerManager::GetInstance();
    profiler_manager.HandleEvent(*event, oss.str());
    queue->finish();
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
  for (GGsize i = 0; i < number_of_solids_; ++i) {
    solids_[i]->PrintInfos();
  }
  materials_->PrintInfos();
  GGcout("GGEMSNavigator", "PrintInfos", 0) << "* Output: " << output_basename_ << GGendl;
  GGcout("GGEMSNavigator", "PrintInfos", 0) << GGendl;
}
