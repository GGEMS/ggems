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

#include "GGEMS/physics/GGEMSMuData.hh"
#include "GGEMS/physics/GGEMSMuDataConstants.hh"
#include "GGEMS/global/GGEMSOpenCLManager.hh"
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
  is_dosimetry_mode_(false),
  is_tle_(0)
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

  is_visible_ = false;
  color_name_ = "";
  mu_tables_ = nullptr;

  GGcout("GGEMSNavigator", "GGEMSNavigator", 3) << "GGEMSNavigator created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSNavigator::~GGEMSNavigator(void)
{
  GGcout("GGEMSNavigator", "~GGEMSNavigator", 3) << "GGEMSNavigator erasing..." << GGendl;
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

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

  if (mu_tables_) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      opencl_manager.Deallocate(mu_tables_[i], sizeof(GGEMSMuMuEnData), i);
    }
    delete[] mu_tables_;
    mu_tables_ = nullptr;
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

void GGEMSNavigator::EnableTLE(bool const& is_activated)
{
  if (is_activated) is_tle_ = 1;
  else is_tle_ = 0 ;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::SetVisible(bool const& is_visible)
{
  is_visible_ = is_visible;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::SetColor(std::string const& color)
{
  color_name_ = color;
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

  // Initialization of mu for materials
  if (is_tle_) Init_Mu_Table();
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
    cl::Event event;
    GGint kernel_status = queue->enqueueNDRangeKernel(*kernel, 0, global_wi, local_wi, nullptr, &event);
    opencl_manager.CheckOpenCLError(kernel_status, "GGEMSNavigator", "ParticleSolidDistance");
    queue->finish();

    // GGEMS Profiling
    GGEMSProfilerManager::GetInstance().HandleEvent(event, oss.str());
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
    cl::Event event;
    GGint kernel_status = queue->enqueueNDRangeKernel(*kernel, 0, global_wi, local_wi, nullptr, &event);
    opencl_manager.CheckOpenCLError(kernel_status, "GGEMSNavigator", "ProjectToSolid");
    queue->finish();

    // GGEMS Profiling
    GGEMSProfilerManager::GetInstance().HandleEvent(event, oss.str());
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
    cl::Buffer* mu_table_d = nullptr;

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
      if (is_tle_){
        mu_table_d = mu_tables_[thread_index] ;// TO TEST
      }
    }

    // Getting kernel, and setting parameters
    cl::Kernel* kernel = solids_[s]->GetKernelTrackThroughSolid(thread_index);
    kernel->setArg(0, number_of_particles);
    kernel->setArg(1, *primary_particles);
    kernel->setArg(2, *randoms);
    kernel->setArg(3, *solid_data);
    if (!label_data) kernel->setArg(4, sizeof(cl_mem), NULL);
    else kernel->setArg(4, *label_data); // Useful only for GGEMSVoxelizedSolid
    kernel->setArg(5, *cross_sections);
    kernel->setArg(6, *materials);
    kernel->setArg(7, threshold_);
    if (data_reg_type == "HISTOGRAM") {
      kernel->setArg(8, *histogram);
      if (!scatter_histogram) kernel->setArg(9, sizeof(cl_mem), NULL);
      else kernel->setArg(9, *scatter_histogram);
    }
    else if (data_reg_type == "DOSIMETRY") {
      kernel->setArg(8, *dosimetry_params);
      kernel->setArg(9, *edep_tracking_dosimetry);

      if (!edep_squared_tracking_dosimetry) kernel->setArg(10, sizeof(cl_mem), NULL);
      else kernel->setArg(10, *edep_squared_tracking_dosimetry);

      if (!hit_tracking_dosimetry) kernel->setArg(11, sizeof(cl_mem), NULL);
      else kernel->setArg(11, *hit_tracking_dosimetry);
      if (!photon_tracking_dosimetry) kernel->setArg(12, sizeof(cl_mem), NULL);
      else kernel->setArg(12, *photon_tracking_dosimetry);
      if (!mu_table_d) kernel->setArg(13, sizeof(cl_mem), NULL);
      else kernel->setArg(13, *mu_table_d);
      kernel->setArg(14, is_tle_);
    }

    // Launching kernel
    cl::Event event;
    GGint kernel_status = queue->enqueueNDRangeKernel(*kernel, 0, global_wi, local_wi, nullptr, &event);
    opencl_manager.CheckOpenCLError(kernel_status, "GGEMSNavigator", "TrackThroughSolid");

    // GGEMS Profiling
    GGEMSProfilerManager::GetInstance().HandleEvent(event, oss.str());
    queue->finish();
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::ComputeDose(GGsize const& thread_index)
{
  if (is_dosimetry_mode_) dose_calculator_->ComputeDose(thread_index);
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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::Init_Mu_Table(void) //GGEMSMuMuEnData* mu_table_device, GGEMSMaterialTables*  material_table_device)
{
  GGcout("GGEMSNavigator", "Init_Mu_Table", 3) << "TLE activated!!!" << GGendl;
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Load mu data from database
  GGfloat* energies = new GGfloat[GGEMSMuDataConstants::kMuNbEnergies];
  GGfloat* mu = new GGfloat[GGEMSMuDataConstants::kMuNbEnergies];
  GGfloat* mu_en = new GGfloat[GGEMSMuDataConstants::kMuNbEnergies];
  GGint* mu_index = new GGint[GGEMSMuDataConstants::kMuNbElements];

  GGint index_table = 0;
  GGint index_data = 0;

  for (GGint i = 0; i <= GGEMSMuDataConstants::kMuNbElements; ++i) {
    GGint nb_energies = GGEMSMuDataConstants::kMuNbEnergyBins[i];
    mu_index[i] = index_table;

      for (GGint j = 0; j < nb_energies; ++j) {
        energies[index_table] = GGEMSMuDataConstants::kMuData[index_data++];
        mu[index_table]       = GGEMSMuDataConstants::kMuData[index_data++];
        mu_en[index_table]    = GGEMSMuDataConstants::kMuData[index_data++];

        index_table++;
      }
  }

  // Loop over the device
  mu_tables_ = new cl::Buffer*[number_activated_devices_];
  for (GGsize d = 0; d < number_activated_devices_; ++d) {
    // Allocating memory on OpenCL device
    mu_tables_[d] = opencl_manager.Allocate(nullptr, sizeof(GGEMSMuMuEnData), d, CL_MEM_READ_WRITE, "GGEMSNavigator");

    // Getting the OpenCL pointer on Mu tables
    GGEMSMuMuEnData* mu_table_device = opencl_manager.GetDeviceBuffer<GGEMSMuMuEnData>(mu_tables_[d], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSMuMuEnData), d);

    cl::Buffer* particle_cs = cross_sections_->GetCrossSections(d);
    GGEMSParticleCrossSections* particle_cs_device =  opencl_manager.GetDeviceBuffer<GGEMSParticleCrossSections>(particle_cs, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSParticleCrossSections), d);

    mu_table_device->nb_mat = static_cast<GGint>(particle_cs_device->number_of_materials_);
    mu_table_device->E_max = particle_cs_device->max_energy_;
    mu_table_device->E_min = particle_cs_device->min_energy_;
    mu_table_device->nb_bins = static_cast<GGint>(particle_cs_device->number_of_bins_);

    opencl_manager.ReleaseDeviceBuffer(particle_cs, particle_cs_device, d);

    // Fill energy table with log scale
    GGfloat slope = log(mu_table_device->E_max / mu_table_device->E_min);
    GGint i = 0;
    while (i < mu_table_device->nb_bins) {
      mu_table_device->E_bins[i] = mu_table_device->E_min * exp(slope * ((GGfloat)i / ((GGfloat)mu_table_device->nb_bins-1)))*MeV;
      ++i;
    }

    cl::Buffer* materials = materials_->GetMaterialTables(d);
    GGEMSMaterialTables* materials_device =  opencl_manager.GetDeviceBuffer<GGEMSMaterialTables>(materials, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSMaterialTables), d);

    // For each material and energy bin compute mu and muen
    GGint imat = 0;
    GGint abs_index, E_index, mu_index_E;
    GGint iZ, Z;
    GGfloat energy, mu_over_rho, mu_en_over_rho, frac;
    while (imat < mu_table_device->nb_mat) {
      // for each energy bin
      i=0;
      while (i < mu_table_device->nb_bins) {
        // absolute index to store data within the table
        abs_index = imat*mu_table_device->nb_bins + i;

        // Energy value
        energy = mu_table_device->E_bins[i];

        // For each element of the material
        mu_over_rho = 0.0f; mu_en_over_rho = 0.0f;
        iZ=0;
        while (iZ < materials_device->number_of_chemical_elements_[ imat ]) {
          // Get Z and mass fraction
          Z = materials_device->atomic_number_Z_[materials_device->index_of_chemical_elements_[ imat ] + iZ];
          frac = materials_device->mass_fraction_[materials_device->index_of_chemical_elements_[ imat ] + iZ];

          // Get energy index
          mu_index_E = GGEMSMuDataConstants::kMuIndexEnergy[Z];
          E_index = BinarySearchLeft(energy, energies, mu_index_E+GGEMSMuDataConstants::kMuNbEnergyBins[Z], 0, mu_index_E);

          // Get mu an mu_en from interpolation
          if ( E_index == mu_index_E ) {
            mu_over_rho += mu[ E_index ];
            mu_en_over_rho += mu_en[ E_index ];
          }
          else
          {
            mu_over_rho += frac * LinearInterpolation(energies[E_index-1], mu[E_index-1], energies[E_index], mu[E_index], energy);
            mu_en_over_rho += frac * LinearInterpolation(energies[E_index-1], mu_en[E_index-1], energies[E_index], mu_en[E_index], energy);
          }
          ++iZ;
        }

        // Store values
        mu_table_device->mu[ abs_index ] = mu_over_rho * materials_device->density_of_material_[ imat ] / (g/cm3);
        mu_table_device->mu_en[ abs_index ] = mu_en_over_rho * materials_device->density_of_material_[ imat ] / (g/cm3);

        ++i;
      } // E bin
      ++imat;
    }
    opencl_manager.ReleaseDeviceBuffer(mu_tables_[d], mu_table_device, d);
    opencl_manager.ReleaseDeviceBuffer(materials, materials_device, d);
  }
}
