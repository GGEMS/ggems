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
  \file GGEMSCrossSections.cc

  \brief GGEMS class handling the cross sections tables

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 31, 2020
*/

#include "GGEMS/physics/GGEMSCrossSections.hh"
#include "GGEMS/physics/GGEMSComptonScattering.hh"
#include "GGEMS/physics/GGEMSPhotoElectricEffect.hh"
#include "GGEMS/physics/GGEMSRayleighScattering.hh"
#include "GGEMS/materials/GGEMSMaterials.hh"
#include "GGEMS/physics/GGEMSProcessesManager.hh"
#include "GGEMS/tools/GGEMSRAMManager.hh"
#include "GGEMS/maths/GGEMSMathAlgorithms.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSCrossSections::GGEMSCrossSections(void)
{
  GGcout("GGEMSCrossSections", "GGEMSCrossSections", 3) << "GGEMSSource creating..." << GGendl;

  is_process_activated_.resize(NUMBER_PROCESSES);
  for (auto&& i : is_process_activated_) i = false;

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  number_activated_devices_ = opencl_manager.GetNumberOfActivatedDevice();

  // Allocating memory for cross section tables on host and device
  particle_cross_sections_ = new cl::Buffer*[number_activated_devices_];
  for (GGsize i = 0; i < number_activated_devices_; ++i) {
    particle_cross_sections_[i] = opencl_manager.Allocate(nullptr, sizeof(GGEMSParticleCrossSections), i, CL_MEM_READ_WRITE, "GGEMSCrossSections");
  }

  // Useful to avoid memory transfer between host and OpenCL
  particle_cross_sections_host_ = new GGEMSParticleCrossSections();

  GGcout("GGEMSCrossSections", "GGEMSCrossSections", 3) << "GGEMSSource created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSCrossSections::~GGEMSCrossSections(void)
{
  GGcout("GGEMSCrossSections", "~GGEMSCrossSections", 3) << "GGEMSSource erasing..." << GGendl;

  if (particle_cross_sections_host_) {
    delete particle_cross_sections_host_;
    particle_cross_sections_host_ = nullptr;
  }

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  if (particle_cross_sections_) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      opencl_manager.Deallocate(particle_cross_sections_[i], sizeof(GGEMSParticleCrossSections), i);
    }
    delete[] particle_cross_sections_;
    particle_cross_sections_ = nullptr;
  }

  GGcout("GGEMSCrossSections", "~GGEMSCrossSections", 3) << "GGEMSSource erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSCrossSections::AddProcess(std::string const& process_name, std::string const& particle_type, bool const& is_secondary)
{
  GGcout("GGEMSCrossSections", "AddProcess", 1) << "Adding " << process_name << " scattering process..." << GGendl;

  if (process_name == "Compton") {
    if (!is_process_activated_.at(COMPTON_SCATTERING)) {
      em_processes_list_.push_back(std::make_shared<GGEMSComptonScattering>(particle_type, is_secondary));
      is_process_activated_.at(COMPTON_SCATTERING) = true;
    }
    else {
      GGwarn("GGEMSCrossSections", "AddProcess", 3) << "Compton scattering process already activated!!!" << GGendl;
    }
  }
  else if (process_name == "Photoelectric") {
    if (!is_process_activated_.at(PHOTOELECTRIC_EFFECT)) {
      em_processes_list_.push_back(std::make_shared<GGEMSPhotoElectricEffect>(particle_type, is_secondary));
      is_process_activated_.at(PHOTOELECTRIC_EFFECT) = true;
    }
    else {
      GGwarn("GGEMSCrossSections", "AddProcess", 3) << "PhotoElectric effect process already activated!!!" << GGendl;
    }
  }
  else if (process_name == "Rayleigh") {
    if (!is_process_activated_.at(RAYLEIGH_SCATTERING)) {
      em_processes_list_.push_back(std::make_shared<GGEMSRayleighScattering>(particle_type, is_secondary));
      is_process_activated_.at(RAYLEIGH_SCATTERING) = true;
    }
    else {
      GGwarn("GGEMSCrossSections", "AddProcess", 3) << "PhotoElectric effect process already activated!!!" << GGendl;
    }
  }
  else {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Unknown process!!! The available processes in GGEMS are:" << std::endl;
    oss << "    * For incident gamma:" << std::endl;
    oss << "        - 'Compton'" << std::endl;
    oss << "        - 'Photoelectric'" << std::endl;
    oss << "        - 'Rayleigh'" << std::endl;
    GGEMSMisc::ThrowException("GGEMSCrossSections", "AddProcess", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSCrossSections::Initialize(GGEMSMaterials const* materials)
{
  GGcout("GGEMSCrossSections", "Initialize", 1) << "Initializing cross section tables..." << GGendl;

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get the process manager
  GGEMSProcessesManager& process_manager = GGEMSProcessesManager::GetInstance();

  // Storing information for process manager
  GGsize number_of_bins = process_manager.GetCrossSectionTableNumberOfBins();
  GGfloat min_energy = process_manager.GetCrossSectionTableMinEnergy();
  GGfloat max_energy = process_manager.GetCrossSectionTableMaxEnergy();

  // Initialize physics on each device
  for (GGsize j = 0; j < number_activated_devices_; ++j) {
    GGEMSParticleCrossSections* particle_cross_sections_device = opencl_manager.GetDeviceBuffer<GGEMSParticleCrossSections>(particle_cross_sections_[j], sizeof(GGEMSParticleCrossSections), j);

    particle_cross_sections_device->number_of_bins_ = number_of_bins;
    particle_cross_sections_device->min_energy_ = min_energy;
    particle_cross_sections_device->max_energy_ = max_energy;
    for (GGsize i = 0; i < materials->GetNumberOfMaterials(); ++i) {
      #ifdef _WIN32
      strcpy_s(reinterpret_cast<char*>(particle_cross_sections_device->material_names_[i]), 32, (materials->GetMaterialName(i)).c_str());
      #else
      strcpy(reinterpret_cast<char*>(particle_cross_sections_device->material_names_[i]), (materials->GetMaterialName(i)).c_str());
      #endif
    }

    // Storing information from materials
    particle_cross_sections_device->number_of_materials_ = static_cast<GGuchar>(materials->GetNumberOfMaterials());

    // Filling energy table with log scale
    GGfloat slope = logf(max_energy/min_energy);
    for (GGsize i = 0; i < number_of_bins; ++i) {
      particle_cross_sections_device->energy_bins_[i] = min_energy * expf(slope * (static_cast<float>(i) / (static_cast<GGfloat>(number_of_bins)-1.0f))) * MeV;
    }

    // Release pointer
    opencl_manager.ReleaseDeviceBuffer(particle_cross_sections_[j], particle_cross_sections_device, j);

    // Loop over the activated physic processes and building tables
    for (auto&& i : em_processes_list_)
      i->BuildCrossSectionTables(particle_cross_sections_[j], materials->GetMaterialTables(j), j);
  }

  // Copy data from device to RAM memory (optimization for python users)
  LoadPhysicTablesOnHost();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSCrossSections::LoadPhysicTablesOnHost(void)
{
  GGcout("GGEMSCrossSections", "LoadPhysicTablesOnHost", 1) << "Loading physic tables from OpenCL device to host (RAM)..." << GGendl;

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  GGEMSParticleCrossSections* particle_cross_sections_device = opencl_manager.GetDeviceBuffer<GGEMSParticleCrossSections>(particle_cross_sections_[0], sizeof(GGEMSParticleCrossSections), 0);

  particle_cross_sections_host_->number_of_bins_ = particle_cross_sections_device->number_of_bins_;
  particle_cross_sections_host_->number_of_materials_ = particle_cross_sections_device->number_of_materials_;
  particle_cross_sections_host_->min_energy_ = particle_cross_sections_device->min_energy_;
  particle_cross_sections_host_->max_energy_ = particle_cross_sections_device->max_energy_;

  for(GGuchar i = 0; i < particle_cross_sections_host_->number_of_materials_; ++i) {
    for(GGuchar j = 0; j < 32; ++j) {
      particle_cross_sections_host_->material_names_[i][j] = particle_cross_sections_device->material_names_[i][j];
    }
  }

  for(GGushort i = 0; i < particle_cross_sections_host_->number_of_bins_; ++i) {
    particle_cross_sections_host_->energy_bins_[i] = particle_cross_sections_device->energy_bins_[i];
  }

  particle_cross_sections_host_->number_of_activated_photon_processes_ = particle_cross_sections_device->number_of_activated_photon_processes_;

  for(GGuchar i = 0; i < NUMBER_PHOTON_PROCESSES; ++i) {
    particle_cross_sections_host_->photon_cs_id_[i] = particle_cross_sections_device->photon_cs_id_[i];
  }

  for(GGuchar j = 0; j < NUMBER_PHOTON_PROCESSES; ++j) {
    for(GGuint i = 0; i < 256*MAX_CROSS_SECTION_TABLE_NUMBER_BINS; ++i) {
      particle_cross_sections_host_->photon_cross_sections_[j][i] = particle_cross_sections_device->photon_cross_sections_[j][i];
    }
  }

  for(GGuchar j = 0; j < NUMBER_PHOTON_PROCESSES; ++j) {
    for(GGuint i = 0; i < 101*MAX_CROSS_SECTION_TABLE_NUMBER_BINS; ++i) {
      particle_cross_sections_host_->photon_cross_sections_per_atom_[j][i] = particle_cross_sections_device->photon_cross_sections_per_atom_[j][i];
    }
  }

  // Release pointer
  opencl_manager.ReleaseDeviceBuffer(particle_cross_sections_[0], particle_cross_sections_device, 0);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSCrossSections::GetPhotonCrossSection(std::string const& process_name, std::string const& material_name, GGfloat const& energy, std::string const& unit) const
{
  // Get min and max energy in the table, and number of bins
  GGfloat min_energy = particle_cross_sections_host_->min_energy_;
  GGfloat max_energy = particle_cross_sections_host_->max_energy_;
  GGsize number_of_bins = particle_cross_sections_host_->number_of_bins_;
  GGsize number_of_materials = particle_cross_sections_host_->number_of_materials_;

  // Converting energy
  GGfloat e_MeV = EnergyUnit(energy, unit);

  if (e_MeV < min_energy || e_MeV > max_energy) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Problem energy: " << e_MeV << " " << unit << " is not in the range [" << min_energy << ", " << max_energy << "] MeV!!!" << std::endl;
    GGEMSMisc::ThrowException("GGEMSCrossSections", "GetPhotonCrossSection", oss.str());
  }

  // Get the process id
  GGuchar process_id = 0;
  if (process_name == "Compton") {
    process_id = COMPTON_SCATTERING;
  }
  else if (process_name == "Photoelectric") {
    process_id = PHOTOELECTRIC_EFFECT;
  }
  else if (process_name == "Rayleigh") {
    process_id = RAYLEIGH_SCATTERING;
  }
  else {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Unknown process!!! The available processes for photon in GGEMS are:" << std::endl;
    oss << "    - 'Compton'" << std::endl;
    oss << "    - 'Photoelectric'" << std::endl;
    oss << "    - 'Rayleigh'" << std::endl;
    GGEMSMisc::ThrowException("GGEMSCrossSections", "GetPhotonCrossSection", oss.str());
  }

  // Get id of material
  GGsize mat_id = 0;
  for (GGsize i = 0; i < number_of_materials; ++i) {
    if (strcmp(material_name.c_str(), reinterpret_cast<char*>(particle_cross_sections_host_->material_names_[i])) == 0) {
      mat_id = i;
      break;
    }
  }

  // Get density of material
  GGEMSMaterialsDatabaseManager& material_database_manager = GGEMSMaterialsDatabaseManager::GetInstance();
  GGfloat density = material_database_manager.GetMaterial(material_name).density_;

  // Computing the energy bin
  GGsize energy_bin = static_cast<GGsize>(BinarySearchLeft(e_MeV, particle_cross_sections_host_->energy_bins_, static_cast<GGint>(number_of_bins), 0, 0));

  // Compute cross section using linear interpolation
  GGfloat energy_a = particle_cross_sections_host_->energy_bins_[energy_bin];
  GGfloat energy_b = particle_cross_sections_host_->energy_bins_[energy_bin+1];
  GGfloat cross_section_a = particle_cross_sections_host_->photon_cross_sections_[process_id][energy_bin + number_of_bins*mat_id];
  GGfloat cross_section_b = particle_cross_sections_host_->photon_cross_sections_[process_id][energy_bin+1 + number_of_bins*mat_id];

  GGfloat cross_section = LinearInterpolation(energy_a, cross_section_a, energy_b, cross_section_b, e_MeV);

  return (cross_section/density) / (cm2/g);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSCrossSections* create_ggems_cross_sections(void)
{
  return new(std::nothrow) GGEMSCrossSections;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void add_process_ggems_cross_sections(GGEMSCrossSections* cross_sections, char const* process_name, char const* particle_name, bool const is_secondary)
{
  cross_sections->AddProcess(process_name, particle_name, is_secondary);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void initialize_ggems_cross_sections(GGEMSCrossSections* cross_sections, GGEMSMaterials* materials)
{
  cross_sections->Initialize(materials);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

float get_cs_cross_sections(GGEMSCrossSections* cross_sections, char const* process_name, char const* material_name, GGfloat const energy, char const* unit)
{
  return cross_sections->GetPhotonCrossSection(process_name, material_name, energy, unit);
}
