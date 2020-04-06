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
#include "GGEMS/physics/GGEMSParticleCrossSectionsStack.hh"
#include "GGEMS/tools/GGEMSTools.hh"
#include "GGEMS/materials/GGEMSMaterials.hh"
#include "GGEMS/physics/GGEMSProcessesManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSCrossSections::GGEMSCrossSections(void)
: opencl_manager_(GGEMSOpenCLManager::GetInstance()),
  process_manager_(GGEMSProcessesManager::GetInstance())
{
  GGcout("GGEMSCrossSections", "GGEMSCrossSections", 3) << "Allocation of GGEMSCrossSections..." << GGendl;
  is_process_activated_.resize(GGEMSProcess::NUMBER_PROCESSES);
  for (auto&& i : is_process_activated_) i = false;

  // Allocating memory for cross section tables
  particle_cross_sections_ = opencl_manager_.Allocate(nullptr, sizeof(GGEMSParticleCrossSections), CL_MEM_READ_WRITE);
  opencl_manager_.AddRAMMemory(sizeof(GGEMSParticleCrossSections));

  process_manager_.AddProcessRAM(sizeof(GGEMSParticleCrossSections));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSCrossSections::~GGEMSCrossSections(void)
{
  GGcout("GGEMSCrossSections", "~GGEMSCrossSections", 3) << "Deallocation of GGEMSCrossSections..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSCrossSections::AddProcess(std::string const& process_name, std::string const& particle_name)
{
  GGcout("GGEMSCrossSections", "AddProcess", 3) << "Adding process " << process_name << "..." << GGendl;

  if (process_name == "Compton") {
    if (!is_process_activated_.at(GGEMSProcess::COMPTON_SCATTERING)) {
      em_processes_list_.push_back(std::make_shared<GGEMSComptonScattering>());
      is_process_activated_.at(GGEMSProcess::COMPTON_SCATTERING) = true;
    }
    else {
      GGwarn("GGEMSCrossSections", "AddProcess", 3) << "Compton scattering process already activated!!!" << GGendl;
    }
  }
  else {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Unknown process!!! The available processes in GGEMS are:" << std::endl;
    oss << "    * For incident gamma:" << std::endl;
    oss << "        - 'Compton'";
    GGEMSMisc::ThrowException("GGEMSCrossSections", "AddProcess", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSCrossSections::Initialize(std::shared_ptr<GGEMSMaterials> const materials)
{
  GGcout("GGEMSCrossSections", "Initialize", 3) << "Initializing cross section tables..." << GGendl;

  // Set missing information in cross section table
  GGEMSParticleCrossSections* particle_cross_sections_device = opencl_manager_.GetDeviceBuffer<GGEMSParticleCrossSections>(particle_cross_sections_, sizeof(GGEMSParticleCrossSections));

  // Storing information for process manager
  GGushort const kNBins = process_manager_.GetCrossSectionTableNumberOfBins();
  GGfloat const kMinEnergy = process_manager_.GetCrossSectionTableMinEnergy();
  GGfloat const kMaxEnergy = process_manager_.GetCrossSectionTableMaxEnergy();

  particle_cross_sections_device->number_of_bins_ = kNBins;
  particle_cross_sections_device->min_energy_ = kMinEnergy;
  particle_cross_sections_device->max_energy_ = kMaxEnergy;

  // Storing information from materials
  particle_cross_sections_device->number_of_materials_ = static_cast<GGuchar>(materials->GetNumberOfMaterials());

  // Filling energy table with log scale
  GGfloat const kSlope = logf(kMaxEnergy/kMinEnergy);
  for (GGushort i = 0; i < kNBins; ++i) {
    particle_cross_sections_device->energy_bins_[i] = kMinEnergy * expf(kSlope * (static_cast<float>(i) / (kNBins-1.0f))) * GGEMSUnits::MeV;
  }

  // Release pointer
  opencl_manager_.ReleaseDeviceBuffer(particle_cross_sections_, particle_cross_sections_device);

  // Loop over the activated physic processes and building tables
  for (auto&& i : em_processes_list_)
    i->BuildCrossSectionTables(particle_cross_sections_, materials->GetMaterialTables());
}
