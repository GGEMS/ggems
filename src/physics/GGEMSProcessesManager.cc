/*!
  \file GGEMSProcessesManager.cc

  \brief GGEMS class managing the processes in GGEMS simulation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday March 9, 2020
*/

#include "GGEMS/physics/GGEMSProcessesManager.hh"
#include "GGEMS/navigators/GGEMSPhantomNavigatorManager.hh"
#include "GGEMS/physics/GGEMSCrossSections.hh"
#include "GGEMS/physics/GGEMSEMProcess.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSProcessesManager::GGEMSProcessesManager(void)
: cross_section_table_number_of_bins_(GGEMSProcessParams::CROSS_SECTION_TABLE_NUMBER_BINS),
  cross_section_table_min_energy_(GGEMSProcessParams::CROSS_SECTION_TABLE_ENERGY_MIN),
  cross_section_table_max_energy_(GGEMSProcessParams::CROSS_SECTION_TABLE_ENERGY_MAX),
  allocated_RAM_for_processes_(0)
{
  GGcout("GGEMSProcessesManager", "GGEMSProcessesManager", 3) << "Allocation of GGEMSProcessesManager..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSProcessesManager::~GGEMSProcessesManager(void)
{
  GGcout("GGEMSProcessesManager", "~GGEMSProcessesManager", 3) << "Deallocation of GGEMSProcessesManager..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProcessesManager::SetCrossSectionTableNumberOfBins(GGushort const& number_of_bins)
{
  cross_section_table_number_of_bins_ = number_of_bins;

  // Checking number of bins, not exceed 1024
  if (cross_section_table_number_of_bins_ > 1024) {
    GGwarn("GGEMSProcessesManager", "SetCrossSectionTableNumberOfBins", 0) << "Warning!!! Number of bins in the cross section table > 1024, the number of bins is reset to 1000." << GGendl;
    cross_section_table_number_of_bins_ = 1024;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProcessesManager::SetCrossSectionTableMinimumEnergy(GGfloat const& energy, char const* unit)
{
  cross_section_table_min_energy_ = GGEMSUnits::EnergyUnit(energy, unit);

  // Checking the min value
  if (cross_section_table_min_energy_ < GGEMSProcessParams::CROSS_SECTION_TABLE_ENERGY_MIN) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "The minimum of energy in the cross section table is 990 eV, yours is " << cross_section_table_min_energy_/GGEMSUnits::eV << " eV!!!";
    GGEMSMisc::ThrowException("GGEMSProcessesManager", "SetCrossSectionTableMinimumEnergy", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProcessesManager::SetCrossSectionTableMaximumEnergy(GGfloat const& energy, char const* unit)
{
  cross_section_table_max_energy_ = GGEMSUnits::EnergyUnit(energy, unit);

  // Checking the max value
  if (cross_section_table_max_energy_ > GGEMSProcessParams::CROSS_SECTION_TABLE_ENERGY_MAX) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "The maximum of energy in the cross section table is 250 MeV, yours is " << cross_section_table_max_energy_/GGEMSUnits::MeV << " MeV!!!";
    GGEMSMisc::ThrowException("GGEMSProcessesManager", "SetCrossSectionTableMaximumEnergy", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProcessesManager::AddProcess(std::string const& process_name, std::string const& particle_name, std::string const& phantom_name)
{
  // Pointer on phantoms
  GGEMSPhantomNavigatorManager& phantom_navigator_manager = GGEMSPhantomNavigatorManager::GetInstance();

  if (phantom_name == "all") {
    // Loop over phantom
    for (size_t i = 0; i < phantom_navigator_manager.GetNumberOfPhantomNavigators(); ++i) {
      std::shared_ptr<GGEMSCrossSections> cross_sections = ((phantom_navigator_manager.GetPhantomNavigators()).at(i))->GetCrossSections();
      cross_sections->AddProcess(process_name, particle_name);
    }
  }
  else {
    std::shared_ptr<GGEMSCrossSections> cross_sections = phantom_navigator_manager.GetPhantomNavigator(phantom_name)->GetCrossSections();
    cross_sections->AddProcess(process_name, particle_name);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProcessesManager::AddProcessRAM(GGulong const& size)
{
  allocated_RAM_for_processes_ += size;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProcessesManager::PrintAllocatedRAM(void) const
{
  GGcout("GGEMSProcessesManager", "PrintAllocatedRAM", 0) << "########################################" << GGendl;
  GGcout("GGEMSProcessesManager", "PrintAllocatedRAM", 0) << "Allocated RAM memory for processes: " << allocated_RAM_for_processes_ << " bytes" << GGendl;
  GGcout("GGEMSProcessesManager", "PrintAllocatedRAM", 0) << "########################################" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProcessesManager::PrintInfos(void) const
{
  // Pointer on phantoms
  GGEMSPhantomNavigatorManager& phantom_navigator_manager = GGEMSPhantomNavigatorManager::GetInstance();

  GGcout("GGEMSProcessesManager", "PrintInfos", 0) << "Cross section table parameters:" << GGendl;
  GGcout("GGEMSProcessesManager", "PrintInfos", 0) << "-------------------------------" << GGendl;
  GGcout("GGEMSProcessesManager", "PrintInfos", 0) << "    * Number of bins for the cross section table: " << cross_section_table_number_of_bins_ << GGendl;
  GGcout("GGEMSProcessesManager", "PrintInfos", 0) << "    * Range in energy of cross section table: [" << cross_section_table_min_energy_/GGEMSUnits::keV << ", " << cross_section_table_max_energy_/GGEMSUnits::keV << "] keV" << GGendl;
  GGcout("GGEMSProcessesManager", "PrintInfos", 0) << GGendl;
  // Loop over all phantoms
  for (size_t i = 0; i < phantom_navigator_manager.GetNumberOfPhantomNavigators(); ++i) {
    std::shared_ptr<GGEMSCrossSections> cross_sections = ((phantom_navigator_manager.GetPhantomNavigators()).at(i))->GetCrossSections();
    GGcout("GGEMSProcessesManager", "PrintInfos", 0) << "Activated processes in phantom: " << ((phantom_navigator_manager.GetPhantomNavigators()).at(i))->GetPhantomName() << GGendl;
    for (size_t j = 0; j < cross_sections->GetProcessesList().size(); ++j) {
      GGcout("GGEMSProcessesManager", "PrintInfos", 0) << "    * " << cross_sections->GetProcessesList().at(j)->GetProcessName() << GGendl;
      GGcout("GGEMSProcessesManager", "PrintInfos", 0) << GGendl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProcessesManager::PrintAvailableProcesses(void) const
{
  GGcout("GGEMSProcessesManager", "PrintAvailableProcesses", 0) << "Available processes:" << GGendl;
  GGcout("GGEMSProcessesManager", "PrintAvailableProcesses", 0) << "--------------------" << GGendl;
  GGcout("GGEMSProcessesManager", "PrintAvailableProcesses", 0) << "    * 'Compton' using Geant4 standard model (G4KleinNishinaCompton)" << GGendl;
  GGcout("GGEMSProcessesManager", "PrintAvailableProcesses", 0) << "        - 'gamma' incident particle" << GGendl;
  GGcout("GGEMSProcessesManager", "PrintAvailableProcesses", 0) << "        - 'e-' secondary particle" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSProcessesManager* get_instance_processes_manager(void)
{
  return &GGEMSProcessesManager::GetInstance();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void add_process_processes_manager(GGEMSProcessesManager* processes_manager, char const* process_name, char const* particle_name, char const* phantom_name)
{
  processes_manager->AddProcess(process_name, particle_name, phantom_name);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_cross_section_table_number_of_bins_processes_manager(GGEMSProcessesManager* processes_manager, GGushort const number_of_bins)
{
  processes_manager->SetCrossSectionTableNumberOfBins(number_of_bins);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_cross_section_table_minimum_energy_processes_manager(GGEMSProcessesManager* processes_manager, GGfloat const energy, char const* unit)
{
  processes_manager->SetCrossSectionTableMinimumEnergy(energy, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_cross_section_table_maximum_energy_processes_manager(GGEMSProcessesManager* processes_manager, GGfloat const energy, char const* unit)
{
  processes_manager->SetCrossSectionTableMaximumEnergy(energy, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void print_infos_processes_manager(GGEMSProcessesManager* processes_manager)
{
  processes_manager->PrintInfos();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void print_available_processes_manager(GGEMSProcessesManager* processes_manager)
{
  processes_manager->PrintAvailableProcesses();
}
