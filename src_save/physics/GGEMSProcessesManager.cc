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
  \file GGEMSProcessesManager.cc

  \brief GGEMS class managing the processes in GGEMS simulation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday March 9, 2020
*/

#include "GGEMS/physics/GGEMSProcessesManager.hh"
#include "GGEMS/navigators/GGEMSNavigatorManager.hh"
#include "GGEMS/physics/GGEMSCrossSections.hh"
#include "GGEMS/physics/GGEMSEMProcess.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSProcessesManager::GGEMSProcessesManager(void)
: cross_section_table_number_of_bins_(CROSS_SECTION_TABLE_NUMBER_BINS),
  cross_section_table_min_energy_(CROSS_SECTION_TABLE_ENERGY_MIN),
  cross_section_table_max_energy_(CROSS_SECTION_TABLE_ENERGY_MAX),
  is_processes_print_tables_(false)
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

void GGEMSProcessesManager::SetCrossSectionTableNumberOfBins(GGsize const& number_of_bins)
{
  cross_section_table_number_of_bins_ = number_of_bins;

  // Checking number of bins
  if (cross_section_table_number_of_bins_ > MAX_CROSS_SECTION_TABLE_NUMBER_BINS) {
    GGwarn("GGEMSProcessesManager", "SetCrossSectionTableNumberOfBins", 0) << "Warning!!! Number of bins in the cross section table > "
      << MAX_CROSS_SECTION_TABLE_NUMBER_BINS << " the number of bins is set to "
      << MAX_CROSS_SECTION_TABLE_NUMBER_BINS << "." << GGendl;
    cross_section_table_number_of_bins_ = MAX_CROSS_SECTION_TABLE_NUMBER_BINS;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProcessesManager::SetCrossSectionTableMinimumEnergy(GGfloat const& energy, char const* unit)
{
  cross_section_table_min_energy_ = EnergyUnit(energy, unit);

  // Checking the min value
  if (cross_section_table_min_energy_ < CROSS_SECTION_TABLE_ENERGY_MIN) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "The minimum of energy in the cross section table is 990 eV, yours is " << BestEnergyUnit(cross_section_table_min_energy_);
    GGEMSMisc::ThrowException("GGEMSProcessesManager", "SetCrossSectionTableMinimumEnergy", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProcessesManager::SetCrossSectionTableMaximumEnergy(GGfloat const& energy, char const* unit)
{
  cross_section_table_max_energy_ = EnergyUnit(energy, unit);

  // Checking the max value
  if (cross_section_table_max_energy_ > CROSS_SECTION_TABLE_ENERGY_MAX) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "The maximum of energy in the cross section table is 250 MeV, yours is " << BestEnergyUnit(cross_section_table_max_energy_);
    GGEMSMisc::ThrowException("GGEMSProcessesManager", "SetCrossSectionTableMaximumEnergy", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProcessesManager::AddProcess(std::string const& process_name, std::string const& particle_name, std::string const& phantom_name)
{
  // Pointer on phantoms
  GGEMSNavigatorManager& navigator_manager = GGEMSNavigatorManager::GetInstance();

  if (phantom_name == "all") {
    // Loop over phantom
    for (size_t i = 0; i < navigator_manager.GetNumberOfNavigators(); ++i) {
      std::shared_ptr<GGEMSCrossSections> cross_sections = ((navigator_manager.GetNavigators()).at(i))->GetCrossSections().lock();
      cross_sections->AddProcess(process_name, particle_name);
    }
  }
  else {
    std::shared_ptr<GGEMSCrossSections> cross_sections = navigator_manager.GetNavigator(phantom_name)->GetCrossSections().lock();
    cross_sections->AddProcess(process_name, particle_name);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProcessesManager::PrintInfos(void) const
{
  // Pointer on phantoms
  GGEMSNavigatorManager& navigator_manager = GGEMSNavigatorManager::GetInstance();

  GGcout("GGEMSProcessesManager", "PrintInfos", 0) << "Cross section table parameters:" << GGendl;
  GGcout("GGEMSProcessesManager", "PrintInfos", 0) << "-------------------------------" << GGendl;
  GGcout("GGEMSProcessesManager", "PrintInfos", 0) << "    * Number of bins for the cross section table: " << cross_section_table_number_of_bins_ << GGendl;
  GGcout("GGEMSProcessesManager", "PrintInfos", 0) << "    * Range in energy of cross section table: [" << BestEnergyUnit(cross_section_table_min_energy_) << ", " << BestEnergyUnit(cross_section_table_max_energy_) << "]" << GGendl;
  GGcout("GGEMSProcessesManager", "PrintInfos", 0) << GGendl;
  // Loop over all phantoms
  for (size_t i = 0; i < navigator_manager.GetNumberOfNavigators(); ++i) {
    std::shared_ptr<GGEMSCrossSections> cross_sections = ((navigator_manager.GetNavigators()).at(i))->GetCrossSections().lock();
    GGcout("GGEMSProcessesManager", "PrintInfos", 0) << "Activated processes in phantom: " << ((navigator_manager.GetNavigators()).at(i))->GetNavigatorName() << GGendl;
    for (size_t j = 0; j < cross_sections->GetProcessesList().size(); ++j) {
      GGcout("GGEMSProcessesManager", "PrintInfos", 0) << "    * " << cross_sections->GetProcessesList().at(j)->GetProcessName() << GGendl;
    }
    GGcout("GGEMSProcessesManager", "PrintInfos", 0) << GGendl;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProcessesManager::PrintAvailableProcesses(void) const
{
  GGcout("GGEMSProcessesManager", "PrintAvailableProcesses", 0) << "Available processes:" << GGendl;
  GGcout("GGEMSProcessesManager", "PrintAvailableProcesses", 0) << "--------------------" << GGendl;
  GGcout("GGEMSProcessesManager", "PrintAvailableProcesses", 0) << "    * 'Compton' scattering (Klein-Nishina model without atomic shell effect)" << GGendl;
  GGcout("GGEMSProcessesManager", "PrintAvailableProcesses", 0) << "        - 'gamma' incident particle" << GGendl;
  GGcout("GGEMSProcessesManager", "PrintAvailableProcesses", 0) << "        - 'e-' secondary particle" << GGendl;
  GGcout("GGEMSProcessesManager", "PrintAvailableProcesses", 0) << GGendl;
  GGcout("GGEMSProcessesManager", "PrintAvailableProcesses", 0) << "    * 'Photoelectric' effect (Sandia table)" << GGendl;
  GGcout("GGEMSProcessesManager", "PrintAvailableProcesses", 0) << "        - 'gamma' incident particle" << GGendl;
  GGcout("GGEMSProcessesManager", "PrintAvailableProcesses", 0) << "        - 'e-' secondary particle" << GGendl;
  GGcout("GGEMSProcessesManager", "PrintAvailableProcesses", 0) << GGendl;
  GGcout("GGEMSProcessesManager", "PrintAvailableProcesses", 0) << "    * 'Rayleigh' scattering (Livermore model)" << GGendl;
  GGcout("GGEMSProcessesManager", "PrintAvailableProcesses", 0) << "        - 'gamma' incident particle" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProcessesManager::PrintPhysicTables(bool const& is_processes_print_tables)
{
  is_processes_print_tables_ = is_processes_print_tables;
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

void set_cross_section_table_number_of_bins_processes_manager(GGEMSProcessesManager* processes_manager, GGsize const number_of_bins)
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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void print_tables_processes_manager(GGEMSProcessesManager* processes_manager, bool const is_processes_print_tables)
{
  processes_manager->PrintPhysicTables(is_processes_print_tables);
}
