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
  \file GGEMSRangeCutsManager.cc

  \brief GGEMS class managing the range cuts in GGEMS simulation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Friday March 6, 2020
*/

#include "GGEMS/physics/GGEMSRangeCutsManager.hh"
#include "GGEMS/physics/GGEMSRangeCuts.hh"
#include "GGEMS/navigators/GGEMSNavigatorManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSRangeCutsManager::GGEMSRangeCutsManager(void)
{
  GGcout("GGEMSRangeCutsManager", "GGEMSRangeCutsManager", 3) << "GGEMSRangeCutsManager creating..." << GGendl;

  GGcout("GGEMSRangeCutsManager", "GGEMSRangeCutsManager", 3) << "GGEMSRangeCutsManager created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSRangeCutsManager::~GGEMSRangeCutsManager(void)
{
  GGcout("GGEMSRangeCutsManager", "~GGEMSRangeCutsManager", 3) << "GGEMSRangeCutsManager erasing..." << GGendl;

  GGcout("GGEMSRangeCutsManager", "~GGEMSRangeCutsManager", 3) << "GGEMSRangeCutsManager erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRangeCutsManager::PrintInfos(void) const
{
  GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "Printing infos about range cuts" << GGendl;

  // Loop over the phantom and printing range cuts by materials in length and energy
  GGEMSNavigatorManager& navigator_manager = GGEMSNavigatorManager::GetInstance();

  for (size_t i = 0; i < navigator_manager.GetNumberOfNavigators(); ++i) {
    // Get pointer on phantom navigator
    std::string name_of_phantom = ((navigator_manager.GetNavigators())[i])->GetNavigatorName();
    // Get the Range cut pointer
    GGEMSRangeCuts* range_cuts = ((navigator_manager.GetNavigators())[i])->GetMaterials()->GetRangeCuts();

    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "Range cuts for phantom navigator: " << name_of_phantom << GGendl;
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "---------------------------------" << GGendl;
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "Length cuts:" << GGendl;
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "    * Photon: " << BestDistanceUnit(range_cuts->GetPhotonDistanceCut()) << GGendl;
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "    * Electron: " << BestDistanceUnit(range_cuts->GetElectronDistanceCut()) << GGendl;
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "    * Positron: " << BestDistanceUnit(range_cuts->GetPositronDistanceCut()) << GGendl;
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "Energy cuts:" << GGendl;
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "    * Photon:" << GGendl;
    EnergyCutUMap energy_cut_of_photon = range_cuts->GetPhotonEnergyCut();
    for (auto&& j : energy_cut_of_photon) {
      GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "        - " << j.first << ": " << BestEnergyUnit(j.second) << GGendl;
    }
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "    * Electron:" << GGendl;
    EnergyCutUMap energy_cut_of_electron = range_cuts->GetElectronEnergyCut();
    for (auto&& j : energy_cut_of_electron) {
      GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "        - " << j.first << ": " << BestEnergyUnit(j.second)<< GGendl;
    }
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "    * Positron:" << GGendl;
    EnergyCutUMap const energy_cut_of_positron = range_cuts->GetPositronEnergyCut();
    for (auto&& j : energy_cut_of_positron) {
      GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "        - " << j.first << ": " << BestEnergyUnit(j.second) << GGendl;
    }
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << GGendl;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRangeCutsManager::SetLengthCut(std::string const& phantom_name, std::string const& particle_name, GGfloat const& value, std::string const& unit)
{
  GGEMSNavigatorManager& navigator_manager = GGEMSNavigatorManager::GetInstance();

  if (phantom_name == "all") {
    // Loop over phantom
    for (size_t i = 0; i < navigator_manager.GetNumberOfNavigators(); ++i) {
      ((navigator_manager.GetNavigators())[i])->GetMaterials()->SetDistanceCut(particle_name, value, unit);
    }
  }
  else {
    navigator_manager.GetNavigator(phantom_name)->GetMaterials()->SetDistanceCut(particle_name, value, unit);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSRangeCutsManager* get_instance_range_cuts_manager(void)
{
  return &GGEMSRangeCutsManager::GetInstance();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_cut_range_cuts_manager(GGEMSRangeCutsManager* range_cut_manager, char const* phantom_name, char const* particle_name, GGfloat const value, char const* unit)
{
  range_cut_manager->SetLengthCut(phantom_name, particle_name, value, unit);
}
