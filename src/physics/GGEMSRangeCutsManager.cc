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
#include "GGEMS/navigators/GGEMSPhantomNavigatorManager.hh"
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSRangeCutsManager::GGEMSRangeCutsManager(void)
{
  GGcout("GGEMSRangeCutsManager", "GGEMSRangeCutsManager", 3) << "Allocation of GGEMSRangeCutsManager..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSRangeCutsManager::~GGEMSRangeCutsManager(void)
{
  GGcout("GGEMSRangeCutsManager", "~GGEMSRangeCutsManager", 3) << "Deallocation of GGEMSRangeCutsManager..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRangeCutsManager::PrintInfos(void) const
{
  GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "Printing infos about range cuts" << GGendl;

  // Loop over the phantom and printing range cuts by materials in length and energy
  GGEMSPhantomNavigatorManager& phantom_navigator_manager = GGEMSPhantomNavigatorManager::GetInstance();

  for (size_t i = 0; i < phantom_navigator_manager.GetNumberOfPhantomNavigators(); ++i) {
    // Get pointer on phantom navigator
    std::string const kPhantomName = ((phantom_navigator_manager.GetPhantomNavigators()).at(i))->GetPhantomName();
    // Get the Range cut pointer
    std::shared_ptr<GGEMSRangeCuts> range_cuts = ((phantom_navigator_manager.GetPhantomNavigators()).at(i))->GetMaterials()->GetRangeCuts();

    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "Range cuts for phantom navigator: " << kPhantomName << GGendl;
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "---------------------------------" << GGendl;
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "Length cuts:" << GGendl;
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "    * Photon: " << range_cuts->GetPhotonDistanceCut()/GGEMSUnits::mm << " mm"<< GGendl;
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "    * Electron: " << range_cuts->GetElectronDistanceCut()/GGEMSUnits::mm << " mm" << GGendl;
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "    * Positron: " << range_cuts->GetPositronDistanceCut()/GGEMSUnits::mm << " mm" << GGendl;
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "Energy cuts:" << GGendl;
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "    * Photon:" << GGendl;
    EnergyCutUMap const kEnergyCutsPhoton = range_cuts->GetPhotonEnergyCut();
    for (auto&& j : kEnergyCutsPhoton) {
      GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "        - " << j.first << ": " << j.second/GGEMSUnits::keV << " keV" << GGendl;
    }
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "    * Electron:" << GGendl;
    EnergyCutUMap const kEnergyCutsElectron = range_cuts->GetElectronEnergyCut();
    for (auto&& j : kEnergyCutsElectron) {
      GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "        - " << j.first << ": " << j.second/GGEMSUnits::keV << " keV" << GGendl;
    }
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "    * Positron:" << GGendl;
    EnergyCutUMap const kEnergyCutsPositron = range_cuts->GetPositronEnergyCut();
    for (auto&& j : kEnergyCutsPositron) {
      GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "        - " << j.first << ": " << j.second/GGEMSUnits::keV << " keV" << GGendl;
    }
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << GGendl;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRangeCutsManager::SetLengthCut(std::string const& phantom_name, std::string const& particle_name, GGfloat const& value, std::string const& unit)
{
  GGEMSPhantomNavigatorManager& phantom_navigator_manager = GGEMSPhantomNavigatorManager::GetInstance();

  if (phantom_name == "all") {
    // Loop over phantom
    for (size_t i = 0; i < phantom_navigator_manager.GetNumberOfPhantomNavigators(); ++i) {
      std::shared_ptr<GGEMSMaterials> materials = ((phantom_navigator_manager.GetPhantomNavigators()).at(i))->GetMaterials();
      materials->SetDistanceCut(particle_name, value, unit);
    }
  }
  else {
    std::shared_ptr<GGEMSMaterials> materials = phantom_navigator_manager.GetPhantomNavigator(phantom_name)->GetMaterials();
    materials->SetDistanceCut(particle_name, value, unit);
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
