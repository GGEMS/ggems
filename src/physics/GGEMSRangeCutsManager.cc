/*!
  \file GGEMSRangeCutsManager.cc

  \brief GGEMS class managing the range cuts in GGEMS simulation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Friday March 6, 2020
*/

#include "GGEMS/navigators/GGEMSPhantomNavigatorManager.hh"

#include "GGEMS/navigators/GGEMSPhantomNavigator.hh"
#include "GGEMS/physics/GGEMSRangeCutsManager.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
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

  GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "Photon cuts:" << GGendl;
  for (auto&& i : photon_cuts_) {
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "    * " << i.first << ": " << i.second/GGEMSUnits::mm << " mm" << GGendl;
  }

  GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "Electron cuts:" << GGendl;
  for (auto&& i : electron_cuts_) {
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "    * " << i.first << ": " << i.second/GGEMSUnits::mm << " mm" << GGendl;
  }

  GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "Positron cuts:" << GGendl;
  for (auto&& i : positron_cuts_) {
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "    * " << i.first << ": " << i.second/GGEMSUnits::mm << " mm" << GGendl;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRangeCutsManager::SetRangeCut(char const* phantom_name, char const* particle_name, GGfloat const& value, char const* unit)
{
  std::string const kParticleName(particle_name);
  std::string const kPhantomName(phantom_name);

  // If "all", get the name of phantom and set the cut of particle to all the same phantom
  if (!kPhantomName.compare("all")) {
    GGEMSPhantomNavigatorManager& phantom_navigators = GGEMSPhantomNavigatorManager::GetInstance();
    for (size_t i = 0; i < phantom_navigators.GetNumberOfPhantomNavigators(); ++i) {
      std::string const kName = ((phantom_navigators.GetPhantomNavigator())[i])->GetPhantomName();
      if (!kParticleName.compare("gamma")) {
        photon_cuts_[kName] = GGEMSUnits::DistanceUnit(value, unit);
      }
      else if (!kParticleName.compare("e-")) {
        electron_cuts_[kName] = GGEMSUnits::DistanceUnit(value, unit);
      }
      else if (!kParticleName.compare("e+")) {
        positron_cuts_[kName] = GGEMSUnits::DistanceUnit(value, unit);
      }
      else {
        GGEMSMisc::ThrowException("GGEMSRangeCutsManager", "SetRangeCut", "The type of particle is wrong!!!");
      }
    }
  }
  else {
    if (!kParticleName.compare("gamma")) {
      photon_cuts_[kPhantomName] = GGEMSUnits::DistanceUnit(value, unit);
    }
    else if (!kParticleName.compare("e-")) {
      electron_cuts_[kPhantomName] = GGEMSUnits::DistanceUnit(value, unit);
    }
    else if (!kParticleName.compare("e+")) {
      positron_cuts_[kPhantomName] = GGEMSUnits::DistanceUnit(value, unit);
    }
    else {
      GGEMSMisc::ThrowException("GGEMSRangeCutsManager", "SetRangeCut", "The type of particle is wrong!!!");
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRangeCutsManager::CheckRangeCuts(void)
{
  GGcout("GGEMSRangeCutsManager", "CheckRangeCuts", 0) << "Checking all the cuts..." << GGendl;
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
  range_cut_manager->SetRangeCut(phantom_name, particle_name, value, unit);
}
