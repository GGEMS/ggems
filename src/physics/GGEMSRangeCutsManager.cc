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
    std::shared_ptr<GGEMSRangeCuts> range_cuts = ((phantom_navigator_manager.GetPhantomNavigators()).at(i))->GetRangeCuts();

    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "Range cuts for phantom navigator: " << kPhantomName << GGendl;
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "---------------------------------" << GGendl;
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "Length cuts:" << GGendl;
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "    * Photon: " << range_cuts->GetPhotonLengthCut()/GGEMSUnits::mm << " mm"<< GGendl;
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "    * Electron: " << range_cuts->GetElectronLengthCut()/GGEMSUnits::mm << " mm" << GGendl;
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "Energy cuts:" << GGendl;
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "    * Photon:" << GGendl;
    // List of materials
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << "    * Electron:" << GGendl;
    // List of materials

    // Printing energy cut by materials
    GGcout("GGEMSRangeCutsManager", "PrintInfos", 0) << GGendl;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRangeCutsManager::SetLengthCut(char const* phantom_name, char const* particle_name, GGfloat const& value, char const* unit)
{
  std::string const kParticleName(particle_name);
  std::string const kPhantomCase(phantom_name);

  GGEMSPhantomNavigatorManager& phantom_navigator_manager = GGEMSPhantomNavigatorManager::GetInstance();

  // Check the particle name  
  if (!kParticleName.compare("gamma")) {
    // Check if all phantoms in same time, or specific phantom
    if (!kPhantomCase.compare("all")) {
      for (size_t i = 0; i < phantom_navigator_manager.GetNumberOfPhantomNavigators(); ++i) {
        std::shared_ptr<GGEMSRangeCuts> range_cuts = ((phantom_navigator_manager.GetPhantomNavigators()).at(i))->GetRangeCuts();
        range_cuts->SetPhotonLengthCut(GGEMSUnits::DistanceUnit(value, unit));
      }
    }
    else {
      std::shared_ptr<GGEMSRangeCuts> range_cuts = phantom_navigator_manager.GetPhantomNavigator(kPhantomCase)->GetRangeCuts();
      range_cuts->SetPhotonLengthCut(GGEMSUnits::DistanceUnit(value, unit));
    }
  }
  else if (!kParticleName.compare("e-")) {
    // Check if all phantoms in same time, or specific phantom
    if (!kPhantomCase.compare("all")) {
      for (size_t i = 0; i < phantom_navigator_manager.GetNumberOfPhantomNavigators(); ++i) {
        std::shared_ptr<GGEMSRangeCuts> range_cuts = ((phantom_navigator_manager.GetPhantomNavigators()).at(i))->GetRangeCuts();
        range_cuts->SetElectronLengthCut(GGEMSUnits::DistanceUnit(value, unit));
      }
    }
    else {
      std::shared_ptr<GGEMSRangeCuts> range_cuts = phantom_navigator_manager.GetPhantomNavigator(kPhantomCase)->GetRangeCuts();
      range_cuts->SetElectronLengthCut(GGEMSUnits::DistanceUnit(value, unit));
    }
  }
  else {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Particle name " << kParticleName << " unknown!!! The particles are:" << std::endl;
    oss << "    - gamma" << std::endl;
    oss << "    - e-";
    GGEMSMisc::ThrowException("GGEMSRangeCutsManager", "SetLengthCut", oss.str());
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
