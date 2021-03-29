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
  \file cross_sections.cc

  \brief Example of cross sections computation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Friday November 6, 2020
*/

#include <cstdlib>

#include "GGEMS/global/GGEMSOpenCLManager.hh"

#include "GGEMS/materials/GGEMSMaterialsDatabaseManager.hh"
#include "GGEMS/materials/GGEMSMaterials.hh"

// #include "GGEMS/physics/GGEMSProcessesManager.hh"
#include "GGEMS/physics/GGEMSCrossSections.hh"

/*!
  \fn void PrintHelpAndQuit(void)
  \brief Print help to terminal and quit
*/
void PrintHelpAndQuit(void)
{
  std::cerr << "Usage: cross_sections <DeviceID> <Material> <Process> <Energy>" << std::endl;
  std::cerr << std::endl;
  std::cerr << "<DeviceID>: OpenCL device id" << std::endl;
  std::cerr << "<Material> : Material defined in data/materials. Example: Water" << std::endl;
  std::cerr << "<Process>  : Process available:" << std::endl;
  std::cerr << "                 * Compton" << std::endl;
  std::cerr << "                 * Photoelectric" << std::endl;
  std::cerr << "                 * Rayleigh" << std::endl;
  std::cerr << "<Energy>   : Energy of particle in MeV" << std::endl;
  exit(EXIT_FAILURE);
}

/*!
  \fn int main(int argc, char** argv)
  \param argc - number of arguments
  \param argv - list of arguments
  \return status of program
  \brief main function of program
*/
int main(int argc, char** argv)
{
  // Checking parameters
  if (argc != 5) {
    std::cerr << "Invalid number of arguments!!!" << std::endl;
    PrintHelpAndQuit();
  }

  // Getting parameters
  GGsize device_id = static_cast<GGsize>(atoi(argv[1]));
  std::string material_name = argv[2];
  std::string process_name = argv[3];
  GGfloat energy_MeV = strtof(argv[4], NULL);

  // Setting verbosity
  GGcout.SetVerbosity(0);
  GGcerr.SetVerbosity(0);
  GGwarn.SetVerbosity(0);

  // Initialization of singletons
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  GGEMSMaterialsDatabaseManager& material_manager = GGEMSMaterialsDatabaseManager::GetInstance();
  // GGEMSProcessesManager& processes_manager = GGEMSProcessesManager::GetInstance();

  try {
    // Set the context id
    opencl_manager.DeviceToActivate(device_id);

    // Enter material database
    material_manager.SetMaterialsDatabase("../../data/materials.txt");

    // Initializing material
    GGEMSMaterials materials;
    // materials.AddMaterial(material_name);
    // materials.Initialize();

    // Printing useful infos
    // std::cout << "Material: " << material_name << std::endl;
    // std::cout << "    Density: " << materials.GetDensity(material_name) << " g.cm-3" << std::endl;
    // std::cout << "    Photon energy cut (for 1 mm distance): " << materials.GetEnergyCut(material_name, "gamma", 1.0, "mm") << " keV" << std::endl;
    // std::cout << "    Electron energy cut (for 1 mm distance): " << materials.GetEnergyCut(material_name, "e-", 1.0, "mm") << " keV" << std::endl;
    // std::cout << "    Positron energy cut (for 1 mm distance): " << materials.GetEnergyCut(material_name, "e+", 1.0, "mm")<< " keV" << std::endl;
    // std::cout << "    Atomic number density: " << materials.GetAtomicNumberDensity(material_name) << " atoms.cm-3" << std::endl;

    // // Defining global parameters for cross-section building
    // processes_manager.SetCrossSectionTableNumberOfBins(220); // Not exceed 2048 bins
    // processes_manager.SetCrossSectionTableMinimumEnergy(1.0f, "keV");
    // processes_manager.SetCrossSectionTableMaximumEnergy(10.0f, "MeV");

    // // Add physical process and initialize it
    // GGEMSCrossSections cross_sections;
    // cross_sections.AddProcess(process_name, "gamma");
    // cross_sections.Initialize(&materials);

    // std::cout << "At " << energy_MeV << " MeV, cross section is " << cross_sections.GetPhotonCrossSection(process_name, material_name, energy_MeV, "MeV") << " cm2.g-1" << std::endl;
  }
  catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
  catch (...) {
    std::cerr << "Unknown exception!!!" << std::endl;
  }

  opencl_manager.Clean();
  exit(EXIT_SUCCESS);
}
