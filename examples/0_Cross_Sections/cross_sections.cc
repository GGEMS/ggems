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
#include "GGEMS/physics/GGEMSProcessesManager.hh"
#include "GGEMS/physics/GGEMSCrossSections.hh"

#ifdef _WIN32
#include "GGEMS/tools/GGEMSWinGetOpt.hh"
#else
#include <getopt.h>
#endif

/*!
  \fn void PrintHelpAndQuit(std::string const& message, char const *p_executable)
  \param message - error message
  \param p_executable - name of the executable
  \brief print the help or the error of the program
*/
void PrintHelpAndQuit(std::string const& message, char const* exec)
{
  std::ostringstream oss(std::ostringstream::out);
  oss << message << std::endl;
  oss << std::endl;
  oss << "-->> 0 - Cross Sections Example <<--\n" << std::endl;
  oss << "Usage: " << exec << " [OPTIONS...]\n" << std::endl;
  oss << "[--help]                   Print the help to the terminal" << std::endl;
  oss << "[--verbose X]              Verbosity level" << std::endl;
  oss << "                           (X=0, default)" << std::endl;
  oss << std::endl;
  oss << "Specific hardware selection:" << std::endl;
  oss << "----------------------------" << std::endl;
  oss << "[--device X]               Index of device type" << std::endl;
  oss << "                           (X=0, by default)" << std::endl;
  oss << std::endl;
  oss << "Physic parameters:" << std::endl;
  oss << "------------------" << std::endl;
  oss << "[--material X]             Material name" << std::endl;
  oss << "[--process X]              Process name:" << std::endl;
  oss << "                               - Compton" << std::endl;
  oss << "                               - Photoelectric" << std::endl;
  oss << "                               - Rayleigh" << std::endl;
  oss << "[--energy X]               Energy in MeV" << std::endl;
  throw std::invalid_argument(oss.str());
}

/*!
  \fn void ParseCommandLine(std::string const& line_option, T* p_buffer)
  \tparam T - type of the array storing the option
  \param line_option - string from the command line
  \param p_buffer - buffer storing the commands
  \brief parse the command with comma
*/
template<typename T>
void ParseCommandLine(std::string const& line_option, T* p_buffer)
{
  std::istringstream iss(line_option);
  T* p = &p_buffer[0];
  while (iss >> *p++) if (iss.peek() == ',') iss.ignore();
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
  try {
    // Verbosity level
    GGint verbosity_level = 0;

    // List of parameters
    GGsize device_id = 0;
    std::string material_name = "";
    std::string process_name = "";
    GGfloat energy_MeV = 0.0f;

    // Loop while there is an argument
    GGint counter(0);
    while (1) {
      // Declaring a structure of the options
      GGint option_index = 0;
      static struct option sLongOptions[] = {
        {"verbose", required_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {"material", required_argument, 0, 'm'},
        {"device", required_argument, 0, 'd'},
        {"process", required_argument, 0, 'p'},
        {"energy", required_argument, 0, 'e'}
      };

      // Getting the options
      counter = getopt_long(argc, argv, "hv:m:d:p:e:", sLongOptions, &option_index);

      // Exit the loop if -1
      if (counter == -1) break;

      // Analyzing each option
      switch (counter) {
        case 0: {
          // If this option set a flag, do nothing else now
          if (sLongOptions[option_index].flag != 0) break;
          break;
        }
        case 'v': {
          ParseCommandLine(optarg, &verbosity_level);
          break;
        }
        case 'h': {
          PrintHelpAndQuit("Printing the help", argv[0]);
          break;
        }
        case 'm': {
          material_name = optarg;
          break;
        }
        case 'd': {
          ParseCommandLine(optarg, &device_id);
          break;
        }
        case 'p': {
          process_name = optarg;
          break;
        }
        case 'e': {
          ParseCommandLine(optarg, &energy_MeV);
          break;
        }
        default: {
          PrintHelpAndQuit("Out of switch options!!!", argv[0]);
          break;
        }
      }
    }

    // Checking parameters
    if (material_name.empty()) {
      PrintHelpAndQuit("Please set a material name!!!", argv[0]);
    }

    if (process_name.empty()) {
      PrintHelpAndQuit("Please set a process name!!!", argv[0]);
    }

    if (energy_MeV == 0.0f) {
      PrintHelpAndQuit("Set an energy in MeV > 0.001!!!", argv[0]);
    }

    // Setting verbosity
    GGcout.SetVerbosity(verbosity_level);
    GGcerr.SetVerbosity(verbosity_level);
    GGwarn.SetVerbosity(verbosity_level);

    // Initialization of singletons
    GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
    GGEMSMaterialsDatabaseManager& material_manager = GGEMSMaterialsDatabaseManager::GetInstance();
    GGEMSProcessesManager& processes_manager = GGEMSProcessesManager::GetInstance();

    // Set the context id
    opencl_manager.DeviceToActivate(device_id);

    // Enter material database
    material_manager.SetMaterialsDatabase("../../data/materials.txt");

    // Initializing material
    GGEMSMaterials materials;
    materials.AddMaterial(material_name);
    materials.Initialize();

    // Printing useful infos
    std::cout << "Material: " << material_name << std::endl;
    std::cout << "    Density: " << materials.GetDensity(material_name) << " g.cm-3" << std::endl;
    std::cout << "    Photon energy cut (for 1 mm distance): " << materials.GetEnergyCut(material_name, "gamma", 1.0, "mm") << " keV" << std::endl;
    std::cout << "    Electron energy cut (for 1 mm distance): " << materials.GetEnergyCut(material_name, "e-", 1.0, "mm") << " keV" << std::endl;
    std::cout << "    Positron energy cut (for 1 mm distance): " << materials.GetEnergyCut(material_name, "e+", 1.0, "mm")<< " keV" << std::endl;
    std::cout << "    Atomic number density: " << materials.GetAtomicNumberDensity(material_name) << " atoms.cm-3" << std::endl;

    // Defining global parameters for cross-section building
    processes_manager.SetCrossSectionTableNumberOfBins(220); // Not exceed 2048 bins
    processes_manager.SetCrossSectionTableMinimumEnergy(1.0f, "keV");
    processes_manager.SetCrossSectionTableMaximumEnergy(10.0f, "MeV");

    // Add physical process and initialize it
    GGEMSCrossSections cross_sections;
    cross_sections.AddProcess(process_name, "gamma");
    cross_sections.Initialize(&materials);

    std::cout << "At " << energy_MeV << " MeV, cross section is " << cross_sections.GetPhotonCrossSection(process_name, material_name, energy_MeV, "MeV") << " cm2.g-1" << std::endl;

    // Cleaning object
    materials.Clean();
    cross_sections.Clean();
  }
  catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    // Exit safely
    GGEMSOpenCLManager::GetInstance().Clean();
  }
  catch (...) {
    std::cerr << "Unknown exception!!!" << std::endl;
    // Exit safely
    GGEMSOpenCLManager::GetInstance().Clean();
  }

  // Exit safely
  GGEMSOpenCLManager::GetInstance().Clean();
  exit(EXIT_SUCCESS);
}
