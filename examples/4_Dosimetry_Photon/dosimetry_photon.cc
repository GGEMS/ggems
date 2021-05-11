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
  \file generate_volume.cc

  \brief Example of volume creation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday November 2, 2020
*/

#include <cstdlib>

#include "GGEMS/global/GGEMSOpenCLManager.hh"
#include "GGEMS/materials/GGEMSMaterialsDatabaseManager.hh"
#include "GGEMS/navigators/GGEMSVoxelizedPhantom.hh"
#include "GGEMS/navigators/GGEMSDosimetryCalculator.hh"
#include "GGEMS/physics/GGEMSProcessesManager.hh"
#include "GGEMS/physics/GGEMSRangeCutsManager.hh"
#include "GGEMS/sources/GGEMSXRaySource.hh"
#include "GGEMS/global/GGEMS.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/geometries/GGEMSVolumeCreatorManager.hh"
#include "GGEMS/geometries/GGEMSTube.hh"

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
  oss << "-->> 4 - Dosimetry Example <<--\n" << std::endl;
  oss << "Usage: " << exec << " [OPTIONS...]\n" << std::endl;
  oss << "[--help]                   Print the help to the terminal" << std::endl;
  oss << "[--verbose X]              Verbosity level" << std::endl;
  oss << "                           (X=0, default)" << std::endl;
  oss << std::endl;
  oss << "Specific hardware selection:" << std::endl;
  oss << "----------------------------" << std::endl;
  oss << "[--device X]               Device type:" << std::endl;
  oss << "                           (X=all, by default)" << std::endl;
  oss << "                               - all (all devices)" << std::endl;
  oss << "                               - cpu (cpu device)" << std::endl;
  oss << "                               - gpu (all gpu devices)" << std::endl;
  oss << "                               - gpu_nvidia (all gpu nvidia devices)" << std::endl;
  oss << "                               - gpu_intel (all gpu intel devices)" << std::endl;
  oss << "                               - gpu_amd (all gpu amd devices)" << std::endl;
  oss << "                               - X;Y;Z ... (index of device)" << std::endl;
  oss << "[--balance X]              Balance computation for device if many devices are selected." << std::endl;
  oss << "                           If 2 devices selected:" << std::endl;
  oss << "                               --balance 0.5;0.5 means 50% of computation on device 0, and 50% of computation on device 1" << std::endl;
  oss << "                               --balance 0.32;0.68 means 32% of computation on device 0, and 68% of computation on device 1" << std::endl;
  oss << "                           Total balance has to be equal to 1" << std::endl;
  oss << std::endl;
  oss << "Simulation parameters:" << std::endl;
  oss << "----------------------" << std::endl;
  oss << "[--n-particles X]         Number of particles" << std::endl;
  oss << "                          (X=1000000, default)" << std::endl;
  oss << "[--seed X]                Seed of pseudo generator number" << std::endl;
  oss << "                          (X=777, default)" << std::endl;
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
    GGsize number_of_particles = 1000000;
    std::string device = "all";
    std::string device_balance = "";
    GGuint seed = 777;

    // Loop while there is an argument
    GGint counter(0);
    while (1) {
      // Declaring a structure of the options
      GGint option_index = 0;
      static struct option sLongOptions[] = {
        {"verbose", required_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {"n-particles", required_argument, 0, 'p'},
        {"device", required_argument, 0, 'd'},
        {"balance", required_argument, 0, 'b'},
        {"seed", required_argument, 0, 's'}
      };

      // Getting the options
      counter = getopt_long(argc, argv, "hv:p:d:b:s:", sLongOptions, &option_index);

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
        case 'p': {
          ParseCommandLine(optarg, &number_of_particles);
          break;
        }
        case 'd': {
          device = optarg;
          break;
        }
        case 'b': {
          device_balance = optarg;
          break;
        }
        case 's': {
          ParseCommandLine(optarg, &seed);
          break;
        }
        default: {
          PrintHelpAndQuit("Out of switch options!!!", argv[0]);
          break;
        }
      }
    }

    // Setting verbosity
    GGcout.SetVerbosity(verbosity_level);
    GGcerr.SetVerbosity(verbosity_level);
    GGwarn.SetVerbosity(verbosity_level);

    // Initialization of singletons
    GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
    GGEMSMaterialsDatabaseManager& material_manager = GGEMSMaterialsDatabaseManager::GetInstance();
    GGEMSVolumeCreatorManager& volume_creator_manager = GGEMSVolumeCreatorManager::GetInstance();
    GGEMSProcessesManager& processes_manager = GGEMSProcessesManager::GetInstance();
    GGEMSRangeCutsManager& range_cuts_manager = GGEMSRangeCutsManager::GetInstance();

    // Activating device
    if (device == "gpu_nvidia") opencl_manager.DeviceToActivate("gpu", "nvidia");
    else if (device == "gpu_amd") opencl_manager.DeviceToActivate("gpu", "amd");
    else if (device == "gpu_intel") opencl_manager.DeviceToActivate("gpu", "intel");
    else opencl_manager.DeviceToActivate(device);

    // Device balancing
    if (!device_balance.empty()) opencl_manager.DeviceBalancing(device_balance);

    // Enter material database
    material_manager.SetMaterialsDatabase("data/materials.txt");

    // Initializing a global voxelized volume
    volume_creator_manager.SetVolumeDimensions(240, 240, 640);
    volume_creator_manager.SetElementSizes(0.5f, 0.5f, 0.5f, "mm");
    volume_creator_manager.SetOutputImageFilename("data/phantom.mhd");
    volume_creator_manager.SetRangeToMaterialDataFilename("data/range_phantom.txt");
    volume_creator_manager.SetMaterial("Air");
    volume_creator_manager.SetDataType("MET_INT");
    volume_creator_manager.Initialize();

    // Creating a box
    GGEMSTube* tube_phantom = new GGEMSTube(50.0f, 50.0f, 300.0f, "mm");
    tube_phantom->SetPosition(0.0f, 0.0f, 0.0f, "mm");
    tube_phantom->SetLabelValue(1);
    tube_phantom->SetMaterial("Water");
    tube_phantom->Initialize();
    tube_phantom->Draw();
    delete tube_phantom;

    // Writing volume
    volume_creator_manager.Write();

    // Phantoms and systems
    GGEMSVoxelizedPhantom phantom("phantom");
    phantom.SetPhantomFile("data/phantom.mhd", "data/range_phantom.txt");
    phantom.SetRotation(0.0f, 0.0f, 0.0f, "deg");
    phantom.SetPosition(0.0f, 0.0f, 0.0f, "mm");

    // Dosimetry
    GGEMSDosimetryCalculator dosimetry;
    dosimetry.AttachToNavigator("phantom");
    dosimetry.SetOutputDosimetryBasename("data/dosimetry");
    dosimetry.SetDoselSizes(0.5f, 0.5f, 0.5f);
    dosimetry.SetWaterReference(false);
    dosimetry.SetMinimumDensity(0.1f, "g/cm3");

    dosimetry.SetUncertainty(true);
    dosimetry.SetPhotonTracking(true);
    dosimetry.SetEdep(true);
    dosimetry.SetEdepSquared(true);
    dosimetry.SetHitTracking(true);

    // Physics
    processes_manager.AddProcess("Compton", "gamma", "all");
    processes_manager.AddProcess("Photoelectric", "gamma", "all");
    processes_manager.AddProcess("Rayleigh", "gamma", "all");

    // Optional options, the following are by default
    processes_manager.SetCrossSectionTableNumberOfBins(220);
    processes_manager.SetCrossSectionTableMinimumEnergy(1.0f, "keV");
    processes_manager.SetCrossSectionTableMaximumEnergy(1.0f, "MeV");

    // Cuts, by default but are 1 um
    range_cuts_manager.SetLengthCut("all", "gamma", 0.1f, "mm");

    // Source
    GGEMSXRaySource point_source("point_source");
    point_source.SetSourceParticleType("gamma");
    point_source.SetNumberOfParticles(number_of_particles);
    point_source.SetPosition(-595.0f, 0.0f, 0.0f, "mm");
    point_source.SetRotation(0.0f, 0.0f, 0.0f, "deg");
    point_source.SetBeamAperture(5.0f, "deg");
    point_source.SetFocalSpotSize(0.0f, 0.0f, 0.0f, "mm");
    point_source.SetPolyenergy("data/spectrum_120kVp_2mmAl.dat");

    // GGEMS simulation
    GGEMS ggems;
    ggems.SetOpenCLVerbose(true);
    ggems.SetMaterialDatabaseVerbose(false);
    ggems.SetNavigatorVerbose(false);
    ggems.SetSourceVerbose(true);
    ggems.SetMemoryRAMVerbose(true);
    ggems.SetProcessVerbose(true);
    ggems.SetRangeCutsVerbose(true);
    ggems.SetRandomVerbose(true);
    ggems.SetProfilingVerbose(true);
    ggems.SetTrackingVerbose(false, 0);

    // Initializing the GGEMS simulation
    ggems.Initialize(seed);

    // Start GGEMS simulation
    ggems.Run();
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
