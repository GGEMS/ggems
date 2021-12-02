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
  \file visualization.cc

  \brief Example of simulation with visualization

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday October 25, 2021
*/

#include <sstream>
#include "GGEMS/global/GGEMSOpenCLManager.hh"
#include "GGEMS/graphics/GGEMSOpenGLManager.hh"
#include "GGEMS/global/GGEMS.hh"
#include "GGEMS/materials/GGEMSMaterialsDatabaseManager.hh"
#include "GGEMS/navigators/GGEMSVoxelizedPhantom.hh"
#include "GGEMS/navigators/GGEMSCTSystem.hh"
#include "GGEMS/physics/GGEMSRangeCutsManager.hh"
#include "GGEMS/physics/GGEMSProcessesManager.hh"
#include "GGEMS/sources/GGEMSXRaySource.hh"
#include "GGEMS/geometries/GGEMSVolumeCreatorManager.hh"
#include "GGEMS/geometries/GGEMSBox.hh"

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
  oss << "-->> 6 - Visualization Example <<--\n" << std::endl;
  oss << "Usage: " << exec << " [OPTIONS...]\n" << std::endl;
  oss << "[--help]                   Print the help to the terminal" << std::endl;
  oss << "[--verbose X]              Verbosity level" << std::endl;
  oss << "                           (X=0, default)" << std::endl;
  oss << std::endl;
  oss << "OpenGL params:" << std::endl;
  oss << "--------------" << std::endl;
  oss << "[--wdims X,Y]              Window dimensions" << std::endl;
  oss << "                           (800,800, by default)" << std::endl;
  oss << "[--msaa X]                 MSAA factor (1x, 2x, 4x or 8x)" << std::endl;
  oss << "                           (8, by default)" << std::endl;
  oss << "[--axis]                   Drawing axis on screen" << std::endl;
  oss << "[--wcolor X]               Window color" << std::endl;
  oss << "                           (black, by default)" << std::endl;
  oss << "Available colors:" << std::endl;
  oss << "                               * black" << std::endl;
  oss << "                               * blue" << std::endl;
  oss << "                               * lime" << std::endl;
  oss << "                               * cyan" << std::endl;
  oss << "                               * red" << std::endl;
  oss << "                               * magenta" << std::endl;
  oss << "                               * yellow" << std::endl;
  oss << "                               * white" << std::endl;
  oss << "                               * gray" << std::endl;
  oss << "                               * silver" << std::endl;
  oss << "                               * maroon" << std::endl;
  oss << "                               * olive" << std::endl;
  oss << "                               * green" << std::endl;
  oss << "                               * purple" << std::endl;
  oss << "                               * teal" << std::endl;
  oss << "                               * navy" << std::endl;
  oss << std::endl;
  oss << "Specific hardware selection:" << std::endl;
  oss << "----------------------------" << std::endl;
  oss << "[--device X]               Device Index:" << std::endl;
  oss << "                           (0, by default)" << std::endl;
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
    GGint msaa = 8;
    GGint window_dims[] = {800, 800};
    std::string window_color = "black";
    GGsize number_of_particles = 1000000;
    GGsize device_index = 0;
    GGuint seed = 777;
    static GGint is_axis = 0;

    // Loop while there is an argument
    GGint counter(0);
    while (1) {
      // Declaring a structure of the options
      GGint option_index = 0;
      static struct option sLongOptions[] = {
        {"verbose", required_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {"msaa", required_argument, 0, 'm'},
        {"wdims", required_argument, 0, 'w'},
        {"wcolor", required_argument, 0, 'c'},
        {"device", required_argument, 0, 'd'},
        {"seed", required_argument, 0, 's'},
        {"n-particles", required_argument, 0, 'n'},
        {"axis", no_argument, &is_axis, 1},
      };

      // Getting the options
      counter = getopt_long(argc, argv, "hv:m:w:c:d:s:n:", sLongOptions, &option_index);

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
          ParseCommandLine(optarg, &msaa);
          break;
        }
        case 'w': {
          ParseCommandLine(optarg, &window_dims[0]);
          break;
        }
        case 'c': {
          window_color = optarg;
          break;
        }
        case 'd': {
          ParseCommandLine(optarg, &device_index);
          break;
        }
        case 's': {
          ParseCommandLine(optarg, &seed);
          break;
        }
        case 'n': {
          ParseCommandLine(optarg, &number_of_particles);
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
    GGEMSOpenGLManager& opengl_manager = GGEMSOpenGLManager::GetInstance();
    GGEMSMaterialsDatabaseManager& material_manager = GGEMSMaterialsDatabaseManager::GetInstance();
    GGEMSVolumeCreatorManager& volume_creator_manager = GGEMSVolumeCreatorManager::GetInstance();
    GGEMSProcessesManager& processes_manager = GGEMSProcessesManager::GetInstance();
    GGEMSRangeCutsManager& range_cuts_manager = GGEMSRangeCutsManager::GetInstance();

    // Visualization params
    opengl_manager.SetMSAA(msaa);
    if (is_axis) opengl_manager.SetDrawAxis(true);
    opengl_manager.SetWindowDimensions(window_dims[0], window_dims[1]);
    opengl_manager.SetBackgroundColor(window_color);
    opengl_manager.SetImageOutput("data/axis");
    opengl_manager.SetWorldSize(2.0f*m,2.0f*m,2.0f*m);
    opengl_manager.Initialize();
    // XXX.SetParticleColor("gamma","red") ...
    // XXX.SetParticles(10000) // Max: 1000000!!!

    // OpenCL params
    opencl_manager.DeviceToActivate(device_index);

    // Enter material database
    material_manager.SetMaterialsDatabase("data/materials.txt");

    // Initializing a global voxelized volume
    volume_creator_manager.SetVolumeDimensions(120, 120, 120);
    volume_creator_manager.SetElementSizes(0.1f, 0.1f, 0.1f, "mm");
    volume_creator_manager.SetOutputImageFilename("data/phantom.mhd");
    volume_creator_manager.SetRangeToMaterialDataFilename("data/range_phantom.txt");
    volume_creator_manager.SetMaterial("Air");
    volume_creator_manager.SetDataType("MET_INT");
    volume_creator_manager.Initialize();

    // Creating a box
    GGEMSBox* box_phantom = new GGEMSBox(10.0f, 10.0f, 10.0f, "mm");
    box_phantom->SetPosition(0.0f, 0.0f, 0.0f, "mm");
    box_phantom->SetLabelValue(1);
    box_phantom->SetMaterial("Water");
    box_phantom->Initialize();
    box_phantom->Draw();
    delete box_phantom;

    // Writing volume
    volume_creator_manager.Write();

    // Phantoms and systems
    GGEMSVoxelizedPhantom phantom("phantom");
    phantom.SetPhantomFile("data/phantom.mhd", "data/range_phantom.txt");
    phantom.SetRotation(0.0f, 0.0f, 0.0f, "deg");
    phantom.SetPosition(0.0f, 0.0f, 0.0f, "mm");
    phantom.SetVisible(true);
    // phantom.SetMaterialColor("Water", false);
    // phantom.SetMaterialColor("Water", 0.34, 0.4, 0.2, true);

    GGEMSCTSystem ct_detector("Stellar");
    ct_detector.SetCTSystemType("curved");
    ct_detector.SetNumberOfModules(1, 46);
    ct_detector.SetNumberOfDetectionElementsInsideModule(64, 16, 1);
    ct_detector.SetSizeOfDetectionElements(0.6f, 0.6f, 0.6f, "mm");
    ct_detector.SetMaterialName("GOS");
    ct_detector.SetSourceDetectorDistance(1085.6f, "mm");
    ct_detector.SetSourceIsocenterDistance(595.0f, "mm");
    ct_detector.SetRotation(0.0f, 0.0f, 0.0f, "deg");
    ct_detector.SetThreshold(10.0f, "keV");
    ct_detector.StoreOutput("data/projection");
    ct_detector.StoreScatter(true);
    ct_detector.SetVisible(true);
    //ct_detector.SetColor("red");

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
    point_source.SetBeamAperture(12.5f, "deg");
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

    opengl_manager.Display();

    // Start GGEMS simulation
    // ggems.Run();
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
}
