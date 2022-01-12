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
#include <sstream>

#include "GGEMS/global/GGEMSOpenCLManager.hh"

#include "GGEMS/geometries/GGEMSVolumeCreatorManager.hh"
#include "GGEMS/geometries/GGEMSBox.hh"
#include "GGEMS/geometries/GGEMSTube.hh"
#include "GGEMS/geometries/GGEMSSphere.hh"
#include "GGEMS/tools/GGEMSRAMManager.hh"
#include "GGEMS/tools/GGEMSProfilerManager.hh"
#include "GGEMS/tools/GGEMSPrint.hh"

#ifdef _WIN32
#include "GGEMS/tools/GGEMSWinGetOpt.hh"
#else
#include <getopt.h>
#endif

namespace {
  /*!
    \fn void PrintHelpAndQuit(std::string const& message, char const *exec)
    \param message - error message
    \param exec - name of the executable
    \brief print the help or the error of the program
  */
  [[noreturn]] void PrintHelpAndQuit(std::string const& message, char const* exec)
  {
    std::ostringstream oss(std::ostringstream::out);
    oss << message << std::endl;
    oss << std::endl;
    oss << "-->> 3 - Generate Volume Example <<--\n" << std::endl;
    oss << "Usage: " << exec << " [OPTIONS...]\n" << std::endl;
    oss << "[--help]                   Print the help to the terminal" << std::endl;
    oss << "[--verbose X]              Verbosity level" << std::endl;
    oss << "                           (X=0, default)" << std::endl;
    oss << std::endl;
    oss << "Specific hardware selection:" << std::endl;
    oss << "----------------------------" << std::endl;
    oss << "[--device X]               Index of device type" << std::endl;
    oss << "                           (X=0, by default)" << std::endl;
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

    // Loop while there is an argument
    GGint counter(0);
    while (1) {
      // Declaring a structure of the options
      GGint option_index = 0;
      static struct option sLongOptions[] = {
        {"verbose", required_argument, nullptr, 'v'},
        {"help", no_argument, nullptr, 'h'},
        {"device", required_argument, nullptr, 'd'}
      };

      // Getting the options
      counter = getopt_long(argc, argv, "hv:d:", sLongOptions, &option_index);

      // Exit the loop if -1
      if (counter == -1) break;

      // Analyzing each option
      switch (counter) {
        case 0: {
          // If this option set a flag, do nothing else now
          if (sLongOptions[option_index].flag != nullptr) break;
          break;
        }
        case 'v': {
          ParseCommandLine(optarg, &verbosity_level);
          break;
        }
        case 'h': {
          PrintHelpAndQuit("Printing the help", argv[0]);
        }
        case 'd': {
          ParseCommandLine(optarg, &device_id);
          break;
        }
        default: {
          PrintHelpAndQuit("Out of switch options!!!", argv[0]);
        }
      }
    }

    // Setting verbosity
    GGcout.SetVerbosity(verbosity_level);
    GGcerr.SetVerbosity(verbosity_level);
    GGwarn.SetVerbosity(verbosity_level);

    // Initialization of singletons
    GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
    GGEMSVolumeCreatorManager& volume_creator_manager = GGEMSVolumeCreatorManager::GetInstance();
    GGEMSProfilerManager& profiler_manager = GGEMSProfilerManager::GetInstance();
    GGEMSRAMManager& ram_manager = GGEMSRAMManager::GetInstance();

    // Set the context id
    opencl_manager.DeviceToActivate(device_id);

    // Initializing a global voxelized volume
    volume_creator_manager.SetVolumeDimensions(450, 450, 450);
    volume_creator_manager.SetElementSizes(0.5f, 0.5f, 0.5f, "mm");
    volume_creator_manager.SetOutputImageFilename("data/volume.mhd");
    volume_creator_manager.SetRangeToMaterialDataFilename("data/range_volume.txt");
    volume_creator_manager.SetMaterial("Air");
    volume_creator_manager.SetDataType("MET_INT");
    volume_creator_manager.Initialize();

    // Creating a box
    GGEMSBox* box = new GGEMSBox(24.0f, 36.0f, 56.0f, "mm");
    box->SetPosition(-70.0f, -30.0f, 10.0f, "mm");
    box->SetLabelValue(1);
    box->SetMaterial("Water");
    box->Initialize();
    box->Draw();
    delete box;

    // Creating a tube
    GGEMSTube* tube = new GGEMSTube(13.0f, 8.0f, 50.0f, "mm");
    tube->SetPosition(20.0f, 10.0f, -2.0f, "mm");
    tube->SetLabelValue(2);
    tube->SetMaterial("Calcium");
    tube->Initialize();
    tube->Draw();
    delete tube;

    // Creating a sphere
    GGEMSSphere* sphere = new GGEMSSphere(14.0f, "mm");
    sphere->SetPosition(30.0f, -30.0f, 8.0f, "mm");
    sphere->SetLabelValue(3);
    sphere->SetMaterial("Lung");
    sphere->Initialize();
    sphere->Draw();
    delete sphere;

    // Printing RAM status
    ram_manager.PrintRAMStatus();

    // Writing volume
    volume_creator_manager.Write();

    // Printing profiler summary
    profiler_manager.PrintSummaryProfile();
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
