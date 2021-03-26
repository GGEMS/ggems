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
  \file multi_platform.cc

  \brief Example of multi platform application in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Friday March 26, 2021
*/

#include <cstdlib>
#include "GGEMS/global/GGEMSOpenCLManager.hh"

/*!
  \fn void PrintHelpAndQuit(void)
  \brief Print help to terminal and quit
*/
void PrintHelpAndQuit(void)
{
  std::cerr << "Usage: multi_platform <Device>" << std::endl;
  std::cerr << std::endl;
  std::cerr << "<Device>: \"all\", \"cpu\", \"gpu\", \"gpu_nvidia\", \"gpu_amd\", \"gpu_intel\"" << std::endl;
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
  if (argc != 2) {
    std::cerr << "Invalid number of arguments!!!" << std::endl;
    PrintHelpAndQuit();
  }

  // Getting parameters
  std::string device = argv[1];

  // Setting verbosity
  GGcout.SetVerbosity(1);
  GGcerr.SetVerbosity(1);
  GGwarn.SetVerbosity(1);

  // Initialization of singletons
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  try {
    // Print infos about platform and device
    opencl_manager.PrintPlatformInfos();
    opencl_manager.PrintDeviceInfos();

    // Activating device
    if (device == "gpu_nvidia") opencl_manager.DeviceToActivate("gpu", "nvidia");
    else if (device == "gpu_amd") opencl_manager.DeviceToActivate("gpu", "amd");
    else if (device == "gpu_intel") opencl_manager.DeviceToActivate("gpu", "intel");
    else opencl_manager.DeviceToActivate(device);

    opencl_manager.PrintActivatedDevices();
  }
  catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
  catch (...) {
    std::cerr << "Unknown exception!!!" << std::endl;
  }

  // opencl_manager.Clean();
  exit(EXIT_SUCCESS);
}
