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
#include <thread>
#include <mutex>

#include "GGEMS/global/GGEMSOpenCLManager.hh"
#include "GGEMS/global/GGEMSConfiguration.hh"
#include "GGEMS/tools/GGEMSRAMManager.hh"
#include "GGEMS/sources/GGEMSSourceManager.hh"
#include "GGEMS/sources/GGEMSXRaySource.hh"
#include "GGEMS/tools/GGEMSProfilerManager.hh"

namespace {
  std::mutex mutex;
}

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

void Run(GGsize const& thread_index)
{
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();

  // Loop over sources
  for (GGsize i = 0; i < source_manager.GetNumberOfSources(); ++i) {
    GGsize device_index = opencl_manager.GetIndexOfActivatedDevice(thread_index);

    mutex.lock();
    GGcout("", "Run", 0) << "## Source " << source_manager.GetNameOfSource(i) << " on " << opencl_manager.GetDeviceName(device_index) << GGendl;
    mutex.unlock();

    // Loop over batch
    GGsize number_of_batchs = source_manager.GetNumberOfBatchs(i, thread_index);
    for (GGsize j = 0; j < number_of_batchs; ++j) {
      GGsize number_of_particles = source_manager.GetNumberOfParticlesInBatch(i, thread_index, j);

      mutex.lock();
      GGcout("", "Run", 0) << "----> Launching batch " << j+1 << "/" << number_of_batchs << GGendl;
      GGcout("", "Run", 0) << "      + Generating " << number_of_particles << " particles..." << GGendl;
      mutex.unlock();

      // Generating particles
      source_manager.GetPrimaries(i, thread_index, number_of_particles);
    }
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
  // Checking parameters
  if (argc != 2) {
    std::cerr << "Invalid number of arguments!!!" << std::endl;
    PrintHelpAndQuit();
  }

  // Getting parameters
  std::string device = argv[1];

  // Setting verbosity
  GGcout.SetVerbosity(3);
  GGcerr.SetVerbosity(3);
  GGwarn.SetVerbosity(3);

  // Initialization of singletons
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  GGEMSRAMManager& ram_manager = GGEMSRAMManager::GetInstance();
  GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();
  GGEMSProfilerManager& profiler_manager = GGEMSProfilerManager::GetInstance();

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

    GGEMSXRaySource point_source("point_source");
    point_source.SetSourceParticleType("gamma");
    point_source.SetNumberOfParticles(10000000);
    point_source.SetPosition(-595.0f, 0.0f, 0.0f, "mm");
    point_source.SetRotation(0.0f, 0.0f, 0.0f, "deg");
    point_source.SetBeamAperture(12.5f, "deg");
    point_source.SetFocalSpotSize(0.0f, 0.0f, 0.0f, "mm");
    point_source.SetMonoenergy(60.0f, "keV");

    GGEMSXRaySource point_source2("point_source2");
    point_source2.SetSourceParticleType("gamma");
    point_source2.SetNumberOfParticles(25000000);
    point_source2.SetPosition(-595.0f, 0.0f, 0.0f, "mm");
    point_source2.SetRotation(0.0f, 0.0f, 90.0f, "deg");
    point_source2.SetBeamAperture(12.5f, "deg");
    point_source2.SetFocalSpotSize(0.0f, 0.0f, 0.0f, "mm");
    point_source2.SetPolyenergy("spectrum_120kVp_2mmAl.dat");

    source_manager.Initialize(777);
    source_manager.PrintInfos();

    ram_manager.PrintRAMStatus();

    GGsize number_of_activated_devices = opencl_manager.GetNumberOfActivatedDevice();
    std::thread* thread_device = new std::thread[number_of_activated_devices];

    for (GGsize i = 0; i < number_of_activated_devices; ++i) {
      thread_device[i] = std::thread(Run, i);
    }

    for (GGsize i = 0; i < number_of_activated_devices; ++i) thread_device[i].join();

    delete[] thread_device;

    profiler_manager.PrintSummaryProfile();
  }
  catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
  catch (...) {
    std::cerr << "Unknown exception!!!" << std::endl;
  }

  // Cleaning OpenCL manager
  source_manager.Clean();
  opencl_manager.Clean();
  exit(EXIT_SUCCESS);
}
