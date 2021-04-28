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
  \file ct_scanner.cc

  \brief Example of ct/cbct scanner simulation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday December 14, 2020
*/

#include <cstdlib>
#include "GGEMS/global/GGEMSOpenCLManager.hh"
#include "GGEMS/global/GGEMS.hh"
#include "GGEMS/materials/GGEMSMaterialsDatabaseManager.hh"
#include "GGEMS/navigators/GGEMSVoxelizedPhantom.hh"
#include "GGEMS/navigators/GGEMSSystem.hh"
#include "GGEMS/navigators/GGEMSCTSystem.hh"
#include "GGEMS/physics/GGEMSRangeCutsManager.hh"
#include "GGEMS/physics/GGEMSProcessesManager.hh"
#include "GGEMS/sources/GGEMSXRaySource.hh"
#include "GGEMS/geometries/GGEMSVolumeCreatorManager.hh"
#include "GGEMS/geometries/GGEMSBox.hh"

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
  GGcout.SetVerbosity(2);
  GGcerr.SetVerbosity(2);
  GGwarn.SetVerbosity(2);

  // Initialization of singletons
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  GGEMSMaterialsDatabaseManager& material_manager = GGEMSMaterialsDatabaseManager::GetInstance();
  GGEMSVolumeCreatorManager& volume_creator_manager = GGEMSVolumeCreatorManager::GetInstance();
  GGEMSProcessesManager& processes_manager = GGEMSProcessesManager::GetInstance();
  GGEMSRangeCutsManager& range_cuts_manager = GGEMSRangeCutsManager::GetInstance();

  try {
    // Activating device
    if (device == "gpu_nvidia") opencl_manager.DeviceToActivate("gpu", "nvidia");
    else if (device == "gpu_amd") opencl_manager.DeviceToActivate("gpu", "amd");
    else if (device == "gpu_intel") opencl_manager.DeviceToActivate("gpu", "intel");
    else opencl_manager.DeviceToActivate(device);

    // Enter material database
    material_manager.SetMaterialsDatabase("../../data/materials.txt");

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
    ct_detector.StoreOutput("data/projection.mhd");

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
    point_source.SetNumberOfParticles(10000);
    point_source.SetPosition(-595.0f, 0.0f, 0.0f, "mm");
    point_source.SetRotation(0.0f, 0.0f, 0.0f, "deg");
    point_source.SetBeamAperture(12.5f, "deg");
    point_source.SetFocalSpotSize(0.0f, 0.0f, 0.0f, "mm");
    point_source.SetPolyenergy("data/spectrum_120kVp_2mmAl.dat");

    // GGEMS simulation
    GGEMS ggems;
    ggems.SetOpenCLVerbose(true);
    ggems.SetNavigatorVerbose(false);
    ggems.SetSourceVerbose(true);
    ggems.SetMemoryRAMVerbose(true);
    ggems.SetProcessVerbose(true);
    ggems.SetRangeCutsVerbose(true);
    ggems.SetRandomVerbose(true);
    ggems.SetProfilingVerbose(true);
    ggems.SetTrackingVerbose(false, 0);

    // Initializing the GGEMS simulation
    ggems.Initialize(777);

    // Start GGEMS simulation
    ggems.Run();
  }
  catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
  catch (...) {
    std::cerr << "Unknown exception!!!" << std::endl;
  }

  // Exit safely
  opencl_manager.Clean();
  exit(EXIT_SUCCESS);
}
