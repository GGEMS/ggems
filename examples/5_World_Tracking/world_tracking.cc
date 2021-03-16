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
  \file world_tracking.cc

  \brief Example of world tracking simulation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 2, 2021
*/

#include <cstdlib>
#include "GGEMS/global/GGEMSOpenCLManager.hh"
#include "GGEMS/global/GGEMSManager.hh"
#include "GGEMS/materials/GGEMSMaterialsDatabaseManager.hh"
#include "GGEMS/navigators/GGEMSVoxelizedPhantom.hh"
#include "GGEMS/navigators/GGEMSCTSystem.hh"
#include "GGEMS/physics/GGEMSRangeCutsManager.hh"
#include "GGEMS/physics/GGEMSProcessesManager.hh"
#include "GGEMS/sources/GGEMSXRaySource.hh"
#include "GGEMS/navigators/GGEMSDosimetryCalculator.hh"
#include "GGEMS/geometries/GGEMSVolumeCreatorManager.hh"
#include "GGEMS/geometries/GGEMSTube.hh"
#include "GGEMS/navigators/GGEMSWorld.hh"

/*!
  \fn void PrintHelpAndQuit(void)
  \brief Print help to terminal and quit
*/
void PrintHelpAndQuit(void)
{
  std::cerr << "Usage: world_tracking <DeviceID>" << std::endl;
  std::cerr << std::endl;
  std::cerr << "<DeviceID>: OpenCL device id" << std::endl;
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
  GGsize device_id = static_cast<GGsize>(atoi(argv[1]));

  // Setting verbosity
  GGcout.SetVerbosity(1);
  GGcerr.SetVerbosity(1);
  GGwarn.SetVerbosity(1);

  // Initialization of singletons
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  GGEMSMaterialsDatabaseManager& material_manager = GGEMSMaterialsDatabaseManager::GetInstance();
  GGEMSVolumeCreatorManager& volume_creator_manager = GGEMSVolumeCreatorManager::GetInstance();
  GGEMSProcessesManager& processes_manager = GGEMSProcessesManager::GetInstance();
  GGEMSRangeCutsManager& range_cuts_manager = GGEMSRangeCutsManager::GetInstance();
  GGEMSManager& ggems_manager = GGEMSManager::GetInstance();

  try {
    // Set the context id
    opencl_manager.DeviceToActivate(device_id);

    // Enter material database
    material_manager.SetMaterialsDatabase("../../data/materials.txt");

    // Initializing a global voxelized volume
    volume_creator_manager.SetVolumeDimensions(240, 240, 240);
    volume_creator_manager.SetElementSizes(1.0f, 1.0f, 1.0f, "mm");
    volume_creator_manager.SetOutputImageFilename("data/phantom.mhd");
    volume_creator_manager.SetRangeToMaterialDataFilename("data/range_phantom.txt");
    volume_creator_manager.SetMaterial("Air");
    volume_creator_manager.SetDataType("MET_INT");
    volume_creator_manager.Initialize();

    // Creating a box
    GGEMSTube* tube_phantom = new GGEMSTube(50.0f, 50.0f, 200.0f, "mm");
    tube_phantom->SetPosition(0.0f, 0.0f, 0.0f, "mm");
    tube_phantom->SetLabelValue(1);
    tube_phantom->SetMaterial("Water");
    tube_phantom->Initialize();
    tube_phantom->Draw();
    delete tube_phantom;

    // Writing volume
    volume_creator_manager.Write();

    // Creating a world
    GGEMSWorld* world = new GGEMSWorld();
    world->SetDimension(400, 400, 400);
    world->SetElementSize(5.0f, 5.0f, 5.0f, "mm");
    world->SetOutputWorldBasename("data/world");
    world->SetEnergyTracking(true);
    world->SetEnergySquaredTracking(true);
    world->SetMomentum(true);
    world->SetPhotonTracking(true);

    // Phantoms and systems
    GGEMSVoxelizedPhantom phantom("phantom");
    phantom.SetPhantomFile("data/phantom.mhd", "data/range_phantom.txt");
    phantom.SetRotation(0.0f, 0.0f, 0.0f, "deg");
    phantom.SetPosition(0.0f, 0.0f, 0.0f, "mm");

    // Dosimetry
    GGEMSDosimetryCalculator dosimetry;
    dosimetry.AttachToNavigator("phantom");
    dosimetry.SetOutputDosimetryBasename("data/dosimetry");
    dosimetry.SetWaterReference(false);
    dosimetry.SetMinimumDensity(0.1f, "g/cm3");
    dosimetry.SetUncertainty(true);
    dosimetry.SetPhotonTracking(true);
    dosimetry.SetEdep(true);
    dosimetry.SetEdepSquared(true);
    dosimetry.SetHitTracking(true);

    GGEMSCTSystem ct_detector("custom");
    ct_detector.SetCTSystemType("flat");
    ct_detector.SetNumberOfModules(1, 1);
    ct_detector.SetNumberOfDetectionElementsInsideModule(400, 400, 1);
    ct_detector.SetSizeOfDetectionElements(1.0f, 1.0f, 10.0f, "mm");
    ct_detector.SetMaterialName("Silicon");
    ct_detector.SetSourceDetectorDistance(1500.0f, "mm");
    ct_detector.SetSourceIsocenterDistance(900.0f, "mm");
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
    point_source.SetNumberOfParticles(1000000);
    point_source.SetPosition(-900.0f, 0.0f, 0.0f, "mm");
    point_source.SetRotation(0.0f, 0.0f, 0.0f, "deg");
    point_source.SetBeamAperture(12.0f, "deg");
    point_source.SetFocalSpotSize(0.0f, 0.0f, 0.0f, "mm");
    point_source.SetMonoenergy(60.0f, "keV");

    // GGEMS simulation
    ggems_manager.SetOpenCLVerbose(false);
    ggems_manager.SetMaterialDatabaseVerbose(false);
    ggems_manager.SetNavigatorVerbose(true);
    ggems_manager.SetSourceVerbose(true);
    ggems_manager.SetMemoryRAMVerbose(true);
    ggems_manager.SetProcessVerbose(true);
    ggems_manager.SetRangeCutsVerbose(true);
    ggems_manager.SetRandomVerbose(true);
    ggems_manager.SetProfilingVerbose(true);
    ggems_manager.SetTrackingVerbose(false, 0);

    // Initializing the GGEMS simulation
    ggems_manager.Initialize();

    // Start GGEMS simulation
    ggems_manager.Run();
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
