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
#include "GGEMS/global/GGEMSManager.hh"
#include "GGEMS/tools/GGEMSPrint.hh"

/*!
  \fn int main(void)
  \return status of program
  \brief main function of program
*/
int main(void)
{
  // Setting verbosity
  GGcout.SetVerbosity(1);
  GGcerr.SetVerbosity(1);
  GGwarn.SetVerbosity(1);

  // Initialization of singletons
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  GGEMSMaterialsDatabaseManager& material_manager = GGEMSMaterialsDatabaseManager::GetInstance();
  GGEMSProcessesManager& processes_manager = GGEMSProcessesManager::GetInstance();
  GGEMSRangeCutsManager& range_cuts_manager = GGEMSRangeCutsManager::GetInstance();
  GGEMSManager& ggems_manager = GGEMSManager::GetInstance();

  // Set the context id
  opencl_manager.ContextToActivate(0);

  // Enter material database
  material_manager.SetMaterialsDatabase("../../data/materials.txt");

  // Phantoms and systems
  GGEMSVoxelizedPhantom phantom("phantom");
  phantom.SetPhantomFile("data/phantom.mhd", "data/range_phantom.txt");
  phantom.SetRotation(0.0f, 0.0f, 0.0f, "deg");
  phantom.SetPosition(0.0f, 0.0f, 0.0f, "mm");

  // Dosimetry
  GGEMSDosimetryCalculator dosimetry("phantom");
  dosimetry.SetOutputDosimetryFilename("data/dosimetry");
  dosimetry.SetDoselSizes(0.5f, 0.5f, 0.5f, "mm");
  dosimetry.SetWaterReference(true);
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
  point_source.SetNumberOfParticles(1000000000);
  point_source.SetPosition(-595.0f, 0.0f, 0.0f, "mm");
  point_source.SetRotation(0.0f, 0.0f, 0.0f, "deg");
  point_source.SetBeamAperture(5.0f, "deg");
  point_source.SetFocalSpotSize(0.0f, 0.0f, 0.0f, "mm");
  point_source.SetPolyenergy("data/spectrum_120kVp_2mmAl.dat");

  // GGEMS simulation
  ggems_manager.SetOpenCLVerbose(true);
  ggems_manager.SetNavigatorVerbose(true);
  ggems_manager.SetSourceVerbose(true);
  ggems_manager.SetMemoryRAMVerbose(true);
  ggems_manager.SetProcessVerbose(true);
  ggems_manager.SetRangeCutsVerbose(true);
  ggems_manager.SetRandomVerbose(true);
  ggems_manager.SetKernelVerbose(true);
  ggems_manager.SetTrackingVerbose(false, 0);

  // Initializing the GGEMS simulation
  ggems_manager.Initialize(777);

  // Start GGEMS simulation
  ggems_manager.Run();

  exit(EXIT_SUCCESS);
}
