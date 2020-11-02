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
#include "GGEMS/geometries/GGEMSVolumeCreatorManager.hh"
#include "GGEMS/geometries/GGEMSBox.hh"
#include "GGEMS/tools/GGEMSPrint.hh"

int main()
{
  // Set verbosity at lower value
  GGcout.SetVerbosity(0);
  GGwarn.SetVerbosity(0);
  GGcerr.SetVerbosity(0);

  // Initialization of singletons
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  GGEMSVolumeCreatorManager& volume_creator_manager = GGEMSVolumeCreatorManager::GetInstance();

  // Set the context id
  opencl_manager.ContextToActivate(0);

  // Initializing a global voxelized volume
  volume_creator_manager.SetVolumeDimensions(500, 500, 128);
  volume_creator_manager.SetElementSizes(0.5f, 0.5f, 0.5f, "mm");
  volume_creator_manager.SetOutputImageFilename("data/volume");
  volume_creator_manager.SetRangeToMaterialDataFilename("data/range_volume");
  volume_creator_manager.SetMaterial("Air");
  volume_creator_manager.SetDataType("MET_INT");
  volume_creator_manager.Initialize();

  // Creating a tube

  // Creating a box
  GGEMSBox* box = new GGEMSBox(24.0f, 36.0f, 12.0f, "mm");
  box->SetPosition(0.0f, 0.0f, 0.0f, "mm");
  box->SetLabelValue(1);
  box->SetMaterial("Water");
  box->Initialize();
  box->Draw();
  delete box;

  // Creating a sphere

  // Creating a triangle

  // Creating an ellipse

  // Writing volume
  volume_creator_manager.Write();

  // Cleaning OpenCL manager
  opencl_manager.Clean();

  exit(EXIT_SUCCESS);
}
