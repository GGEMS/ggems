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
  \file GGEMSVoxelizedPhantom.cc

  \brief Child GGEMS class handling voxelized phantom

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \date Tuesday October 20, 2020
*/

#include "GGEMS/navigators/GGEMSVoxelizedPhantom.hh"
#include "GGEMS/navigators/GGEMSDosimetryCalculator.hh"
#include "GGEMS/navigators/GGEMSDoseParams.hh"
#include "GGEMS/geometries/GGEMSVoxelizedSolid.hh"
#include "GGEMS/io/GGEMSMHDImage.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVoxelizedPhantom::GGEMSVoxelizedPhantom(std::string const& voxelized_phantom_name)
: GGEMSNavigator(voxelized_phantom_name),
  voxelized_phantom_filename_(""),
  range_data_filename_("")
{
  GGcout("GGEMSVoxelizedPhantom", "GGEMSVoxelizedPhantom", 3) << "GGEMSVoxelizedPhantom creating..." << GGendl;

  GGcout("GGEMSVoxelizedPhantom", "GGEMSVoxelizedPhantom", 3) << "GGEMSVoxelizedPhantom created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVoxelizedPhantom::~GGEMSVoxelizedPhantom(void)
{
  GGcout("GGEMSVoxelizedPhantom", "~GGEMSVoxelizedPhantom", 3) << "GGEMSVoxelizedPhantom erasing..." << GGendl;

  GGcout("GGEMSVoxelizedPhantom", "~GGEMSVoxelizedPhantom", 3) << "GGEMSVoxelizedPhantom erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedPhantom::CheckParameters(void) const
{
  GGcout("GGEMSVoxelizedPhantom", "CheckParameters", 3) << "Checking the mandatory parameters..." << GGendl;

  // Checking voxelized phantom files (mhd+range data)
  if (voxelized_phantom_filename_.empty()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a mhd file containing the voxelized phantom!!!";
    GGEMSMisc::ThrowException("GGEMSVoxelizedPhantom", "CheckParameters", oss.str());
  }

  // Checking the phantom name
  if (range_data_filename_.empty()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a file with the range to material data!!!";
    GGEMSMisc::ThrowException("GGEMSVoxelizedPhantom", "CheckParameters", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedPhantom::Initialize(void)
{
  GGcout("GGEMSVoxelizedPhantom", "Initialize", 3) << "Initializing a GGEMS voxelized phantom..." << GGendl;

  CheckParameters();

  // Getting the current number of registered solid
  GGEMSNavigatorManager& navigator_manager = GGEMSNavigatorManager::GetInstance();

  // Get the number of already registered buffer
  GGsize number_of_registered_solids = navigator_manager.GetNumberOfRegisteredSolids();

  // Allocation of memory for solid, 1 solid in case of voxelized phantom
  solids_ = new GGEMSSolid*[1];
  number_of_solids_ = 1;

  // Initializing voxelized solid for geometric navigation
  if (is_dosimetry_mode_) {
    solids_[0] = new GGEMSVoxelizedSolid(voxelized_phantom_filename_, range_data_filename_, "DOSIMETRY");
  }
  else {
    solids_[0] = new GGEMSVoxelizedSolid(voxelized_phantom_filename_, range_data_filename_);
  }

  // Enabling tracking if necessary
  if (is_tracking_) solids_[0]->EnableTracking();

  // Load voxelized phantom from MHD file and storing materials
  solids_[0]->Initialize(materials_);

  // Perform rotation before position
  if (is_update_rot_) solids_[0]->SetRotation(rotation_xyz_);
  if (is_update_pos_) solids_[0]->SetPosition(position_xyz_);

  for (GGsize j = 0; j < number_activated_devices_; ++j) {
    solids_[0]->SetSolidID<GGEMSVoxelizedSolidData>(number_of_registered_solids, j);
    // Store the transformation matrix in solid object
    solids_[0]->UpdateTransformationMatrix(j);
  }

  // Initialize parent class
  GGEMSNavigator::Initialize();

  // Checking if dosimetry mode activated
  if (is_dosimetry_mode_) dose_calculator_->Initialize();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedPhantom::SaveResults(void)
{
  if (is_dosimetry_mode_) {
    GGcout("GGEMSVoxelizedPhantom", "SaveResults", 2) << "Saving dosimetry results in MHD format..." << GGendl;

    // Compute dose and save results
    dose_calculator_->SaveResults();
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedPhantom::SetPhantomFile(std::string const& voxelized_phantom_filename, std::string const& range_data_filename)
{
  voxelized_phantom_filename_ = voxelized_phantom_filename;
  range_data_filename_ = range_data_filename;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVoxelizedPhantom* create_ggems_voxelized_phantom(char const* voxelized_phantom_name)
{
  return new(std::nothrow) GGEMSVoxelizedPhantom(voxelized_phantom_name);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_phantom_file_ggems_voxelized_phantom(GGEMSVoxelizedPhantom* voxelized_phantom, char const* phantom_filename, char const* range_data_filename)
{
  voxelized_phantom->SetPhantomFile(phantom_filename, range_data_filename);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_position_ggems_voxelized_phantom(GGEMSVoxelizedPhantom* voxelized_phantom, GGfloat const position_x, GGfloat const position_y, GGfloat const position_z, char const* unit)
{
  voxelized_phantom->SetPosition(position_x, position_y, position_z, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_rotation_ggems_voxelized_phantom(GGEMSVoxelizedPhantom* voxelized_phantom, GGfloat const rx, GGfloat const ry, GGfloat const rz, char const* unit)
{
  voxelized_phantom->SetRotation(rx, ry, rz, unit);
}
