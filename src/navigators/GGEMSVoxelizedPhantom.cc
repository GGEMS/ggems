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
#include "GGEMS/geometries/GGEMSVoxelizedSolid.hh"
#include "GGEMS/global/GGEMSManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVoxelizedPhantom::GGEMSVoxelizedPhantom(std::string const& voxelized_phantom_name)
: GGEMSNavigator(voxelized_phantom_name),
  voxelized_phantom_filename_(""),
  range_data_filename_(""),
  is_dosimetry_mode_(false),
  dosel_sizes_({-1.0f, -1.0f, -1.0f}),
  dosimetry_output_filename("dosimetry_output.mhd")
{
  GGcout("GGEMSVoxelizedPhantom", "GGEMSVoxelizedPhantom", 3) << "Allocation of GGEMSVoxelizedPhantom..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVoxelizedPhantom::~GGEMSVoxelizedPhantom(void)
{
  GGcout("GGEMSVoxelizedPhantom", "~GGEMSVoxelizedPhantom", 3) << "Deallocation of GGEMSVoxelizedPhantom..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedPhantom::SetDosimetryMode(bool const& dosimetry_mode)
{
  is_dosimetry_mode_ = dosimetry_mode;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedPhantom::SetDoselSizes(float const& dosel_x, float const& dosel_y, float const& dosel_z, std::string const& unit)
{
  dosel_sizes_.s[0] = DistanceUnit(dosel_x, unit);
  dosel_sizes_.s[1] = DistanceUnit(dosel_y, unit);
  dosel_sizes_.s[2] = DistanceUnit(dosel_z, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedPhantom::SetOutputDosimetryFilename(std::string const& output_filename)
{
  dosimetry_output_filename = output_filename;
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

  // Initializing voxelized solid for geometric navigation
  solids_.emplace_back(new GGEMSVoxelizedSolid(voxelized_phantom_filename_, range_data_filename_));

  // Enabling tracking if necessary
  if (GGEMSManager::GetInstance().IsTrackingVerbose()) solids_.at(0)->EnableTracking();

  // Getting the current number of registered solid
  GGEMSNavigatorManager& navigator_manager = GGEMSNavigatorManager::GetInstance();
  // Get the number of already registered buffer, we take the total number of solids (including the all current solids) minus all current solids
  std::size_t number_of_registered_solids = navigator_manager.GetNumberOfRegisteredSolids() - solids_.size();

  solids_.at(0)->SetSolidID<GGEMSVoxelizedSolidData>(number_of_registered_solids);

  // Load voxelized phantom from MHD file and storing materials
  solids_.at(0)->Initialize(materials_);

  // Perform rotation before position
  if (is_update_rot_) solids_.at(0)->SetRotation(rotation_xyz_);
  if (is_update_pos_) solids_.at(0)->SetPosition(position_xyz_);

  // Store the transformation matrix in solid object
  solids_.at(0)->GetTransformationMatrix();

  // Checking if dosimetry mode activated
  if (is_dosimetry_mode_) {
    dose_calculator_.reset(new GGEMSDosimetryCalculator());
    dose_calculator_->Initialize();
  }

  // Initialize parent class
  GGEMSNavigator::Initialize();
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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_dosimetry_mode_voxelized_phantom(GGEMSVoxelizedPhantom* voxelized_phantom, bool const is_dosimetry_mode)
{
  voxelized_phantom->SetDosimetryMode(is_dosimetry_mode);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_dosel_size_voxelized_phantom(GGEMSVoxelizedPhantom* voxelized_phantom, GGfloat const dose_x, GGfloat const dose_y, GGfloat const dose_z, char const* unit)
{
  voxelized_phantom->SetDoselSizes(dose_x, dose_y, dose_z, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_dose_output_voxelized_phantom(GGEMSVoxelizedPhantom* voxelized_phantom, char const* dose_output_filename)
{
  voxelized_phantom->SetOutputDosimetryFilename(dose_output_filename);
}
