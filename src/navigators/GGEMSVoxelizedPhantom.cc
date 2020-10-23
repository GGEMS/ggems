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
#include "GGEMS/geometries/GGEMSVoxelizedSolid.hh"
#include "GGEMS/tools/GGEMSPrint.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVoxelizedPhantom::GGEMSVoxelizedPhantom(std::string const& voxelized_phantom_name)
: GGEMSNavigator(voxelized_phantom_name),
  voxelized_phantom_filename_(""),
  range_data_filename_("")
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

  // Checking the parameters of phantom
  CheckParameters();

  // Initializing Solid for geometric navigation depending on type of navigator
  solid_.emplace_back(new GGEMSVoxelizedSolid(voxelized_phantom_filename_, range_data_filename_));

  // Enabling tracking if necessary
  if (is_tracking_) solid_.at(0)->EnableTracking();

  // Getting the current number of registered solid
  GGEMSNavigatorManager& navigator_manager = GGEMSNavigatorManager::GetInstance();
  // Get the number of already registered buffer, we take the total number of solids (including the all current solids)
  // minus all current solids
  std::size_t const kNumberOfAlreadyRegisteredSolids = navigator_manager.GetNumberOfRegisteredSolids() - solid_.size();
  solid_.at(0)->SetSolidID(0+kNumberOfAlreadyRegisteredSolids); // Only 1 solid!!!
 
  solid_.at(0)->Initialize(materials_); // Load voxelized phantom from MHD file

  // Updating or setting a position, rotation, or local axis for each solid
  if (is_update_pos_) solid_.at(0)->SetPosition(position_xyz_);
  if (is_update_rot_) solid_.at(0)->SetRotation(rotation_xyz_);
  if (is_update_axis_) solid_.at(0)->SetLocalAxis(local_axis_);

  // Update the transformation matrix
  solid_.at(0)->UpdateTransformationMatrix();

  // Initialize parent class
  GGEMSNavigator::Initialize();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedPhantom::SetVoxelizedPhantomFile(std::string const& voxelized_phantom_filename, std::string const& range_data_filename)
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

void set_voxelized_phantom_file_ggems_voxelized_phantom(GGEMSVoxelizedPhantom* voxelized_phantom, char const* phantom_filename, char const* range_data_filename)
{
  voxelized_phantom->SetVoxelizedPhantomFile(phantom_filename, range_data_filename);
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

void set_local_axis_ggems_voxelized_phantom(GGEMSVoxelizedPhantom* voxelized_phantom, GGfloat const m00, GGfloat const m01, GGfloat const m02, GGfloat const m10, GGfloat const m11, GGfloat const m12, GGfloat const m20, GGfloat const m21, GGfloat const m22)
{
  voxelized_phantom->SetLocalAxis({m00, m01, m02}, {m10, m11, m12}, {m20, m21, m22});
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_rotation_ggems_voxelized_phantom(GGEMSVoxelizedPhantom* voxelized_phantom, GGfloat const rx, GGfloat const ry, GGfloat const rz, char const* unit)
{
  voxelized_phantom->SetRotation(rx, ry, rz, unit);
}

