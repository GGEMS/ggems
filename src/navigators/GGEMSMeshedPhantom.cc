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
  \file GGEMSMeshedPhantom.cc

  \brief Child GGEMS class handling meshed phantom

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \date Tuesday June 14, 2022
*/

#include "GGEMS/navigators/GGEMSMeshedPhantom.hh"
#include "GGEMS/navigators/GGEMSDosimetryCalculator.hh"
#include "GGEMS/navigators/GGEMSNavigatorManager.hh"
#include "GGEMS/geometries/GGEMSMeshedSolid.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMeshedPhantom::GGEMSMeshedPhantom(std::string const& meshed_phantom_name)
: GGEMSNavigator(meshed_phantom_name),
  meshed_phantom_filename_("")
{
  GGcout("GGEMSMeshedPhantom", "GGEMSMeshedPhantom", 3) << "GGEMSMeshedPhantom creating..." << GGendl;

  GGcout("GGEMSMeshedPhantom", "GGEMSMeshedPhantom", 3) << "GGEMSMeshedPhantom created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMeshedPhantom::~GGEMSMeshedPhantom(void)
{
  GGcout("GGEMSMeshedPhantom", "~GGEMSMeshedPhantom", 3) << "GGEMSMeshedPhantom erasing..." << GGendl;

  GGcout("GGEMSMeshedPhantom", "~GGEMSMeshedPhantom", 3) << "GGEMSMeshedPhantom erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMeshedPhantom::CheckParameters(void) const
{
  GGcout("GGEMSMeshedPhantom", "CheckParameters", 3) << "Checking the mandatory parameters..." << GGendl;

  // Checking meshed phantom files
  if (meshed_phantom_filename_.empty()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a mesh file in STL format!!!";
    GGEMSMisc::ThrowException("GGEMSMeshedPhantom", "CheckParameters", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMeshedPhantom::Initialize(void)
{
  GGcout("GGEMSMeshedPhantom", "Initialize", 3) << "Initializing a GGEMS meshed phantom..." << GGendl;

  CheckParameters();

  // Getting the current number of registered solid
  GGEMSNavigatorManager& navigator_manager = GGEMSNavigatorManager::GetInstance();

  // Get the number of already registered buffer
  GGsize number_of_registered_solids = navigator_manager.GetNumberOfRegisteredSolids();

  // Allocation of memory for solid, 1 solid in case of meshed phantom
  solids_ = new GGEMSSolid*[1];
  number_of_solids_ = 1;

  // Initializing meshed solid for geometric navigation
  // if (is_dosimetry_mode_) {
  //   solids_[0] = new GGEMSMeshedSolid(meshed_phantom_filename_, "DOSIMETRY");
  // }
  // else {
  solids_[0] = new GGEMSMeshedSolid(meshed_phantom_filename_);
  // }

  // Enabling tracking if necessary
  // if (is_tracking_) solids_[0]->EnableTracking();

  // Enabling TLE
  // if (is_tle_) solids_[0]->AddKernelOption(" -DTLE");

  // Load meshed phantom from STL file and storing materials
  solids_[0]->Initialize(materials_);
  // solids_[0]->SetCustomMaterialColor(custom_material_rgb_);
  // solids_[0]->SetMaterialVisible(material_visible_);

  // // Perform rotation before position
  // if (is_update_rot_) {
  //   solids_[0]->SetRotation(rotation_xyz_);
  //   #ifdef OPENGL_VISUALIZATION
  //   solids_[0]->SetXUpdateAngleOpenGL(rotation_xyz_.s[0]);
  //   solids_[0]->SetYUpdateAngleOpenGL(rotation_xyz_.s[1]);
  //   solids_[0]->SetZUpdateAngleOpenGL(rotation_xyz_.s[2]);
  //   #endif
  // }
  // if (is_update_pos_) solids_[0]->SetPosition(position_xyz_);

  // for (GGsize j = 0; j < number_activated_devices_; ++j) {
  //   solids_[0]->SetSolidID<GGEMSVoxelizedSolidData>(number_of_registered_solids, j);
  //   // Store the transformation matrix in solid object
  //   solids_[0]->UpdateTransformationMatrix(j);
  // }

  // #ifdef OPENGL_VISUALIZATION
  // solids_[0]->SetVisible(is_visible_);
  // solids_[0]->BuildOpenGL();
  // #endif

  // Initialize parent class
  GGEMSNavigator::Initialize();

  // // Checking if dosimetry mode activated
  // if (is_dosimetry_mode_) dose_calculator_->Initialize();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMeshedPhantom::SetPhantomFile(std::string const& meshed_phantom_filename)
{
  meshed_phantom_filename_ = meshed_phantom_filename;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMeshedPhantom::SaveResults(void)
{
  if (is_dosimetry_mode_) {
    GGcout("GGEMSMeshedPhantom", "SaveResults", 2) << "Saving dosimetry results in MHD format..." << GGendl;

    // Compute dose and save results
    dose_calculator_->SaveResults();
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMeshedPhantom* create_ggems_meshed_phantom(char const* meshed_phantom_name)
{
  return new(std::nothrow) GGEMSMeshedPhantom(meshed_phantom_name);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_phantom_file_ggems_meshed_phantom(GGEMSMeshedPhantom* meshed_phantom, char const* phantom_filename)
{
  meshed_phantom->SetPhantomFile(phantom_filename);
}
