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
  \file GGEMSCTSystem.cc

  \brief GGEMS class managing CT/CBCT detector ct_system_name in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \date Monday October 19, 2020
*/

#include "GGEMS/navigators/GGEMSCTSystem.hh"
#include "GGEMS/tools/GGEMSPrint.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSCTSystem::GGEMSCTSystem(std::string const& ct_system_name)
: GGEMSSystem(ct_system_name),
  ct_system_type_(""),
  source_isocenter_distance_(0.0f),
  source_detector_distance_(0.0f)
{
  GGcout("GGEMSCTSystem", "GGEMSCTSystem", 3) << "Allocation of GGEMSCTSystem..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSCTSystem::~GGEMSCTSystem(void)
{
  GGcout("GGEMSCTSystem", "~GGEMSCTSystem", 3) << "Deallocation of GGEMSCTSystem..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSCTSystem::SetCTSystemType(std::string const& ct_system_type)
{
  ct_system_type_ = ct_system_type;

  // Transform string to low letter
  std::transform(ct_system_type_.begin(), ct_system_type_.end(), ct_system_type_.begin(), ::tolower);

  // Checking the CT type
  if (ct_system_type_ != "curved" && ct_system_type_ != "flat") {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Available CT system types: 'curved' or 'flat'";
    GGEMSMisc::ThrowException("GGEMSCTSystem", "SetCTSystemType", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSCTSystem::SetSourceIsocenterDistance(GGfloat const& source_isocenter_distance, std::string const& unit)
{
  source_isocenter_distance_ = DistanceUnit(source_isocenter_distance, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSCTSystem::SetSourceDetectorDistance(GGfloat const& source_detector_distance, std::string const& unit)
{
  source_detector_distance_ = DistanceUnit(source_detector_distance, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSCTSystem::CheckParameters(void) const
{
  GGcout("GGEMSCTSystem", "CheckParameters", 3) << "Checking the mandatory parameters..." << GGendl;

  // Checking the CT type
  if (ct_system_type_ != "curved" && ct_system_type_ != "flat") {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Available CT system types: 'curved' or 'flat'";
    GGEMSMisc::ThrowException("GGEMSCTSystem", "CheckParameters", oss.str());
  }

  // Checking Source Isocenter Distance (SID)
  if (source_isocenter_distance_ == 0.0f) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "For CT system, source isocenter distance (SID) has to be > 0.0 mm!!!";
    GGEMSMisc::ThrowException("GGEMSCTSystem", "CheckParameters", oss.str());
  }

  // Checking Source Detector Distance (SDD)
  if (source_detector_distance_ == 0.0f) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "For CT system, source detector distance (SDD) has to be > 0.0 mm!!!";
    GGEMSMisc::ThrowException("GGEMSCTSystem", "CheckParameters", oss.str());
  }

  // Call parent class
  GGEMSSystem::CheckParameters();
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSCTSystem::Initialize(void)
{
  GGcout("GGEMSCTSystem", "Initialize", 3) << "Initializing a GGEMS CT system..." << GGendl;

  // Checking the parameters
  CheckParameters();

  // Build CT system depending on input parameters
  // A CT system is composed by solid boxes

  // // Initializing Solid for geometric navigation depending on type of navigator
  // solid_.emplace_back(new GGEMSVoxelizedSolid(voxelized_phantom_filename_, range_data_filename_));

  // // Enabling tracking if necessary
  // if (is_tracking_) solid_.at(0)->EnableTracking();

  // // Getting the current number of registered solid
  // GGEMSNavigatorManager& navigator_manager = GGEMSNavigatorManager::GetInstance();
  // // Get the number of already registered buffer, we take the total number of solids (including the all current solids)
  // // minus all current solids
  // std::size_t const kNumberOfAlreadyRegisteredSolids = navigator_manager.GetNumberOfRegisteredSolids() - solid_.size();
  // solid_.at(0)->SetSolidID(0+kNumberOfAlreadyRegisteredSolids); // Only 1 solid!!!
 
  // solid_.at(0)->Initialize(materials_); // Load voxelized phantom from MHD file

  // // Updating or setting a position, rotation, or local axis for each solid
  // if (is_update_pos_) solid_.at(0)->SetPosition(position_xyz_);
  // if (is_update_rot_) solid_.at(0)->SetRotation(rotation_xyz_);
  // if (is_update_axis_) solid_.at(0)->SetLocalAxis(local_axis_);

  // // Update the transformation matrix
  // solid_.at(0)->UpdateTransformationMatrix();

  // Adding material
  materials_->AddMaterial(material_name_);

  // Initialize parent class
  GGEMSNavigator::Initialize();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSCTSystem* create_ggems_ct_system(char const* ct_system_name)
{
  return new(std::nothrow) GGEMSCTSystem(ct_system_name);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_number_of_modules_ggems_ct_system(GGEMSCTSystem* ct_system, GGuint const module_x, GGuint const module_y)
{
  ct_system->SetNumberOfModules(module_x, module_y);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_ct_system_type_ggems_ct_system(GGEMSCTSystem* ct_system, char const* ct_system_type)
{
  ct_system->SetCTSystemType(ct_system_type);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_number_of_detection_elements_ggems_ct_system(GGEMSCTSystem* ct_system, GGuint const n_detection_element_x, GGuint const n_detection_element_y)
{
  ct_system->SetNumberOfDetectionElementsInsideModule(n_detection_element_x, n_detection_element_y);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_size_of_detection_elements_ggems_ct_system(GGEMSCTSystem* ct_system, GGfloat const size_of_detection_element_x, GGfloat const size_of_detection_element_y, GGfloat const size_of_detection_element_z, char const* unit)
{
  ct_system->SetSizeOfDetectionElements(size_of_detection_element_x, size_of_detection_element_y, size_of_detection_element_z, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_material_name_ggems_ct_system(GGEMSCTSystem* ct_system, char const* material_name)
{
  ct_system->SetMaterialName(material_name);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_global_position_ggems_ct_system(GGEMSCTSystem* ct_system, GGfloat global_position_x, GGfloat const global_position_y, GGfloat const global_position_z, char const* unit)
{
  ct_system->SetGlobalPosition(global_position_x, global_position_y, global_position_z, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_source_isocenter_distance_ggems_ct_system(GGEMSCTSystem* ct_system, GGfloat const source_isocenter_distance, char const* unit)
{
  ct_system->SetSourceIsocenterDistance(source_isocenter_distance, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_source_detector_distance_ggems_ct_system(GGEMSCTSystem* ct_system, GGfloat const source_detector_distance, char const* unit)
{
  ct_system->SetSourceDetectorDistance(source_detector_distance, unit);
}
