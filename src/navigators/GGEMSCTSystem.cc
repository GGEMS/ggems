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
#include "GGEMS/geometries/GGEMSSolidBox.hh"
#include "GGEMS/geometries/GGEMSSolidBoxData.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSCTSystem::GGEMSCTSystem(std::string const& ct_system_name)
: GGEMSSystem(ct_system_name),
  ct_system_type_(""),
  source_isocenter_distance_(0.0f),
  source_detector_distance_(0.0f)
{
  GGcout("GGEMSCTSystem", "GGEMSCTSystem", 3) << "GGEMSCTSystem creating..." << GGendl;

  GGcout("GGEMSCTSystem", "GGEMSCTSystem", 3) << "GGEMSCTSystem created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSCTSystem::~GGEMSCTSystem(void)
{
  GGcout("GGEMSCTSystem", "~GGEMSCTSystem", 3) << "GGEMSCTSystem erasing..." << GGendl;

  GGcout("GGEMSCTSystem", "~GGEMSCTSystem", 3) << "GGEMSCTSystem erased!!!" << GGendl;
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

  if (ct_system_type_ != "curved" && ct_system_type_ != "flat") {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Available CT system types: 'curved' or 'flat'";
    GGEMSMisc::ThrowException("GGEMSCTSystem", "CheckParameters", oss.str());
  }

  if (source_detector_distance_ == 0.0f) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "For CT system, source detector distance (SDD) has to be > 0.0 mm!!!";
    GGEMSMisc::ThrowException("GGEMSCTSystem", "CheckParameters", oss.str());
  }

  if (source_isocenter_distance_ > source_detector_distance_) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Source isocenter distance must be inferior to source detector distance!!!";
    GGEMSMisc::ThrowException("GGEMSCTSystem", "CheckParameters", oss.str());
  }

  // Call parent parameters
  GGEMSSystem::CheckParameters();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSCTSystem::InitializeCurvedGeometry(void)
{
  // Computing the angle 'alpha' between x-ray source and Y borders of a module
  // Using Pythagore algorithm in a regular polygone
  // rho = Hypotenuse
  // h = Apothem or source detector distance in our case
  // c = Half-distance of module in Y
  GGfloat c = (static_cast<GGfloat>(number_of_detection_elements_inside_module_xyz_.y_)*size_of_detection_elements_xyz_.s[1])*0.5f;
  GGfloat rho = std::sqrt(source_detector_distance_*source_detector_distance_ + c*c);
  GGfloat alpha = 2.0f*std::asin(c/rho);

  // Center of rotation O (ox, oy) is the source
  GGfloat ox = -source_isocenter_distance_;
  GGfloat oy = 0.0f;

  // Center of module P (px, py) is source detector distance minus source isocenter distance plus half size of module in Z (module referential)
  GGfloat px = source_detector_distance_ - source_isocenter_distance_;
  GGfloat py = 0.0f;

  // Loop over each module in X and Y, and compute angle of each module in around Z
  for (GGsize j = 0; j < number_of_modules_xy_.y_; ++j) { // for Y modules
    GGfloat step_angle = alpha * (static_cast<GGfloat>(j) + 0.5f*(1.0f - static_cast<GGfloat>(number_of_modules_xy_.y_)));

    // Computing the X and Y positions in global position (isocenter)
    GGfloat global_position_x = (px-ox)*std::cos(step_angle) - (py-oy)*std::sin(step_angle) + ox;
    GGfloat global_position_y = (px-ox)*std::sin(step_angle) + (py-oy)*std::cos(step_angle) + oy;

    for (GGsize i = 0; i < number_of_modules_xy_.x_; ++i) { // for X modules
      // Computing the Z position in global position (isocenter)
      GGfloat global_position_z = static_cast<GGfloat>(number_of_detection_elements_inside_module_xyz_.x_)*size_of_detection_elements_xyz_.s[0]*(static_cast<GGfloat>(i)+0.5f*(1.0f-static_cast<GGfloat>(number_of_modules_xy_.x_)));

      GGsize global_index = i+j*number_of_modules_xy_.x_;

      GGfloat3 rotation;
      rotation.s[0] = 0.0f;
      rotation.s[1] = 0.0f;
      rotation.s[2] = step_angle;
      solids_[global_index]->SetRotation(rotation);

      GGfloat3 position;
      position.s[0] = global_position_x + global_system_position_xyz_.s[0];
      position.s[1] = global_position_y + global_system_position_xyz_.s[1];
      position.s[2] = global_position_z + global_system_position_xyz_.s[2];
      solids_[global_index]->SetPosition(position);

      // Rotation for OpenGL volume
      #ifdef OPENGL_VISUALIZATION
      solids_[global_index]->SetZAngleOpenGL(step_angle);
      #endif
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSCTSystem::InitializeFlatGeometry(void)
{
    // Computing the X, Y and Z positions in global position (isocenter)
  GGfloat global_position_x = source_detector_distance_ - source_isocenter_distance_;
  GGfloat global_position_y = 0.0f;
  GGfloat global_position_z = 0.0f;

  // Consider flat geometry for CBCT configuration
  for (GGsize j = 0; j < number_of_modules_xy_.y_; ++j) { // Y modules
    global_position_y = static_cast<GGfloat>(number_of_detection_elements_inside_module_xyz_.y_)*size_of_detection_elements_xyz_.s[1]*(static_cast<GGfloat>(j)+0.5f*(1.0f-static_cast<GGfloat>(number_of_modules_xy_.y_)));
    for (GGsize i = 0; i < number_of_modules_xy_.x_; ++i) { // X modules
      global_position_z = static_cast<GGfloat>(number_of_detection_elements_inside_module_xyz_.x_)*size_of_detection_elements_xyz_.s[0]*(static_cast<GGfloat>(i)+0.5f*(1.0f-static_cast<GGfloat>(number_of_modules_xy_.x_)));

      GGsize global_index = i+j*number_of_modules_xy_.x_;

      // No rotation of module
      GGfloat3 rotation;
      rotation.s[0] = 0.0f;
      rotation.s[1] = 0.0f;
      rotation.s[2] = 0.0f;
      solids_[global_index]->SetRotation(rotation);

      GGfloat3 position;
      position.s[0] = global_position_x + global_system_position_xyz_.s[0];
      position.s[1] = global_position_y + global_system_position_xyz_.s[1];
      position.s[2] = global_position_z + global_system_position_xyz_.s[2];
      solids_[global_index]->SetPosition(position);
    }
  }
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
  // Getting the current number of registered solid
  GGEMSNavigatorManager& navigator_manager = GGEMSNavigatorManager::GetInstance();
  GGsize number_of_registered_solids = navigator_manager.GetNumberOfRegisteredSolids();

  // Creating all solids, solid box for CT
  number_of_solids_ = static_cast<GGsize>(number_of_modules_xy_.x_ * number_of_modules_xy_.y_);

  // Allocation of memory for solid
  solids_ = new GGEMSSolid*[number_of_solids_];

  for (GGsize i = 0; i < number_of_solids_; ++i) { // In CT system only "HISTOGRAM"
    solids_[i] = new GGEMSSolidBox(
      number_of_detection_elements_inside_module_xyz_.x_,
      number_of_detection_elements_inside_module_xyz_.y_,
      number_of_detection_elements_inside_module_xyz_.z_,
      static_cast<GGfloat>(number_of_detection_elements_inside_module_xyz_.x_) * size_of_detection_elements_xyz_.s[0],
      static_cast<GGfloat>(number_of_detection_elements_inside_module_xyz_.y_) * size_of_detection_elements_xyz_.s[1],
      static_cast<GGfloat>(number_of_detection_elements_inside_module_xyz_.z_) * size_of_detection_elements_xyz_.s[2],
      "HISTOGRAM"
    );
    solids_[i]->SetVisible(is_visible_);
    solids_[i]->SetMaterialName(materials_->GetMaterialName(0));
    solids_[i]->SetCustomMaterialColor(custom_material_rgb_);
    solids_[i]->SetMaterialVisible(material_visible_);

    // Enabling scatter if necessary
    if (is_scatter_) solids_[i]->EnableScatter();

    // Enabling tracking if necessary
    if (is_tracking_) solids_[i]->EnableTracking();

    // Initialize kernels
    solids_[i]->Initialize(nullptr);
  }

  // Initialize of the geometry depending on type of CT system
  if (ct_system_type_ == "curved") {
    InitializeCurvedGeometry();
  }
  else if (ct_system_type_ == "flat") {
    InitializeFlatGeometry();
  }

  // Performing a global rotation when source and system rotate
  if (is_update_rot_) {
    for (GGsize i = 0; i < number_of_solids_; ++i) {
      solids_[i]->SetRotation(rotation_xyz_);
      #ifdef OPENGL_VISUALIZATION
      solids_[i]->SetXUpdateAngleOpenGL(rotation_xyz_.s[0]);
      solids_[i]->SetYUpdateAngleOpenGL(rotation_xyz_.s[1]);
      solids_[i]->SetZUpdateAngleOpenGL(rotation_xyz_.s[2]);
      #endif
    }
  }

  // Get the final transformation matrix
  for (GGsize j = 0; j < number_activated_devices_; ++j) {
    for (GGsize i = 0; i < number_of_solids_; ++i) {
      // Set solid id
      solids_[i]->SetSolidID<GGEMSSolidBoxData>(number_of_registered_solids+i, j);
      solids_[i]->UpdateTransformationMatrix(j);
    }
  }

  #ifdef OPENGL_VISUALIZATION
  for (GGsize i = 0; i < number_of_solids_; ++i) solids_[i]->BuildOpenGL();
  #endif

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

void set_number_of_modules_ggems_ct_system(GGEMSCTSystem* ct_system, GGsize const module_x, GGsize const module_y)
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

void set_number_of_detection_elements_ggems_ct_system(GGEMSCTSystem* ct_system, GGsize const n_detection_element_x, GGsize const n_detection_element_y, GGsize const n_detection_element_z)
{
  ct_system->SetNumberOfDetectionElementsInsideModule(n_detection_element_x, n_detection_element_y, n_detection_element_z);
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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_rotation_ggems_ct_system(GGEMSCTSystem* ct_system, GGfloat const rx, GGfloat const ry, GGfloat const rz, char const* unit)
{
  ct_system->SetRotation(rx, ry, rz, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_save_ggems_ct_system(GGEMSCTSystem* ct_system, char const* basename)
{
  ct_system->StoreOutput(basename);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_threshold_ggems_ct_system(GGEMSCTSystem* ct_system, GGfloat const threshold, char const* unit)
{
  ct_system->SetThreshold(threshold, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void store_scatter_ggems_ct_system(GGEMSCTSystem* ct_system, bool const is_scatter)
{
  ct_system->StoreScatter(is_scatter);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_visible_ggems_ct_system(GGEMSCTSystem* ct_system, bool const flag)
{
  ct_system->SetVisible(flag);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_material_visible_ggems_ct_system(GGEMSCTSystem* ct_system, char const* material_name, bool const flag)
{
  ct_system->SetMaterialVisible(material_name, flag);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_material_color_ggems_ct_system(GGEMSCTSystem* ct_system, char const* material_name, GGuchar const red, GGuchar const green, GGuchar const blue)
{
  ct_system->SetMaterialColor(material_name, red, green, blue);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_material_color_name_ggems_ct_system(GGEMSCTSystem* ct_system, char const* material_name, char const* color_name)
{
  ct_system->SetMaterialColor(material_name, color_name);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_global_system_position_ggems_ct_system(GGEMSCTSystem* ct_system, GGfloat const global_system_position_x, GGfloat const global_system_position_y, GGfloat const global_system_position_z, char const* unit)
{
  ct_system->SetGlobalSystemPosition(global_system_position_x, global_system_position_y, global_system_position_z, unit);
}
