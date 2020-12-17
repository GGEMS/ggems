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
#include "GGEMS/global/GGEMSManager.hh"

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

  if (ct_system_type_ != "curved" && ct_system_type_ != "flat") {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Available CT system types: 'curved' or 'flat'";
    GGEMSMisc::ThrowException("GGEMSCTSystem", "CheckParameters", oss.str());
  }

  if (source_isocenter_distance_ == 0.0f) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "For CT system, source isocenter distance (SID) has to be > 0.0 mm!!!";
    GGEMSMisc::ThrowException("GGEMSCTSystem", "CheckParameters", oss.str());
  }

  if (source_detector_distance_ == 0.0f) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "For CT system, source detector distance (SDD) has to be > 0.0 mm!!!";
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
  GGfloat c = (number_of_detection_elements_inside_module_xyz_.s1*size_of_detection_elements_xyz_.s1)*0.5f;
  GGfloat rho = std::sqrt(source_detector_distance_*source_detector_distance_ + c*c);
  GGfloat alpha = 2.0f*std::asin(c/rho);

  // Center of rotation O (ox, oy) is the source
  GGfloat ox = -source_isocenter_distance_;
  GGfloat oy = 0.0f;

  // Center of module P (px, py) is source detector distance minus source isocenter distance plus half size of module in Z (module referential)
  GGfloat px = source_detector_distance_ - source_isocenter_distance_ + 0.5f*number_of_detection_elements_inside_module_xyz_.s2*size_of_detection_elements_xyz_.s2;
  GGfloat py = 0.0f;

  // Loop over each module in X and Y, and compute angle of each module in around Z
  for (GGint j = 0; j < number_of_modules_xy_.s1; ++j) { // for Y modules
    GGfloat step_angle = alpha * (j + 0.5f*(1-number_of_modules_xy_.s1));

    // Computing the X and Y positions in global position (isocenter)
    GGfloat global_position_x = (px-ox)*std::cos(step_angle) - (py-oy)*std::sin(step_angle) + ox;
    GGfloat global_position_y = (px-ox)*std::sin(step_angle) + (py-oy)*std::cos(step_angle) + oy;

    for (GGint i = 0; i < number_of_modules_xy_.s0; ++i) { // for X modules
      // Computing the Z position in global position (isocenter)
      GGfloat global_position_z = number_of_detection_elements_inside_module_xyz_.s0*size_of_detection_elements_xyz_.s0*(i+0.5f*(1-number_of_modules_xy_.s0));

      solids_.at(i+j*number_of_modules_xy_.s0)->SetRotation({0.0f, 0.0f, step_angle});
      solids_.at(i+j*number_of_modules_xy_.s0)->SetPosition({global_position_x, global_position_y, global_position_z});
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSCTSystem::InitializeFlatGeometry(void)
{
    // Computing the X, Y and Z positions in global position (isocenter)
  GGfloat global_position_x = source_detector_distance_ - source_isocenter_distance_ + 0.5f*number_of_detection_elements_inside_module_xyz_.s2*size_of_detection_elements_xyz_.s2;
  GGfloat global_position_y = 0.0f;
  GGfloat global_position_z = 0.0f;

  // Consider flat geometry for CBCT configuration
  for (GGint j = 0; j < number_of_modules_xy_.s1; ++j) { // Y modules
    global_position_y = number_of_detection_elements_inside_module_xyz_.s1*size_of_detection_elements_xyz_.s1*(j+0.5f*(1-number_of_modules_xy_.s1));
    for (GGint i = 0; i < number_of_modules_xy_.s0; ++i) { // X modules
      global_position_z = number_of_detection_elements_inside_module_xyz_.s0*size_of_detection_elements_xyz_.s0*(i+0.5f*(1-number_of_modules_xy_.s0));
      // No rotation of module
      solids_.at(i+j*number_of_modules_xy_.s0)->SetRotation({0.0f, 0.0f, 0.0f});
      solids_.at(i+j*number_of_modules_xy_.s0)->SetPosition({global_position_x, global_position_y, global_position_z});
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
  std::size_t number_of_registered_solids = navigator_manager.GetNumberOfRegisteredSolids() - solids_.size();

  // Creating all solids, solid box for CT
  GGint number_of_solids = number_of_modules_xy_.x * number_of_modules_xy_.y;
  for (GGint i = 0; i < number_of_solids; ++i) {
    // In CT system only "HISTOGRAM"
    solids_.emplace_back(new GGEMSSolidBox(
      number_of_detection_elements_inside_module_xyz_.x,
      number_of_detection_elements_inside_module_xyz_.y,
      number_of_detection_elements_inside_module_xyz_.z,
      number_of_detection_elements_inside_module_xyz_.x * size_of_detection_elements_xyz_.x,
      number_of_detection_elements_inside_module_xyz_.y * size_of_detection_elements_xyz_.y,
      number_of_detection_elements_inside_module_xyz_.z * size_of_detection_elements_xyz_.z,
      "HISTOGRAM")
    );

    // Enabling tracking if necessary
    if (GGEMSManager::GetInstance().IsTrackingVerbose()) solids_.at(i)->EnableTracking();

    // Set solid id
    solids_.at(i)->SetSolidID<GGEMSSolidBoxData>(number_of_registered_solids+i);

    // // Initialize kernels
    solids_.at(i)->Initialize(std::weak_ptr<GGEMSMaterials>());
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
    for (auto&& i : solids_) i->SetRotation(rotation_xyz_);
  }

  // Get the final transformation matrix
  for (auto&& i : solids_) i->GetTransformationMatrix();

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

void set_number_of_modules_ggems_ct_system(GGEMSCTSystem* ct_system, GGint const module_x, GGint const module_y)
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

void set_number_of_detection_elements_ggems_ct_system(GGEMSCTSystem* ct_system, GGint const n_detection_element_x, GGint const n_detection_element_y, GGint const n_detection_element_z)
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
