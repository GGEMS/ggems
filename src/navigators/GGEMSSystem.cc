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
  \file GGEMSSystem.hh

  \brief GGEMS class managing detector system in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \date Monday October 19, 2020
*/

#include "GGEMS/navigators/GGEMSSystem.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/navigators/GGEMSNavigatorManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSystem::GGEMSSystem(std::string const& system_name)
: GGEMSNavigator(system_name),
  number_of_modules_xy_({0, 0}),
  number_of_detection_elements_inside_module_xy_({0, 0}),
  size_of_detection_elements_xyz_({0.0f, 0.0f, 0.0f}),
  material_name_("")
{
  GGcout("GGEMSSystem", "GGEMSSystem", 3) << "Allocation of GGEMSSystem..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSystem::~GGEMSSystem(void)
{
  GGcout("GGEMSSystem", "~GGEMSSystem", 3) << "Deallocation of GGEMSSystem..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSystem::SetNumberOfModules(GGuint const& n_module_x, GGuint const& n_module_y)
{
  number_of_modules_xy_.s[0] = n_module_x;
  number_of_modules_xy_.s[1] = n_module_y;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSystem::SetNumberOfDetectionElementsInsideModule(GGuint const& n_detection_element_x, GGuint const& n_detection_element_y)
{
  number_of_detection_elements_inside_module_xy_.s[0] = n_detection_element_x;
  number_of_detection_elements_inside_module_xy_.s[1] = n_detection_element_y;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSystem::SetSizeOfDetectionElements(GGfloat const& detection_element_x, GGfloat const& detection_element_y, GGfloat const& detection_element_z, std::string const& unit)
{
  size_of_detection_elements_xyz_.s[0] = DistanceUnit(detection_element_x, unit);
  size_of_detection_elements_xyz_.s[1] = DistanceUnit(detection_element_y, unit);
  size_of_detection_elements_xyz_.s[2] = DistanceUnit(detection_element_z, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSystem::SetGlobalPosition(GGfloat const& global_position_x, GGfloat const& global_position_y, GGfloat const& global_position_z, std::string const& unit)
{
  position_.s[0] = DistanceUnit(global_position_x, unit);
  position_.s[1] = DistanceUnit(global_position_y, unit);
  position_.s[2] = DistanceUnit(global_position_z, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSystem::SetMaterialName(std::string const& material_name)
{
  material_name_ = material_name;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSystem::CheckParameters(void) const
{
  GGcout("GGEMSSystem", "CheckParameters", 3) << "Checking the mandatory parameters..." << GGendl;

  if (number_of_modules_xy_.s[0] == 0 || number_of_modules_xy_.s[1] == 0) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "In system parameters, number of module in x and y axis (local axis) has to be > 0!!!";
    GGEMSMisc::ThrowException("GGEMSSystem", "CheckParameters", oss.str());
  }

  if (number_of_detection_elements_inside_module_xy_.s[0] == 0 || number_of_detection_elements_inside_module_xy_.s[1] == 0) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "In system parameters, number of detection elements in x and y axis (local axis) has to be > 0!!!";
    GGEMSMisc::ThrowException("GGEMSSystem", "CheckParameters", oss.str());
  }

  if (size_of_detection_elements_xyz_.s[0] == 0.0f || size_of_detection_elements_xyz_.s[1] == 0.0f || size_of_detection_elements_xyz_.s[2] == 0.0f) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "In system parameters, size of detection elements (local axis) has to be > 0.0 mm!!!";
    GGEMSMisc::ThrowException("GGEMSSystem", "CheckParameters", oss.str());
  }

  if (material_name_.empty()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "In system parameters, a material has to be defined!!!";
    GGEMSMisc::ThrowException("GGEMSSystem", "CheckParameters", oss.str());
  }

  // Call parent class
  GGEMSNavigator::CheckParameters();
}
