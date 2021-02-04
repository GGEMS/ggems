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
  \file GGEMSSystem.cc

  \brief GGEMS class managing detector system in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \date Monday October 19, 2020
*/

#include "GGEMS/navigators/GGEMSSystem.hh"
#include "GGEMS/geometries/GGEMSSolid.hh"
#include "GGEMS/io/GGEMSMHDImage.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSystem::GGEMSSystem(std::string const& system_name)
: GGEMSNavigator(system_name)
{
  GGcout("GGEMSSystem", "GGEMSSystem", 3) << "Allocation of GGEMSSystem..." << GGendl;

  number_of_modules_xy_.x = 0;
  number_of_modules_xy_.y = 0;

 number_of_detection_elements_inside_module_xyz_.x = 0;
 number_of_detection_elements_inside_module_xyz_.y = 0;
 number_of_detection_elements_inside_module_xyz_.z = 0;

  size_of_detection_elements_xyz_.x = 0.0f;
  size_of_detection_elements_xyz_.y = 0.0f;
  size_of_detection_elements_xyz_.z = 0.0f;
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

void GGEMSSystem::SetNumberOfModules(GGsize const& n_module_x, GGsize const& n_module_y)
{
  number_of_modules_xy_.x = n_module_x;
  number_of_modules_xy_.y = n_module_y;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSystem::SetNumberOfDetectionElementsInsideModule(GGsize const& n_detection_element_x, GGsize const& n_detection_element_y, GGsize const& n_detection_element_z)
{
  number_of_detection_elements_inside_module_xyz_.x = n_detection_element_x;
  number_of_detection_elements_inside_module_xyz_.y = n_detection_element_y;
  number_of_detection_elements_inside_module_xyz_.z = n_detection_element_z;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSystem::SetSizeOfDetectionElements(GGfloat const& detection_element_x, GGfloat const& detection_element_y, GGfloat const& detection_element_z, std::string const& unit)
{
  size_of_detection_elements_xyz_.x = DistanceUnit(detection_element_x, unit);
  size_of_detection_elements_xyz_.y = DistanceUnit(detection_element_y, unit);
  size_of_detection_elements_xyz_.z = DistanceUnit(detection_element_z, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSystem::SetMaterialName(std::string const& material_name)
{
  materials_->AddMaterial(material_name);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSystem::CheckParameters(void) const
{
  GGcout("GGEMSSystem", "CheckParameters", 3) << "Checking the mandatory parameters..." << GGendl;

  if (number_of_modules_xy_.x == 0 || number_of_modules_xy_.y == 0) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "In system parameters, number of module in x and y axis (local axis) has to be > 0!!!";
    GGEMSMisc::ThrowException("GGEMSSystem", "CheckParameters", oss.str());
  }

  if (number_of_detection_elements_inside_module_xyz_.x == 0 || number_of_detection_elements_inside_module_xyz_.y == 0 || number_of_detection_elements_inside_module_xyz_.z == 0) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "In system parameters, number of detection elements in x, y and z axis (local axis) has to be > 0!!!";
    GGEMSMisc::ThrowException("GGEMSSystem", "CheckParameters", oss.str());
  }

  if (size_of_detection_elements_xyz_.x == 0.0f || size_of_detection_elements_xyz_.y == 0.0f || size_of_detection_elements_xyz_.z == 0.0f) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "In system parameters, size of detection elements (local axis) has to be > 0.0 mm!!!";
    GGEMSMisc::ThrowException("GGEMSSystem", "CheckParameters", oss.str());
  }

  if (materials_->GetNumberOfMaterials() == 0) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "In system parameters, a material has to be defined!!!";
    GGEMSMisc::ThrowException("GGEMSSystem", "CheckParameters", oss.str());
  }

  GGEMSNavigator::CheckParameters();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSystem::SaveResults(void)
{
  GGcout("GGEMSSystem", "SaveResults", 2) << "Saving results in MHD format..." << GGendl;

  GGsize3 total_dim;
  total_dim.x = number_of_modules_xy_.x*number_of_detection_elements_inside_module_xyz_.x;
  total_dim.y = number_of_modules_xy_.y*number_of_detection_elements_inside_module_xyz_.y;
  total_dim.z = number_of_detection_elements_inside_module_xyz_.z;

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  GGint* output = new GGint[total_dim.x*total_dim.y*total_dim.z];
  std::memset(output, 0, total_dim.x*total_dim.y*total_dim.z*sizeof(GGint));

  GGEMSMHDImage mhdImage;
  mhdImage.SetBaseName(output_basename_);
  mhdImage.SetDataType("MET_INT");
  mhdImage.SetDimensions(total_dim);
  mhdImage.SetElementSizes(size_of_detection_elements_xyz_);

  // Getting all the counts from solid on OpenCL device
  for (GGsize jj = 0; jj < number_of_modules_xy_.y; ++jj) {
    for (GGsize ii = 0; ii < number_of_modules_xy_.x; ++ii) {
      cl::Buffer* histogram = solids_.at(ii + jj* number_of_modules_xy_.x)->GetHistogram()->histogram_.get();

      GGint* histogram_device = opencl_manager.GetDeviceBuffer<GGint>(histogram, number_of_detection_elements_inside_module_xyz_.x*number_of_detection_elements_inside_module_xyz_.y*sizeof(GGint));

      // Storing data on host
      for (GGsize jjj = 0; jjj < number_of_detection_elements_inside_module_xyz_.y; ++jjj) {
        for (GGsize iii = 0; iii < number_of_detection_elements_inside_module_xyz_.x; ++iii) {
          output[(iii+ii*number_of_detection_elements_inside_module_xyz_.x) + (jjj+jj*number_of_detection_elements_inside_module_xyz_.y)*total_dim.x] =
            histogram_device[iii + jjj*number_of_detection_elements_inside_module_xyz_.x];
        }
      }

      opencl_manager.ReleaseDeviceBuffer(histogram, histogram_device);
    }
  }

  mhdImage.Write<GGint>(output, total_dim.x*total_dim.y*total_dim.z);
  delete[] output;
}
