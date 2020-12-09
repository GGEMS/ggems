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
#include "GGEMS/geometries/GGEMSSolid.hh"
#include "GGEMS/io/GGEMSMHDImage.hh"
#include "GGEMS/io/GGEMSHitCollection.hh"
#include "GGEMS/navigators/GGEMSNavigatorManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSystem::GGEMSSystem(std::string const& system_name)
: GGEMSNavigator(system_name),
  number_of_modules_xy_({0, 0}),
  number_of_detection_elements_inside_module_xyz_({0, 0, 0}),
  size_of_detection_elements_xyz_({0.0f, 0.0f, 0.0f})
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

void GGEMSSystem::SetNumberOfModules(GGint const& n_module_x, GGint const& n_module_y)
{
  number_of_modules_xy_.s0 = n_module_x;
  number_of_modules_xy_.s1 = n_module_y;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSystem::SetNumberOfDetectionElementsInsideModule(GGint const& n_detection_element_x, GGint const& n_detection_element_y, GGint const& n_detection_element_z)
{
  number_of_detection_elements_inside_module_xyz_.s0 = n_detection_element_x;
  number_of_detection_elements_inside_module_xyz_.s1 = n_detection_element_y;
  number_of_detection_elements_inside_module_xyz_.s2 = n_detection_element_z;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSystem::SetSizeOfDetectionElements(GGfloat const& detection_element_x, GGfloat const& detection_element_y, GGfloat const& detection_element_z, std::string const& unit)
{
  size_of_detection_elements_xyz_.s0 = DistanceUnit(detection_element_x, unit);
  size_of_detection_elements_xyz_.s1 = DistanceUnit(detection_element_y, unit);
  size_of_detection_elements_xyz_.s2 = DistanceUnit(detection_element_z, unit);
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

  if (number_of_modules_xy_.s0 == 0 || number_of_modules_xy_.s1 == 0) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "In system parameters, number of module in x and y axis (local axis) has to be > 0!!!";
    GGEMSMisc::ThrowException("GGEMSSystem", "CheckParameters", oss.str());
  }

  if (number_of_detection_elements_inside_module_xyz_.s0 == 0 || number_of_detection_elements_inside_module_xyz_.s1 == 0 || number_of_detection_elements_inside_module_xyz_.s2 == 0) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "In system parameters, number of detection elements in x, y and z axis (local axis) has to be > 0!!!";
    GGEMSMisc::ThrowException("GGEMSSystem", "CheckParameters", oss.str());
  }

  if (size_of_detection_elements_xyz_.s0 == 0.0f || size_of_detection_elements_xyz_.s1 == 0.0f || size_of_detection_elements_xyz_.s2 == 0.0f) {
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

  GGint3 total_dim = {
    number_of_modules_xy_.s0*number_of_detection_elements_inside_module_xyz_.s0,
    number_of_modules_xy_.s1*number_of_detection_elements_inside_module_xyz_.s1,
    number_of_detection_elements_inside_module_xyz_.s2
  };

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  GGint* output = new GGint[total_dim.s0*total_dim.s1*total_dim.s2];
  std::memset(output, 0, total_dim.s0*total_dim.s1*total_dim.s2*sizeof(GGint));

  GGEMSMHDImage mhdImage;
  mhdImage.SetBaseName(output_basename_);
  mhdImage.SetDataType("MET_INT");
  mhdImage.SetDimensions(total_dim);
  mhdImage.SetElementSizes(size_of_detection_elements_xyz_);

  // Getting all the counts from solid on OpenCL device
  for (std::size_t jj = 0; jj < number_of_modules_xy_.s1; ++jj) {
    for (std::size_t ii = 0; ii < number_of_modules_xy_.s0; ++ii) {
      cl::Buffer* hit = solids_.at(ii + jj* number_of_modules_xy_.s0)->GetHitCollection()->hit_cl_.get();

      GGint* hit_device = opencl_manager.GetDeviceBuffer<GGint>(hit, number_of_detection_elements_inside_module_xyz_.s0*number_of_detection_elements_inside_module_xyz_.s1*sizeof(GGint));

      // Storing data on host
      for (GGint jjj = 0; jjj < number_of_detection_elements_inside_module_xyz_.s1; ++jjj) {
        for (GGint iii = 0; iii < number_of_detection_elements_inside_module_xyz_.s0; ++iii) {
          output[(iii+ii*number_of_detection_elements_inside_module_xyz_.s0) + (jjj+jj*number_of_detection_elements_inside_module_xyz_.s1)*total_dim.s0] =
            hit_device[iii + jjj*number_of_detection_elements_inside_module_xyz_.s0];
        }
      }

      opencl_manager.ReleaseDeviceBuffer(hit, hit_device);
    }
  }

  mhdImage.Write<GGint>(output, total_dim.s0*total_dim.s1*total_dim.s2);
  delete[] output;
}
