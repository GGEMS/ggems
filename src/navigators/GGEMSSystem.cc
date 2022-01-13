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
  GGcout("GGEMSSystem", "GGEMSSystem", 3) << "GGEMSSystem creating..." << GGendl;

  number_of_modules_xy_.x_ = 0;
  number_of_modules_xy_.y_ = 0;

  number_of_detection_elements_inside_module_xyz_.x_ = 0;
  number_of_detection_elements_inside_module_xyz_.y_ = 0;
  number_of_detection_elements_inside_module_xyz_.z_ = 0;

  size_of_detection_elements_xyz_.s[0] = 0.0f;
  size_of_detection_elements_xyz_.s[1] = 0.0f;
  size_of_detection_elements_xyz_.s[2] = 0.0f;

  is_scatter_ = false;

  global_system_position_xyz_.s[0] = 0.0f;
  global_system_position_xyz_.s[1] = 0.0f;
  global_system_position_xyz_.s[2] = 0.0f;

  GGcout("GGEMSSystem", "GGEMSSystem", 3) << "GGEMSSystem created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSystem::~GGEMSSystem(void)
{
  GGcout("GGEMSSystem", "~GGEMSSystem", 3) << "GGEMSSystem erasing..." << GGendl;

  GGcout("GGEMSSystem", "~GGEMSSystem", 3) << "GGEMSSystem erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSystem::SetNumberOfModules(GGsize const& n_module_x, GGsize const& n_module_y)
{
  number_of_modules_xy_.x_ = n_module_x;
  number_of_modules_xy_.y_ = n_module_y;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSystem::SetNumberOfDetectionElementsInsideModule(GGsize const& n_detection_element_x, GGsize const& n_detection_element_y, GGsize const& n_detection_element_z)
{
  number_of_detection_elements_inside_module_xyz_.x_ = n_detection_element_x;
  number_of_detection_elements_inside_module_xyz_.y_ = n_detection_element_y;
  number_of_detection_elements_inside_module_xyz_.z_ = n_detection_element_z;
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

void GGEMSSystem::SetMaterialName(std::string const& material_name)
{
  materials_->AddMaterial(material_name);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSystem::SetGlobalSystemPosition(GGfloat const& global_system_position_x, GGfloat const& global_system_position_y, GGfloat const& global_system_position_z, std::string const& unit)
{
  global_system_position_xyz_.s[0] = DistanceUnit(global_system_position_x, unit);
  global_system_position_xyz_.s[1] = DistanceUnit(global_system_position_y, unit);
  global_system_position_xyz_.s[2] = DistanceUnit(global_system_position_z, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSystem::StoreScatter(bool const& is_scatter)
{
  is_scatter_ = is_scatter;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSystem::CheckParameters(void) const
{
  GGcout("GGEMSSystem", "CheckParameters", 3) << "Checking the mandatory parameters..." << GGendl;

  if (number_of_modules_xy_.x_ == 0 || number_of_modules_xy_.y_ == 0) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "In system parameters, number of module in x and y axis (local axis) has to be > 0!!!";
    GGEMSMisc::ThrowException("GGEMSSystem", "CheckParameters", oss.str());
  }

  if (number_of_detection_elements_inside_module_xyz_.x_ == 0 || number_of_detection_elements_inside_module_xyz_.y_ == 0 || number_of_detection_elements_inside_module_xyz_.z_ == 0) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "In system parameters, number of detection elements in x, y and z axis (local axis) has to be > 0!!!";
    GGEMSMisc::ThrowException("GGEMSSystem", "CheckParameters", oss.str());
  }

  if (size_of_detection_elements_xyz_.s[0] == 0.0f || size_of_detection_elements_xyz_.s[1] == 0.0f || size_of_detection_elements_xyz_.s[2] == 0.0f) {
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
  total_dim.x_ = number_of_modules_xy_.x_*number_of_detection_elements_inside_module_xyz_.x_;
  total_dim.y_ = number_of_modules_xy_.y_*number_of_detection_elements_inside_module_xyz_.y_;
  total_dim.z_ = number_of_detection_elements_inside_module_xyz_.z_;

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  GGint* output = new GGint[total_dim.x_*total_dim.y_*total_dim.z_];
  std::memset(output, 0, total_dim.x_*total_dim.y_*total_dim.z_*sizeof(GGint));

  GGEMSMHDImage mhdImage;
  mhdImage.SetOutputFileName(output_basename_);
  mhdImage.SetDataType("MET_INT");
  mhdImage.SetDimensions(total_dim);
  mhdImage.SetElementSizes(size_of_detection_elements_xyz_);

  // Getting all the counts from solid from all OpenCL devices
  for (GGsize i = 0; i < number_activated_devices_; ++i) {
    for (GGsize jj = 0; jj < number_of_modules_xy_.y_; ++jj) {
      for (GGsize ii = 0; ii < number_of_modules_xy_.x_; ++ii) {
        cl::Buffer* histogram = solids_[ii + jj* number_of_modules_xy_.x_]->GetHistogram(i);

        GGint* histogram_device = opencl_manager.GetDeviceBuffer<GGint>(histogram, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, number_of_detection_elements_inside_module_xyz_.x_*number_of_detection_elements_inside_module_xyz_.y_*sizeof(GGint), i);

        // Storing data on host
        for (GGsize jjj = 0; jjj < number_of_detection_elements_inside_module_xyz_.y_; ++jjj) {
          for (GGsize iii = 0; iii < number_of_detection_elements_inside_module_xyz_.x_; ++iii) {
            output[(iii+ii*number_of_detection_elements_inside_module_xyz_.x_) + (jjj+jj*number_of_detection_elements_inside_module_xyz_.y_)*total_dim.x_] +=
              histogram_device[iii + jjj*number_of_detection_elements_inside_module_xyz_.x_];
          }
        }

        opencl_manager.ReleaseDeviceBuffer(histogram, histogram_device, i);
      }
    }
  }

  mhdImage.Write<GGint>(output);

  // Cleaning output buffer
  std::memset(output, 0, total_dim.x_*total_dim.y_*total_dim.z_*sizeof(GGint));

  // If scatter output if necessary
  if (is_scatter_) {
    // From output file add '-scatter' extension
    std::string scatter_output_filename = output_basename_;

    // Checking if there is .mhd suffix
    GGsize found_mhd = output_basename_.find(".mhd");

    if (found_mhd == std::string::npos) { // "add '-scatter.mhd' at the end of file"
      scatter_output_filename += "-scatter.mhd";
    }
    else { // If suffix found, add '-scatter' between end of filename and suffix
      scatter_output_filename = scatter_output_filename.substr(0, found_mhd) + "-scatter.mhd";
    }

    GGEMSMHDImage mhdImageScatter;
    mhdImageScatter.SetOutputFileName(scatter_output_filename);
    mhdImageScatter.SetDataType("MET_INT");
    mhdImageScatter.SetDimensions(total_dim);
    mhdImageScatter.SetElementSizes(size_of_detection_elements_xyz_);

    // Getting all the counts from solid from all OpenCL devices
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      for (GGsize jj = 0; jj < number_of_modules_xy_.y_; ++jj) {
        for (GGsize ii = 0; ii < number_of_modules_xy_.x_; ++ii) {
          cl::Buffer* scatter_histogram = solids_[ii + jj* number_of_modules_xy_.x_]->GetScatterHistogram(i);

          GGint* scatter_histogram_device = opencl_manager.GetDeviceBuffer<GGint>(scatter_histogram, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, number_of_detection_elements_inside_module_xyz_.x_*number_of_detection_elements_inside_module_xyz_.y_*sizeof(GGint), i);

          // Storing data on host
          for (GGsize jjj = 0; jjj < number_of_detection_elements_inside_module_xyz_.y_; ++jjj) {
            for (GGsize iii = 0; iii < number_of_detection_elements_inside_module_xyz_.x_; ++iii) {
              output[(iii+ii*number_of_detection_elements_inside_module_xyz_.x_) + (jjj+jj*number_of_detection_elements_inside_module_xyz_.y_)*total_dim.x_] +=
                scatter_histogram_device[iii + jjj*number_of_detection_elements_inside_module_xyz_.x_];
            }
          }

          opencl_manager.ReleaseDeviceBuffer(scatter_histogram, scatter_histogram_device, i);
        }
      }
    }

    mhdImageScatter.Write<GGint>(output);
  }

  delete[] output;
}
