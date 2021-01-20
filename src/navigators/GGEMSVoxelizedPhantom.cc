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
#include "GGEMS/navigators/GGEMSDoseParams.hh"
#include "GGEMS/geometries/GGEMSVoxelizedSolid.hh"
#include "GGEMS/global/GGEMSManager.hh"
#include "GGEMS/io/GGEMSMHDImage.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVoxelizedPhantom::GGEMSVoxelizedPhantom(std::string const& voxelized_phantom_name)
: GGEMSNavigator(voxelized_phantom_name),
  voxelized_phantom_filename_(""),
  range_data_filename_(""),
  is_photon_tracking_(false)
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

void GGEMSVoxelizedPhantom::SetPhotonTracking(bool const& is_activated)
{
  is_photon_tracking_ = is_activated;
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
  if (is_dosimetry_mode_) {
    solids_.emplace_back(new GGEMSVoxelizedSolid(voxelized_phantom_filename_, range_data_filename_, "DOSIMETRY"));
  }
  else {
    solids_.emplace_back(new GGEMSVoxelizedSolid(voxelized_phantom_filename_, range_data_filename_));
  }

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

  // Initialize parent class
  GGEMSNavigator::Initialize();

  // Checking if dosimetry mode activated
  if (is_dosimetry_mode_) {
    dose_calculator_.reset(new GGEMSDosimetryCalculator());
    dose_calculator_->SetOutputDosimetryFilename(dosimetry_output_filename);
    dose_calculator_->SetDoselSizes(dosel_sizes_);
    dose_calculator_->SetNavigator(navigator_name_);
    dose_calculator_->Initialize();
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedPhantom::SaveResults(void)
{
  if (is_dosimetry_mode_) {
    GGcout("GGEMSSystem", "GGEMSVoxelizedPhantom", 2) << "Saving dosimetry results in MHD format..." << GGendl;

    // Get the OpenCL manager
    GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

    // Get pointer on OpenCL device for dose parameters
    GGEMSDoseParams* dose_params_device = opencl_manager.GetDeviceBuffer<GGEMSDoseParams>(dose_calculator_->GetDoseParams().get(), sizeof(GGEMSDoseParams));

    // Params
    GGint total_number_of_dosels = dose_params_device->total_number_of_dosels_;
    GGint3 number_of_dosels = dose_params_device->number_of_dosels_;
    GGfloat3 size_of_dosels = dose_params_device->size_of_dosels_;

    if(is_photon_tracking_) {
      GGint* photon_tracking = new GGint[total_number_of_dosels];
      std::memset(photon_tracking, 0, total_number_of_dosels*sizeof(GGint));

      GGEMSMHDImage mhdImage;
      mhdImage.SetBaseName(dosimetry_output_filename + "_photon_tracking");
      mhdImage.SetDataType("MET_INT");
      mhdImage.SetDimensions(number_of_dosels);
      mhdImage.SetElementSizes(size_of_dosels);

      cl::Buffer* photon_tracking_dosimetry_cl = dose_calculator_->GetPhotonTrackingBuffer().get();
      GGint* photon_tracking_device = opencl_manager.GetDeviceBuffer<GGint>(photon_tracking_dosimetry_cl, total_number_of_dosels*sizeof(GGint));

      for (GGint i = 0; i < total_number_of_dosels; ++i) {
        photon_tracking[i] = photon_tracking_device[i];
      }

      // Writing data
      mhdImage.Write<GGint>(photon_tracking, total_number_of_dosels);
      opencl_manager.ReleaseDeviceBuffer(photon_tracking_dosimetry_cl, photon_tracking_device);
      delete[] photon_tracking;
    }

    // Release the pointer
    opencl_manager.ReleaseDeviceBuffer(dose_calculator_->GetDoseParams().get(), dose_params_device);
  }

  // GGint3 total_dim = {
  //   number_of_modules_xy_.s0*number_of_detection_elements_inside_module_xyz_.s0,
  //   number_of_modules_xy_.s1*number_of_detection_elements_inside_module_xyz_.s1,
  //   number_of_detection_elements_inside_module_xyz_.s2
  // };

  // GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  // GGint* output = new GGint[total_dim.s0*total_dim.s1*total_dim.s2];
  // std::memset(output, 0, total_dim.s0*total_dim.s1*total_dim.s2*sizeof(GGint));

  // GGEMSMHDImage mhdImage;
  // mhdImage.SetBaseName(output_basename_);
  // mhdImage.SetDataType("MET_INT");
  // mhdImage.SetDimensions(total_dim);
  // mhdImage.SetElementSizes(size_of_detection_elements_xyz_);

  // // Getting all the counts from solid on OpenCL device
  // for (std::size_t jj = 0; jj < number_of_modules_xy_.s1; ++jj) {
  //   for (std::size_t ii = 0; ii < number_of_modules_xy_.s0; ++ii) {
  //     cl::Buffer* histogram_cl = solids_.at(ii + jj* number_of_modules_xy_.s0)->GetHistogram()->histogram_cl_.get();

  //     GGint* histogram_device = opencl_manager.GetDeviceBuffer<GGint>(histogram_cl, number_of_detection_elements_inside_module_xyz_.s0*number_of_detection_elements_inside_module_xyz_.s1*sizeof(GGint));

  //     // Storing data on host
  //     for (GGint jjj = 0; jjj < number_of_detection_elements_inside_module_xyz_.s1; ++jjj) {
  //       for (GGint iii = 0; iii < number_of_detection_elements_inside_module_xyz_.s0; ++iii) {
  //         output[(iii+ii*number_of_detection_elements_inside_module_xyz_.s0) + (jjj+jj*number_of_detection_elements_inside_module_xyz_.s1)*total_dim.s0] =
  //           histogram_device[iii + jjj*number_of_detection_elements_inside_module_xyz_.s0];
  //       }
  //     }

  //     opencl_manager.ReleaseDeviceBuffer(histogram_cl, histogram_device);
  //   }
  // }

  // mhdImage.Write<GGint>(output, total_dim.s0*total_dim.s1*total_dim.s2);
  // delete[] output;
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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void dose_photon_tracking_voxelized_phantom(GGEMSVoxelizedPhantom* voxelized_phantom, bool const is_activated)
{
  voxelized_phantom->SetPhotonTracking(is_activated);
}
