#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSVOXELIZEDSOLID_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSVOXELIZEDSOLID_HH

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
  \file GGEMSVoxelizedSolid.hh

  \brief GGEMS class for voxelized solid

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday June 10, 2020
*/

#include "GGEMS/geometries/GGEMSVoxelizedSolidData.hh"
#include "GGEMS/geometries/GGEMSSolid.hh"

/*!
  \class GGEMSVoxelizedSolid
  \brief GGEMS class for voxelized solid
*/
class GGEMS_EXPORT GGEMSVoxelizedSolid : public GGEMSSolid
{
  public:
    /*!
      \param volume_header_filename - header file for volume
      \param range_filename - file with range value
      \param data_reg_type - type of registration for voxelized solid
      \brief GGEMSVoxelizedSolid constructor
    */
    GGEMSVoxelizedSolid(std::string const& volume_header_filename, std::string const& range_filename, std::string const& data_reg_type = "");

    /*!
      \brief GGEMSVoxelizedSolid destructor
    */
    ~GGEMSVoxelizedSolid(void);

    /*!
      \fn GGEMSVoxelizedSolid(GGEMSVoxelizedSolid const& voxelized_solid) = delete
      \param voxelized_solid - reference on the GGEMS voxelized solid
      \brief Avoid copy by reference
    */
    GGEMSVoxelizedSolid(GGEMSVoxelizedSolid const& voxelized_solid) = delete;

    /*!
      \fn GGEMSVoxelizedSolid& operator=(GGEMSVoxelizedSolid const& voxelized_solid) = delete
      \param voxelized_solid - reference on the GGEMS voxelized solid
      \brief Avoid assignement by reference
    */
    GGEMSVoxelizedSolid& operator=(GGEMSVoxelizedSolid const& voxelized_solid) = delete;

    /*!
      \fn GGEMSVoxelizedSolid(GGEMSVoxelizedSolid const&& voxelized_solid) = delete
      \param voxelized_solid - rvalue reference on the GGEMS voxelized solid
      \brief Avoid copy by rvalue reference
    */
    GGEMSVoxelizedSolid(GGEMSVoxelizedSolid const&& voxelized_solid) = delete;

    /*!
      \fn GGEMSVoxelizedSolid& operator=(GGEMSVoxelizedSolid const&& voxelized_solid) = delete
      \param voxelized_solid - rvalue reference on the GGEMS voxelized solid
      \brief Avoid copy by rvalue reference
    */
    GGEMSVoxelizedSolid& operator=(GGEMSVoxelizedSolid const&& voxelized_solid) = delete;

    /*!
      \fn void Initialize(GGEMSMaterials* materials)
      \param materials - pointer on materials
      \brief Initialize solid for geometric navigation
    */
    void Initialize(GGEMSMaterials* materials) override;

    /*!
      \fn void EnableScatter(void)
      \brief Activate scatter registration
    */
    void EnableScatter(void) override {;};

    /*!
      \fn void PrintInfos(void) const
      \brief printing infos about voxelized solid
    */
    void PrintInfos(void) const override;

    /*!
      \fn void LoadVolumeImage(GGEMSMaterials* materials)
      \param materials - pointer on material for a phantom
      \brief load volume image to GGEMS and create a volume of label in GGEMS for voxelized solid
    */
    void LoadVolumeImage(GGEMSMaterials* materials);

    /*!
      \fn void UpdateTransformationMatrix(GGsize const& thread_index)
      \param thread_index - index of the thread (= activated device index)
      \brief Update transformation matrix for solid box object
    */
    void UpdateTransformationMatrix(GGsize const& thread_index) override;

    /*!
      \fn GGfloat3 GetVoxelSizes(GGsize const& thread_index) const
      \param thread_index - index of the thread (= activated device index)
      \return size of voxels in voxelized solid
      \brief get the size of voxels in voxelized solid
    */
    GGfloat3 GetVoxelSizes(GGsize const& thread_index) const;

    /*!
      \fn GGEMSOBB GetOBBGeometry(GGsize const& thread_index) const
      \param thread_index - index of the thread (= activated device index)
      \return OBB params for the object
      \brief return the parameters about OBB geometry
    */
    GGEMSOBB GetOBBGeometry(GGsize const& thread_index) const;

  private:
    /*!
      \fn template <typename T> void ConvertImageToLabel(std::string const& raw_data_filename, std::string const& range_data_filename, GGEMSMaterials* materials)
      \tparam T - type of data
      \param raw_data_filename - raw data filename from mhd
      \param range_data_filename - name of the file containing the range to material data
      \param materials - pointer on material for a phantom
      \brief convert image data to label data
    */
    template <typename T>
    void ConvertImageToLabel(std::string const& raw_data_filename, std::string const& range_data_filename, GGEMSMaterials* materials);

    /*!
      \fn void InitializeKernel(void)
      \brief Initialize kernel for particle solid distance
    */
    void InitializeKernel(void) override;

  private:
    std::string volume_header_filename_; /*!< Filename of MHD file for phantom */
    std::string range_filename_; /*!< Filename of file for range data */
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
void GGEMSVoxelizedSolid::ConvertImageToLabel(std::string const& raw_data_filename, std::string const& range_data_filename, GGEMSMaterials* materials)
{
  GGcout("GGEMSVoxelizedSolid", "ConvertImageToLabel", 3) << "Converting image material data to label data..." << GGendl;

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  for (GGsize d = 0; d < number_activated_devices_; ++d) {
    // Get pointer on OpenCL device
    GGEMSVoxelizedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data_[d], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSVoxelizedSolidData), d);

    // Get information about mhd file
    number_of_voxels_ = static_cast<GGsize>(solid_data_device->number_of_voxels_);

    // Release the pointer
    opencl_manager.ReleaseDeviceBuffer(solid_data_[d], solid_data_device, d);

    // Checking if file exists
    std::ifstream in_raw_stream(raw_data_filename, std::ios::in | std::ios::binary);
    GGEMSFileStream::CheckInputStream(in_raw_stream, raw_data_filename);

    // Reading data to a tmp buffer
    std::vector<T> tmp_raw_data;
    tmp_raw_data.resize(number_of_voxels_);
    in_raw_stream.read(reinterpret_cast<char*>(&tmp_raw_data[0]), static_cast<std::streamsize>(number_of_voxels_ * sizeof(T)));

    // Closing file
    in_raw_stream.close();

    // Allocating memory on OpenCL device
    label_data_[d] = opencl_manager.Allocate(nullptr, number_of_voxels_ * sizeof(GGuchar), d, CL_MEM_READ_WRITE, "GGEMSVoxelizedSolid");

    // Get pointer on OpenCL device
    GGuchar* label_data_device = opencl_manager.GetDeviceBuffer<GGuchar>(label_data_[d], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, number_of_voxels_ * sizeof(GGuchar), d);

    // Set value to max of GGuchar
    std::fill(label_data_device, label_data_device + number_of_voxels_, std::numeric_limits<GGuchar>::max());

    // Opening range data file
    std::ifstream in_range_stream(range_data_filename, std::ios::in);
    GGEMSFileStream::CheckInputStream(in_range_stream, range_data_filename);

    // Values in the range file
    GGfloat first_label_value = 0.0f;
    GGfloat last_label_value = 0.0f;
    GGuchar label_index = 0;
    std::string material_name("");

    // Reading range file
    std::string line("");
    while (std::getline(in_range_stream, line)) {
      // Check if blank line
      if (GGEMSTextReader::IsBlankLine(line)) continue;

      // Getting the value in string stream
      std::istringstream iss = GGEMSRangeReader::ReadRangeMaterial(line);
      iss >> first_label_value >> last_label_value >> material_name;

      // Adding the material only once
      if (d == 0) materials->AddMaterial(material_name);

      // Setting the label
      for (GGsize i = 0; i < number_of_voxels_; ++i) {
        // Getting the value of phantom
        GGfloat value = static_cast<GGfloat>(tmp_raw_data[i]);
        if (((value == first_label_value) && (value == last_label_value)) || ((value >= first_label_value) && (value < last_label_value))) {
          label_data_device[i] = label_index;
        }
      }

      // Increment the label index
      ++label_index;
    }

    // Final loop checking if a value is still max of GGuchar
    bool all_converted = true;
    for (GGsize i = 0; i < number_of_voxels_; ++i) {
      if (label_data_device[i] == std::numeric_limits<GGuchar>::max()) all_converted = false;
    }

    // Closing file
    in_range_stream.close();
    tmp_raw_data.clear();

    // Release the pointer
    opencl_manager.ReleaseDeviceBuffer(label_data_[d], label_data_device, d);

    // Checking if all voxels converted
    if (all_converted) {
      GGcout("GGEMSVoxelizedSolid", "ConvertImageToLabel", 2) << "All your voxels are converted to label..." << GGendl;
    }
    else {
      GGEMSMisc::ThrowException("GGEMSVoxelizedSolid", "ConvertImageToLabel", "Errors(s) in the range data file!!!");
    }
  }
}

#endif // End of GUARD_GGEMS_GEOMETRIES_GGEMSVOXELIZEDSOLID_HH
