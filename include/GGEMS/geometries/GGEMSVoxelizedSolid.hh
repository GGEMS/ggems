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

#include "GGEMS/global/GGEMSExport.hh"

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
      \brief GGEMSVoxelizedSolid constructor
    */
    GGEMSVoxelizedSolid(std::string const& volume_header_filename, std::string const& range_filename);

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
      \fn void Initialize(std::weak_ptr<GGEMSMaterials> materials)
      \param materials - pointer on materials
      \brief Initialize solid for geometric navigation
    */
    void Initialize(std::weak_ptr<GGEMSMaterials> materials) override;

    /*!
      \fn void PrintInfos(void) const
      \brief printing infos about voxelized solid
    */
    void PrintInfos(void) const override;

    /*!
      \fn void LoadVolumeImage(std::weak_ptr<GGEMSMaterials> materials)
      \param materials - pointer on material for a phantom
      \brief load volume image to GGEMS and create a volume of label in GGEMS for voxelized solid
    */
    void LoadVolumeImage(std::weak_ptr<GGEMSMaterials> materials);

    /*!
      \fn void GetTransformationMatrix(void)
      \brief Get the transformation matrix for voxelized solid object
    */
    void GetTransformationMatrix(void) override;

  private:
    /*!
      \fn template <typename T> void ConvertImageToLabel(std::string const& raw_data_filename, std::string const& range_data_filename, std::weak_ptr<GGEMSMaterials> materials)
      \tparam T - type of data
      \param raw_data_filename - raw data filename from mhd
      \param range_data_filename - name of the file containing the range to material data
      \param materials - pointer on material for a phantom
      \brief convert image data to label data
    */
    template <typename T>
    void ConvertImageToLabel(std::string const& raw_data_filename, std::string const& range_data_filename, std::weak_ptr<GGEMSMaterials> materials);

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
void GGEMSVoxelizedSolid::ConvertImageToLabel(std::string const& raw_data_filename, std::string const& range_data_filename, std::weak_ptr<GGEMSMaterials> materials)
{
  GGcout("GGEMSVoxelizedSolid", "ConvertImageToLabel", 3) << "Converting image material data to label data..." << GGendl;

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Checking if file exists
  std::ifstream in_raw_stream(raw_data_filename, std::ios::in | std::ios::binary);
  GGEMSFileStream::CheckInputStream(in_raw_stream, raw_data_filename);

  // Get pointer on OpenCL device
  GGEMSVoxelizedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data_cl_.get(), sizeof(GGEMSVoxelizedSolidData));

  // Get information about mhd file
  GGint const kNumberOfVoxels = solid_data_device->number_of_voxels_;

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(solid_data_cl_.get(), solid_data_device);

  // Reading data to a tmp buffer
  std::vector<T> tmp_raw_data;
  tmp_raw_data.resize(kNumberOfVoxels);
  in_raw_stream.read(reinterpret_cast<char*>(&tmp_raw_data[0]), kNumberOfVoxels * sizeof(T));

  // Closing file
  in_raw_stream.close();

  // Allocating memory on OpenCL device
  label_data_cl_ = opencl_manager.Allocate(nullptr, kNumberOfVoxels * sizeof(GGshort), CL_MEM_READ_WRITE);

  // Get pointer on OpenCL device
  GGshort* label_data_device = opencl_manager.GetDeviceBuffer<GGshort>(label_data_cl_.get(), kNumberOfVoxels * sizeof(GGshort));

  // Set value to max of GGshort
  std::fill(label_data_device, label_data_device + kNumberOfVoxels, std::numeric_limits<GGshort>::max());

  // Opening range data file
  std::ifstream in_range_stream(range_data_filename, std::ios::in);
  GGEMSFileStream::CheckInputStream(in_range_stream, range_data_filename);

  // Values in the range file
  GGfloat first_label_value = 0.0f;
  GGfloat last_label_value = 0.0f;
  GGshort label_index = 0;
  std::string material_name("");

  // Reading range file
  std::string line("");
  while (std::getline(in_range_stream, line)) {
    // Check if blank line
    if (GGEMSTextReader::IsBlankLine(line)) continue;

    // Getting the value in string stream
    std::istringstream iss = GGEMSRangeReader::ReadRangeMaterial(line);
    iss >> first_label_value >> last_label_value >> material_name;

    // Adding the material
    materials.lock()->AddMaterial(material_name);

    // Setting the label
    for (GGint i = 0; i < kNumberOfVoxels; ++i) {
      // Getting the value of phantom
      GGfloat const kValue = static_cast<GGfloat>(tmp_raw_data[i]);
      if (((kValue == first_label_value) && (kValue == last_label_value)) || ((kValue >= first_label_value) && (kValue < last_label_value))) {
        label_data_device[i] = label_index;
      }
    }

    // Increment the label index
    ++label_index;
  }

  // Final loop checking if a value is still max of GGuchar
  bool all_converted = true;
  for (GGint i = 0; i < kNumberOfVoxels; ++i) {
    if (label_data_device[i] == std::numeric_limits<GGuchar>::max()) all_converted = false;
  }

  // Closing file
  in_range_stream.close();
  tmp_raw_data.clear();

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(label_data_cl_.get(), label_data_device);

  // Checking if all voxels converted
  if (all_converted) {
    GGcout("GGEMSVoxelizedSolid", "ConvertImageToLabel", 0) << "All your voxels are converted to label..." << GGendl;
  }
  else {
    GGEMSMisc::ThrowException("GGEMSVoxelizedSolid", "ConvertImageToLabel", "Errors(s) in the range data file!!!");
  }
}

#endif // End of GUARD_GGEMS_GEOMETRIES_GGEMSVOXELIZEDSOLID_HH
