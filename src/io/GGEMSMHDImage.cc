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
  \file GGEMSMHDImage.cc

  \brief I/O class handling MHD file

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday January 14, 2020
*/

#include <vector>

#include "GGEMS/geometries/GGEMSVoxelizedSolidData.hh"
#include "GGEMS/io/GGEMSMHDImage.hh"
#include "GGEMS/io/GGEMSTextReader.hh"
#include "GGEMS/tools/GGEMSTools.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMHDImage::GGEMSMHDImage(void)
: mhd_header_file_(""),
  mhd_raw_file_(""),
  output_dir_(""),
  mhd_data_type_("MET_FLOAT")
{
  GGcout("GGEMSMHDImage", "GGEMSMHDImage", 3) << "GGEMSMHDImage creating..." << GGendl;


  element_sizes_.x = 0.0f;
  element_sizes_.y = 0.0f;
  element_sizes_.z = 0.0f;

  dimensions_.x_ = 0;
  dimensions_.y_ = 0;
  dimensions_.z_ = 0;

  GGcout("GGEMSMHDImage", "GGEMSMHDImage", 3) << "GGEMSMHDImage created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMHDImage::~GGEMSMHDImage(void)
{
  GGcout("GGEMSMHDImage", "~GGEMSMHDImage", 3) << "GGEMSMHDImage erasing!!!" << GGendl;

  GGcout("GGEMSMHDImage", "~GGEMSMHDImage", 3) << "GGEMSMHDImage erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMHDImage::SetOutputFileName(std::string const& filename)
{
  // Checking *.mhd suffixe
  GGsize found_mhd = filename.find(".mhd");
  if (found_mhd == std::string::npos) {
    mhd_header_file_ += filename + ".mhd";
  }
  else {
    mhd_header_file_ = filename;
  }

  GGsize found_dir = filename.find_last_of("/\\");
  if (found_dir != std::string::npos) {
    output_dir_ = filename.substr(0, found_dir+1);
    mhd_raw_file_ = filename.substr(found_dir+1, found_mhd-found_dir-1) + ".raw";
  }
  else {
    mhd_raw_file_ = filename.substr(0, found_mhd) + ".raw";
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMHDImage::SetElementSizes(GGfloat3 const& element_sizes)
{
  element_sizes_ = element_sizes;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMHDImage::SetDataType(std::string const& data_type)
{
  mhd_data_type_ = data_type;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMHDImage::SetDimensions(GGsize3 const& dimensions)
{
  dimensions_ = dimensions;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMHDImage::Read(std::string const& image_mhd_header_filename, cl::Buffer* solid_data, GGsize const& thread_index)
{
  GGcout("GGEMSMHDImage", "Read", 2) << "Reading MHD Image..." << GGendl;

  // Checking if file exists
  std::ifstream in_header_stream(image_mhd_header_filename, std::ios::in);

  GGEMSFileStream::CheckInputStream(in_header_stream, image_mhd_header_filename);

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device
  GGEMSVoxelizedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSVoxelizedSolidData), thread_index);

  // Getting output directory
  std::size_t found_dir = image_mhd_header_filename.find_last_of("/\\");
  if (found_dir != std::string::npos) {
    output_dir_ = image_mhd_header_filename.substr(0, found_dir+1);
  }

  // Read the file
  std::string line("");
  while (std::getline(in_header_stream, line)) {
    // Skip comment
    GGEMSTextReader::SkipComment(in_header_stream, line);
    // Check if blank line
    if (GGEMSTextReader::IsBlankLine(line)) continue;

    // Getting the key
    std::string const kKey = GGEMSMHDReader::ReadKey(line);

    // Getting the value in string stream
    std::istringstream iss = GGEMSMHDReader::ReadValue(line);

    // Compare key and store data if valid
    if (!kKey.compare("DimSize")) {
      iss >> solid_data_device->number_of_voxels_xyz_.x >> solid_data_device->number_of_voxels_xyz_.y >> solid_data_device->number_of_voxels_xyz_.z;
      // Computing number of voxels
      solid_data_device->number_of_voxels_ = solid_data_device->number_of_voxels_xyz_.x * solid_data_device->number_of_voxels_xyz_.y * solid_data_device->number_of_voxels_xyz_.z;
    }
    else if (!kKey.compare("ElementSpacing")) {
      iss >> solid_data_device->voxel_sizes_xyz_.x >> solid_data_device->voxel_sizes_xyz_.y >> solid_data_device->voxel_sizes_xyz_.z;
    }
    else if (!kKey.compare("ElementType")) {
      iss >> mhd_data_type_;
    }
    else if (!kKey.compare("ElementDataFile")) {
      iss >> mhd_raw_file_;
    }
  }

  // Closing the input header
  in_header_stream.close();

  // Checking the values
  if (solid_data_device->number_of_voxels_xyz_.x <= 0 || solid_data_device->number_of_voxels_xyz_.y <= 0 || solid_data_device->number_of_voxels_xyz_.z <= 0) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Dimension invalid for the key 'DimSize'!!! The values have to be > 0";
    GGEMSMisc::ThrowException("GGEMSMHDImage", "Read", oss.str());
  }

  if (solid_data_device->voxel_sizes_xyz_.x == 0.0 || solid_data_device->voxel_sizes_xyz_.y == 0.0 || solid_data_device->voxel_sizes_xyz_.z == 0.0) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Voxel size invalid for the key 'ElementSpacing'!!! The values have to be > 0";
    GGEMSMisc::ThrowException("GGEMSMHDImage", "Read", oss.str());
  }

  if (mhd_data_type_.empty() && mhd_data_type_.compare("MET_DOUBLE") && mhd_data_type_.compare("MET_FLOAT") && mhd_data_type_.compare("MET_SHORT") && mhd_data_type_.compare("MET_USHORT") && mhd_data_type_.compare("MET_UCHAR") && mhd_data_type_.compare("MET_CHAR") && mhd_data_type_.compare("MET_UINT") && mhd_data_type_.compare("MET_INT")) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Value invalid for the key 'ElementType'!!! The value have to be 'MET_DOUBLE' or 'MET_FLOAT' or 'MET_SHORT' or 'MET_USHORT' or 'MET_UCHAR' or 'MET_CHAR' or 'MET_UINT' or 'MET_INT'";
    GGEMSMisc::ThrowException("GGEMSMHDImage", "Read", oss.str());
  }

  if (mhd_raw_file_.empty()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Value invalid for the key 'ElementDataFile'!!! A filename for raw data has to be given";
    GGEMSMisc::ThrowException("GGEMSMHDImage", "Read", oss.str());
  }

  // Computing bounding box borders automatically at isocenter
  for (GGsize i = 0; i < 3; ++i) {
    solid_data_device->obb_geometry_.border_min_xyz_.s[i] = -static_cast<GGfloat>(solid_data_device->number_of_voxels_xyz_.s[i]) * solid_data_device->voxel_sizes_xyz_.s[i] * 0.5f;
    solid_data_device->obb_geometry_.border_max_xyz_.s[i] = static_cast<GGfloat>(solid_data_device->number_of_voxels_xyz_.s[i]) * solid_data_device->voxel_sizes_xyz_.s[i] * 0.5f;
  }

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(solid_data, solid_data_device, thread_index);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMHDImage::Write(cl::Buffer* image, GGsize const& thread_index) const
{
  GGcout("GGEMSMHDImage", "Write", 1) << "Writing MHD Image: " <<  mhd_header_file_ << "..." << GGendl;

  // Checking parameters before to write
  CheckParameters();

  // header data
  std::ofstream out_header_stream(mhd_header_file_, std::ios::out);
  out_header_stream << "ObjectType = Image" << std::endl;
  out_header_stream << "BinaryDataByteOrderMSB = False" << std::endl;
  out_header_stream << "NDims = 3" << std::endl;
  out_header_stream << "ElementSpacing = " << element_sizes_.x << " " << element_sizes_.y << " " << element_sizes_.z << std::endl;
  out_header_stream << "DimSize = " << dimensions_.x_ << " " << dimensions_.y_ << " " << dimensions_.z_ << std::endl;
  out_header_stream << "ElementType = " << mhd_data_type_ << std::endl;
  out_header_stream << "ElementDataFile = " << mhd_raw_file_ << std::endl;
  out_header_stream.close();

  // Writing raw data to file
  if (!mhd_data_type_.compare("MET_CHAR")) WriteRaw<char>(image, thread_index);
  else if (!mhd_data_type_.compare("MET_UCHAR")) WriteRaw<unsigned char>(image, thread_index);
  else if (!mhd_data_type_.compare("MET_SHORT")) WriteRaw<GGshort>(image, thread_index);
  else if (!mhd_data_type_.compare("MET_USHORT")) WriteRaw<GGushort>(image, thread_index);
  else if (!mhd_data_type_.compare("MET_INT")) WriteRaw<GGint>(image, thread_index);
  else if (!mhd_data_type_.compare("MET_UINT")) WriteRaw<GGuint>(image, thread_index);
  else if (!mhd_data_type_.compare("MET_FLOAT")) WriteRaw<GGfloat>(image, thread_index);
  else if (!mhd_data_type_.compare("MET_DOUBLE")) WriteRaw<GGfloat>(image, thread_index);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMHDImage::CheckParameters(void) const
{
  if (mhd_header_file_.empty()) {
    GGEMSMisc::ThrowException("GGEMSMHDImage", "CheckParameters", "MHD header filename is empty!!!");
  }

  if (mhd_raw_file_.empty()) {
    GGEMSMisc::ThrowException("GGEMSMHDImage", "CheckParameters", "MHD raw filename is empty!!!");
  }

  // Checking phantom dimensions
  if (dimensions_.x_ == 0 && dimensions_.y_ == 0 && dimensions_.z_ == 0) {
    GGEMSMisc::ThrowException("GGEMSMHDImage", "CheckParameters", "Phantom dimensions have to be > 0!!!");
  }

  // Checking size of voxels
  if (element_sizes_.x == 0.0f && element_sizes_.y == 0.0f && element_sizes_.z == 0.0f) {
    GGEMSMisc::ThrowException("GGEMSMHDImage", "CheckParameters", "Phantom voxel sizes have to be > 0.0!!!");
  }
}
