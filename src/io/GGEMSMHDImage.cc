/*!
  \file GGEMSMHDImage.cc

  \brief I/O class handling MHD file

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday January 14, 2020
*/

#include <fstream>
#include <vector>
#include <sstream>

#include "GGEMS/geometries/GGEMSVoxelizedSolidStack.hh"
#include "GGEMS/io/GGEMSMHDImage.hh"
#include "GGEMS/tools/GGEMSTools.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/io/GGEMSTextReader.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMHDImage::GGEMSMHDImage(void)
: mhd_header_file_(""),
  mhd_raw_file_(""),
  mhd_data_type_("MET_FLOAT"),
  element_sizes_(GGfloat3{{0.0, 0.0, 0.0}}),
  dimensions_(GGuint3{{0, 0, 0}})
{
  GGcout("GGEMSMHDImage", "GGEMSMHDImage", 3) << "Allocation of GGEMSMHDImage..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMHDImage::~GGEMSMHDImage(void)
{
  GGcout("GGEMSMHDImage", "~GGEMSMHDImage", 3) << "Deallocation of GGEMSMHDImage..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMHDImage::SetBaseName(std::string const& basename)
{
  mhd_header_file_ = basename + ".mhd";
  mhd_raw_file_ = basename + ".raw";
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

void GGEMSMHDImage::SetDimensions(GGuint3 const& dimensions)
{
  dimensions_ = dimensions;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMHDImage::Read(std::string const& image_mhd_header_filename, std::shared_ptr<cl::Buffer> solid_data)
{
  GGcout("GGEMSMHDImage", "Read", 0) << "Reading MHD Image..." << GGendl;

  // Checking if file exists
  std::ifstream in_header_stream(image_mhd_header_filename, std::ios::in);
  GGEMSFileStream::CheckInputStream(in_header_stream, image_mhd_header_filename);

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device
  GGEMSVoxelizedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data, sizeof(GGEMSVoxelizedSolidData));

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
    if (!kKey.compare("ObjectType")) {
      GGwarn("GGEMSMHDImage", "Read", 0) << "The key 'ObjectType' is useless in GGEMS." << GGendl;
    }
    else if (!kKey.compare("DimSize")) {
      iss >> solid_data_device->number_of_voxels_xyz_.s[0] >> solid_data_device->number_of_voxels_xyz_.s[1] >> solid_data_device->number_of_voxels_xyz_.s[2];
      // Computing number of voxels
      solid_data_device->number_of_voxels_ = solid_data_device->number_of_voxels_xyz_.s[0] * solid_data_device->number_of_voxels_xyz_.s[1] * solid_data_device->number_of_voxels_xyz_.s[2];
    }
    else if (!kKey.compare("ElementSpacing")) {
      iss >> solid_data_device->voxel_sizes_xyz_.s[0] >> solid_data_device->voxel_sizes_xyz_.s[1] >> solid_data_device->voxel_sizes_xyz_.s[2];
    }
    else if (!kKey.compare("ElementType")) {
      iss >> mhd_data_type_;
    }
    else if (!kKey.compare("ElementDataFile")) {
      iss >> mhd_raw_file_;
    }
    else if (!kKey.compare("Offset")) {
      GGwarn("GGEMSMHDImage", "Read", 0) << "The key 'Offset' is useless in GGEMS." << GGendl;
    }
    else if (!kKey.compare("NDims")) {
      GGwarn("GGEMSMHDImage", "Read", 0) << "The key 'NDims' is useless in GGEMS." << GGendl;
    }
    else if (!kKey.compare("BinaryData")) {
      GGwarn("GGEMSMHDImage", "Read", 0) << "The key 'BinaryData' is useless in GGEMS." << GGendl;
    }
    else if (!kKey.compare("CompressedData")) {
      GGwarn("GGEMSMHDImage", "Read", 0) << "The key 'CompressedData' is useless in GGEMS." << GGendl;
    }
    else if (!kKey.compare("BinaryDataByteOrderMSB")) {
      GGwarn("GGEMSMHDImage", "Read", 0) << "The key 'BinaryDataByteOrderMSB' is useless in GGEMS." << GGendl;
    }
    else if (!kKey.compare("BinaryDataByteOrderMSB")) {
      GGwarn("GGEMSMHDImage", "Read", 0) << "The key 'BinaryDataByteOrderMSB' is useless in GGEMS." << GGendl;
    }
    else {
      std::ostringstream oss(std::ostringstream::out);
      oss << "Unknown MHD key: " << kKey << "'!!!";
      GGEMSMisc::ThrowException("GGEMSMHDImage", "Read", oss.str());
    }
  }

  // Closing the input header
  in_header_stream.close();

  // Checking the values
  if (solid_data_device->number_of_voxels_xyz_.s[0] <= 0 || solid_data_device->number_of_voxels_xyz_.s[1] <= 0 || solid_data_device->number_of_voxels_xyz_.s[2] <= 0) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Dimension invalid for the key 'DimSize'!!! The values have to be > 0";
    GGEMSMisc::ThrowException("GGEMSMHDImage", "Read", oss.str());
  }

  if (GGEMSMisc::IsEqual(solid_data_device->voxel_sizes_xyz_.s[0], 0.0) || GGEMSMisc::IsEqual(solid_data_device->voxel_sizes_xyz_.s[1], 0.0) || GGEMSMisc::IsEqual(solid_data_device->voxel_sizes_xyz_.s[2], 0.0)) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Voxel size invalid for the key 'ElementSpacing'!!! The values have to be > 0";
    GGEMSMisc::ThrowException("GGEMSMHDImage", "Read", oss.str());
  }

  if (mhd_data_type_.empty() && mhd_data_type_.compare("MET_FLOAT") && mhd_data_type_.compare("MET_SHORT") && mhd_data_type_.compare("MET_USHORT") && mhd_data_type_.compare("MET_UCHAR") && mhd_data_type_.compare("MET_CHAR") && mhd_data_type_.compare("MET_UINT") && mhd_data_type_.compare("MET_INT")) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Value invalid for the key 'ElementType'!!! The value have to be 'MET_FLOAT' or 'MET_SHORT' or 'MET_USHORT' or 'MET_UCHAR' or 'MET_CHAR' or 'MET_UINT' or 'MET_INT'";
    GGEMSMisc::ThrowException("GGEMSMHDImage", "Read", oss.str());
  }

  if (mhd_raw_file_.empty()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Value invalid for the key 'ElementDataFile'!!! A filename for raw data has to be given";
    GGEMSMisc::ThrowException("GGEMSMHDImage", "Read", oss.str());
  }

  // Computing the offset and bounding box
  for (GGuint i = 0; i < 3; ++i) {
    solid_data_device->position_xyz_.s[i] = solid_data_device->number_of_voxels_xyz_.s[i] * solid_data_device->voxel_sizes_xyz_.s[i] * 0.5;

    solid_data_device->border_min_xyz_.s[i] = -solid_data_device->position_xyz_.s[i];
    solid_data_device->border_max_xyz_.s[i] = solid_data_device->position_xyz_.s[i];
  }

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(solid_data, solid_data_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMHDImage::Write(std::shared_ptr<cl::Buffer> image) const
{
  GGcout("GGEMSMHDImage", "Write", 0) << "Writing MHD Image..." << GGendl;

  // Checking parameters before to write
  CheckParameters();

  // header data
  std::ofstream out_header_stream(mhd_header_file_, std::ios::out);
  out_header_stream << "ElementSpacing = " << element_sizes_.s[0] << " " << element_sizes_.s[1] << " " << element_sizes_.s[2] << std::endl;
  out_header_stream << "DimSize = " << dimensions_.s[0] << " " << dimensions_.s[1] << " " << dimensions_.s[2] << std::endl;
  out_header_stream << "ElementType = " << mhd_data_type_ << std::endl;
  out_header_stream << "ElementDataFile = " << mhd_raw_file_ << std::endl;
  out_header_stream.close();

  // Writing raw data to file
  if (!mhd_data_type_.compare("MET_CHAR")) WriteRaw<char>(image);
  else if (!mhd_data_type_.compare("MET_UCHAR")) WriteRaw<unsigned char>(image);
  else if (!mhd_data_type_.compare("MET_SHORT")) WriteRaw<GGshort>(image);
  else if (!mhd_data_type_.compare("MET_USHORT")) WriteRaw<GGushort>(image);
  else if (!mhd_data_type_.compare("MET_INT")) WriteRaw<GGint>(image);
  else if (!mhd_data_type_.compare("MET_UINT")) WriteRaw<GGuint>(image);
  else if (!mhd_data_type_.compare("MET_FLOAT")) WriteRaw<GGfloat>(image);
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
  if (dimensions_.s[0] == 0 && dimensions_.s[1] == 0 && dimensions_.s[2] == 0) {
    GGEMSMisc::ThrowException("GGEMSMHDImage", "CheckParameters", "Phantom dimensions have to be > 0!!!");
  }

  // Checking size of voxels
  if (GGEMSMisc::IsEqual(element_sizes_.s[0], 0.0f) && GGEMSMisc::IsEqual(element_sizes_.s[1], 0.0f) && GGEMSMisc::IsEqual(element_sizes_.s[2], 0.0f)) {
    GGEMSMisc::ThrowException("GGEMSMHDImage", "CheckParameters", "Phantom voxel sizes have to be > 0.0!!!");
  }
}
