/*!
  \file GGEMSMHDImage.hh

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
  element_sizes_(GGdouble3{{0.0, 0.0, 0.0}}),
  dimensions_(GGuint3{{0, 0, 0}}),
  offsets_(GGdouble3{{0.0, 0.0, 0.0}}),
  opencl_manager_(GGEMSOpenCLManager::GetInstance())
{
  GGcout("GGEMSMHDImage", "GGEMSMHDImage", 3) << "Allocation of GGEMSMHDImage..." << GGendl;

  // Initialization of the header data structure
  mhd_header_data_.reset(new GGEMSMHDHeaderData);
  mhd_header_data_->object_type_ = "Image";
  mhd_header_data_->number_of_voxels_xyz_.s[0] = 0;
  mhd_header_data_->number_of_voxels_xyz_.s[1] = 0;
  mhd_header_data_->number_of_voxels_xyz_.s[2] = 0;
  mhd_header_data_->number_of_voxels_ = 0;
  mhd_header_data_->voxel_sizes_xyz_.s[0] = 0.0;
  mhd_header_data_->voxel_sizes_xyz_.s[1] = 0.0;
  mhd_header_data_->voxel_sizes_xyz_.s[2] = 0.0;
  mhd_header_data_->offsets_xyz_.s[0] = 0.0;
  mhd_header_data_->offsets_xyz_.s[1] = 0.0;
  mhd_header_data_->offsets_xyz_.s[2] = 0.0;
  mhd_header_data_->border_min_xyz_.s[0] = 0.0;
  mhd_header_data_->border_min_xyz_.s[1] = 0.0;
  mhd_header_data_->border_min_xyz_.s[2] = 0.0;
  mhd_header_data_->border_max_xyz_.s[0] = 0.0;
  mhd_header_data_->border_max_xyz_.s[1] = 0.0;
  mhd_header_data_->border_max_xyz_.s[2] = 0.0;
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

void GGEMSMHDImage::SetElementSizes(GGdouble3 const& element_sizes)
{
  element_sizes_ = element_sizes;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMHDImage::SetOffsets(GGdouble3 const& offsets)
{
  offsets_ = offsets;
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

void GGEMSMHDImage::Read(std::string const& image_mhd_header_filename)
{
  GGcout("GGEMSMHDImage", "Read", 0) << "Reading MHD Image..." << GGendl;

  // Checking if file exists
  std::ifstream in_header_stream(image_mhd_header_filename, std::ios::in);
  GGEMSFileStream::CheckInputStream(in_header_stream, image_mhd_header_filename);

  // Strings for type and raw data
  std::string data_type("");
  std::string raw_data_filename("");

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
      iss >> mhd_header_data_->object_type_;

    }
    else if (!kKey.compare("DimSize")) {
      iss >> mhd_header_data_->number_of_voxels_xyz_.s[0] >> mhd_header_data_->number_of_voxels_xyz_.s[1] >> mhd_header_data_->number_of_voxels_xyz_.s[2];
      // Computing number of voxels
      mhd_header_data_->number_of_voxels_ = mhd_header_data_->number_of_voxels_xyz_.s[0] * mhd_header_data_->number_of_voxels_xyz_.s[1] * mhd_header_data_->number_of_voxels_xyz_.s[2];
    }
    else if (!kKey.compare("Offset")) {
      iss >> mhd_header_data_->offsets_xyz_.s[0] >> mhd_header_data_->offsets_xyz_.s[1] >> mhd_header_data_->offsets_xyz_.s[2];
    }
    else if (!kKey.compare("ElementSpacing")) {
      iss >> mhd_header_data_->voxel_sizes_xyz_.s[0] >> mhd_header_data_->voxel_sizes_xyz_.s[1] >> mhd_header_data_->voxel_sizes_xyz_.s[2];
    }
    else if (!kKey.compare("ElementType")) {
      iss >> data_type;
    }
    else if (!kKey.compare("ElementDataFile")) {
      iss >> raw_data_filename;
    }
    else if (!kKey.compare("NDims")) {
      GGwarn("GGEMSMHDImage", "Read", 1) << "The key 'NDims' is useless in GGEMS." << GGendl;
    }
    else if (!kKey.compare("BinaryData")) {
      GGwarn("GGEMSMHDImage", "Read", 1) << "The key 'BinaryData' is useless in GGEMS." << GGendl;
    }
    else if (!kKey.compare("CompressedData")) {
      GGwarn("GGEMSMHDImage", "Read", 1) << "The key 'CompressedData' is useless in GGEMS." << GGendl;
    }
    else if (!kKey.compare("BinaryDataByteOrderMSB")) {
      GGwarn("GGEMSMHDImage", "Read", 1) << "The key 'BinaryDataByteOrderMSB' is useless in GGEMS." << GGendl;
    }
    else if (!kKey.compare("BinaryDataByteOrderMSB")) {
      GGwarn("GGEMSMHDImage", "Read", 1) << "The key 'BinaryDataByteOrderMSB' is useless in GGEMS." << GGendl;
    }
    else {
      std::ostringstream oss(std::ostringstream::out);
      oss << "Unknown MHD key: " << kKey << "'!!!";
      GGEMSMisc::ThrowException("GGEMSMHDImage", "Read", oss.str());
    }
  }

  // Checking the values
  if (mhd_header_data_->object_type_.compare("Image")) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Value invalid for the key 'ObjectType'!!! The value have to be 'Image'";
    GGEMSMisc::ThrowException("GGEMSMHDImage", "Read", oss.str());
  }

  if (mhd_header_data_->number_of_voxels_xyz_.s[0] <= 0 || mhd_header_data_->number_of_voxels_xyz_.s[1] <= 0 || mhd_header_data_->number_of_voxels_xyz_.s[2] <= 0) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Dimension invalid for the key 'DimSize'!!! The values have to be > 0";
    GGEMSMisc::ThrowException("GGEMSMHDImage", "Read", oss.str());
  }

  if (GGEMSMisc::IsEqual(mhd_header_data_->voxel_sizes_xyz_.s[0], 0.0) || GGEMSMisc::IsEqual(mhd_header_data_->voxel_sizes_xyz_.s[1], 0.0) || GGEMSMisc::IsEqual(mhd_header_data_->voxel_sizes_xyz_.s[2], 0.0)) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Voxel size invalid for the key 'ElementSpacing'!!! The values have to be > 0";
    GGEMSMisc::ThrowException("GGEMSMHDImage", "Read", oss.str());
  }

  if (data_type.empty() && data_type.compare("MET_FLOAT") && data_type.compare("MET_SHORT") && data_type.compare("MET_USHORT") && data_type.compare("MET_UCHAR") && data_type.compare("MET_CHAR") && data_type.compare("MET_UINT") && data_type.compare("MET_INT")) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Value invalid for the key 'ElementType'!!! The value have to be 'MET_FLOAT' or 'MET_SHORT' or 'MET_USHORT' or 'MET_UCHAR' or 'MET_CHAR' or 'MET_UINT' or 'MET_INT'";
    GGEMSMisc::ThrowException("GGEMSMHDImage", "Read", oss.str());
  }

  if (raw_data_filename.empty()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Value invalid for the key 'ElementDataFile'!!! A filename for raw data has to be given";
    GGEMSMisc::ThrowException("GGEMSMHDImage", "Read", oss.str());
  }

  // Computing border for bounding box

  GGcout("GGEMSMHDImage", "Read", 1) << "Header of the MHD Input Image..." << GGendl;
  GGcout("GGEMSMHDImage", "Read", 1) << "    *Object type: " << mhd_header_data_->object_type_ << GGendl;
  GGcout("GGEMSMHDImage", "Read", 1) << "    *Dimension: " << mhd_header_data_->number_of_voxels_xyz_.s[0] << " " << mhd_header_data_->number_of_voxels_xyz_.s[1] << " " << mhd_header_data_->number_of_voxels_xyz_.s[2] << GGendl;
  GGcout("GGEMSMHDImage", "Read", 1) << "    *Number of voxels: " << mhd_header_data_->number_of_voxels_ << GGendl;
  GGcout("GGEMSMHDImage", "Read", 1) << "    *Size of voxels: (" << mhd_header_data_->voxel_sizes_xyz_.s[0] << "x" << mhd_header_data_->voxel_sizes_xyz_.s[1] << "x" << mhd_header_data_->voxel_sizes_xyz_.s[2] << ") mm3" << GGendl;
  GGcout("GGEMSMHDImage", "Read", 1) << "    *Offset: (" << mhd_header_data_->offsets_xyz_.s[0] << "x" << mhd_header_data_->offsets_xyz_.s[1] << "x" << mhd_header_data_->offsets_xyz_.s[2] << ") mm3" << GGendl;
  GGcout("GGEMSMHDImage", "Read", 1) << "    *Type: " << data_type << GGendl;
  GGcout("GGEMSMHDImage", "Read", 1) << "    *Raw filename: " << raw_data_filename << GGendl;

  // Closing the input header
  in_header_stream.close();
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
  out_header_stream << "ObjectType = Image" << std::endl;
  out_header_stream << "NDims = 3" << std::endl;
  out_header_stream << "BinaryData = True" << std::endl;
  out_header_stream << "CompressedData = False" << std::endl;
  out_header_stream << "Offset = " << offsets_.s[0] << " " << offsets_.s[1] << " " << offsets_.s[2] << std::endl;
  out_header_stream << "ElementSpacing = " << element_sizes_.s[0] << " " << element_sizes_.s[1] << " " << element_sizes_.s[2] << std::endl;
  out_header_stream << "DimSize = " << dimensions_.s[0] << " " << dimensions_.s[1] << " " << dimensions_.s[2] << std::endl;
  out_header_stream << "ElementType = MET_FLOAT" << std::endl;
  out_header_stream << "ElementDataFile = " << mhd_raw_file_ << std::endl;
  out_header_stream.close();

  // raw data
  std::ofstream out_raw_stream(mhd_raw_file_, std::ios::out | std::ios::binary);

  // Mapping data
  GGfloat* data_image = opencl_manager_.GetDeviceBuffer<GGfloat>(image, dimensions_.s[0] * dimensions_.s[1] * dimensions_.s[2] * sizeof(GGfloat));

  // Writing data on file
  out_raw_stream.write(reinterpret_cast<char*>(data_image), dimensions_.s[0] * dimensions_.s[1] * dimensions_.s[2] * sizeof(GGfloat));

  // Release the pointers
  opencl_manager_.ReleaseDeviceBuffer(image, data_image);
  out_raw_stream.close();
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
  if (GGEMSMisc::IsEqual(element_sizes_.s[0], 0.0) && GGEMSMisc::IsEqual(element_sizes_.s[1], 0.0) && GGEMSMisc::IsEqual(element_sizes_.s[2], 0.0)) {
    GGEMSMisc::ThrowException("GGEMSMHDImage", "CheckParameters", "Phantom voxel sizes have to be > 0.0!!!");
  }
}
