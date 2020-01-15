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

#include "GGEMS/io/GGEMSMHDImage.hh"
#include "GGEMS/tools/GGEMSTools.hh"
#include "GGEMS/tools/GGEMSPrint.hh"

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
  GGcout("GGEMSMHDImage", "GGEMSMHDImage", 3)
    << "Allocation of GGEMSMHDImage..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMHDImage::~GGEMSMHDImage(void)
{
  GGcout("GGEMSMHDImage", "~GGEMSMHDImage", 3)
    << "Deallocation of GGEMSMHDImage..." << GGendl;
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

void GGEMSMHDImage::Write(cl::Buffer* p_image) const
{
  GGcout("GGEMSMHDImage", "~GGEMSMHDImage", 0)
    << "Writing MHD Image..." << GGendl;

  // Checking parameters before to write
  CheckParameters();

  // header data
  std::ofstream outHeaderStream(mhd_header_file_, std::ios::out);
  outHeaderStream << "ObjectType = Image" << std::endl;
  outHeaderStream << "NDims = 3" << std::endl;
  outHeaderStream << "BinaryData = True" << std::endl;
  outHeaderStream << "CompressedData = False" << std::endl;
  outHeaderStream << "Offset = " << offsets_.s[0] << " " << offsets_.s[1]
    << " " << offsets_.s[2] << std::endl;
  outHeaderStream << "ElementSpacing = " << element_sizes_.s[0] << " "
    << element_sizes_.s[1] << " " << element_sizes_.s[2] << std::endl;
  outHeaderStream << "DimSize = " << dimensions_.s[0] << " " << dimensions_.s[1]
    << " " << dimensions_.s[2] << std::endl;
  outHeaderStream << "ElementType = MET_FLOAT" << std::endl;
  outHeaderStream << "ElementDataFile = " << mhd_raw_file_ << std::endl;
  outHeaderStream.close();

  // raw data
  std::ofstream outRawStream(mhd_raw_file_, std::ios::out | std::ios::binary);

  // Mapping data
  GGfloat* p_data_image = opencl_manager_.GetDeviceBuffer<GGfloat>(
    p_image,
    dimensions_.s[0] * dimensions_.s[1] * dimensions_.s[2] * sizeof(GGfloat)
  );

  // Writing data on file
  outRawStream.write(reinterpret_cast<char*>(p_data_image),
    dimensions_.s[0] * dimensions_.s[1] * dimensions_.s[2] * sizeof(GGfloat));

  // Release the pointers
  opencl_manager_.ReleaseDeviceBuffer(p_image, p_data_image);
  outRawStream.close();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMHDImage::CheckParameters(void) const
{
  if (mhd_header_file_.empty()) {
    GGEMSMisc::ThrowException("GGEMSMHDImage", "CheckParameters",
      "MHD header filename is empty!!!");
  }

  if (mhd_raw_file_.empty()) {
    GGEMSMisc::ThrowException("GGEMSMHDImage", "CheckParameters",
      "MHD raw filename is empty!!!");
  }

  // Checking phantom dimensions
  if (dimensions_.s[0] == 0 && dimensions_.s[1] == 0 && dimensions_.s[2] == 0) {
    GGEMSMisc::ThrowException("GGEMSMHDImage", "CheckParameters",
      "Phantom dimensions have to be > 0!!!");
  }

  // Checking size of voxels
  if (GGEMSMisc::IsEqual(element_sizes_.s[0], 0.0) &&
    GGEMSMisc::IsEqual(element_sizes_.s[1], 0.0) &&
    GGEMSMisc::IsEqual(element_sizes_.s[2], 0.0)) {
    GGEMSMisc::ThrowException("GGEMSMHDImage", "CheckParameters",
      "Phantom voxel sizes have to be > 0.0!!!");
  }
}
