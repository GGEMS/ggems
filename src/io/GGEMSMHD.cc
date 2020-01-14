/*!
  \file GGEMSMHD.hh

  \brief I/O class handling MHD file

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday January 14, 2020
*/

#include <fstream>

#include "GGEMS/io/GGEMSMHD.hh"
#include "GGEMS/tools/GGEMSTools.hh"
#include "GGEMS/tools/GGEMSPrint.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMHD::GGEMSMHD(void)
: mhd_header_file_(""),
  mhd_raw_file_(""),
  element_sizes_(GGdouble3{0.0, 0.0, 0.0}),
  dimensions_(GGuint3{0, 0, 0})
{
  GGcout("GGEMSMHD", "GGEMSMHD", 3)
    << "Allocation of GGEMSMHD..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMHD::~GGEMSMHD(void)
{
  GGcout("GGEMSMHD", "~GGEMSMHD", 3)
    << "Deallocation of GGEMSMHD..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMHD::SetBaseName(std::string const& basename)
{
  mhd_header_file_ = basename + ".mhd";
  mhd_raw_file_ = basename + ".raw";
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMHD::SetElementSizes(GGdouble3 const& element_sizes)
{
  element_sizes_ = element_sizes;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMHD::SetDimensions(GGuint3 const& dimensions)
{
  dimensions_ = dimensions;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMHD::Write(void)
{
  // Checking parameters before to write
  CheckParameters();

  // header
  std::ofstream outHeaderStream(mhd_header_file_, std::ios::out);

  outHeaderStream << "ObjectType = Image" << std::endl;
  outHeaderStream << "NDims = 3" << std::endl;
  outHeaderStream << "BinaryData = True" << std::endl;
  outHeaderStream << "BinaryDataByteOrderMSB = False" << std::endl;
  outHeaderStream << "CompressedData = False" << std::endl;
  outHeaderStream << "TransformMatrix = 1 0 0 0 1 0 0 0 1" << std::endl;
  outHeaderStream << "CenterOfRotation = 0 0 0" << std::endl;
  outHeaderStream << "ElementSpacing = " << element_sizes_.s[0] << " "
    << element_sizes_.s[1] << " " << element_sizes_.s[2] << std::endl;
  outHeaderStream << "DimSize = " << dimensions_.s[0] << " " << dimensions_.s[1]
    << " " << dimensions_.s[2] << std::endl;
  outHeaderStream << "ElementDataFile = " << mhd_raw_file_ << std::endl;

  outHeaderStream.close();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMHD::CheckParameters(void) const
{
  if (mhd_header_file_.empty()) {
    GGEMSMisc::ThrowException("GGEMSMHD", "CheckParameters",
      "MHD header filename is empty!!!");
  }

  if (mhd_raw_file_.empty()) {
    GGEMSMisc::ThrowException("GGEMSMHD", "CheckParameters",
      "MHD raw filename is empty!!!");
  }

  // Checking phantom dimensions
  if (dimensions_.s[0] == 0 && dimensions_.s[1] == 0 && dimensions_.s[2] == 0) {
    GGEMSMisc::ThrowException("GGEMSMHD", "CheckParameters",
      "Phantom dimensions have to be > 0!!!");
  }

  // Checking size of voxels
  if (GGEMSMisc::IsEqual(element_sizes_.s[0], 0.0) &&
    GGEMSMisc::IsEqual(element_sizes_.s[1], 0.0) &&
    GGEMSMisc::IsEqual(element_sizes_.s[2], 0.0)) {
    GGEMSMisc::ThrowException("GGEMSMHD", "CheckParameters",
      "Phantom voxel sizes have to be > 0.0!!!");
    }
}
