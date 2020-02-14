/*!
  \file GGEMSPhantomNavigator.cc

  \brief GGEMS mother class for phantom navigation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday February 11, 2020
*/

#include <sstream>

#include "GGEMS/geometries/GGEMSPhantomNavigator.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"
#include "GGEMS/tools/GGEMSTools.hh"
#include "GGEMS/global/GGEMSConstants.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPhantomNavigator::GGEMSPhantomNavigator(void)
: phantom_navigator_name_(""),
  phantom_mhd_header_filename_(""),
  range_data_filename_(""),
  geometry_tolerance_(GGEMSTolerance::GEOMETRY)
{
  GGcout("GGEMSPhantomNavigator", "GGEMSPhantomNavigator", 3)
    << "Allocation of GGEMSPhantomNavigator..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPhantomNavigator::~GGEMSPhantomNavigator(void)
{
  GGcout("GGEMSPhantomNavigator", "~GGEMSPhantomNavigator", 3)
    << "Deallocation of GGEMSPhantomNavigator..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPhantomNavigator::SetPhantomName(char const* phantom_navigator_name)
{
  phantom_navigator_name_ = phantom_navigator_name;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPhantomNavigator::SetPhantomFile(char const* phantom_filename)
{
  phantom_mhd_header_filename_ = phantom_filename;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPhantomNavigator::SetRangeToMaterialFile(
  char const* range_data_filename)
{
  range_data_filename_ = range_data_filename;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPhantomNavigator::SetGeometryTolerance(GGdouble const& distance,
  char const* unit)
{
  // Convert char* to string
  std::string const kUnit = unit;

  if (kUnit == "nm") {
    geometry_tolerance_ = distance * GGEMSUnits::nm;
  }
  else if (kUnit == "um") {
    geometry_tolerance_ = distance * GGEMSUnits::um;
  }
  else if (kUnit == "mm") {
    geometry_tolerance_ = distance * GGEMSUnits::mm;
  }
  else if (kUnit == "cm") {
    geometry_tolerance_ = distance * GGEMSUnits::cm;
  }
  else {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Unknown unit!!! You have choice between:" << std::endl;
    oss << "    - \"nm\": nanometer" << std::endl;
    oss << "    - \"um\": micrometer" << std::endl;
    oss << "    - \"mm\": millimeter" << std::endl;
    oss << "    - \"cm\": centimeter";
    GGEMSMisc::ThrowException("GGEMSPhantomNavigator", "SetGeometryTolerance",
      oss.str());
  }
}
