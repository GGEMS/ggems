/*!
  \file GGEMSPhantomNavigator.cc

  \brief GGEMS mother class for phantom navigation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday February 11, 2020
*/

#include "GGEMS/navigators/GGEMSPhantomNavigatorManager.hh"

#include "GGEMS/navigators/GGEMSSolidPhantom.hh"
#include "GGEMS/physics/GGEMSRangeCuts.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPhantomNavigator::GGEMSPhantomNavigator(GGEMSPhantomNavigator* phantom_navigator)
: phantom_navigator_name_(""),
  phantom_mhd_header_filename_(""),
  range_data_filename_(""),
  geometry_tolerance_(GGEMSTolerance::GEOMETRY),
  offset_xyz_(MakeFloat3Zeros()),
  is_offset_flag_(false)
{
  GGcout("GGEMSPhantomNavigator", "GGEMSPhantomNavigator", 3) << "Allocation of GGEMSPhantomNavigator..." << GGendl;

  // Store the phantom navigator in phantom navigator manager
  GGEMSPhantomNavigatorManager::GetInstance().Store(phantom_navigator);

  // Allocation of materials
  materials_.reset(new GGEMSMaterials());

  // Allocation of cuts
  range_cuts_.reset(new GGEMSRangeCuts());

  // Allocation of solid phantom
  solid_phantom_.reset(new GGEMSSolidPhantom());
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPhantomNavigator::~GGEMSPhantomNavigator(void)
{
  GGcout("GGEMSPhantomNavigator", "~GGEMSPhantomNavigator", 3) << "Deallocation of GGEMSPhantomNavigator..." << GGendl;
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

void GGEMSPhantomNavigator::SetRangeToMaterialFile(char const* range_data_filename)
{
  range_data_filename_ = range_data_filename;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPhantomNavigator::SetGeometryTolerance(GGfloat const& distance, char const* unit)
{
  geometry_tolerance_ = GGEMSUnits::DistanceUnit(distance, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPhantomNavigator::SetOffset(GGfloat const offset_x, GGfloat const offset_y, GGfloat const offset_z, char const* unit)
{
  offset_xyz_.s[0] = GGEMSUnits::DistanceUnit(offset_x, unit);
  offset_xyz_.s[1] = GGEMSUnits::DistanceUnit(offset_y, unit);
  offset_xyz_.s[2] = GGEMSUnits::DistanceUnit(offset_z, unit);
  is_offset_flag_ = true;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPhantomNavigator::CheckParameters(void) const
{
  GGcout("GGEMSPhantomNavigator", "CheckParameters", 3) << "Checking the mandatory parameters..." << GGendl;

  // Checking the phantom name
  if (phantom_navigator_name_.empty()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a name for the phantom!!!";
    GGEMSMisc::ThrowException("GGEMSPhantomNavigator", "CheckParameters", oss.str());
  }

  // Checking the phantom name
  if (phantom_mhd_header_filename_.empty()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a mhd file containing the phantom!!!";
    GGEMSMisc::ThrowException("GGEMSPhantomNavigator", "CheckParameters", oss.str());
  }

  // Checking the phantom name
  if (range_data_filename_.empty()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a file with the range to material data!!!";
    GGEMSMisc::ThrowException("GGEMSPhantomNavigator", "CheckParameters", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPhantomNavigator::Initialize(void)
{
  GGcout("GGEMSPhantomNavigator", "Initialize", 3) << "Initializing a GGEMS phantom..." << GGendl;

  // Checking the parameters of phantom
  CheckParameters();

  // Loading the phantom and convert image to material data, and adding material to GGEMS
  solid_phantom_->LoadPhantomImage(phantom_mhd_header_filename_, range_data_filename_, materials_);

  // Apply offset
  if (is_offset_flag_) solid_phantom_->ApplyOffset(offset_xyz_);

  // Converting length cut to energy cut
  //range_cuts_->ConvertCuts(materials_);

  // Loading the materials to OpenCL device and converting cuts
  materials_->Initialize();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPhantomNavigator::PrintInfos(void) const
{
  GGcout("GGEMSPhantomNavigator", "PrintInfos", 0) << GGendl;
  GGcout("GGEMSPhantomNavigator", "PrintInfos", 0) << "GGEMSPhantomNavigator Infos: " << GGendl;
  GGcout("GGEMSPhantomNavigator", "PrintInfos", 0) << "--------------------------------------------" << GGendl;
  solid_phantom_->PrintInfos();
  materials_->PrintInfos();
  GGcout("GGEMSPhantomNavigator", "PrintInfos", 0) << GGendl;
}
