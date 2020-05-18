/*!
  \file GGEMSNavigator.cc

  \brief GGEMS mother class for navigation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday February 11, 2020
*/

#include "GGEMS/navigators/GGEMSNavigatorManager.hh"
#include "GGEMS/geometries/GGEMSSolid.hh"
#include "GGEMS/physics/GGEMSCrossSections.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSNavigator::GGEMSNavigator(GGEMSNavigator* navigator)
: navigator_name_(""),
  phantom_mhd_header_filename_(""),
  range_data_filename_(""),
  geometry_tolerance_(GGEMSTolerance::GEOMETRY),
  offset_xyz_(MakeFloat3Zeros()),
  is_offset_flag_(false)
{
  GGcout("GGEMSNavigator", "GGEMSNavigator", 3) << "Allocation of GGEMSNavigator..." << GGendl;

  // Store the phantom navigator in phantom navigator manager
  GGEMSNavigatorManager::GetInstance().Store(navigator);

  // Allocation of materials
  materials_.reset(new GGEMSMaterials());

  // Allocation of solid phantom
  solid_.reset(new GGEMSSolid());

  // Allocation of cross sections including physics
  cross_sections_.reset(new GGEMSCrossSections());
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSNavigator::~GGEMSNavigator(void)
{
  GGcout("GGEMSNavigator", "~GGEMSNavigator", 3) << "Deallocation of GGEMSNavigator..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::SetNavigatorName(std::string const& navigator_name)
{
  navigator_name_ = navigator_name;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::SetPhantomFile(std::string const& phantom_filename)
{
  phantom_mhd_header_filename_ = phantom_filename;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::SetRangeToMaterialFile(std::string const& range_data_filename)
{
  range_data_filename_ = range_data_filename;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::SetGeometryTolerance(GGfloat const& distance, std::string const& unit)
{
  geometry_tolerance_ = GGEMSUnits::DistanceUnit(distance, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::SetOffset(GGfloat const offset_x, GGfloat const offset_y, GGfloat const offset_z, std::string const& unit)
{
  offset_xyz_.s[0] = GGEMSUnits::DistanceUnit(offset_x, unit);
  offset_xyz_.s[1] = GGEMSUnits::DistanceUnit(offset_y, unit);
  offset_xyz_.s[2] = GGEMSUnits::DistanceUnit(offset_z, unit);
  is_offset_flag_ = true;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::CheckParameters(void) const
{
  GGcout("GGEMSNavigator", "CheckParameters", 3) << "Checking the mandatory parameters..." << GGendl;

  // Checking the phantom name
  if (navigator_name_.empty()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a name for the phantom!!!";
    GGEMSMisc::ThrowException("GGEMSNavigator", "CheckParameters", oss.str());
  }

  // Checking the phantom name
  if (phantom_mhd_header_filename_.empty()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a mhd file containing the phantom!!!";
    GGEMSMisc::ThrowException("GGEMSNavigator", "CheckParameters", oss.str());
  }

  // Checking the phantom name
  if (range_data_filename_.empty()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a file with the range to material data!!!";
    GGEMSMisc::ThrowException("GGEMSNavigator", "CheckParameters", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::Initialize(void)
{
  GGcout("GGEMSNavigator", "Initialize", 3) << "Initializing a GGEMS phantom..." << GGendl;

  // Checking the parameters of phantom
  CheckParameters();

  // Loading the phantom and convert image to material data, and adding material to GGEMS
  solid_->LoadPhantomImage(phantom_mhd_header_filename_, range_data_filename_, materials_);

  // Apply offset
  if (is_offset_flag_) solid_->ApplyOffset(offset_xyz_);

  // Loading the materials and building tables to OpenCL device and converting cuts
  materials_->Initialize();

  // Initialization of electromagnetic process and building cross section tables for each particles and materials
  cross_sections_->Initialize(materials_.get());
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::ComputeParticleNavigatorDistance(void) const
{
  // Compute the distance between particles and navigator using the solid information
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::PrintInfos(void) const
{
  GGcout("GGEMSNavigator", "PrintInfos", 0) << GGendl;
  GGcout("GGEMSNavigator", "PrintInfos", 0) << "GGEMSNavigator Infos: " << GGendl;
  GGcout("GGEMSNavigator", "PrintInfos", 0) << "--------------------------------------------" << GGendl;
  solid_->PrintInfos();
  materials_->PrintInfos();
  GGcout("GGEMSNavigator", "PrintInfos", 0) << GGendl;
}
