/*!
  \file GGEMSVoxelizedPhantomNavigatorImagery.cc

  \brief GGEMS class managing voxelized phantom navigator for imagery application

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday February 11, 2020
*/

#include "GGEMS/navigators/GGEMSNavigatorManager.hh"
#include "GGEMS/navigators/GGEMSVoxelizedPhantomNavigatorImagery.hh"
#include "GGEMS/geometries/GGEMSSolid.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVoxelizedPhantomNavigatorImagery::GGEMSVoxelizedPhantomNavigatorImagery(void)
: GGEMSNavigator(this)
{
  GGcout("GGEMSVoxelizedPhantomNavigatorImagery", "GGEMSVoxelizedPhantomNavigatorImagery", 3) << "Allocation of GGEMSVoxelizedPhantomNavigatorImagery..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVoxelizedPhantomNavigatorImagery::~GGEMSVoxelizedPhantomNavigatorImagery(void)
{
  GGcout("GGEMSVoxelizedPhantomNavigatorImagery", "~GGEMSVoxelizedPhantomNavigatorImagery", 3) << "Deallocation of GGEMSVoxelizedPhantomNavigatorImagery..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedPhantomNavigatorImagery::Initialize(void)
{
  GGcout("GGEMSVoxelizedPhantomNavigatorImagery", "Initialize", 3) << "Initializing the voxelized phantom navigator..." << GGendl;

  // Initialize the phantom navigator
  GGEMSNavigator::Initialize();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSVoxelizedPhantomNavigatorImagery::PrintInfos(void) const
{
  // Infos about GGEMSVoxelizedPhantomNavigatorImagery
  GGcout("GGEMSVoxelizedPhantomNavigatorImagery", "PrintInfos", 0) << GGendl;
  GGcout("GGEMSVoxelizedPhantomNavigatorImagery", "PrintInfos", 0) << "GGEMSVoxelizedPhantomNavigatorImagery Infos: " << GGendl;
  GGcout("GGEMSVoxelizedPhantomNavigatorImagery", "PrintInfos", 0) << "--------------------------------------------" << GGendl;
  GGcout("GGEMSVoxelizedPhantomNavigatorImagery", "PrintInfos", 0) << "*Phantom navigator name: " << navigator_name_ << GGendl;
  GGcout("GGEMSVoxelizedPhantomNavigatorImagery", "PrintInfos", 0) << "*Phantom header filename: " << phantom_mhd_header_filename_ << GGendl;
  GGcout("GGEMSVoxelizedPhantomNavigatorImagery", "PrintInfos", 0) << "*Range label to material filename: " << range_data_filename_ << GGendl;
  GGcout("GGEMSVoxelizedPhantomNavigatorImagery", "PrintInfos", 0) << "*Geometry tolerance: " << geometry_tolerance_/GGEMSUnits::mm << " mm" << GGendl;
  GGcout("GGEMSVoxelizedPhantomNavigatorImagery", "PrintInfos", 0) << GGendl;

  // Printing infos about GGEMSNavigator
  GGEMSNavigator::PrintInfos();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSVoxelizedPhantomNavigatorImagery* create_ggems_voxelized_phantom_navigator_imagery(void)
{
  return new(std::nothrow) GGEMSVoxelizedPhantomNavigatorImagery;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_phantom_name_ggems_voxelized_phantom_navigator_imagery(GGEMSVoxelizedPhantomNavigatorImagery* voxelized_phantom_navigator_imagery, char const* phantom_navigator_name)
{
  voxelized_phantom_navigator_imagery->SetNavigatorName(phantom_navigator_name);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_phantom_file_ggems_voxelized_phantom_navigator_imagery(GGEMSVoxelizedPhantomNavigatorImagery* voxelized_phantom_navigator_imagery, char const* phantom_filename)
{
  voxelized_phantom_navigator_imagery->SetPhantomFile(phantom_filename);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_range_to_material_filename_ggems_voxelized_phantom_navigator_imagery(GGEMSVoxelizedPhantomNavigatorImagery* voxelized_phantom_navigator_imagery, char const* range_data_filename)
{
  voxelized_phantom_navigator_imagery->SetRangeToMaterialFile(range_data_filename);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_geometry_tolerance_ggems_voxelized_phantom_navigator_imagery(GGEMSVoxelizedPhantomNavigatorImagery* voxelized_phantom_navigator_imagery, GGfloat const distance, char const* unit)
{
  voxelized_phantom_navigator_imagery->SetGeometryTolerance(distance, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_offset_ggems_voxelized_phantom_navigator_imagery(GGEMSVoxelizedPhantomNavigatorImagery* voxelized_phantom_navigator_imagery, GGfloat const offset_x, GGfloat const offset_y, GGfloat const offset_z, char const* unit)
{
  voxelized_phantom_navigator_imagery->SetOffset(offset_x, offset_y, offset_z, unit);
}
