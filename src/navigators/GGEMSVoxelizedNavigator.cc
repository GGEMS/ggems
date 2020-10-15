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
  \file GGEMSVoxelizedNavigator.cc

  \brief GGEMS class managing voxelized phantom navigator for imagery application

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday February 11, 2020
*/

// #include "GGEMS/navigators/GGEMSNavigatorManager.hh"
// #include "GGEMS/navigators/GGEMSVoxelizedNavigator.hh"
// #include "GGEMS/geometries/GGEMSSolid.hh"
// #include "GGEMS/geometries/GGEMSVoxelizedSolid.hh"

// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////

// GGEMSVoxelizedNavigator::GGEMSVoxelizedNavigator(void)
// : GGEMSNavigator(),
//   phantom_mhd_header_filename_(""),
//   range_data_filename_("")
// {
//   GGcout("GGEMSVoxelizedNavigator", "GGEMSVoxelizedNavigator", 3) << "Allocation of GGEMSVoxelizedNavigator..." << GGendl;
// }

// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////

// GGEMSVoxelizedNavigator::~GGEMSVoxelizedNavigator(void)
// {
//   GGcout("GGEMSVoxelizedNavigator", "~GGEMSVoxelizedNavigator", 3) << "Deallocation of GGEMSVoxelizedNavigator..." << GGendl;
// }

// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////

// void GGEMSVoxelizedNavigator::SetPhantomFile(std::string const& phantom_filename)
// {
//   phantom_mhd_header_filename_ = phantom_filename;
// }

// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////

// void GGEMSVoxelizedNavigator::SetRangeToMaterialFile(std::string const& range_data_filename)
// {
//   range_data_filename_ = range_data_filename;
// }

// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////

// void GGEMSVoxelizedNavigator::CheckParameters(void) const
// {
//   GGcout("GGEMSVoxelizedNavigator", "CheckParameters", 3) << "Checking the mandatory parameters..." << GGendl;

//   // Checking the phantom name
//   if (phantom_mhd_header_filename_.empty()) {
//     std::ostringstream oss(std::ostringstream::out);
//     oss << "You have to set a mhd file containing the phantom!!!";
//     GGEMSMisc::ThrowException("GGEMSVoxelizedNavigator", "CheckParameters", oss.str());
//   }

//   // Checking the phantom name
//   if (range_data_filename_.empty()) {
//     std::ostringstream oss(std::ostringstream::out);
//     oss << "You have to set a file with the range to material data!!!";
//     GGEMSMisc::ThrowException("GGEMSVoxelizedNavigator", "CheckParameters", oss.str());
//   }
// }

// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////

// void GGEMSVoxelizedNavigator::Initialize(void)
// {
//   GGcout("GGEMSVoxelizedNavigator", "Initialize", 3) << "Initializing the voxelized phantom navigator..." << GGendl;

//   // Checking parameters
//   CheckParameters();

//   // Building specific solid for navigator. Here voxelized solid
//   solid_.reset(new GGEMSVoxelizedSolid(phantom_mhd_header_filename_, range_data_filename_));

//   // Initialize the phantom navigator
//   GGEMSNavigator::Initialize();
// }

// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////

// void GGEMSVoxelizedNavigator::PrintInfos(void) const
// {
//   // Printing infos about GGEMSNavigator
//   GGEMSNavigator::PrintInfos();

//   // Infos about GGEMSVoxelizedNavigator
//   GGcout("GGEMSVoxelizedNavigator", "PrintInfos", 0) << GGendl;
//   GGcout("GGEMSVoxelizedNavigator", "PrintInfos", 0) << "GGEMSVoxelizedNavigator Infos:" << GGendl;
//   GGcout("GGEMSVoxelizedNavigator", "PrintInfos", 0) << "------------------------------" << GGendl;
//   GGcout("GGEMSVoxelizedNavigator", "PrintInfos", 0) << "*Phantom header filename: " << phantom_mhd_header_filename_ << GGendl;
//   GGcout("GGEMSVoxelizedNavigator", "PrintInfos", 0) << "*Range label to material filename: " << range_data_filename_ << GGendl;
//   GGcout("GGEMSVoxelizedNavigator", "PrintInfos", 0) << GGendl;
// }

// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////

// GGEMSVoxelizedNavigator* create_ggems_voxelized_navigator(void)
// {
//   return new(std::nothrow) GGEMSVoxelizedNavigator;
// }

// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////

// void set_phantom_name_ggems_voxelized_navigator(GGEMSVoxelizedNavigator* voxelized_navigator, char const* phantom_navigator_name)
// {
//   voxelized_navigator->SetNavigatorName(phantom_navigator_name);
// }

// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////

// void set_phantom_file_ggems_voxelized_navigator(GGEMSVoxelizedNavigator* voxelized_navigator, char const* phantom_filename)
// {
//   voxelized_navigator->SetPhantomFile(phantom_filename);
// }

// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////

// void set_range_to_material_filename_ggems_voxelized_navigator(GGEMSVoxelizedNavigator* voxelized_navigator, char const* range_data_filename)
// {
//   voxelized_navigator->SetRangeToMaterialFile(range_data_filename);
// }

// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////

// void set_geometry_tolerance_ggems_voxelized_navigator(GGEMSVoxelizedNavigator* voxelized_navigator, GGfloat const distance, char const* unit)
// {
//   voxelized_navigator->SetGeometryTolerance(distance, unit);
// }

// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////

// void set_position_ggems_voxelized_navigator(GGEMSVoxelizedNavigator* voxelized_navigator, GGfloat const position_x, GGfloat const position_y, GGfloat const position_z, char const* unit)
// {
//   voxelized_navigator->SetPosition(position_x, position_y, position_z, unit);
// }
