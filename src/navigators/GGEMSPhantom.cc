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
  \file GGEMSPhantom.cc

  \brief GGEMS class initializing a phantom and setting type of navigator

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \date Thrusday October 15, 2020
*/

#include "GGEMS/navigators/GGEMSPhantom.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/navigators/GGEMSNavigator.hh"
#include "GGEMS/navigators/GGEMSNavigatorManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPhantom::GGEMSPhantom(std::string const& phantom_name, std::string const& phantom_type)
{
  GGcout("GGEMSPhantom", "GGEMSPhantom", 3) << "Allocation of GGEMSPhantom..." << GGendl;

  // Allocation of navigator depending of type
  new GGEMSNavigator();

  // Get pointer on last created navigator
  navigator_ = GGEMSNavigatorManager::GetInstance().GetLastNavigator();
  navigator_.lock()->SetNavigatorName(phantom_name);
  navigator_.lock()->SetNavigatorType(phantom_type);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPhantom::~GGEMSPhantom(void)
{
  GGcout("GGEMSPhantom", "~GGEMSPhantom", 3) << "Deallocation of GGEMSPhantom..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPhantom::SetPosition(GGfloat const& position_x, GGfloat const& position_y, GGfloat const& position_z, std::string const& unit)
{
  navigator_.lock()->SetPosition(position_x, position_y, position_z, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSPhantom::SetVoxelizedPhantomFile(std::string const& filename, std::string const& range_data_filename)
{
  navigator_.lock()->SetVoxelizedNavigatorFile(filename, range_data_filename);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSPhantom* create_ggems_phantom(char const* phantom_name, char const* phantom_type)
{
  return new(std::nothrow) GGEMSPhantom(phantom_name, phantom_type);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_phantom_file_ggems_phantom(GGEMSPhantom* phantom, char const* phantom_filename, char const* range_data_filename)
{
  phantom->SetVoxelizedPhantomFile(phantom_filename, range_data_filename);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_position_ggems_phantom(GGEMSPhantom* phantom, GGfloat const position_x, GGfloat const position_y, GGfloat const position_z, char const* unit)
{
  phantom->SetPosition(position_x, position_y, position_z, unit);
}
