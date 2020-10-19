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
  \file GGEMSSystem.hh

  \brief GGEMS class managing detector system in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \date Monday October 19, 2020
*/

#include "GGEMS/systems/GGEMSSystem.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/navigators/GGEMSNavigator.hh"
#include "GGEMS/navigators/GGEMSNavigatorManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSystem::GGEMSSystem(std::string const& system_name)
{
  GGcout("GGEMSSystem", "GGEMSSystem", 3) << "Allocation of GGEMSSystem..." << GGendl;

  // Allocation of navigator depending of type
  new GGEMSNavigator();

  // Get pointer on last created navigator
  navigator_ = GGEMSNavigatorManager::GetInstance().GetLastNavigator();
  navigator_.lock()->SetNavigatorName(system_name);
  //navigator_.lock()->SetNavigatorType("voxelized");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSystem::~GGEMSSystem(void)
{
  GGcout("GGEMSSystem", "~GGEMSSystem", 3) << "Deallocation of GGEMSSystem..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*void GGEMSSystem::SetPosition(GGfloat const& position_x, GGfloat const& position_y, GGfloat const& position_z, std::string const& unit)
{
  ;//navigator_.lock()->SetPosition(position_x, position_y, position_z, unit);
}*/
