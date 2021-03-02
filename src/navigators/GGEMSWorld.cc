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
  \file GGEMSWorld.cc

  \brief GGEMS class handling global world (space between navigators) in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 11, 2021
*/

#include "GGEMS/navigators/GGEMSNavigatorManager.hh"
#include "GGEMS/navigators/GGEMSWorld.hh"
#include "GGEMS/tools/GGEMSPrint.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSWorld::GGEMSWorld()
{
  GGcout("GGEMSWorld", "GGEMSWorld", 3) << "Allocation of GGEMSWorld..." << GGendl;

  GGEMSNavigatorManager::GetInstance().StoreWorld(this);

  dimensions_.x = 0;
  dimensions_.y = 0;
  dimensions_.z = 0;

  sizes_.x = -1.0f;
  sizes_.y = -1.0f;
  sizes_.z = -1.0f;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSWorld::~GGEMSWorld(void)
{
  GGcout("GGEMSWorld", "~GGEMSWorld", 3) << "Deallocation of GGEMSWorld..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSWorld::CheckParameters(void) const
{
  GGcout("GGEMSWorld", "CheckParameters", 3) << "Checking the mandatory parameters..." << GGendl;

  // Checking world dimensions
  if (dimensions_.x == 0 || dimensions_.y == 0 || dimensions_.z == 0) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Dimensions of world have to be set";
    GGEMSMisc::ThrowException("GGEMSWorld", "CheckParameters", oss.str());
  }

  // Checking elements size in world
  if (sizes_.x < 0.0 || sizes_.y < 0.0 || sizes_.z < 0.0) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Size of elements in world";
    GGEMSMisc::ThrowException("GGEMSWorld", "CheckParameters", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSWorld::SetDimension(GGint const& dimension_x, GGint const& dimension_y, GGint const& dimension_z)
{
  dimensions_.x = dimension_x;
  dimensions_.y = dimension_y;
  dimensions_.z = dimension_z;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSWorld::SetElementSize(GGfloat const& size_x, GGfloat const& size_y, GGfloat const& size_z, std::string const& unit)
{
  sizes_.x = DistanceUnit(size_x, unit);
  sizes_.y = DistanceUnit(size_y, unit);
  sizes_.z = DistanceUnit(size_z, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSWorld::Initialize(void)
{
  GGcout("GGEMSWorld", "Initialize", 3) << "Initializing a GGEMS world..." << GGendl;

  // Checking the parameters of world
  CheckParameters();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSWorld* create_ggems_world(void)
{
  return new(std::nothrow) GGEMSWorld();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_dimension_ggems_world(GGEMSWorld* world, GGint const dimension_x, GGint const dimension_y, GGint const dimension_z)
{
  world->SetDimension(dimension_x, dimension_y, dimension_z);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_size_ggems_world(GGEMSWorld* world, GGfloat const size_x, GGfloat const size_y, GGfloat const size_z, char const* unit)
{
  world->SetElementSize(size_x, size_y, size_z, unit);
}
