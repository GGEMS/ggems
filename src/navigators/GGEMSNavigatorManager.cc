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
  \file GGEMSNavigatorManager.cc

  \brief GGEMS class handling the navigators (detector + phantom) in GGEMS

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday February 11, 2020
*/

#include "GGEMS/physics/GGEMSRangeCutsManager.hh"

#include "GGEMS/geometries/GGEMSSolid.hh"

#include "GGEMS/sources/GGEMSSourceManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSNavigatorManager::GGEMSNavigatorManager(void)
: navigators_(0),
  world_(nullptr)
{
  GGcout("GGEMSNavigatorManager", "GGEMSNavigatorManager", 3) << "Allocation of GGEMS navigator manager..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSNavigatorManager::~GGEMSNavigatorManager(void)
{
  // Freeing memory
  navigators_.clear();

  GGcout("GGEMSNavigatorManager", "~GGEMSNavigatorManager", 3) << "Deallocation of GGEMS navigator manager..." << GGendl;

  if (world_) {
    delete world_;
    world_ = nullptr;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigatorManager::Store(GGEMSNavigator* navigator)
{
  GGcout("GGEMSNavigatorManager", "Store", 3) << "Storing new navigator in GGEMS..." << GGendl;

  // Set index of navigator and store the pointer
  navigator->SetNavigatorID(navigators_.size());
  navigators_.emplace_back(navigator);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigatorManager::StoreWorld(GGEMSWorld* world)
{
  GGcout("GGEMSNavigatorManager", "StoreWorld", 3) << "Storing world in GGEMS..." << GGendl;
  world_ = world;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigatorManager::SaveResults(void) const
{
  for (auto&& i : navigators_) i->SaveResults();

  // Checking if world exists
  if (world_) world_->SaveResults();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigatorManager::Initialize(void) const
{
  GGcout("GGEMSNavigatorManager", "Initialize", 3) << "Initializing the GGEMS navigator(s)..." << GGendl;

  // A navigator must be declared
  if (navigators_.empty()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "A navigator (detector or phantom) has to be declared!!!";
    GGEMSMisc::ThrowException("GGEMSNavigatorManager", "Initialize", oss.str());
  }

  // Initialization of world
  if (world_) world_->Initialize();

  // Initialization of phantoms
  for (auto&& i : navigators_) i->Initialize();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigatorManager::PrintInfos(void) const
{
  GGcout("GGEMSNavigatorManager", "PrintInfos", 0) << "Printing infos about phantom navigators" << GGendl;
  GGcout("GGEMSNavigatorManager", "PrintInfos", 0) << "Number of navigator(s): " << navigators_.size() << GGendl;

  for (auto&&i : navigators_) i->PrintInfos();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigatorManager::FindSolid(void) const
{
  for (auto&& n : navigators_) n->ParticleSolidDistance();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigatorManager::ProjectToSolid(void) const
{
  for (auto&& n : navigators_) n->ProjectToSolid();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigatorManager::TrackThroughSolid(void) const
{
  for (auto&& n : navigators_) n->TrackThroughSolid();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigatorManager::WorldTracking(void) const
{
  // Checking if world exists
  if (world_) world_->Tracking();
}
