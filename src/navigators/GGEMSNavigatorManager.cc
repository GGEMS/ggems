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
: navigators_(nullptr),
  number_of_navigators_(0),
  world_(nullptr)
{
  GGcout("GGEMSNavigatorManager", "GGEMSNavigatorManager", 3) << "GGEMSNavigatorManager creating..." << GGendl;

  GGcout("GGEMSNavigatorManager", "GGEMSNavigatorManager", 3) << "GGEMSNavigatorManager created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSNavigatorManager::~GGEMSNavigatorManager(void)
{
  GGcout("GGEMSNavigatorManager", "~GGEMSNavigatorManager", 3) << "GGEMSNavigatorManager erasing..." << GGendl;

  if (navigators_) {
    delete[] navigators_;
    navigators_ = nullptr;
  }

  if (world_) {
    delete world_;
    world_ = nullptr;
  }

  GGcout("GGEMSNavigatorManager", "~GGEMSNavigatorManager", 3) << "GGEMSNavigatorManager erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigatorManager::Clean(void)
{
  GGcout("GGEMSNavigatorManager", "Clean", 3) << "GGEMSNavigatorManager cleaning..." << GGendl;

  GGcout("GGEMSNavigatorManager", "Clean", 3) << "GGEMSNavigatorManager cleaned!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigatorManager::Store(GGEMSNavigator* navigator)
{
  GGcout("GGEMSNavigatorManager", "Store", 3) << "Storing new navigator in GGEMS..." << GGendl;

  // Set index of navigator and store the pointer
  navigator->SetNavigatorID(number_of_navigators_);

  if (number_of_navigators_ == 0) {
    navigators_ = new GGEMSNavigator*[1];
    navigators_[0] = navigator;
  }
  else {
    GGEMSNavigator** tmp = new GGEMSNavigator*[number_of_navigators_+1];
    for (GGsize i = 0; i < number_of_navigators_; ++i) {
      tmp[i] = navigators_[i];
    }

    tmp[number_of_navigators_] = navigator;

    delete[] navigators_;
    navigators_ = tmp;
  }

  number_of_navigators_++;
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
  for (GGsize i = 0; i < number_of_navigators_; ++i) {
    navigators_[i]->SaveResults();
  }

  // Checking if world exists
  if (world_) world_->SaveResults();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigatorManager::Initialize(bool const& is_tracking) const
{
  GGcout("GGEMSNavigatorManager", "Initialize", 3) << "Initializing the GGEMS navigator(s)..." << GGendl;

  // A navigator must be declared
  if (number_of_navigators_ == 0) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "A navigator (detector or phantom) has to be declared!!!";
    GGEMSMisc::ThrowException("GGEMSNavigatorManager", "Initialize", oss.str());
  }

  // Initialization of world
  if (world_) {
    if (is_tracking) world_->EnableTracking();
    world_->Initialize();
  }

  // Initialization of phantoms
  for (GGsize i = 0; i < number_of_navigators_; ++i) {
    if (is_tracking) navigators_[i]->EnableTracking();
    navigators_[i]->Initialize();
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigatorManager::PrintInfos(void) const
{
  GGcout("GGEMSNavigatorManager", "PrintInfos", 0) << "Printing infos about phantom navigators" << GGendl;
  GGcout("GGEMSNavigatorManager", "PrintInfos", 0) << "Number of navigator(s): " << number_of_navigators_ << GGendl;

  for (GGsize i = 0; i < number_of_navigators_; ++i) {
    navigators_[i]->PrintInfos();
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigatorManager::FindSolid(GGsize const& thread_index) const
{
  for (GGsize i = 0; i < number_of_navigators_; ++i) {
    navigators_[i]->ParticleSolidDistance(thread_index);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigatorManager::ProjectToSolid(GGsize const& thread_index) const
{
  for (GGsize i = 0; i < number_of_navigators_; ++i) {
    navigators_[i]->ProjectToSolid(thread_index);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigatorManager::TrackThroughSolid(GGsize const& thread_index) const
{
  for (GGsize i = 0; i < number_of_navigators_; ++i) {
    navigators_[i]->TrackThroughSolid(thread_index);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigatorManager::WorldTracking(GGsize const& thread_index) const
{
  // Checking if world exists
  if (world_) world_->Tracking(thread_index);
}
