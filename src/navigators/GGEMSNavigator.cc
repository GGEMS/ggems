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
#include "GGEMS/geometries/GGEMSVoxelizedSolid.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSNavigator::GGEMSNavigator(std::string const& navigator_name)
: navigator_name_(navigator_name),
  position_xyz_({0.0f, 0.0f, 0.0f}),
  rotation_xyz_({0.0f, 0.0f, 0.0f}),
  navigator_id_(-1),
  is_update_pos_(false),
  is_update_rot_(false)
{
  GGcout("GGEMSNavigator", "GGEMSNavigator", 3) << "Allocation of GGEMSNavigator..." << GGendl;

  // Store the phantom navigator in phantom navigator manager
  GGEMSNavigatorManager::GetInstance().Store(this);

  // Allocation of materials
  materials_.reset(new GGEMSMaterials());

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

void GGEMSNavigator::SetPosition(GGfloat const& position_x, GGfloat const& position_y, GGfloat const& position_z, std::string const& unit)
{
  is_update_pos_ = true;

  position_xyz_.s0 = DistanceUnit(position_x, unit);
  position_xyz_.s1 = DistanceUnit(position_y, unit);
  position_xyz_.s2 = DistanceUnit(position_z, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::SetRotation(GGfloat const& rx, GGfloat const& ry, GGfloat const& rz, std::string const& unit)
{
  is_update_rot_ = true;

  rotation_xyz_.s0 = AngleUnit(rx, unit);
  rotation_xyz_.s1 = AngleUnit(ry, unit);
  rotation_xyz_.s2 = AngleUnit(rz, unit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::SetNavigatorID(std::size_t const& navigator_id)
{
  navigator_id_ = navigator_id;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::CheckParameters(void) const
{
  GGcout("GGEMSNavigator", "CheckParameters", 3) << "Checking the mandatory parameters..." << GGendl;

  // Checking id of the navigator
  if (navigator_id_ == -1) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Id of the navigator is not set!!!";
    GGEMSMisc::ThrowException("GGEMSNavigator", "CheckParameters", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::Initialize(void)
{
  GGcout("GGEMSNavigator", "Initialize", 3) << "Initializing a GGEMS navigator..." << GGendl;

  // Checking the parameters of phantom
  CheckParameters();

  // Loading the materials and building tables to OpenCL device and converting cuts
  materials_->Initialize();

  // Initialization of electromagnetic process and building cross section tables for each particles and materials
  cross_sections_->Initialize(materials_.get());
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

DurationNano GGEMSNavigator::GetAllKernelParticleSolidDistanceTimer(void) const
{
  DurationNano total_duration = GGEMSChrono::Zero();
  for (auto&& i : solid_) total_duration += i->GetKernelParticleSolidDistanceTimer();
  return total_duration;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

DurationNano GGEMSNavigator::GetAllKernelProjectToSolidTimer(void) const
{
  DurationNano total_duration = GGEMSChrono::Zero();
  for (auto&& i : solid_) total_duration += i->GetKernelProjectToSolidTimer();
  return total_duration;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

DurationNano GGEMSNavigator::GetAllKernelTrackThroughSolidTimer(void) const
{
  DurationNano total_duration = GGEMSChrono::Zero();
  for (auto&& i : solid_) total_duration += i->GetKernelTrackThroughSolidTimer();
  return total_duration;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::ParticleSolidDistance(void) const
{
  for (auto&& i : solid_) i->ParticleSolidDistance();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::ProjectToSolid(void) const
{
  for (auto&& i : solid_) i->ProjectToSolid();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::TrackThroughSolid(void) const
{
  for (auto&& i : solid_) i->TrackThroughSolid(cross_sections_, materials_);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSNavigator::PrintInfos(void) const
{
  GGcout("GGEMSNavigator", "PrintInfos", 0) << GGendl;
  GGcout("GGEMSNavigator", "PrintInfos", 0) << "GGEMSNavigator Infos:" << GGendl;
  GGcout("GGEMSNavigator", "PrintInfos", 0) << "---------------------" << GGendl;
  GGcout("GGEMSNavigator", "PrintInfos", 0) << "* Navigator name: " << navigator_name_ << GGendl;
  for (std::size_t i = 0; i < solid_.size(); ++i) {
    solid_.at(i)->PrintInfos();
  }
  materials_->PrintInfos();
  GGcout("GGEMSNavigator", "PrintInfos", 0) << GGendl;
}
