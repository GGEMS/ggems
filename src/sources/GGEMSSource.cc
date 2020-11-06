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
  \file GGEMSSource.cc

  \brief GGEMS mother class for the source

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday October 15, 2019
*/

#include <sstream>
#include <algorithm>

#include "GGEMS/global/GGEMSConstants.hh"
#include "GGEMS/sources/GGEMSSourceManager.hh"
#include "GGEMS/maths/GGEMSGeometryTransformation.hh"
#include "GGEMS/tools/GGEMSTools.hh"
#include "GGEMS/physics/GGEMSPrimaryParticles.hh"
#include "GGEMS/randoms/GGEMSRandom.hh"
#include "GGEMS/physics/GGEMSParticles.hh"
#include "GGEMS/global/GGEMSManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSource::GGEMSSource(std::string const& source_name)
: source_name_(source_name),
  number_of_particles_(0),
  number_of_particles_in_batch_(0),
  particle_type_(99),
  tracking_kernel_option_("")
{
  GGcout("GGEMSSource", "GGEMSSource", 3) << "Allocation of GGEMSSource..." << GGendl;

  // Allocation of geometry transformation
  geometry_transformation_.reset(new GGEMSGeometryTransformation());

  // Store the source in source manager
  GGEMSSourceManager::GetInstance().Store(this);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSource::~GGEMSSource(void)
{
  GGcout("GGEMSSource", "~GGEMSSource", 3) << "Deallocation of GGEMSSource..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSource::EnableTracking(void)
{
  tracking_kernel_option_ = "-DGGEMS_TRACKING";
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSource::SetPosition(GGfloat const& pos_x, GGfloat const& pos_y, GGfloat const& pos_z, std::string const& unit)
{
  geometry_transformation_->SetTranslation({DistanceUnit(pos_x, unit), DistanceUnit(pos_y, unit), DistanceUnit(pos_z, unit)});
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSource::SetNumberOfParticles(GGlong const& number_of_particles)
{
  number_of_particles_ = number_of_particles;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSource::SetSourceParticleType(std::string const& particle_type)
{
  if (particle_type == "gamma") {
    particle_type_ = PHOTON;
  }
  else if (particle_type == "e-") {
    particle_type_ = ELECTRON;
  }
  else if (particle_type == "e+") {
    particle_type_ = POSITRON;
  }
  else
  {
    GGEMSMisc::ThrowException("GGEMSSourceManager", "SetParticleType", "Unknown particle!!!");
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSource::SetLocalAxis(GGfloat3 const& m0, GGfloat3 const& m1, GGfloat3 const& m2)
{
  geometry_transformation_->SetAxisTransformation(m0, m1, m2);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSource::SetRotation(GGfloat const& rx, GGfloat const& ry, GGfloat const& rz, std::string const& unit)
{
  geometry_transformation_->SetRotation({AngleUnit(rx, unit), AngleUnit(ry, unit), AngleUnit(rz, unit)});
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSource::CheckParameters(void) const
{
  GGcout("GGEMSSource", "CheckParameters", 3) << "Checking the mandatory parameters..." << GGendl;

  // Checking the type of particles
  if (particle_type_ == 99) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a particle type for the source:" << std::endl;
    oss << "    - Photon" << std::endl;
    oss << "    - Electron" << std::endl;
    oss << "    - Positron" << std::endl;
    GGEMSMisc::ThrowException("GGEMSSource", "CheckParameters", oss.str());
  }

  // Checking name of the source
  if (source_name_.empty()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a name for the source!!!";
    GGEMSMisc::ThrowException("GGEMSSource", "CheckParameters", oss.str());
  }

  // Checking the number of particles
  if (number_of_particles_ == 0) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a number of particles > 0!!!";
    GGEMSMisc::ThrowException("GGEMSSource", "CheckParameters", oss.str());
  }

  // Checking the position of particles
  GGfloat3 const kPosition = geometry_transformation_->GetPosition();
  if (kPosition.s[0] == std::numeric_limits<float>::min() || kPosition.s[1] == std::numeric_limits<float>::min() || kPosition.s[2] == std::numeric_limits<float>::min()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a position for the source!!!";
    GGEMSMisc::ThrowException("GGEMSSource", "CheckParameters", oss.str());
  }

  // Checking the rotation of particles
  GGfloat3 const kRotation = geometry_transformation_->GetRotation();
  if (kRotation.s[0] == std::numeric_limits<float>::min() || kRotation.s[1] == std::numeric_limits<float>::min() || kRotation.s[2] == std::numeric_limits<float>::min()) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a rotation for the source!!!";
    GGEMSMisc::ThrowException("GGEMSSource", "CheckParameters", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSource::OrganizeParticlesInBatch(void)
{
  GGcout("GGEMSSource", "OrganizeParticlesInBatch", 3) << "Organizing the number of particles in batch..." << GGendl;

  // Computing the number of batch depending on the number of simulated
  // particles and the maximum simulated particles defined during GGEMS
  // compilation
  std::size_t number_of_batchs = static_cast<std::size_t>(std::ceil(static_cast<GGfloat>(number_of_particles_) / MAXIMUM_PARTICLES));

  // Resizing vector storing the number of particles in batch
  number_of_particles_in_batch_.resize(number_of_batchs, 0);

  // Computing the number of simulated particles in batch
  if (number_of_batchs == 1) {
    number_of_particles_in_batch_[0] = number_of_particles_;
  }
  else {
    for (auto&& i : number_of_particles_in_batch_) {
      i = number_of_particles_ / number_of_batchs;
    }

    // Adding the remaing particles
    for (std::size_t i = 0; i < number_of_particles_ % number_of_batchs; ++i) {
      number_of_particles_in_batch_[i]++;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSource::Initialize(void)
{
  GGcout("GGEMSSource", "Initialize", 3) << "Initializing the a GGEMS source..." << GGendl;

  // Checking the parameters of Source
  CheckParameters();

  // Activate tracking
  if (GGEMSManager::GetInstance().IsTrackingVerbose()) EnableTracking();

  // Organize the particles in batch
  OrganizeParticlesInBatch();
  GGcout("GGEMSSource", "Initialize", 0) << "Particles arranged in batch OK" << GGendl;
}
