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
#include "GGEMS/physics/GGEMSPrimaryParticlesStack.hh"
#include "GGEMS/randoms/GGEMSRandomStack.hh"
#include "GGEMS/physics/GGEMSParticles.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSource::GGEMSSource(std::string const& source_name)
: source_name_(source_name),
  number_of_particles_(0),
  number_of_particles_in_batch_(0),
  particle_type_(99)
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

void GGEMSSource::SetPosition(GGfloat const& pos_x, GGfloat const& pos_y, GGfloat const& pos_z, std::string const& unit)
{
  geometry_transformation_->SetTranslation(MakeFloat3(DistanceUnit(pos_x, unit), DistanceUnit(pos_y, unit), DistanceUnit(pos_z, unit)));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSource::SetNumberOfParticles(GGulong const& number_of_particles)
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
  geometry_transformation_->SetRotation(MakeFloat3(
    AngleUnit(rx, unit),
    AngleUnit(ry, unit),
    AngleUnit(rz, unit))
  );
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

void GGEMSSource::CheckMemoryForParticles(void) const
{
  // By security the particle allocation by batch should not exceed 10% of
  // RAM memory

  // Compute the RAM memory percentage allocated for primary particles
  GGfloat const kRAMParticles = static_cast<GGfloat>(sizeof(GGEMSPrimaryParticles)) + static_cast<GGfloat>(sizeof(GGEMSRandom)) + 4.0f; // 4 bytes is for particle tracking id

  // Getting the RAM memory on activated device
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  GGfloat const kMaxRAM = static_cast<GGfloat>(opencl_manager.GetMaxRAMMemoryOnActivatedContext());

  // Computing the ratio of used RAM memory on device
  GGfloat const kMaxRatioUsedRAM = kRAMParticles / kMaxRAM;

  // Computing a theoric max. number of particles depending on activated
  // device and advice this number to the user. 10% of RAM memory for particles
  GGulong const kTheoricMaxNumberOfParticles = static_cast<GGulong>(0.1 * kMaxRAM / (kRAMParticles/MAXIMUM_PARTICLES));

  if (kMaxRatioUsedRAM > 0.1) { // Printing warning
    GGwarn("GGEMSSourceManager", "CheckMemoryForParticles", 0) << "Warning!!! The number of particles in a batch defined during GGEMS compilation is maybe to high. We recommand to not use more than 10% of RAM memory for particles allocation. Your theoric number of particles is " << kTheoricMaxNumberOfParticles << ". Recompile GGEMS " << "with this number of particles is recommended." << GGendl;
  }
  else { // Printing theoric number of particle
    GGcout("GGEMSSourceManager", "CheckMemoryForParticles", 0) << "The number of particles in a batch defined during the compilation is correct. We recommend to not used more than 10% of memory for particle allocation. Your theoric number of particles is " << kTheoricMaxNumberOfParticles << ". Recompile GGEMS with this number of particles is recommended" << GGendl;
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
  std::size_t const kNumberOfBatchs = static_cast<std::size_t>(std::ceil(static_cast<GGfloat>(number_of_particles_) / MAXIMUM_PARTICLES));

  // Resizing vector storing the number of particles in batch
  number_of_particles_in_batch_.resize(kNumberOfBatchs, 0);

  // Computing the number of simulated particles in batch
  if (kNumberOfBatchs == 1) {
    number_of_particles_in_batch_[0] = number_of_particles_;
  }
  else {
    for (auto&& i : number_of_particles_in_batch_) {
      i = number_of_particles_ / kNumberOfBatchs;
    }

    // Adding the remaing particles
    for (std::size_t i = 0; i < number_of_particles_ % kNumberOfBatchs; ++i) {
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

  // Checking the RAM memory for particle and propose a new MAXIMUM_PARTICLE number
  CheckMemoryForParticles();

  // Organize the particles in batch
  OrganizeParticlesInBatch();
  GGcout("GGEMSSource", "Initialize", 0) << "Particles arranged in batch OK" << GGendl;
}
