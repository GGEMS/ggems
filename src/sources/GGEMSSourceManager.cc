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
  \file GGEMSSourceManager.cc

  \brief GGEMS class handling the source(s)

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thursday January 16, 2020
*/

#include "GGEMS/sources/GGEMSSourceManager.hh"
#include "GGEMS/physics/GGEMSPrimaryParticles.hh"
#include "GGEMS/randoms/GGEMSPseudoRandomGenerator.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSourceManager::GGEMSSourceManager(void)
: sources_(nullptr),
  number_of_sources_(0)
{
  GGcout("GGEMSSourceManager", "GGEMSSourceManager", 3) << "Allocation of GGEMSSourceManager..." << GGendl;

  // Allocation of Particle object
  particles_ = new GGEMSParticles;

  // Allocation of pseudo random generator object
  pseudo_random_generator_ = new GGEMSPseudoRandomGenerator;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSourceManager::~GGEMSSourceManager(void)
{
  GGcout("GGEMSSourceManager", "~GGEMSSourceManager", 3) << "Deallocation of GGEMSSourceManager..." << GGendl;

  // Freeing memory
  delete particles_;
  delete pseudo_random_generator_;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSourceManager::Store(GGEMSSource* source)
{
  GGcout("GGEMSSourceManager", "Store", 3) << "Storing new source in GGEMS source manager..." << GGendl;

  if (number_of_sources_ == 0) {
    sources_ = new GGEMSSource*[1];
    sources_[0] = source;
  }
  else {
    GGEMSSource** tmp = new GGEMSSource*[number_of_sources_+1];
    for (std::size_t i = 0; i < number_of_sources_; ++i) {
      tmp[i] = sources_[i];
    }

    tmp[number_of_sources_] = source;

    delete[] sources_;
    sources_ = tmp;
  }

  number_of_sources_++;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSourceManager::PrintInfos(void) const
{
  GGcout("GGEMSSourceManager", "PrintInfos", 0) << "Printing infos about sources" << GGendl;
  GGcout("GGEMSSourceManager", "PrintInfos", 0) << "Number of source(s): " << number_of_sources_ << GGendl;

  // Printing infos about each source
  for (GGsize i = 0; i < number_of_sources_; ++i ) {
    sources_[i]->PrintInfos();
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSourceManager::Initialize(GGuint const& seed, bool const& is_tracking, GGint const& particle_tracking_id) const
{
  GGcout("GGEMSSourceManager", "Initialize", 3) << "Initializing the GGEMS source(s)..." << GGendl;

  // Checking number of source, if 0 kill the simulation
  if (number_of_sources_ == 0) {
    GGEMSMisc::ThrowException("GGEMSSourceManager", "Initialize", "You have to define a source before to run GGEMS!!!");
  }

  // Initialization of particle stack and random stack
  particles_->Initialize();
  GGcout("GGEMSSourceManager", "Initialize", 0) << "Initialization of particles OK" << GGendl;

  pseudo_random_generator_->Initialize(seed);
  GGcout("GGEMSSourceManager", "Initialize", 0) << "Initialization of GGEMS pseudo random generator OK" << GGendl;

  // Initialization of sources
  for (GGsize i = 0; i < number_of_sources_; ++i) {
    sources_[i]->Initialize();
  }

  // If tracking activated, set the particle id to track
  if (is_tracking) {
    // Get the OpenCL manager
    GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

    // Loop over activated device
    for (GGsize i = 0; i < opencl_manager.GetNumberOfActivatedDevice(); ++i) {
      // Get pointer on OpenCL device for particles
      GGEMSPrimaryParticles* primary_particles_device = opencl_manager.GetDeviceBuffer<GGEMSPrimaryParticles>(particles_->GetPrimaryParticles(i), sizeof(GGEMSPrimaryParticles), i);

      primary_particles_device->particle_tracking_id = particle_tracking_id;

      // Release the pointer
      opencl_manager.ReleaseDeviceBuffer(particles_->GetPrimaryParticles(i), primary_particles_device, i);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

bool GGEMSSourceManager::IsAlive(GGsize const& index) const
{
  // Check if all particles are DEAD in OpenCL particle buffer
  return particles_->IsAlive(index);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSourceManager* get_instance_ggems_source_manager(void)
{
  return &GGEMSSourceManager::GetInstance();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void initialize_source_manager(GGEMSSourceManager* source_manager, GGuint const& seed)
{
  source_manager->Initialize(seed);
}
