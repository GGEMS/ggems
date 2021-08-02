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
  GGcout("GGEMSSourceManager", "GGEMSSourceManager", 3) << "GGEMSSourceManager creating..." << GGendl;

  // Allocation of Particle object
  particles_ = new GGEMSParticles;

  // Allocation of pseudo random generator object
  pseudo_random_generator_ = new GGEMSPseudoRandomGenerator;

  GGcout("GGEMSSourceManager", "GGEMSSourceManager", 3) << "GGEMSSourceManager created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSourceManager::~GGEMSSourceManager(void)
{
  GGcout("GGEMSSourceManager", "~GGEMSSourceManager", 3) << "GGEMSSourceManager erasing..." << GGendl;

  GGcout("GGEMSSourceManager", "~GGEMSSourceManager", 3) << "GGEMSSourceManager erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSourceManager::Clean(void)
{
  GGcout("GGEMSSourceManager", "Clean", 3) << "GGEMSSourceManager cleaning..." << GGendl;

  if (particles_) {
    delete particles_;
    particles_ = nullptr;
  }

  if (pseudo_random_generator_) {
    delete pseudo_random_generator_;
    pseudo_random_generator_ = nullptr;
  }

  GGcout("GGEMSSourceManager", "Clean", 3) << "GGEMSSourceManager cleaned!!!" << GGendl;
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
    for (GGsize i = 0; i < number_of_sources_; ++i) {
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
  for (GGsize i = 0; i < number_of_sources_; ++i ) sources_[i]->PrintInfos();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGsize GGEMSSourceManager::GetTotalNumberOfBatchs(void) const
{
  // Getting the number of activated device
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  GGsize number_of_activated_devices = opencl_manager.GetNumberOfActivatedDevice();

  // Loop over number of sources
  GGsize total_number_of_batchs = 0;
  for (GGsize i = 0; i < number_of_sources_; ++i) {
    // Loop over the number of activated devices
    for (GGsize j = 0; j < number_of_activated_devices; ++j) {
      total_number_of_batchs += sources_[i]->GetNumberOfBatchs(j);
    }
  }

  return total_number_of_batchs;
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
  for (GGsize i = 0; i < number_of_sources_; ++i) sources_[i]->Initialize(is_tracking);

  // If tracking activated, set the particle id to track
  if (is_tracking) {
    // Get the OpenCL manager
    GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

    // Loop over activated device
    for (GGsize i = 0; i < opencl_manager.GetNumberOfActivatedDevice(); ++i) {
      // Get pointer on OpenCL device for particles
      GGEMSPrimaryParticles* primary_particles_device = opencl_manager.GetDeviceBuffer<GGEMSPrimaryParticles>(particles_->GetPrimaryParticles(i), CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, sizeof(GGEMSPrimaryParticles), i);

      primary_particles_device->particle_tracking_id = particle_tracking_id;

      // Release the pointer
      opencl_manager.ReleaseDeviceBuffer(particles_->GetPrimaryParticles(i), primary_particles_device, i);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

bool GGEMSSourceManager::IsAlive(GGsize const& thread_index) const
{
  // Check if all particles are DEAD in OpenCL particle buffer
  return particles_->IsAlive(thread_index);
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

void initialize_source_manager(GGEMSSourceManager* source_manager, GGuint const seed)
{
  source_manager->Initialize(seed);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void print_infos_source_manager(GGEMSSourceManager* source_manager)
{
  source_manager->PrintInfos();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void clean_source_manager(GGEMSSourceManager* source_manager)
{
  source_manager->Clean();
}
