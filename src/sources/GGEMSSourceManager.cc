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

#include <algorithm>
#include <sstream>

#include "GGEMS/sources/GGEMSSourceManager.hh"

#include "GGEMS/tools/GGEMSTools.hh"

#include "GGEMS/physics/GGEMSParticles.hh"
#include "GGEMS/physics/GGEMSPrimaryParticles.hh"

#include "GGEMS/randoms/GGEMSPseudoRandomGenerator.hh"

#include "GGEMS/global/GGEMSManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSourceManager::GGEMSSourceManager(void)
: sources_(0),
  particles_(nullptr),
  pseudo_random_generator_(nullptr)
{
  GGcout("GGEMSSourceManager", "GGEMSSourceManager", 3) << "Allocation of GGEMSSourceManager..." << GGendl;

  // Allocation of Particle object
  particles_.reset(new GGEMSParticles());

  // Allocation of pseudo random generator object
  pseudo_random_generator_.reset(new GGEMSPseudoRandomGenerator());
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSourceManager::~GGEMSSourceManager(void)
{
  // Deleting source
  sources_.clear();

  GGcout("GGEMSSourceManager", "~GGEMSSourceManager", 3) << "Deallocation of GGEMSSourceManager..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSourceManager::Store(GGEMSSource* source)
{
  GGcout("GGEMSSourceManager", "Store", 3) << "Storing new source in GGEMS source manager..." << GGendl;
  sources_.emplace_back(source);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSourceManager::PrintInfos(void) const
{
  GGcout("GGEMSSourceManager", "PrintInfos", 0) << "Printing infos about sources" << GGendl;
  GGcout("GGEMSSourceManager", "PrintInfos", 0) << "Number of source(s): " << sources_.size() << GGendl;

  // Printing infos about each navigator
  for (auto&& i : sources_) i->PrintInfos();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSourceManager::Initialize(void) const
{
  GGcout("GGEMSSourceManager", "Initialize", 3) << "Initializing the GGEMS source(s)..." << GGendl;

  // Checking number of source, if 0 kill the simulation
  if (sources_.empty()) {
    GGEMSMisc::ThrowException("GGEMSSourceManager", "Initialize", "You have to define a source before to run GGEMS!!!");
  }

  // Initialization of particle stack and random stack
  particles_->Initialize();
  GGcout("GGEMSSourceManager", "Initialize", 0) << "Initialization of particles OK" << GGendl;

  pseudo_random_generator_->Initialize();
  GGcout("GGEMSSourceManager", "Initialize", 0) << "Initialization of GGEMS pseudo random generator OK" << GGendl;

  // Initialization of sources
  for (auto&& i : sources_) i->Initialize();

  // If tracking activated, set the particle id to track
  if (GGEMSManager::GetInstance().IsTrackingVerbose()) {
    // Get the OpenCL manager
    GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

    // Get pointer on OpenCL device for particles
    GGEMSPrimaryParticles* primary_particles_device = opencl_manager.GetDeviceBuffer<GGEMSPrimaryParticles>(particles_->GetPrimaryParticles(), sizeof(GGEMSPrimaryParticles));

    primary_particles_device->particle_tracking_id = GGEMSManager::GetInstance().GetParticleTrackingID();

    // Release the pointer
    opencl_manager.ReleaseDeviceBuffer(particles_->GetPrimaryParticles(), primary_particles_device);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

bool GGEMSSourceManager::IsAlive(void) const
{
  // Check if all particles are DEAD in OpenCL particle buffer
  return particles_->IsAlive();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSourceManager::PrintKernelElapsedTime(void) const
{
  DurationNano total_duration = GGEMSChrono::Zero();
  for (auto&& i : sources_) {
    total_duration += i->GetKernelGetPrimariesTimer();
  }

  GGEMSChrono::DisplayTime(total_duration, "Get Primaries");
}
